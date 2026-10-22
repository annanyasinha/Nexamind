"""
NexaMind MCP Client.

Provides a synchronous wrapper around the async MCP Python SDK client so
the existing synchronous NexaMindAgent can call MCP tools without
restructuring its execution loop.

Architecture (persistent session)
----------------------------------

NexaMindAgent.run()                 (synchronous caller)
        │
        ▼
MCPClient.call_tool()               (synchronous; submits to background loop)
        │
        ▼  asyncio.run_coroutine_threadsafe()
        │
   [background daemon thread — owns a dedicated asyncio event loop]
        │
        ▼
MCPClient._async_call_tool()        (async; reuses the open ClientSession)
        │
    stdio transport (kept open for the lifetime of MCPClient)
        │
        ▼
NexaMind MCP Server                 (subprocess, started ONCE at session open)
        │
        ▼
Existing Tool (GitHub / Web / YouTube)
        │
        ▼
Tool Result dict

The server subprocess and ClientSession are created once inside
``_open_session()`` and live until ``close()`` is called (or the process
exits — the thread is a daemon, so it is killed automatically).

Usage (standalone discovery test)
----------------------------------
    PYTHONPATH=src python -m core.mcp_client

"""

import asyncio
import concurrent.futures
import json
import sys
import os
import threading
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

# Ensure src/ is on the path when run directly.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from utils.logger import logger


# ---------------------------------------------------------------------------
# Server launch parameters
# ---------------------------------------------------------------------------

def _get_server_params() -> StdioServerParameters:
    """
    Build StdioServerParameters that tell the MCP client how to start the
    NexaMind MCP server subprocess.
    """
    return StdioServerParameters(
        command=sys.executable,          # same Python interpreter that runs the client
        args=["-m", "core.mcp_server"],  # module entrypoint
        env={
            **os.environ,               # forward ALL environment variables (API keys etc.)
            "PYTHONPATH": _SRC_DIR,
        },
    )


# ---------------------------------------------------------------------------
# Synchronous public interface
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Synchronous MCP client with a *persistent* server session.

    On first use a background daemon thread is started.  That thread owns a
    dedicated ``asyncio`` event loop which:

    1. Opens the stdio transport to the MCP server subprocess.
    2. Creates and initialises a ``ClientSession``.
    3. Runs ``loop.run_forever()`` to service subsequent coroutine submissions.

    All public methods (``list_tools``, ``call_tool``, …) submit async work to
    that loop via ``asyncio.run_coroutine_threadsafe`` and block until the
    result is ready.  This means:

    * The server subprocess is started **once** per ``MCPClient`` instance.
    * No per-call startup / teardown overhead.
    * The session is inherently single-threaded within the background loop, so
      no additional locking is required for correctness.

    Parameters
    ----------
    timeout_seconds:
        Maximum wall-clock seconds to wait for a single tool call (or session
        startup).  Defaults to 120 s (enough for GitHub indexing on first call).
    """

    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

        # --- Internal session state (touched only by background thread) ---
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None

        # --- Threading primitives ---
        self._start_lock = threading.Lock()          # guards _start_session()
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()              # set when session is up
        self._startup_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Tool discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict[str, Any]]:
        """
        Return metadata for every tool registered on the MCP server.

        Returns
        -------
        list of dict, each with keys: name, description, input_schema.
        """
        return self._run_async(self._async_list_tools())

    # ------------------------------------------------------------------
    # Tool invocation
    # ------------------------------------------------------------------

    def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call *tool_name* on the NexaMind MCP server.

        Parameters
        ----------
        tool_name:
            One of ``search_github_repository``, ``search_web``,
            ``search_youtube_transcript``.
        arguments:
            Dict matching the tool's JSON schema.

        Returns
        -------
        dict with at minimum ``output`` and ``sources`` keys, mirroring the
        format returned by the underlying NexaMind tool.

        Raises
        ------
        RuntimeError
            If the MCP server fails to start or the tool call errors.
        """
        logger.info(f"[MCPClient] Calling MCP tool '{tool_name}' | args={arguments}")

        try:
            result = self._run_async(self._async_call_tool(tool_name, arguments))
            logger.info(f"[MCPClient] Tool '{tool_name}' returned successfully.")
            return result
        except Exception as exc:
            logger.error(f"[MCPClient] Tool '{tool_name}' failed: {exc}")
            raise RuntimeError(
                f"MCP tool '{tool_name}' failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Convenience shortcuts
    # ------------------------------------------------------------------

    def search_github(self, query: str, repo_url: str) -> Dict[str, Any]:
        """Convenience wrapper around the ``search_github_repository`` MCP tool."""
        return self.call_tool(
            "search_github_repository",
            {"query": query, "repo_url": repo_url},
        )

    def search_web(self, query: str) -> Dict[str, Any]:
        """Convenience wrapper around the ``search_web`` MCP tool."""
        return self.call_tool("search_web", {"query": query})

    def search_youtube(
        self, query: str, video_url_or_id: str
    ) -> Dict[str, Any]:
        """Convenience wrapper around the ``search_youtube_transcript`` MCP tool."""
        return self.call_tool(
            "search_youtube_transcript",
            {"query": query, "video_url_or_id": video_url_or_id},
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """
        Gracefully shut down the persistent session and background thread.

        Safe to call multiple times.  After ``close()`` the client cannot be
        reused — create a new ``MCPClient`` instance if needed.
        """
        if self._loop is None or self._session is None:
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                self._close_session(), self._loop
            )
            future.result(timeout=10)
        except Exception as exc:
            logger.warning(f"[MCPClient] Error during session close: {exc}")
        finally:
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._session = None
            self._exit_stack = None

    # ------------------------------------------------------------------
    # Internal: session management
    # ------------------------------------------------------------------

    def _ensure_started(self) -> None:
        """
        Start the background thread and open the MCP session on first call.
        Subsequent calls return immediately (fast path: ``_session is not None``).
        """
        # Fast path — session already open.
        if self._session is not None:
            return

        with self._start_lock:
            # Re-check under the lock to handle concurrent callers.
            if self._session is not None:
                return

            self._loop = asyncio.new_event_loop()
            self._ready.clear()
            self._startup_error = None

            self._thread = threading.Thread(
                target=self._run_loop,
                daemon=True,
                name="nexamind-mcp-session",
            )
            self._thread.start()

            # Block until the session is ready or startup fails.
            if not self._ready.wait(timeout=30):
                raise RuntimeError(
                    "MCP session did not start within 30 seconds"
                )
            if self._startup_error:
                raise self._startup_error

    def _run_loop(self) -> None:
        """Entry point for the background session thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._open_session())
        # Only enter run_forever() if startup succeeded.
        if self._startup_error is None:
            self._loop.run_forever()

    async def _open_session(self) -> None:
        """
        Open the stdio transport and initialise a ``ClientSession``.

        Uses ``AsyncExitStack`` so both the ``stdio_client`` context manager
        and the ``ClientSession`` context manager are kept alive after this
        coroutine returns.  ``_close_session()`` calls ``stack.aclose()``
        to tear them down in reverse order.
        """
        try:
            stack = AsyncExitStack()
            server_params = _get_server_params()

            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params)
            )
            session = ClientSession(read_stream, write_stream)
            self._session = await stack.enter_async_context(session)
            await self._session.initialize()

            self._exit_stack = stack
            logger.info("[MCPClient] Persistent MCP session opened.")
        except Exception as exc:
            self._startup_error = exc
            logger.error(f"[MCPClient] Session startup failed: {exc}")
        finally:
            # Always unblock _ensure_started(), even on failure.
            self._ready.set()

    async def _close_session(self) -> None:
        """Tear down the AsyncExitStack (closes session + stdio transport)."""
        if self._exit_stack:
            await self._exit_stack.aclose()
            self._exit_stack = None
            self._session = None
            logger.info("[MCPClient] Persistent MCP session closed.")

    # ------------------------------------------------------------------
    # Internal: submitting work to the background loop
    # ------------------------------------------------------------------

    def _submit(self, coro) -> Any:
        """
        Ensure the session is started, then submit *coro* to the background
        event loop and block until the result is available.

        Every coroutine is wrapped in ``asyncio.wait_for`` *inside* the
        background loop so that when the outer ``future.result()`` timeout fires
        the async task also receives a cooperative cancellation signal at its
        next ``await`` checkpoint.  Without this inner wrap, ``future.cancel()``
        only cancels the ``Future`` shell — the underlying coroutine continues
        running in the background loop until it naturally returns.

        Raises
        ------
        RuntimeError
            On timeout or if the background loop is unavailable.
        """
        self._ensure_started()

        async def _with_timeout():
            return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

        future = asyncio.run_coroutine_threadsafe(_with_timeout(), self._loop)
        try:
            # Add a small wall-clock buffer on top of the async timeout so that
            # asyncio.wait_for always fires first, giving a clean CancelledError
            # before the thread-level TimeoutError.
            return future.result(timeout=self.timeout_seconds + 5)
        except concurrent.futures.TimeoutError:
            future.cancel()
            logger.error(
                f"[MCPClient] Operation exceeded wall-clock timeout "
                f"({self.timeout_seconds + 5}s); cancelled."
            )
            raise RuntimeError(
                f"MCP operation timed out after {self.timeout_seconds}s"
            )

    def _run_async(self, coro):
        """Compatibility shim — delegates to :meth:`_submit`.

        Some tests and legacy call-sites patch ``_run_async`` directly.  This
        thin wrapper keeps them working without altering the new persistent-
        session architecture.
        """
        return self._submit(coro)

    # ------------------------------------------------------------------
    # Internal: async tool helpers (run inside the background loop)
    # ------------------------------------------------------------------

    async def _async_list_tools(self) -> List[Dict[str, Any]]:
        """List tools using the persistent session."""
        result = await self._session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema,  # SDK v2 snake_case
            }
            for tool in result.tools
        ]

    async def _async_call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Call *tool_name* using the persistent session and return a parsed result.

        Error handling (in order):
        1. ``is_error=True`` on the MCP result → raise ``RuntimeError``
           so the agent's direct-tool fallback fires.
        2. ``structured_content`` dict → return directly (SDK v2 fast path,
           avoids JSON round-trip).
        3. Text content items → reassemble and parse as JSON; wrap as plain
           text on parse failure.
        """
        mcp_result = await self._session.call_tool(tool_name, arguments=arguments)

        # 1. Propagate MCP-level tool errors.
        if mcp_result.is_error:
            error_text = "\n".join(
                item.text
                for item in mcp_result.content
                if getattr(item, "text", None)
            )
            raise RuntimeError(
                error_text or f"MCP tool '{tool_name}' failed"
            )

        # 2. Prefer structured_content when available (SDK v2+).
        structured = getattr(mcp_result, "structured_content", None)
        if isinstance(structured, dict):
            return structured

        # 3. Collect text items and parse as JSON.
        raw_parts: List[str] = []
        for item in mcp_result.content:
            if hasattr(item, "text") and item.text:
                raw_parts.append(item.text)

        combined = "\n".join(raw_parts)

        try:
            return json.loads(combined)
        except (json.JSONDecodeError, ValueError):
            # Last resort: wrap raw text so the caller always gets a dict.
            return {
                "tool": tool_name,
                "output": combined,
                "sources": [],
            }


# ---------------------------------------------------------------------------
# Module-level singleton for use from agent.py
# ---------------------------------------------------------------------------

_default_client: Optional[MCPClient] = None


def get_mcp_client() -> MCPClient:
    """Return the module-level singleton MCPClient (created on first call)."""
    global _default_client
    if _default_client is None:
        _default_client = MCPClient()
    return _default_client


# ---------------------------------------------------------------------------
# Standalone test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = MCPClient()

    print("\n" + "=" * 60)
    print("NexaMind MCP Client — Tool Discovery Test")
    print("=" * 60)

    tools = client.list_tools()
    for tool in tools:
        print(f"\n  ▸ {tool['name']}")
        print(f"    {tool['description']}")
        print(f"    Input schema: {json.dumps(tool['input_schema'], indent=6)}")

    print("\n" + "=" * 60)
    print("NexaMind MCP Client — Web Search Tool Test")
    print("=" * 60)

    try:
        result = client.search_web("What is the Model Context Protocol (MCP)?")
        print(f"\nOutput:\n{result.get('output', '')[:800]}")
        print(f"\nSources returned: {len(result.get('sources', []))}")
    except Exception as e:
        print(f"Web search test failed: {e}")

    client.close()
    print("\nDone.\n")
