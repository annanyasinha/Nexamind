"""
NexaMind MCP Client.

Provides a synchronous wrapper around the async MCP Python SDK client so
the existing synchronous NexaMindAgent can call MCP tools without
restructuring its execution loop.

Architecture
------------

NexaMindAgent.run()          (synchronous)
        │
        ▼
MCPClient.call_tool()        (synchronous wrapper)
        │
        ▼
_async_call_tool()           (async, runs inside a dedicated event loop)
        │
    stdio transport
        │
        ▼
NexaMind MCP Server          (subprocess, started on demand)
        │
        ▼
Existing Tool (GitHub / Web / YouTube)
        │
        ▼
Tool Result dict

Usage (standalone discovery test)
----------------------------------
    PYTHONPATH=src python -m core.mcp_client

"""

import asyncio
import sys
import os
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
# Async helpers
# ---------------------------------------------------------------------------

async def _async_list_tools() -> List[Dict[str, Any]]:
    """Start the MCP server and return the list of available tools."""
    server_params = _get_server_params()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.list_tools()

            tools_info = []
            for tool in result.tools:
                tools_info.append({
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                })

            return tools_info


async def _async_call_tool(
    tool_name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Start the MCP server, call *tool_name* with *arguments*, and return
    the parsed result dict.

    The server process lives only for the duration of this call; each
    invocation is self-contained.  For a long-running production service
    you would keep the session open across calls.
    """
    server_params = _get_server_params()

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            mcp_result = await session.call_tool(tool_name, arguments=arguments)

            import json

            # ------------------------------------------------------------------
            # 1. Propagate MCP-level tool errors so the agent's fallback fires.
            #    session.call_tool() does NOT raise for tool-side failures;
            #    instead it sets is_error=True on the result object.
            # ------------------------------------------------------------------
            if mcp_result.is_error:
                error_text = "\n".join(
                    item.text
                    for item in mcp_result.content
                    if getattr(item, "text", None)
                )
                raise RuntimeError(
                    error_text or f"MCP tool '{tool_name}' failed"
                )

            # ------------------------------------------------------------------
            # 2. Prefer structured_content when the SDK provides it (SDK v2+).
            #    This avoids an unnecessary JSON round-trip.
            # ------------------------------------------------------------------
            structured = getattr(mcp_result, "structured_content", None)
            if isinstance(structured, dict):
                return structured

            # ------------------------------------------------------------------
            # 3. Fallback: collect text content items and parse as JSON.
            #    The MCP server serialises its returned dict as a JSON text item.
            # ------------------------------------------------------------------
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
# Synchronous public interface
# ---------------------------------------------------------------------------

class MCPClient:
    """
    Synchronous MCP client for use inside the existing NexaMindAgent.

    Each call to :meth:`call_tool` spins up a fresh event loop (or re-uses
    an existing one where possible) so the caller does not need to be async.

    Parameters
    ----------
    timeout_seconds:
        Maximum wall-clock seconds to wait for a tool call to complete.
        Defaults to 120 s (enough for GitHub indexing on first call).
    """

    def __init__(self, timeout_seconds: int = 120):
        self.timeout_seconds = timeout_seconds

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
        return self._run_async(_async_list_tools())

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
            result = self._run_async(
                _async_call_tool(tool_name, arguments)
            )
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
    # Internal: event-loop management
    # ------------------------------------------------------------------

    def _run_async(self, coro):
        """
        Run *coro* synchronously, enforcing ``self.timeout_seconds`` in both
        execution paths.

        Two situations arise:

        1. **No running event loop** (typical when called from a plain sync
           context, the CLI, or a pytest test):
           ``asyncio.run()`` is used directly.  ``asyncio.wait_for`` wraps the
           coroutine so the timeout is always respected.

        2. **Running event loop already exists** (FastAPI async endpoint,
           Jupyter notebook):
           A new loop is started in a fresh thread to avoid the
           "This event loop is already running" error.  The thread-level
           ``future.result(timeout=…)`` provides the outer wall-clock guard;
           ``asyncio.wait_for`` inside the thread provides the async-level
           cancellation signal.
        """
        import concurrent.futures

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Running inside an event loop (e.g. FastAPI async endpoint).
            # Offload to a thread so we don't deadlock.
            async def _with_timeout():
                return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _with_timeout())
                # The outer timeout adds a wall-clock guard on top of the
                # async cancellation; both must match for clean teardown.
                return future.result(timeout=self.timeout_seconds + 5)
        else:
            # No running loop — use asyncio.run() with an async timeout.
            async def _with_timeout():
                return await asyncio.wait_for(coro, timeout=self.timeout_seconds)

            return asyncio.run(_with_timeout())


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
    import json

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

    print("\nDone.\n")
