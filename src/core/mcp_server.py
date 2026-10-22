"""
NexaMind MCP Server.

Exposes NexaMind's external tool capabilities (GitHub RAG, Web Search,
YouTube RAG) as MCP tools over stdio transport so any MCP-compatible
client can discover and invoke them.

Run directly for local testing:
    PYTHONPATH=src python -m core.mcp_server
"""

import sys
import os

# Ensure the src directory is on the Python path when the module is run
# directly (e.g. `python -m core.mcp_server`).
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from mcp.server.mcpserver import MCPServer

from core.tools import GitHubRAGTool, WebSearchTool, YouTubeRAGTool
from utils.logger import logger

# ---------------------------------------------------------------------------
# Server instantiation
# MCPServer is the v2 successor to FastMCP.  The .tool() decorator and
# .run(transport="stdio") interface are identical to the v1 API.
# ---------------------------------------------------------------------------

mcp = MCPServer("NexaMind MCP Server")

# Instantiate tools once at module level so they are reused across requests.
# Each tool is self-contained and thread-safe for read operations.
_github_tool = GitHubRAGTool()
_web_tool = WebSearchTool()
_youtube_tool = YouTubeRAGTool()


# ---------------------------------------------------------------------------
# Tool: GitHub Repository Search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_github_repository(
    query: str,
    repo_url: str,
) -> dict:
    """
    Search a public GitHub repository for source code and documentation
    relevant to the user's query using FAISS cosine-similarity.

    Args:
        query:    The natural-language question or search query.
        repo_url: The full GitHub repository URL
                  (e.g. https://github.com/owner/repo).

    Returns:
        A dict with keys:
          - tool       (str): Always "github_rag".
          - query      (str): The original query.
          - repository (str): The repo URL that was searched.
          - output     (str): Retrieved code / documentation snippets.
          - sources    (list): Metadata for each retrieved chunk.
    """
    logger.info(
        f"[MCP] search_github_repository called | repo={repo_url} | query={query!r}"
    )

    result = _github_tool.run(query=query, repo_url=repo_url)

    if result.get("success") is False:
        raise RuntimeError(
            result.get("output", "GitHub RAG tool execution failed")
        )

    return {
        "tool": "github_rag",
        "query": query,
        "repository": repo_url,
        "output": result["output"],
        "sources": result.get("sources", []),
        "execution_time_ms": result.get("execution_time_ms", 0),
    }


# ---------------------------------------------------------------------------
# Tool: Web Search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_web(query: str) -> dict:
    """
    Search the live public web for real-time and up-to-date information.

    Tries Google Search grounding first, then falls back to DuckDuckGo.

    Args:
        query: The natural-language question or topic to search for.

    Returns:
        A dict with keys:
          - tool    (str): Always "web_search".
          - query   (str): The original query.
          - output  (str): Search result snippets.
          - sources (list): Structured source metadata (title, url, snippet).
    """
    logger.info(f"[MCP] search_web called | query={query!r}")

    result = _web_tool.run(query=query)

    return {
        "tool": "web_search",
        "query": query,
        "output": result["output"],
        "sources": result.get("sources", []),
        "execution_time_ms": result.get("execution_time_ms", 0),
    }


# ---------------------------------------------------------------------------
# Tool: YouTube Transcript Search
# ---------------------------------------------------------------------------

@mcp.tool()
def search_youtube_transcript(
    query: str,
    video_url_or_id: str,
) -> dict:
    """
    Search a YouTube video's spoken transcript for content relevant
    to the user's query using FAISS cosine-similarity.

    Args:
        query:           The natural-language question or search query.
        video_url_or_id: A full YouTube URL or bare video ID
                         (e.g. "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                          or simply "dQw4w9WgXcQ").

    Returns:
        A dict with keys:
          - tool    (str): Always "youtube_rag".
          - query   (str): The original query.
          - video   (str): The video URL/ID that was searched.
          - output  (str): Timestamped transcript snippets.
          - sources (list): Metadata for each retrieved segment.
    """
    logger.info(
        f"[MCP] search_youtube_transcript called | video={video_url_or_id} | query={query!r}"
    )

    result = _youtube_tool.run(query=query, url_or_id=video_url_or_id)

    if result.get("success") is False:
        raise RuntimeError(
            result.get("output", "YouTube RAG tool execution failed")
        )

    return {
        "tool": "youtube_rag",
        "query": query,
        "video": video_url_or_id,
        "output": result["output"],
        "sources": result.get("sources", []),
        "execution_time_ms": result.get("execution_time_ms", 0),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # stdio transport is the standard for local MCP development.
    # The process communicates via stdin / stdout using the MCP wire protocol.
    logger.info("[MCP] NexaMind MCP Server starting on stdio transport …")
    mcp.run(transport="stdio")
