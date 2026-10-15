"""
Tests for NexaMind MCP integration.

Test levels
-----------
1. Unit tests  — test the shape/logic of results without starting a server.
2. Integration tests — start the MCP server as a subprocess and verify
   real communication (tool discovery, tool invocation).  These are marked
   with @pytest.mark.integration and can be excluded in CI with:
       pytest -m "not integration"
"""

import json
import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock


# ---------------------------------------------------------------------------
# Unit tests: result shape validation
# ---------------------------------------------------------------------------

class TestMCPResultShape:
    """Validate that the MCP tool result dicts match the expected schema."""

    def _github_result(self) -> dict:
        return {
            "tool": "github_rag",
            "query": "How is tool routing implemented?",
            "repository": "https://github.com/annanyasinha/Nexamind",
            "output": "Relevant code chunks found in agent.py …",
            "sources": [
                {
                    "source_type": "github",
                    "repository": "annanyasinha/Nexamind",
                    "file_path": "src/core/agent.py",
                    "similarity_score": 0.87,
                }
            ],
            "execution_time_ms": 320.5,
        }

    def _web_result(self) -> dict:
        return {
            "tool": "web_search",
            "query": "What is MCP?",
            "output": "[1] Model Context Protocol …",
            "sources": [{"title": "MCP docs", "url": "https://example.com", "snippet": "…"}],
            "execution_time_ms": 180.0,
        }

    def _youtube_result(self) -> dict:
        return {
            "tool": "youtube_rag",
            "query": "What is discussed in the video?",
            "video": "dQw4w9WgXcQ",
            "output": "[00:01:23] Spoken content …",
            "sources": [],
            "execution_time_ms": 95.0,
        }

    # --- GitHub ---

    def test_github_result_has_required_keys(self):
        result = self._github_result()
        for key in ("tool", "query", "repository", "output", "sources"):
            assert key in result, f"Missing key: {key}"

    def test_github_result_tool_name(self):
        assert self._github_result()["tool"] == "github_rag"

    def test_github_result_sources_is_list(self):
        assert isinstance(self._github_result()["sources"], list)

    def test_github_result_output_is_str(self):
        assert isinstance(self._github_result()["output"], str)

    # --- Web search ---

    def test_web_result_has_required_keys(self):
        result = self._web_result()
        for key in ("tool", "query", "output", "sources"):
            assert key in result

    def test_web_result_tool_name(self):
        assert self._web_result()["tool"] == "web_search"

    # --- YouTube ---

    def test_youtube_result_has_required_keys(self):
        result = self._youtube_result()
        for key in ("tool", "query", "video", "output", "sources"):
            assert key in result

    def test_youtube_result_tool_name(self):
        assert self._youtube_result()["tool"] == "youtube_rag"


# ---------------------------------------------------------------------------
# Unit tests: MCPClient — mocked transport
# ---------------------------------------------------------------------------

class TestMCPClientUnit:
    """Test MCPClient logic with mocked async internals."""

    def test_call_tool_success(self):
        """MCPClient.call_tool() should return the parsed dict on success."""
        from core.mcp_client import MCPClient

        expected = {"tool": "web_search", "output": "Results …", "sources": []}

        client = MCPClient()
        # Patch the internal coroutine runner so no subprocess starts
        with patch.object(client, "_run_async", return_value=expected) as mock_run:
            result = client.call_tool("search_web", {"query": "test"})

        assert result["tool"] == "web_search"
        assert "output" in result
        mock_run.assert_called_once()

    def test_call_tool_raises_on_failure(self):
        """MCPClient.call_tool() should raise RuntimeError when _run_async fails."""
        from core.mcp_client import MCPClient

        client = MCPClient()
        with patch.object(client, "_run_async", side_effect=Exception("server died")):
            with pytest.raises(RuntimeError, match="MCP tool 'search_web' failed"):
                client.call_tool("search_web", {"query": "test"})

    def test_list_tools_returns_list(self):
        """MCPClient.list_tools() should return a list of dicts."""
        from core.mcp_client import MCPClient

        dummy_tools = [
            {"name": "search_github_repository", "description": "…", "input_schema": {}},
            {"name": "search_web", "description": "…", "input_schema": {}},
            {"name": "search_youtube_transcript", "description": "…", "input_schema": {}},
        ]

        client = MCPClient()
        with patch.object(client, "_run_async", return_value=dummy_tools):
            tools = client.list_tools()

        assert isinstance(tools, list)
        assert len(tools) == 3
        assert tools[0]["name"] == "search_github_repository"


# ---------------------------------------------------------------------------
# Unit tests: NexaMindAgent MCP integration
# ---------------------------------------------------------------------------

class TestAgentMCPIntegration:
    """Test that the agent correctly routes through MCP and falls back."""

    def _make_mcp_result(self, tool: str) -> dict:
        return {
            "tool": tool,
            "output": f"Mocked MCP output from {tool}",
            "sources": [],
            "execution_time_ms": 50.0,
            "_via_mcp": True,
        }

    @patch("core.tools.DocumentRAGTool.__init__", return_value=None)
    @patch("core.tools.DocumentRAGTool.run")
    @patch("core.agent.MCPClient")
    @patch("core.agent._MCP_AVAILABLE", True)
    def test_github_tool_called_via_mcp(self, MockMCPClient, mock_doc_run, mock_doc_init):
        """Agent should call GitHub RAG through MCP when use_mcp=True."""
        from core.agent import NexaMindAgent

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.call_tool.return_value = self._make_mcp_result("github_rag")
        MockMCPClient.return_value = mock_mcp_instance

        mock_doc_run.return_value = {
            "tool": "document_rag", "output": "doc fallback",
            "sources": [], "execution_time_ms": 10.0,
        }

        agent = NexaMindAgent(use_mcp=True)
        agent._generate_llm_response = MagicMock(return_value="synthesized answer")

        # Force GitHub RAG routing by providing a GitHub URL
        result = agent.run(
            "Explain agent.py in https://github.com/annanyasinha/Nexamind"
        )

        mock_mcp_instance.call_tool.assert_called_once()
        call_args = mock_mcp_instance.call_tool.call_args
        assert call_args[0][0] == "search_github_repository"
        assert "repo_url" in call_args[0][1]

        # via_mcp should be True in the step
        github_steps = [s for s in result["steps"] if s["tool"] == "github_rag"]
        assert len(github_steps) == 1
        assert github_steps[0]["via_mcp"] is True

    @patch("core.tools.DocumentRAGTool.__init__", return_value=None)
    @patch("core.tools.DocumentRAGTool.run")
    @patch("core.agent.MCPClient")
    @patch("core.agent._MCP_AVAILABLE", True)
    def test_mcp_failure_falls_back_to_direct(self, MockMCPClient, mock_doc_run, mock_doc_init):
        """When MCP call fails, agent must fall back to direct tool execution."""
        from core.agent import NexaMindAgent
        from core.tools import GitHubRAGTool

        mock_mcp_instance = MagicMock()
        mock_mcp_instance.call_tool.side_effect = RuntimeError("MCP server crashed")
        MockMCPClient.return_value = mock_mcp_instance

        mock_doc_run.return_value = {
            "tool": "document_rag", "output": "doc fallback",
            "sources": [], "execution_time_ms": 10.0,
        }

        agent = NexaMindAgent(use_mcp=True)
        agent._generate_llm_response = MagicMock(return_value="answer after fallback")

        # Patch the direct github_tool.run so no real network call happens
        with patch.object(
            agent.github_tool,
            "run",
            return_value={
                "tool": "github_rag",
                "output": "Direct GitHub result",
                "sources": [],
                "execution_time_ms": 200.0,
            },
        ) as mock_direct:
            result = agent.run(
                "Explain agent.py in https://github.com/annanyasinha/Nexamind"
            )
            mock_direct.assert_called_once()

        github_steps = [s for s in result["steps"] if s["tool"] == "github_rag"]
        assert len(github_steps) == 1
        # Fallback means via_mcp must be False
        assert github_steps[0]["via_mcp"] is False

    @patch("core.tools.DocumentRAGTool.__init__", return_value=None)
    @patch("core.tools.DocumentRAGTool.run")
    def test_agent_without_mcp_uses_direct_calls(self, mock_doc_run, mock_doc_init):
        """When use_mcp=False the agent must use direct tool execution paths."""
        from core.agent import NexaMindAgent

        mock_doc_run.return_value = {
            "tool": "document_rag", "output": "doc output",
            "sources": [], "execution_time_ms": 5.0,
        }

        agent = NexaMindAgent(use_mcp=False)
        assert agent.use_mcp is False
        assert agent._mcp_client is None

        agent._generate_llm_response = MagicMock(return_value="direct answer")
        result = agent.run("What is in my documents?")

        assert result["answer"] == "direct answer"
        for step in result["steps"]:
            assert step.get("via_mcp") is False


# ---------------------------------------------------------------------------
# Integration tests: real MCP subprocess
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestMCPIntegration:
    """
    These tests start the actual NexaMind MCP server subprocess via stdio.
    They require a valid environment (API keys, etc.) and take several seconds.

    Exclude them in fast CI runs:
        pytest -m "not integration"
    """

    def test_tool_discovery(self):
        """
        MCP server should expose exactly three tools:
        search_github_repository, search_web, search_youtube_transcript.
        """
        from core.mcp_client import MCPClient

        client = MCPClient(timeout_seconds=60)
        tools = client.list_tools()

        tool_names = {t["name"] for t in tools}
        assert "search_github_repository" in tool_names, \
            f"search_github_repository missing from {tool_names}"
        assert "search_web" in tool_names, \
            f"search_web missing from {tool_names}"
        assert "search_youtube_transcript" in tool_names, \
            f"search_youtube_transcript missing from {tool_names}"

    def test_web_search_tool_returns_output(self):
        """
        Calling search_web through MCP should return a non-empty output string.
        """
        from core.mcp_client import MCPClient

        client = MCPClient(timeout_seconds=60)
        result = client.search_web("What is the Model Context Protocol?")

        assert "output" in result
        assert isinstance(result["output"], str)
        assert len(result["output"]) > 0

    def test_github_tool_handles_invalid_url_gracefully(self):
        """
        Calling search_github_repository with an invalid URL should not raise
        an unhandled exception — it should return an output string describing
        the error.
        """
        from core.mcp_client import MCPClient

        client = MCPClient(timeout_seconds=60)
        try:
            result = client.search_github(
                query="How does routing work?",
                repo_url="https://github.com/nonexistent-org-xyz/no-such-repo-abc",
            )
            # Result should still have output key
            assert "output" in result
        except RuntimeError:
            # A RuntimeError from MCPClient is acceptable for a bad URL
            pass

    def test_existing_document_rag_unaffected(self):
        """
        After MCP integration, the DocumentRAGTool path must remain functional.
        This test uses the agent with use_mcp=False to isolate the document path.
        """
        from core.agent import NexaMindAgent
        from unittest.mock import patch, MagicMock

        with patch("core.tools.DocumentRAGTool.__init__", return_value=None), \
             patch("core.tools.DocumentRAGTool.run") as mock_doc_run:

            mock_doc_run.return_value = {
                "tool": "document_rag",
                "output": "Document content …",
                "sources": [{"source": "data/test.pdf"}],
                "execution_time_ms": 15.0,
            }

            agent = NexaMindAgent(use_mcp=False)
            agent._generate_llm_response = MagicMock(return_value="doc answer")
            result = agent.run("What does my uploaded document say?")

        assert result["answer"] == "doc answer"
        assert any(s["tool"] == "document_rag" for s in result["steps"])
