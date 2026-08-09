import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from api.main import app
from core.tools import DocumentRAGTool, YouTubeRAGTool, WebSearchTool
from core.agent import NexaMindAgent

client = TestClient(app)


def test_document_rag_tool():
    """Tests execution of Document RAG tool."""
    tool = DocumentRAGTool()
    res = tool.run("test query")
    assert res["tool"] == "document_rag"
    assert "output" in res
    assert "sources" in res
    assert "execution_time_ms" in res


def test_youtube_rag_tool():
    """Tests execution of YouTube RAG tool."""
    tool = YouTubeRAGTool()
    res = tool.run("python tutorial", url_or_id="J5_-l7WIO_w")
    assert res["tool"] == "youtube_rag"
    assert "output" in res
    assert "execution_time_ms" in res


def test_web_search_tool():
    """Tests execution of Web Search tool."""
    tool = WebSearchTool()
    res = tool.run("python release news")
    assert res["tool"] == "web_search"
    assert "output" in res
    assert "execution_time_ms" in res


def test_agent_plan_tool_calls():
    """Tests heuristic and intent tool routing in NexaMindAgent."""
    agent = NexaMindAgent()
    
    # Query with YouTube link should trigger YouTube RAG
    tools_yt = agent._plan_tool_calls("What is discussed in https://www.youtube.com/watch?v=J5_-l7WIO_w")
    assert any(t["name"] == "youtube_rag" for t in tools_yt)

    # Query with latest news keyword should trigger Web Search
    tools_web = agent._plan_tool_calls("What are the latest news on AI today?")
    assert any(t["name"] == "web_search" for t in tools_web)

    # Document query should trigger Document RAG
    tools_doc = agent._plan_tool_calls("What does the resume PDF say?")
    assert any(t["name"] == "document_rag" for t in tools_doc)


@patch.object(NexaMindAgent, "_generate_llm_response", return_value="Synthesized mock agent answer")
def test_agent_run_execution(mock_llm):
    """Tests full agent execution flow."""
    agent = NexaMindAgent()
    res = agent.run("What is in my uploaded documents and what is latest in AI?")
    
    assert res["query"] == "What is in my uploaded documents and what is latest in AI?"
    assert res["answer"] == "Synthesized mock agent answer"
    assert len(res["steps"]) > 0
    assert "execution_time_ms" in res


@patch.object(NexaMindAgent, "_generate_llm_response", return_value="API agent response")
def test_agent_api_endpoint(mock_llm):
    """Tests /agent/query REST endpoint."""
    response = client.post(
        "/agent/query",
        json={"query": "Test agent endpoint query", "enabled_tools": ["document_rag"]}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Test agent endpoint query"
    assert data["answer"] == "API agent response"
    assert len(data["steps"]) >= 1
    assert data["steps"][0]["tool"] == "document_rag"
