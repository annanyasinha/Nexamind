"""
NexaMind Autonomous AI Agent Orchestrator.
Routes query intent to tools (Document RAG, YouTube RAG, Web Search) and synthesizes LLM answers.
"""

import re
import time
import json
from typing import List, Dict, Any, Optional
from google import genai
from core.tools import DocumentRAGTool, YouTubeRAGTool, WebSearchTool
from core.youtube_loader import extract_youtube_id
from core.prompt import format_chat_history
from config import settings
from utils.logger import logger

AGENT_PLANNER_PROMPT = """You are NexaMind AI Agent Planning Router.
Given the user query, determine which tool(s) should be called to retrieve information.

Available Tools:
1. document_rag: Use ONLY when the query specifically asks about uploaded files, PDFs, local documents, personal resume/CV, or internal dataset.
2. youtube_rag: Use when a YouTube link or video ID is present, or when the user asks about video transcripts or spoken video content.
3. web_search: Use when the query asks about real-world topics, current events, public news, general concepts, protests, weather, external internet information, or live topics.

Instructions:
- If a query asks about public events, real-world topics, or news (e.g. "jharkhand protest", "latest news", "weather"), select `web_search`.
- If a query asks about uploaded documents or internal dataset files, select `document_rag`.
- You can select multiple tools if needed (e.g. document_rag AND web_search).
- Respond in strictly valid JSON format:
{
  "thought": "<short reasoning of why tools are selected>",
  "tools": [
    {"name": "document_rag" | "youtube_rag" | "web_search", "query": "<search query>", "youtube_url": "<url or id if applicable or empty>"}
  ]
}

User Query: {query}
JSON Output:"""

AGENT_SYNTHESIS_PROMPT = """You are NexaMind AI Agent. Answer the user query using the retrieved tool observations below.
Maintain a helpful, clear, and professional tone. Cite sources appropriately (e.g. Document source, YouTube timestamp, or Web search URL).

{history_text}
Tool Observations Gathered by Agent:
{observations}

Current User Query: {query}

Synthesized Answer:"""


class NexaMindAgent:
    """
    Autonomous Agent orchestrating Document RAG, YouTube RAG, and Web Search tools.
    """
    def __init__(self, llm_model: str = None):
        self.llm_model = llm_model or settings.DEFAULT_LLM_MODEL
        self.doc_tool = DocumentRAGTool()
        self.yt_tool = YouTubeRAGTool()
        self.web_tool = WebSearchTool()
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else genai.Client()

    def run(
        self, 
        query: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        enabled_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Executes the AI Agent lifecycle:
        1. Query analysis & tool selection
        2. Tool execution (Document RAG / YouTube RAG / Web Search)
        3. LLM synthesis of final response
        """
        start_time = time.time()
        logger.info(f"NexaMind Agent receiving query: '{query}'")

        history_text, _ = format_chat_history(chat_history)
        selected_tools = self._plan_tool_calls(query, enabled_tools)

        steps = []
        observations_blocks = []
        all_sources = []

        # Execute selected tools
        for t_info in selected_tools:
            t_name = t_info.get("name")
            t_query = t_info.get("query", query)
            yt_url = t_info.get("youtube_url")

            if t_name == "document_rag" and (not enabled_tools or "document_rag" in enabled_tools):
                res = self.doc_tool.run(t_query)
                steps.append({
                    "tool": "document_rag",
                    "input": t_query,
                    "output": res["output"],
                    "execution_time_ms": res["execution_time_ms"]
                })
                observations_blocks.append(f"=== Document RAG Observation ===\n{res['output']}")
                all_sources.extend(res.get("sources", []))

            elif t_name == "youtube_rag" and (not enabled_tools or "youtube_rag" in enabled_tools):
                res = self.yt_tool.run(t_query, url_or_id=yt_url)
                steps.append({
                    "tool": "youtube_rag",
                    "input": f"{t_query} (URL: {yt_url})" if yt_url else t_query,
                    "output": res["output"],
                    "execution_time_ms": res["execution_time_ms"]
                })
                observations_blocks.append(f"=== YouTube RAG Observation ===\n{res['output']}")
                all_sources.extend(res.get("sources", []))

            elif t_name == "web_search" and (not enabled_tools or "web_search" in enabled_tools):
                res = self.web_tool.run(t_query)
                steps.append({
                    "tool": "web_search",
                    "input": t_query,
                    "output": res["output"],
                    "execution_time_ms": res["execution_time_ms"]
                })
                observations_blocks.append(f"=== Web Search Observation ===\n{res['output']}")
                all_sources.extend(res.get("sources", []))

        # Default fallback if no tools produced results or no tools selected
        if not observations_blocks:
            res = self.doc_tool.run(query)
            steps.append({
                "tool": "document_rag",
                "input": query,
                "output": res["output"],
                "execution_time_ms": res["execution_time_ms"]
            })
            observations_blocks.append(f"=== Document RAG Observation ===\n{res['output']}")
            all_sources.extend(res.get("sources", []))

        observations = "\n\n".join(observations_blocks)
        synthesis_prompt = AGENT_SYNTHESIS_PROMPT.format(
            history_text=history_text,
            observations=observations,
            query=query
        )

        answer = self._generate_llm_response(synthesis_prompt)
        total_duration_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "query": query,
            "answer": answer,
            "steps": steps,
            "sources": all_sources,
            "execution_time_ms": total_duration_ms
        }

    def _plan_tool_calls(self, query: str, enabled_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Determines which tool(s) to call using LLM planning router with query heuristic fallback."""
        tools = []
        
        # 1. Try LLM Planning Router for dynamic multi-tool routing
        try:
            planner_prompt = AGENT_PLANNER_PROMPT.format(query=query)
            for model in settings.LLM_MODEL_CANDIDATES:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=planner_prompt
                    )
                    if response and response.text:
                        raw_text = response.text.strip()
                        if "```" in raw_text:
                            raw_text = re.sub(r"```json?\n?|\n?```", "", raw_text).strip()
                        parsed = json.loads(raw_text)
                        if isinstance(parsed, dict) and "tools" in parsed and isinstance(parsed["tools"], list):
                            for t in parsed["tools"]:
                                if isinstance(t, dict) and "name" in t:
                                    tools.append({
                                        "name": t["name"],
                                        "query": t.get("query", query),
                                        "youtube_url": t.get("youtube_url", "")
                                    })
                            if tools:
                                logger.info(f"LLM Planner successfully selected {len(tools)} tool(s): {[t['name'] for t in tools]}")
                                break
                except Exception as e:
                    logger.warning(f"LLM planner attempt failed for {model}: {e}")
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        time.sleep(1.5)
        except Exception as ex:
            logger.warning(f"Error during LLM tool planning: {ex}")

        # 2. Comprehensive Heuristic Fallback if LLM planner returned no tools
        if not tools:
            yt_id = None
            try:
                yt_id = extract_youtube_id(query)
            except Exception:
                pass

            if yt_id or any(kw in query.lower() for kw in ["youtube", "video", "transcript", "watch", "v="]):
                tools.append({"name": "youtube_rag", "query": query, "youtube_url": yt_id or query})

            web_keywords = ["latest", "news", "today", "current", "weather", "web", "online", "price", "http", "search", "market", "trend", "recent", "developments", "internet", "protest", "who is", "what is", "tell me about", "explain", "info"]
            if any(kw in query.lower() for kw in web_keywords):
                tools.append({"name": "web_search", "query": query})

            doc_keywords = ["doc", "pdf", "file", "index", "dataset", "resume", "skills", "uploaded", "knowledge", "shubham", "annanya", "document", "codebase"]
            if any(kw in query.lower() for kw in doc_keywords) or not tools:
                tools.append({"name": "document_rag", "query": query})

        # 3. Filter by enabled_tools if specified in UI
        if enabled_tools:
            filtered = [t for t in tools if t["name"] in enabled_tools]
            if filtered:
                tools = filtered
            else:
                tools = [{"name": enabled_tools[0], "query": query}]

        return tools

    def _generate_llm_response(self, prompt: str) -> str:
        """Sends prompt to Gemini model candidates with automatic retry and rate limit backoff."""
        errors = []
        for model in settings.LLM_MODEL_CANDIDATES:
            for attempt in range(2):
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=prompt
                    )
                    if response and response.text:
                        logger.info(f"NexaMind Agent successfully generated answer with model {model}")
                        return response.text
                except Exception as e:
                    err_str = str(e)
                    errors.append(f"{model}: {err_str}")
                    logger.warning(f"LLM generation failed for {model} (attempt {attempt+1}): {err_str}")
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        time.sleep(1.5)
                    else:
                        break

        err_detail = " | ".join(errors[:2]) if errors else "API response empty"
        return f"NexaMind Agent gathered context from tools, but encountered a temporary Gemini API rate limit. Please retry in a few seconds.\n\nDetail: {err_detail}"
