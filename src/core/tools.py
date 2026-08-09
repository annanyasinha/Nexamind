"""
NexaMind Tool Definitions for AI Agent Orchestration.
Contains tools for Document RAG, YouTube RAG, and Web Search.
"""

import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from google import genai
from google.genai import types
from core.vectorstore import FaissVectorStore
from core.youtube_loader import fetch_youtube_transcript, save_transcript_to_dataset
from core.document_loader import load_all_documents
from config import settings
from utils.logger import logger

# Try importing LangChain DuckDuckGo tools
try:
    from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
    HAS_LANGCHAIN_DDG = True
except Exception:
    HAS_LANGCHAIN_DDG = False

# Try importing duckduckgo-search for live web search fallback
try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except Exception:
    HAS_DDG = False


class DocumentRAGTool:
    """
    Tool for searching indexed PDF, TXT, DOCX documents via FAISS Vector Store.
    """
    name = "document_rag"
    description = "Searches local indexed knowledge base documents (PDFs, text files, documentation) using FAISS vector similarity."

    def __init__(self, vectorstore: Optional[FaissVectorStore] = None):
        self.vectorstore = vectorstore or FaissVectorStore(settings.FAISS_STORE_DIR, settings.EMBEDDING_MODEL)
        faiss_path = settings.FAISS_STORE_DIR / "faiss.index"
        meta_path = settings.FAISS_STORE_DIR / "metadata.pkl"
        if not (faiss_path.exists() and meta_path.exists()):
            logger.info("Initializing vector store from DATA_DIR for Document RAG Tool...")
            docs = load_all_documents(settings.DATA_DIR)
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()

    def run(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        """Executes vector similarity search on document store."""
        start_time = time.time()
        results = self.vectorstore.query(query, top_k=top_k)
        
        snippets = []
        sources = []
        for r in results:
            meta = r.get("metadata", {})
            text = meta.get("text", "")
            src_file = meta.get("source", "Unknown Document")
            if text:
                snippets.append(f"[Source: {src_file}]\n{text}")
                sources.append({
                    "source": src_file,
                    "distance": float(r.get("distance", 0.0)),
                    "text": text
                })

        context_text = "\n\n---\n\n".join(snippets) if snippets else "No matching documents found in vector store."
        duration_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "tool": self.name,
            "query": query,
            "output": context_text,
            "sources": sources,
            "execution_time_ms": duration_ms
        }


class YouTubeRAGTool:
    """
    Tool for extracting YouTube transcripts and searching spoken video content.
    """
    name = "youtube_rag"
    description = "Fetches spoken audio transcripts for YouTube video links/IDs and searches transcript segments for answers."

    def run(self, query: str, url_or_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetches YouTube transcript and searches spoken transcript text."""
        start_time = time.time()
        transcript_data = None
        snippets = []
        sources = []

        if url_or_id:
            try:
                logger.info(f"Fetching YouTube transcript for tool query: '{url_or_id}'")
                transcript_data = fetch_youtube_transcript(url_or_id)
                # Auto-save and index into DATA_DIR
                save_transcript_to_dataset(transcript_data, settings.DATA_DIR)
            except Exception as e:
                logger.warning(f"Failed fetching YouTube transcript for {url_or_id}: {e}")

        if transcript_data and transcript_data.get("segments"):
            video_id = transcript_data["video_id"]
            # Search relevant segments
            raw_query_words = set(query.lower().split())
            matching_segments = []
            for seg in transcript_data["segments"]:
                seg_text = seg["text"].lower()
                overlap = sum(1 for w in raw_query_words if w in seg_text)
                matching_segments.append((overlap, seg))

            # Sort by keyword match overlap
            matching_segments.sort(key=lambda x: x[0], reverse=True)
            top_segs = [s[1] for s in matching_segments[:5]] if matching_segments else transcript_data["segments"][:5]

            for s in top_segs:
                seg_fmt = f"[{s['timestamp']}] {s['text']}"
                snippets.append(seg_fmt)
                sources.append({
                    "source": f"YouTube Video ({video_id})",
                    "timestamp": s["timestamp"],
                    "text": s["text"]
                })
            
            output_text = f"YouTube Video ID: {video_id}\n\nSpoken Segments:\n" + "\n".join(snippets)
        else:
            # Fallback: Query document store for YouTube transcripts
            from core.vectorstore import FaissVectorStore
            vstore = FaissVectorStore(settings.FAISS_STORE_DIR, settings.EMBEDDING_MODEL)
            vstore.load()
            res = vstore.query(f"YouTube transcript {query}", top_k=4)
            for r in res:
                meta = r.get("metadata", {})
                text = meta.get("text", "")
                if "YouTube" in meta.get("source", "") or "youtube" in text.lower():
                    snippets.append(f"[{meta.get('source')}]\n{text}")
                    sources.append({"source": meta.get("source"), "text": text})

            output_text = "\n\n".join(snippets) if snippets else "No YouTube transcript content found matching the query."

        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "tool": self.name,
            "query": query,
            "output": output_text,
            "sources": sources,
            "execution_time_ms": duration_ms
        }


class WebSearchTool:
    """
    Tool for searching live web results using LangChain's DuckDuckGoSearchRun.
    """
    name = "web_search"
    description = "Searches the live public web for up-to-date real-time information and external references using LangChain DuckDuckGoSearchRun."

    def __init__(self, client: Optional[genai.Client] = None):
        self.client = client or (genai.Client(api_key=settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else genai.Client())
        self.ddg_runner = None
        self.ddg_results = None
        if HAS_LANGCHAIN_DDG:
            try:
                self.ddg_runner = DuckDuckGoSearchRun()
                self.ddg_results = DuckDuckGoSearchResults(output_format="list")
            except Exception as e:
                logger.warning(f"Failed to initialize LangChain DuckDuckGoSearchRun: {e}")

    def run(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """Executes live web search prioritizing Google Search grounding for real-time news and live web results."""
        start_time = time.time()
        sources = []
        output_text = ""

        # 1. Primary: Native Google Search Grounding for real-time live news & web results
        try:
            for model in settings.LLM_MODEL_CANDIDATES:
                try:
                    response = self.client.models.generate_content(
                        model=model,
                        contents=f"Perform a live real-time web and news search for: '{query}'. Provide current news updates, headlines, dates, and source URLs.",
                        config=types.GenerateContentConfig(
                            tools=[{"google_search": {}}]
                        )
                    )
                    if response and response.text:
                        output_text = response.text
                        if hasattr(response, "candidates") and response.candidates:
                            candidate = response.candidates[0]
                            grounding_meta = getattr(candidate, "grounding_metadata", None)
                            if grounding_meta and hasattr(grounding_meta, "grounding_chunks"):
                                for chunk in grounding_meta.grounding_chunks:
                                    web_info = getattr(chunk, "web", None)
                                    if web_info:
                                        sources.append({
                                            "title": getattr(web_info, "title", "Google Search Result"),
                                            "url": getattr(web_info, "uri", ""),
                                            "snippet": getattr(web_info, "title", "")
                                        })
                        logger.info(f"WebSearchTool executed successfully using Google Search grounding ({model})")
                        break
                except Exception as e:
                    logger.warning(f"Google Search grounding failed for model {model}: {e}")
        except Exception as ex:
            logger.warning(f"Google Search grounding error: {ex}")

        # 2. Secondary: LangChain DuckDuckGoSearchRun / DuckDuckGoSearchResults Fallback
        if not output_text and HAS_LANGCHAIN_DDG:
            try:
                if self.ddg_results:
                    raw_res = self.ddg_results.invoke(query)
                    if isinstance(raw_res, list):
                        snippets = []
                        for idx, item in enumerate(raw_res[:max_results]):
                            title = item.get("title", f"Result #{idx+1}")
                            link = item.get("link", "")
                            snippet = item.get("snippet", "")
                            snippets.append(f"[{idx+1}] {title}\nURL: {link}\nSnippet: {snippet}")
                            sources.append({"title": title, "url": link, "snippet": snippet})
                        output_text = "\n\n".join(snippets)
                    elif isinstance(raw_res, str):
                        output_text = raw_res
                        
                if not output_text and self.ddg_runner:
                    output_text = self.ddg_runner.invoke(query)
                    
                if output_text and "No good DuckDuckGo" not in output_text and "returned no results" not in output_text:
                    logger.info(f"WebSearchTool executed successfully using LangChain DuckDuckGoSearchRun")
                else:
                    output_text = ""
            except Exception as e:
                logger.warning(f"LangChain DuckDuckGoSearchRun execution failed: {e}")
                output_text = ""

        # 3. Tertiary: Direct DuckDuckGo HTTP package fallback
        if not output_text:
            output_text, sources = self._fallback_ddg_search(query, max_results=max_results)

        duration_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "tool": self.name,
            "query": query,
            "output": output_text,
            "sources": sources,
            "execution_time_ms": duration_ms
        }

    def _fallback_ddg_search(self, query: str, max_results: int = 5) -> tuple:
        """Fallback search using DuckDuckGo package or HTTP endpoint."""
        sources = []
        output_snippets = []

        if HAS_DDG:
            try:
                try:
                    with DDGS() as ddgs:
                        ddg_results = list(ddgs.text(query, max_results=max_results))
                except Exception:
                    ddg_results = list(DDGS().text(query, max_results=max_results))

                for idx, res in enumerate(ddg_results):
                    title = res.get("title", "No Title")
                    snippet = res.get("body", "")
                    href = res.get("href", "")
                    output_snippets.append(f"[{idx+1}] {title}\nURL: {href}\nSnippet: {snippet}")
                    sources.append({"title": title, "url": href, "snippet": snippet})
            except Exception as e:
                logger.warning(f"DuckDuckGo package search failed for query '{query}': {e}. Using HTTP fallback.")
                sources = self._http_fallback_search(query, max_results=max_results)
        else:
            sources = self._http_fallback_search(query, max_results=max_results)

        if not output_snippets and sources:
            for idx, res in enumerate(sources):
                output_snippets.append(f"[{idx+1}] {res['title']}\nURL: {res['url']}\nSnippet: {res['snippet']}")

        output_text = "\n\n".join(output_snippets) if output_snippets else f"Web search for '{query}' returned no results."
        return output_text, sources

    def _http_fallback_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Fallback HTTP DuckDuckGo web search using requests."""
        import requests
        import urllib.parse
        results = []
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json"
            headers = {"User-Agent": "Mozilla/5.0"}
            resp = requests.get(url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("AbstractText", "")
                heading = data.get("Heading", "Summary")
                abstract_url = data.get("AbstractURL", "")
                if abstract:
                    results.append({"title": heading, "url": abstract_url, "snippet": abstract})

                for topic in data.get("RelatedTopics", [])[:max_results]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({
                            "title": topic.get("Text")[:60] + "...",
                            "url": topic.get("FirstURL", ""),
                            "snippet": topic.get("Text", "")
                        })
        except Exception as e:
            logger.warning(f"HTTP fallback web search error: {e}")
        return results


def get_tool_registry() -> Dict[str, Any]:
    """Returns singleton dictionary of available NexaMind tools."""
    return {
        DocumentRAGTool.name: DocumentRAGTool(),
        YouTubeRAGTool.name: YouTubeRAGTool(),
        WebSearchTool.name: WebSearchTool()
    }
