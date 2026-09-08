from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from google import genai

from config import settings
from core.document_loader import load_all_documents, load_single_document
from core.prompt import NO_CONTEXT_FOUND_MESSAGE, build_rag_prompt, format_chat_history
from core.vectorstore import FaissVectorStore
from utils.logger import logger


class RAGSearch:
    """
    RAG Search Engine orchestrating retrieval from FAISS vectorstore
    and response generation via Google Gemini models.
    """
    def __init__(
        self, 
        persist_dir: Union[str, Path] = None, 
        embedding_model: str = None, 
        llm_model: str = None
    ):
        """Initializes FAISS vector store and Google GenAI client."""
        self.persist_dir = Path(persist_dir) if persist_dir else settings.FAISS_STORE_DIR
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.llm_model = llm_model or settings.DEFAULT_LLM_MODEL
        self.vectorstore = FaissVectorStore(self.persist_dir, self.embedding_model)
        self.client = genai.Client(api_key=settings.GOOGLE_API_KEY) if settings.GOOGLE_API_KEY else genai.Client()

        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"
        if not (faiss_path.exists() and meta_path.exists()):
            logger.info("Vector store files not found. Initializing index from document dataset...")
            try:
                docs = load_all_documents(settings.DATA_DIR)
                self.vectorstore.build_from_documents(docs)
            except Exception as e:
                logger.warning(f"Failed to auto-build vector store index on initialization: {e}")
        else:
            self.vectorstore.load()

    def search_with_sources(
        self, 
        query: str, 
        top_k: int = 5, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> dict:
        """Retrieves vector context and generates a synthesized answer summary."""
        history_text, last_q = format_chat_history(chat_history)
        retrieval_query = f"{last_q} {query}" if last_q else query

        results = self.vectorstore.query(
            retrieval_query, 
            top_k=top_k, 
            min_similarity=settings.MIN_SIMILARITY_SCORE
        )
        sources = []
        formatted_context_blocks = []
        for idx, r in enumerate(results, start=1):
            if r.get("metadata"):
                meta = r["metadata"]
                text = meta.get("text", "")
                fn = meta.get("filename", "unknown")
                pg = meta.get("page_number")
                page_str = f" | Page {pg}" if pg is not None else ""
                header = f"--- Context [Source {idx} | {fn}{page_str}] ---"
                formatted_context_blocks.append(f"{header}\n{text}")
                sources.append({
                    "index": int(r.get("index", -1)),
                    "similarity_score": float(r.get("similarity_score", 0.0)),
                    "text": text,
                    "metadata": meta
                })

        context = "\n\n".join(formatted_context_blocks)
        if not context:
            return {
                "query": query,
                "summary": NO_CONTEXT_FOUND_MESSAGE,
                "sources": []
            }

        prompt = build_rag_prompt(query=query, context=context, history_text=history_text)
        summary = None
        errors = []

        for model in settings.LLM_MODEL_CANDIDATES:
            try:
                response = self.client.models.generate_content(
                    model=model, 
                    contents=prompt
                )
                if response and response.text:
                    summary = response.text
                    logger.info(f"Successfully generated summary using model: {model}")
                    break
            except Exception as e:
                errors.append(f"{model}: {e!s}")
                logger.warning(f"Generation attempt with {model} failed: {e}")

        if not summary:
            summary = f"Retrieved {len(sources)} context chunk(s), but summary generation encountered an issue.\n\n" + \
                      "\n".join(errors)

        return {
            "query": query,
            "summary": summary,
            "sources": sources
        }

    def search_with_sources_stream(
        self, 
        query: str, 
        top_k: int = 5, 
        chat_history: Optional[List[Dict[str, str]]] = None
    ):
        """Yields retrieved context sources and real-time generated response tokens."""
        history_text, last_q = format_chat_history(chat_history)
        retrieval_query = f"{last_q} {query}" if last_q else query

        results = self.vectorstore.query(
            retrieval_query, 
            top_k=top_k, 
            min_similarity=settings.MIN_SIMILARITY_SCORE
        )
        sources = []
        formatted_context_blocks = []
        for idx, r in enumerate(results, start=1):
            if r.get("metadata"):
                meta = r["metadata"]
                text = meta.get("text", "")
                fn = meta.get("filename", "unknown")
                pg = meta.get("page_number")
                page_str = f" | Page {pg}" if pg is not None else ""
                header = f"--- Context [Source {idx} | {fn}{page_str}] ---"
                formatted_context_blocks.append(f"{header}\n{text}")
                sources.append({
                    "index": int(r.get("index", -1)),
                    "similarity_score": float(r.get("similarity_score", 0.0)),
                    "text": text,
                    "metadata": meta
                })

        yield {"type": "sources", "sources": sources}

        context = "\n\n".join(formatted_context_blocks)
        if not context:
            yield {"type": "token", "content": NO_CONTEXT_FOUND_MESSAGE}
            yield {"type": "done", "full_summary": NO_CONTEXT_FOUND_MESSAGE}
            return

        prompt = build_rag_prompt(query=query, context=context, history_text=history_text)

        full_summary = ""
        stream_success = False
        errors = []

        for model in settings.LLM_MODEL_CANDIDATES:
            try:
                response = self.client.models.generate_content_stream(
                    model=model, 
                    contents=prompt
                )
                for chunk in response:
                    if chunk and chunk.text:
                        full_summary += chunk.text
                        yield {"type": "token", "content": chunk.text}
                
                if full_summary:
                    logger.info(f"Successfully streamed response using model: {model}")
                    stream_success = True
                    break
            except Exception as e:
                errors.append(f"{model}: {e!s}")
                logger.warning(f"Streaming attempt with {model} failed: {e}")

        if not stream_success:
            err_msg = f"Retrieved {len(sources)} context chunk(s), but summary generation encountered an issue.\n\n" + "\n".join(errors)
            full_summary = err_msg
            yield {"type": "token", "content": err_msg}

        yield {"type": "done", "full_summary": full_summary}

    def search_and_summarize(self, query: str, top_k: int = 5, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Executes RAG search and returns only the generated answer text."""
        res = self.search_with_sources(query, top_k=top_k, chat_history=chat_history)
        return res["summary"]

    def rebuild_index(self, data_dir: Union[str, Path] = None):
        """Re-scans document dataset directory and rebuilds FAISS index."""
        target_dir = Path(data_dir) if data_dir else settings.DATA_DIR
        docs = load_all_documents(target_dir)
        if docs:
            self.vectorstore.build_from_documents(docs)
            return len(docs)
        else:
            self.vectorstore.clear()
            return 0

    def add_documents(self, documents: List[Any]) -> int:
        """Incrementally adds document objects to FAISS vector store with SHA-256 deduplication."""
        return self.vectorstore.add_documents(documents)

    def index_single_file(self, file_path: Union[str, Path]) -> int:
        """Loads and incrementally indexes a single document file into FAISS vector store."""
        docs = load_single_document(file_path)
        if docs:
            return self.vectorstore.add_documents(docs)
        return 0

