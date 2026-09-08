import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, List, Union

import faiss
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings
from core.embeddings import EmbeddingPipeline
from utils.logger import logger

def _extract_rich_metadata(chunk: Any, chunk_idx: int) -> dict:
    """Extracts and preserves full provenance metadata from a document chunk."""
    meta = dict(getattr(chunk, "metadata", {}) or {})
    text = getattr(chunk, "page_content", str(chunk))
    meta["text"] = text

    page_raw = meta.get("page")
    if page_raw is not None and isinstance(page_raw, int):
        meta["page"] = page_raw
        meta["page_number"] = page_raw + 1
    elif "page_number" in meta and isinstance(meta["page_number"], int):
        meta["page"] = meta["page_number"] - 1

    src_val = meta.get("source", "unknown")
    meta["source"] = src_val
    if isinstance(src_val, (str, Path)):
        meta["filename"] = Path(src_val).name
    else:
        meta["filename"] = str(src_val)

    if "chunk_id" not in meta:
        meta["chunk_id"] = chunk_idx + 1

    if "checksum" not in meta or not meta["checksum"]:
        meta["checksum"] = hashlib.sha256(text.encode("utf-8")).hexdigest()

    if "document_type" not in meta:
        ext = Path(meta["filename"]).suffix.lstrip(".").lower()
        meta["document_type"] = meta.get("type") or ext or "document"

    return meta


class FaissVectorStore:
    """
    FAISS-backed vector store for indexing, saving, loading, and performing similarity search using Cosine Similarity.
    """
    def __init__(
        self, 
        persist_dir: Union[str, Path] = None, 
        embedding_model: str = None, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """Initializes FAISS IndexFlatIP storage and Gemini embedding client."""
        self.persist_dir = Path(persist_dir) if persist_dir else settings.FAISS_STORE_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.model = GoogleGenerativeAIEmbeddings(model=self.embedding_model, api_key=settings.GOOGLE_API_KEY)
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(f"Initialized FaissVectorStore persistence path: {self.persist_dir}")

    def get_indexed_checksums(self) -> set:
        """Returns set of all document content SHA-256 checksums currently stored in metadata."""
        checksums = set()
        for meta in self.metadata:
            if isinstance(meta, dict) and meta.get("checksum"):
                checksums.add(meta["checksum"])
        return checksums

    def build_from_documents(self, documents: List[Any]):
        """Chunks documents, generates L2-normalized embeddings, and builds FAISS IndexFlatIP store."""
        logger.info(f"Building vector store from {len(documents)} raw document(s)...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model, 
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(documents)

        if not chunks:
            logger.warning("No document chunks available to index.")
            return

        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [_extract_rich_metadata(chunk, idx) for idx, chunk in enumerate(chunks)]
        self.index = None
        self.metadata = []
        self.add_embeddings(embeddings.astype("float32"), metadatas)
        self.save()
        logger.info(f"Vector store successfully built and saved to {self.persist_dir}")

    def add_documents(self, documents: List[Any]) -> int:
        """
        Incrementally chunks, embeds, and appends new documents to existing FAISS index
        with SHA-256 deduplication and rich metadata preservation. Returns count of newly added docs.
        """
        if not documents:
            logger.info("No documents provided for incremental indexing.")
            return 0

        if self.index is None:
            self.load()

        existing_checksums = self.get_indexed_checksums()
        new_docs = []

        for doc in documents:
            content_str = getattr(doc, "page_content", str(doc))
            checksum = hashlib.sha256(content_str.encode("utf-8")).hexdigest()
            doc_source = getattr(doc, "metadata", {}).get("source", "unknown")

            if checksum in existing_checksums:
                logger.info(f"Skipping duplicate document content from '{doc_source}' (checksum: {checksum[:8]}...)")
                continue

            if hasattr(doc, "metadata") and isinstance(doc.metadata, dict):
                doc.metadata["checksum"] = checksum
            new_docs.append(doc)

        if not new_docs:
            logger.info("All provided documents are already indexed (deduplicated).")
            return 0

        logger.info(f"Incrementally indexing {len(new_docs)} new document(s)...")
        emb_pipe = EmbeddingPipeline(
            model_name=self.embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = emb_pipe.chunk_documents(new_docs)

        if not chunks:
            logger.warning("No new document chunks generated.")
            return 0

        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [_extract_rich_metadata(chunk, idx) for idx, chunk in enumerate(chunks)]
        self.add_embeddings(embeddings.astype("float32"), metadatas)
        self.save()
        logger.info(f"Successfully added {len(new_docs)} document(s) ({len(chunks)} chunks) to FAISS store.")
        return len(new_docs)

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """Adds L2-normalized vector embeddings and metadata to FAISS IndexFlatIP index for Cosine Similarity."""
        if embeddings.size == 0:
            logger.warning("Attempted to add empty embeddings array.")
            return

        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info(f"Added {embeddings.shape[0]} normalized vector(s) to FAISS IndexFlatIP index.")

    def save(self):
        """Persists the FAISS index and metadata pickle file to disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if self.index is not None:
            faiss.write_index(self.index, faiss_path)
            with open(meta_path, "wb") as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):
        """Loads the persisted FAISS index and metadata pickle file from disk."""
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            logger.info(f"Successfully loaded FAISS index with {self.index.ntotal} vectors from {self.persist_dir}")
        else:
            logger.warning(f"Index or metadata file not found in {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5, min_similarity: float = None):
        """Executes Inner Product (Cosine Similarity) search against FAISS index and filters by min_similarity threshold."""
        if self.index is None:
            logger.warning("FAISS index is not initialized or loaded.")
            return []

        scores, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            score_val = float(score)
            if min_similarity is not None and score_val < min_similarity:
                continue
            meta = self.metadata[idx]
            results.append({
                "index": int(idx),
                "similarity_score": score_val,
                "metadata": meta
            })
        return results

    def clear(self):
        """Clears in-memory vectors and deletes index persistence files on disk."""
        self.index = None
        self.metadata = []
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path):
            try:
                os.remove(faiss_path)
            except Exception as e:
                logger.warning(f"Failed to delete faiss.index: {e}")
        if os.path.exists(meta_path):
            try:
                os.remove(meta_path)
            except Exception as e:
                logger.warning(f"Failed to delete metadata.pkl: {e}")
        logger.info(f"Cleared FAISS vector store index files at {self.persist_dir}")

    def query(self, query_text: str, top_k: int = 5, min_similarity: float = None):
        """Generates L2-normalized query vector embedding and retrieves top-k matching document chunks."""
        logger.info(f"Querying vector store: '{query_text}' (top_k={top_k}, min_similarity={min_similarity})")
        query_emb = np.array(self.model.embed_query(query_text), dtype="float32").reshape(1, -1)
        faiss.normalize_L2(query_emb)
        return self.search(query_emb, top_k=top_k, min_similarity=min_similarity)


