import os
from pathlib import Path
import faiss
import numpy as np
import pickle
from typing import List, Any, Union
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from core.embeddings import EmbeddingPipeline
from config import settings
from utils.logger import logger


class FaissVectorStore:
    """
    FAISS-backed vector store for indexing, saving, loading, and performing similarity search.
    """
    def __init__(
        self, 
        persist_dir: Union[str, Path] = None, 
        embedding_model: str = None, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """Initializes FAISS index storage and Gemini embedding client."""
        self.persist_dir = Path(persist_dir) if persist_dir else settings.FAISS_STORE_DIR
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model or settings.EMBEDDING_MODEL
        self.model = GoogleGenerativeAIEmbeddings(model=self.embedding_model, api_key=settings.GOOGLE_API_KEY)
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        logger.info(f"Initialized FaissVectorStore persistence path: {self.persist_dir}")

    def build_from_documents(self, documents: List[Any]):
        """Chunks documents, generates embeddings, and builds FAISS index."""
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
        metadatas = [
            {
                "text": getattr(chunk, "page_content", str(chunk)),
                "source": getattr(chunk, "metadata", {}).get("source", "unknown")
            } 
            for chunk in chunks
        ]
        self.index = None
        self.metadata = []
        self.add_embeddings(embeddings.astype("float32"), metadatas)
        self.save()
        logger.info(f"Vector store successfully built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        """Adds floating point vector embeddings and metadata to FAISS index."""
        if embeddings.size == 0:
            logger.warning("Attempted to add empty embeddings array.")
            return

        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        logger.info(f"Added {embeddings.shape[0]} vector(s) to FAISS index.")

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

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Executes L2 distance similarity search against FAISS index."""
        if self.index is None:
            logger.warning("FAISS index is not initialized or loaded.")
            return []

        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = self.metadata[idx]
            results.append({"index": int(idx), "distance": float(dist), "metadata": meta})
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

    def query(self, query_text: str, top_k: int = 5):
        """Generates query vector embedding and retrieves top-k matching document chunks."""
        logger.info(f"Querying vector store: '{query_text}' (top_k={top_k})")
        query_emb = np.array(self.model.embed_query(query_text), dtype="float32").reshape(1, -1)
        return self.search(query_emb, top_k=top_k)


