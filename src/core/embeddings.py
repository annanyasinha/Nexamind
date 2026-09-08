from typing import Any, List

import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from utils.logger import logger


class EmbeddingPipeline:
    """
    Handles chunking of documents and vector embedding generation.
    """
    def __init__(
        self, 
        model_name: str = None, 
        chunk_size: int = None, 
        chunk_overlap: int = None
    ):
        """Initializes the document chunking splitter and Google GenAI embedding model."""
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=self.model_name, 
            api_key=settings.GOOGLE_API_KEY
        )
        logger.info(f"Initialized embedding pipeline using model: {self.model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Splits raw documents into smaller textual chunks using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} document(s) into {len(chunks)} chunk(s).")
        return chunks

    def embed_chunks(self, chunks: List[Any], batch_size: int = 10) -> np.ndarray:
        """Generates dense vector embeddings for document chunks using the Gemini embedding API in batches with retries."""
        import time

        if not chunks:
            logger.info("No chunks provided for embedding.")
            return np.empty((0, 0), dtype="float32")

        texts = [getattr(chunk, "page_content", str(chunk)) for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s) in batches of {batch_size}...")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    batch_embeddings = self.embedding_model.embed_documents(batch)
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        if attempt < max_retries - 1:
                            wait_time = (attempt + 1) * 3
                            logger.warning(f"Rate limit reached (429). Retrying batch {i // batch_size + 1}/{(len(texts) + batch_size - 1) // batch_size} in {wait_time}s...")
                            time.sleep(wait_time)
                        else:
                            logger.error(f"Batch {i // batch_size + 1} failed after {max_retries} retries: {e}")
                            raise e
                    else:
                        raise e

        logger.info(f"Embeddings successfully generated. Shape: ({len(all_embeddings)}, {len(all_embeddings[0]) if all_embeddings else 0})")
        return np.array(all_embeddings, dtype="float32")

