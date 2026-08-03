from typing import List, Any
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Generates dense vector embeddings for document chunks using the Gemini embedding API."""
        if not chunks:
            logger.info("No chunks provided for embedding.")
            return np.empty((0, 0), dtype="float32")

        texts = [getattr(chunk, "page_content", str(chunk)) for chunk in chunks]
        logger.info(f"Generating embeddings for {len(texts)} chunk(s)...")
        embeddings = self.embedding_model.embed_documents(texts)
        logger.info(f"Embeddings successfully generated. Shape: ({len(embeddings)}, {len(embeddings[0]) if embeddings else 0})")
        return np.array(embeddings, dtype="float32")

