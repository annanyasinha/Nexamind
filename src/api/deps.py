from typing import Optional
from core.search_engine import RAGSearch
from utils.logger import logger

_rag_search_instance: Optional[RAGSearch] = None

def get_rag_search() -> RAGSearch:
    """
    Dependency injector for singleton RAGSearch instance.
    """
    global _rag_search_instance
    if _rag_search_instance is None:
        logger.info("Initializing RAGSearch singleton instance for API...")
        _rag_search_instance = RAGSearch()
    return _rag_search_instance
