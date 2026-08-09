from typing import Optional
from core.search_engine import RAGSearch
from core.agent import NexaMindAgent
from utils.logger import logger

_rag_search_instance: Optional[RAGSearch] = None
_agent_instance: Optional[NexaMindAgent] = None

def get_rag_search() -> RAGSearch:
    """
    Dependency injector for singleton RAGSearch instance.
    """
    global _rag_search_instance
    if _rag_search_instance is None:
        logger.info("Initializing RAGSearch singleton instance for API...")
        _rag_search_instance = RAGSearch()
    return _rag_search_instance


def get_nexamind_agent() -> NexaMindAgent:
    """
    Dependency injector for singleton NexaMindAgent instance.
    """
    global _agent_instance
    if _agent_instance is None:
        logger.info("Initializing NexaMindAgent singleton instance for API...")
        _agent_instance = NexaMindAgent()
    return _agent_instance

