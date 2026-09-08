"""
FastAPI Routes package
"""
from .agent import router as agent_router
from .documents import router as documents_router
from .health import router as health_router
from .rag import router as rag_router
from .sessions import router as sessions_router
from .youtube import router as youtube_router

__all__ = [
    "agent_router",
    "documents_router",
    "health_router",
    "rag_router",
    "sessions_router",
    "youtube_router"
]
