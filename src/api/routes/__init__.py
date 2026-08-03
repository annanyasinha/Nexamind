"""
FastAPI Routes package
"""
from .health import router as health_router
from .sessions import router as sessions_router
from .rag import router as rag_router
from .documents import router as documents_router

__all__ = [
    "health_router",
    "sessions_router",
    "rag_router",
    "documents_router"
]
