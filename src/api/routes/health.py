import os

from fastapi import APIRouter

from api.deps import get_rag_search
from api.schemas import SystemStatusResponse
from config import settings
from core.database import mongo_db
from core.session_manager import session_manager

router = APIRouter(tags=["Health & Info"])

@router.get("/")
def root():
    """Returns basic system API metadata and registered endpoints."""
    return {
        "name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "search": "/search (POST)",
            "sessions": "/sessions (GET, POST)",
            "session_detail": "/sessions/{session_id} (GET, DELETE)",
            "documents": "/documents (GET)",
            "upload": "/upload (POST)",
            "reindex": "/reindex (POST)"
        }
    }

@router.get("/health", response_model=SystemStatusResponse)
def health_check():
    """Returns live system health status, vector count, and dataset statistics."""
    rag = get_rag_search()
    files = []
    if settings.DATA_DIR.exists():
        files = [f for f in os.listdir(settings.DATA_DIR) if not f.startswith(".")]
    
    total_vectors = len(rag.vectorstore.metadata) if rag.vectorstore and rag.vectorstore.metadata else 0
    
    # Verify MongoDB connectivity
    db_ok = mongo_db.ping()
    status_str = "ok" if db_ok else "warning: mongodb ping failed"
    
    sessions_list = session_manager.list_sessions()
    return SystemStatusResponse(
        status=status_str,
        persist_dir=str(rag.vectorstore.persist_dir),
        total_vectors=total_vectors,
        data_files_count=len(files),
        active_sessions=len(sessions_list),
        data_files=files
    )
