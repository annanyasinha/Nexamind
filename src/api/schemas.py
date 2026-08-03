from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., description="User query string for RAG", example="Where did Shubham study?")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of top context chunks to retrieve")
    session_id: Optional[str] = Field(default=None, description="Optional Session ID to automatically maintain chat history")
    chat_history: Optional[List[dict]] = Field(default=[], description="Explicit chat history (used if session_id is not provided)")


class SourceItem(BaseModel):
    index: int
    distance: float
    text: str
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    query: str
    summary: str
    session_id: Optional[str] = None
    sources: List[SourceItem]


class RawSearchResult(BaseModel):
    index: int
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class RawSearchResponse(BaseModel):
    query: str
    results: List[RawSearchResult]


class SystemStatusResponse(BaseModel):
    status: str
    persist_dir: str
    total_vectors: int
    data_files_count: int
    active_sessions: int
    data_files: List[str]


class UploadResponse(BaseModel):
    message: str
    saved_files: List[str]
    indexed_documents_count: int


class SessionCreateRequest(BaseModel):
    session_id: Optional[str] = Field(None, description="Optional custom session ID")
    name: Optional[str] = Field(None, description="Optional session title/name")
