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


class YouTubeTranscriptRequest(BaseModel):
    url: str = Field(..., description="YouTube video URL or Video ID", example="https://www.youtube.com/watch?v=jNQXAC9IVRw")
    save_to_dataset: bool = Field(default=True, description="Whether to save transcript text into settings.DATA_DIR dataset")
    auto_reindex: bool = Field(default=True, description="Whether to automatically rebuild FAISS index after adding transcript")


class YouTubeSegment(BaseModel):
    start: float
    duration: float
    timestamp: str
    text: str


class YouTubeTranscriptResponse(BaseModel):
    video_id: str
    url: str
    full_text: str
    raw_text: str
    segment_count: int
    segments: List[YouTubeSegment]
    saved_file: Optional[str] = None
    indexed_documents_count: int = 0


class AgentQueryRequest(BaseModel):
    query: str = Field(..., description="User query for NexaMind AI Agent", example="What is discussed in video J5_-l7WIO_w and what are the latest news on AI?")
    session_id: Optional[str] = Field(default=None, description="Optional Session ID to automatically maintain chat history")
    chat_history: Optional[List[dict]] = Field(default=[], description="Explicit chat history")
    enabled_tools: Optional[List[str]] = Field(default=None, description="List of tools to enable: document_rag, youtube_rag, web_search")


class ToolCallStep(BaseModel):
    tool: str
    input: str
    output: str
    execution_time_ms: float


class AgentQueryResponse(BaseModel):
    query: str
    answer: str
    session_id: Optional[str] = None
    steps: List[ToolCallStep]
    sources: List[Dict[str, Any]]
    execution_time_ms: float


