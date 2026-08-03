import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.schemas import QueryRequest, QueryResponse, SourceItem, RawSearchResponse, RawSearchResult
from api.deps import get_rag_search
from core.session_manager import session_manager

router = APIRouter(tags=["RAG Services"])

@router.post("/query", response_model=QueryResponse)
def rag_query(request: QueryRequest):
    """Executes session-aware RAG vector search and LLM summary generation."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    rag = get_rag_search()
    chat_history = request.chat_history or []
    target_session_id = request.session_id
    
    if target_session_id:
        existing = session_manager.get_session(target_session_id)
        if not existing:
            session_manager.create_session(session_id=target_session_id)
        chat_history = session_manager.get_history(target_session_id)

    try:
        res = rag.search_with_sources(
            query=request.query, 
            top_k=request.top_k, 
            chat_history=chat_history
        )
        
        if target_session_id:
            session_manager.add_message_pair(
                session_id=target_session_id,
                user_query=res["query"],
                assistant_summary=res["summary"],
                sources=res["sources"]
            )

        return QueryResponse(
            query=res["query"],
            summary=res["summary"],
            session_id=target_session_id,
            sources=[
                SourceItem(
                    index=s["index"],
                    distance=s["distance"],
                    text=s["text"],
                    metadata=s["metadata"]
                ) for s in res["sources"]
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query execution failed: {str(e)}")

@router.post("/query/stream")
def rag_query_stream(request: QueryRequest):
    """Streams session-aware RAG vector context and LLM response tokens as Server-Sent Events."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    rag = get_rag_search()
    chat_history = request.chat_history or []
    target_session_id = request.session_id
    
    if target_session_id:
        existing = session_manager.get_session(target_session_id)
        if not existing:
            session_manager.create_session(session_id=target_session_id)
        chat_history = session_manager.get_history(target_session_id)

    def event_generator():
        """Yields JSON-formatted event frames for SSE streaming."""
        accumulated_summary = ""
        retrieved_sources = []
        
        for event in rag.search_with_sources_stream(
            query=request.query,
            top_k=request.top_k,
            chat_history=chat_history
        ):
            if event["type"] == "sources":
                retrieved_sources = event.get("sources", [])
            elif event["type"] == "token":
                accumulated_summary += event.get("content", "")
            elif event["type"] == "done":
                if target_session_id:
                    session_manager.add_message_pair(
                        session_id=target_session_id,
                        user_query=request.query,
                        assistant_summary=accumulated_summary,
                        sources=retrieved_sources
                    )
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/search", response_model=RawSearchResponse)
def vector_search(request: QueryRequest):
    """Executes raw FAISS vector similarity search without LLM summarization."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")
    
    rag = get_rag_search()
    try:
        raw_results = rag.vectorstore.query(request.query, top_k=request.top_k)
        formatted_results = [
            RawSearchResult(
                index=int(r.get("index", -1)),
                distance=float(r.get("distance", 0.0)),
                metadata=r.get("metadata")
            ) for r in raw_results
        ]
        return RawSearchResponse(query=request.query, results=formatted_results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector search failed: {str(e)}")

