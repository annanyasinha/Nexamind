from fastapi import APIRouter, HTTPException
from api.schemas import SessionCreateRequest
from core.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Session Management"])

@router.get("")
@router.get("/")
def list_sessions():
    """Returns a list of all active chat sessions and turn metrics."""
    return {"sessions": session_manager.list_sessions()}

@router.post("")
@router.post("/")
def create_session(req: SessionCreateRequest = None):
    """Creates a new active conversation session."""
    session_id = req.session_id if req else None
    name = req.name if req else None
    s = session_manager.create_session(session_id=session_id, name=name)
    return {"message": "Session created successfully", "session": s}

@router.get("/{session_id}")
def get_session(session_id: str):
    """Retrieves session details and full Q&A turn history for a given session ID."""
    s = session_manager.get_session(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return s

@router.delete("/{session_id}")
def delete_session(session_id: str):
    """Deletes a chat session or clears turn history if default."""
    success = session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or cannot be deleted")
    return {"message": f"Session '{session_id}' cleared/deleted successfully"}

