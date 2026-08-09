import uuid
import datetime
from typing import Dict, List, Any, Optional
from utils.logger import logger


class SessionManager:
    """
    Manages multi-session conversation states, chat histories, and memory retention.
    """
    def __init__(self):
        """Initializes the session store with a default active session."""
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.create_session(session_id="default", name="Default Session")

    def create_session(self, session_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        """Creates and registers a new chat session with a unique ID."""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"
        
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        if not name:
            name = f"Session {len(self.sessions) + 1}"
            
        session_data = {
            "session_id": session_id,
            "name": name,
            "created_at": datetime.datetime.now().isoformat(),
            "history": []
        }
        self.sessions[session_id] = session_data
        logger.info(f"Created chat session: '{session_id}' ({name})")
        return session_data

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session data for the specified session ID."""
        return self.sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns a list of all active session metadata and turn counts."""
        result = []
        for s_id, s_data in self.sessions.items():
            result.append({
                "session_id": s_id,
                "name": s_data["name"],
                "created_at": s_data["created_at"],
                "message_count": len(s_data["history"])
            })
        return result

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches the conversation turn history for a given session ID."""
        session = self.get_session(session_id)
        if session:
            return session["history"]
        return []

    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Appends a single message turn (user or assistant) to the session history."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id=session_id)
            
        entry = {
            "role": role,
            "content": content,
            "sources": sources or []
        }
        if role == "assistant":
            entry["summary"] = content
        elif role == "user":
            entry["query"] = content

        session["history"].append(entry)
        logger.info(f"Added {role} message to session '{session_id}'. Total turns: {len(session['history'])}")

    def add_message_pair(
        self, 
        session_id: str, 
        user_query: str, 
        assistant_summary: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Appends a user query and assistant response turn to the session history."""
        session = self.get_session(session_id)
        if not session:
            session = self.create_session(session_id=session_id)
            
        entry = {
            "role": "user",
            "query": user_query,
            "content": user_query,
            "summary": assistant_summary,
            "sources": sources or []
        }
        session["history"].append(entry)
        logger.info(f"Added Q&A turn to session '{session_id}'. Total turns: {len(session['history'])}")

    def clear_session(self, session_id: str):
        """Clears all conversation turns for the specified session."""
        session = self.get_session(session_id)
        if session:
            session["history"] = []
            logger.info(f"Cleared turn history for session '{session_id}'")

    def delete_session(self, session_id: str) -> bool:
        """Deletes a session from the session store or resets it if default."""
        if session_id in self.sessions and session_id != "default":
            del self.sessions[session_id]
            logger.info(f"Deleted chat session '{session_id}'")
            return True
        elif session_id == "default":
            self.sessions["default"]["history"] = []
            logger.info("Cleared default chat session history.")
            return True
        return False



# Global instance export
session_manager = SessionManager()
