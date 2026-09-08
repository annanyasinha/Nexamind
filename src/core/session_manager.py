import datetime
import uuid
from typing import Any, Dict, List, Optional

from core.database import mongo_db
from utils.logger import logger


class SessionManager:
    """
    Manages persistent multi-session conversation states and chat history in MongoDB Atlas.
    """
    def __init__(self, db_instance=None):
        """Initializes collections and creates database indexes."""
        self._db = db_instance or mongo_db
        self._init_indexes()
        self._ensure_default_session()

    @property
    def sessions_collection(self):
        return self._db.sessions

    @property
    def messages_collection(self):
        return self._db.messages

    def _init_indexes(self):
        """Creates indexes for fast lookup and session uniqueness."""
        try:
            self.sessions_collection.create_index("session_id", unique=True)
            self.messages_collection.create_index([("session_id", 1), ("created_at", 1)])
            logger.info("MongoDB session and message collection indexes initialized.")
        except Exception as e:
            logger.warning(f"Database index creation warning: {e}")

    def _ensure_default_session(self):
        """Ensures the default chat session exists on startup."""
        try:
            if not self.get_session("default"):
                self.create_session(session_id="default", name="Default Session")
        except Exception as e:
            logger.warning(f"Could not initialize default session: {e}")

    def create_session(self, session_id: Optional[str] = None, name: Optional[str] = None) -> Dict[str, Any]:
        """Creates and registers a new chat session with a unique ID in MongoDB."""
        if not session_id:
            session_id = f"session_{uuid.uuid4().hex[:8]}"

        existing = self.sessions_collection.find_one({"session_id": session_id})
        if existing:
            existing.pop("_id", None)
            return existing

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        session_doc = {
            "session_id": session_id,
            "name": name or f"Session {self.sessions_collection.count_documents({}) + 1}",
            "created_at": now,
            "updated_at": now
        }
        self.sessions_collection.insert_one(session_doc)
        session_doc.pop("_id", None)
        logger.info(f"Created persistent chat session: '{session_id}' ({session_doc['name']})")
        return session_doc

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves session document from MongoDB by session_id."""
        session = self.sessions_collection.find_one({"session_id": session_id})
        if session:
            session.pop("_id", None)
        return session

    def list_sessions(self) -> List[Dict[str, Any]]:
        """Returns metadata and message counts for all active sessions in MongoDB."""
        result = []
        for session in self.sessions_collection.find():
            s_id = session["session_id"]
            count = self.messages_collection.count_documents({"session_id": s_id})
            result.append({
                "session_id": s_id,
                "name": session.get("name", "Session"),
                "created_at": str(session.get("created_at", "")),
                "message_count": count
            })
        return result

    def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Fetches chronologically sorted conversation message history for a given session ID."""
        cursor = self.messages_collection.find(
            {"session_id": session_id},
            {"_id": 0}
        ).sort([("created_at", 1), ("_id", 1)])
        return list(cursor)

    def add_message(
        self, 
        session_id: str, 
        role: str, 
        content: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Appends a standardized user or assistant message document to MongoDB history."""
        if not self.get_session(session_id):
            self.create_session(session_id=session_id)

        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        msg_doc = {
            "session_id": session_id,
            "role": role,
            "content": content,
            "sources": sources or [],
            "created_at": now
        }
        self.messages_collection.insert_one(msg_doc)
        self.sessions_collection.update_one(
            {"session_id": session_id},
            {"$set": {"updated_at": now}}
        )
        logger.info(f"Added '{role}' message to MongoDB session '{session_id}'.")

    def add_message_pair(
        self, 
        session_id: str, 
        user_query: str, 
        assistant_summary: str, 
        sources: Optional[List[Dict[str, Any]]] = None
    ):
        """Appends separate user query and assistant response message turns to MongoDB session history."""
        self.add_message(session_id=session_id, role="user", content=user_query)
        self.add_message(session_id=session_id, role="assistant", content=assistant_summary, sources=sources)

    def clear_session(self, session_id: str):
        """Deletes all message documents for a specific session ID while keeping session metadata intact."""
        self.messages_collection.delete_many({"session_id": session_id})
        logger.info(f"Cleared message history for session '{session_id}' in MongoDB.")

    def delete_session(self, session_id: str) -> bool:
        """Deletes session metadata and message documents from MongoDB."""
        if session_id == "default":
            self.clear_session("default")
            logger.info("Cleared default session messages in MongoDB.")
            return True

        res = self.sessions_collection.delete_one({"session_id": session_id})
        self.messages_collection.delete_many({"session_id": session_id})
        logger.info(f"Deleted session '{session_id}' and its messages from MongoDB.")
        return res.deleted_count > 0


# Singleton session_manager instance
session_manager = SessionManager()
