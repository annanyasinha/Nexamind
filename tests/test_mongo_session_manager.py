import mongomock
import pytest

from core.database import MongoDatabase
from core.session_manager import SessionManager


@pytest.fixture
def mock_mongo_db():
    """Provides a clean mongomock database instance for isolated unit testing."""
    client = mongomock.MongoClient()
    db_wrapper = MongoDatabase()
    db_wrapper.set_client(client)
    return db_wrapper


def test_create_session(mock_mongo_db):
    """Test 1: Create session -> MongoDB document exists in sessions collection."""
    sm = SessionManager(db_instance=mock_mongo_db)
    session = sm.create_session(session_id="test_sess_01", name="Test Session 1")
    
    assert session["session_id"] == "test_sess_01"
    assert session["name"] == "Test Session 1"
    
    # Verify directly in MongoDB collection
    doc = mock_mongo_db.sessions.find_one({"session_id": "test_sess_01"})
    assert doc is not None
    assert doc["name"] == "Test Session 1"


def test_add_user_message(mock_mongo_db):
    """Test 2: add_message() -> messages collection contains user message document."""
    sm = SessionManager(db_instance=mock_mongo_db)
    sm.create_session(session_id="test_sess_02")
    sm.add_message(session_id="test_sess_02", role="user", content="Hello NexaMind!")
    
    messages = list(mock_mongo_db.messages.find({"session_id": "test_sess_02"}))
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello NexaMind!"


def test_conversation_order(mock_mongo_db):
    """Test 3: Messages returned in exact chronological order (user, assistant, user, assistant)."""
    sm = SessionManager(db_instance=mock_mongo_db)
    session_id = "test_sess_03"
    
    sm.add_message_pair(session_id, user_query="What is RAG?", assistant_summary="RAG stands for Retrieval-Augmented Generation.")
    sm.add_message_pair(session_id, user_query="How does FAISS work?", assistant_summary="FAISS performs vector similarity indexing.")
    
    history = sm.get_history(session_id)
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "What is RAG?"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "RAG stands for Retrieval-Augmented Generation."
    assert history[2]["role"] == "user"
    assert history[2]["content"] == "How does FAISS work?"
    assert history[3]["role"] == "assistant"
    assert history[3]["content"] == "FAISS performs vector similarity indexing."


def test_persistence_simulation(mock_mongo_db):
    """Test 4: Reinstantiating SessionManager against same DB preserves session and history."""
    sm_a = SessionManager(db_instance=mock_mongo_db)
    sm_a.create_session(session_id="persist_sess", name="Persistent Session")
    sm_a.add_message_pair("persist_sess", user_query="Query 1", assistant_summary="Summary 1")
    
    # Simulate application restart with new SessionManager instance
    sm_b = SessionManager(db_instance=mock_mongo_db)
    retrieved_session = sm_b.get_session("persist_sess")
    retrieved_history = sm_b.get_history("persist_sess")
    
    assert retrieved_session is not None
    assert retrieved_session["name"] == "Persistent Session"
    assert len(retrieved_history) == 2
    assert retrieved_history[0]["content"] == "Query 1"
    assert retrieved_history[1]["content"] == "Summary 1"


def test_clear_session(mock_mongo_db):
    """Test 5: clear_session() keeps session metadata but deletes messages."""
    sm = SessionManager(db_instance=mock_mongo_db)
    session_id = "test_clear_sess"
    sm.create_session(session_id=session_id, name="Clearable Session")
    sm.add_message(session_id, role="user", content="Will be cleared")
    
    sm.clear_session(session_id)
    
    assert sm.get_session(session_id) is not None  # Metadata preserved
    assert len(sm.get_history(session_id)) == 0   # Messages wiped


def test_delete_session(mock_mongo_db):
    """Test 6: delete_session() deletes both metadata and messages."""
    sm = SessionManager(db_instance=mock_mongo_db)
    session_id = "test_del_sess"
    sm.create_session(session_id=session_id)
    sm.add_message(session_id, role="user", content="To be deleted")
    
    result = sm.delete_session(session_id)
    assert result is True
    assert sm.get_session(session_id) is None
    assert len(sm.get_history(session_id)) == 0


def test_duplicate_session_id(mock_mongo_db):
    """Test 7: Attempting to create duplicate session returns existing without error/duplicates."""
    sm = SessionManager(db_instance=mock_mongo_db)
    s1 = sm.create_session(session_id="dup_sess", name="Original Name")
    s2 = sm.create_session(session_id="dup_sess", name="Different Name")
    
    assert s1["session_id"] == s2["session_id"]
    assert s2["name"] == "Original Name"  # Preserved original
    count = mock_mongo_db.sessions.count_documents({"session_id": "dup_sess"})
    assert count == 1
