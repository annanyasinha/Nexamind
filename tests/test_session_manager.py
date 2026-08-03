from core.session_manager import SessionManager

def test_session_creation():
    """Tests session instantiation and registration."""
    mgr = SessionManager()
    session = mgr.create_session(session_id="test_sess", name="Test Session")
    assert session["session_id"] == "test_sess"
    assert session["name"] == "Test Session"
    assert session["history"] == []

def test_add_message_pair():
    """Tests adding Q&A message turns to a session history."""
    mgr = SessionManager()
    mgr.add_message_pair("test_sess", "Hi", "Hello!", sources=[])
    history = mgr.get_history("test_sess")
    assert len(history) == 1
    assert history[0]["query"] == "Hi"
    assert history[0]["summary"] == "Hello!"

def test_delete_session():
    """Tests removing a session from the session store."""
    mgr = SessionManager()
    mgr.create_session(session_id="test_del", name="To Delete")
    assert mgr.get_session("test_del") is not None
    deleted = mgr.delete_session("test_del")
    assert deleted is True
    assert mgr.get_session("test_del") is None

