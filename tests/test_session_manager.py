from core.session_manager import SessionManager

def test_session_creation():
    """Tests session instantiation and registration."""
    mgr = SessionManager()
    session = mgr.create_session(session_id="test_sess_legacy", name="Test Session")
    assert session["session_id"] == "test_sess_legacy"
    assert session["name"] == "Test Session"
    assert mgr.get_history("test_sess_legacy") == []

def test_add_message_pair():
    """Tests adding Q&A message turns to a session history."""
    mgr = SessionManager()
    mgr.add_message_pair("test_sess_legacy", "Hi", "Hello!", sources=[])
    history = mgr.get_history("test_sess_legacy")
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[0]["content"] == "Hi"
    assert history[1]["role"] == "assistant"
    assert history[1]["content"] == "Hello!"

def test_delete_session():
    """Tests removing a session from the session store."""
    mgr = SessionManager()
    mgr.create_session(session_id="test_del_legacy", name="To Delete")
    assert mgr.get_session("test_del_legacy") is not None
    deleted = mgr.delete_session("test_del_legacy")
    assert deleted is True
    assert mgr.get_session("test_del_legacy") is None

