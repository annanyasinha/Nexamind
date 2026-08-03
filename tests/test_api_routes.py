from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_root_endpoint():
    """Tests the root metadata API endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "endpoints" in data

def test_list_sessions():
    """Tests listing active sessions via API."""
    response = client.get("/sessions")
    assert response.status_code == 200
    data = response.json()
    assert "sessions" in data

def test_create_session_endpoint():
    """Tests session creation API endpoint."""
    response = client.post("/sessions", json={"name": "API Test Session"})
    assert response.status_code == 200
    data = response.json()
    assert "session" in data
    assert data["session"]["name"] == "API Test Session"

def test_list_documents_endpoint():
    """Tests document list API endpoint."""
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data

def test_delete_document_not_found():
    """Tests 404 response for deleting non-existent file."""
    response = client.delete("/documents/non_existent_file_123.txt")
    assert response.status_code == 404

def test_delete_and_clear_documents(tmp_path, monkeypatch):
    """Tests deleting single file and clearing all files via API."""
    import config as cfg
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    (test_data_dir / "sample.txt").write_text("Sample document text for test")
    
    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)
    
    # Verify file is listed
    res_list = client.get("/documents")
    assert res_list.status_code == 200
    assert any(d["filename"] == "sample.txt" for d in res_list.json()["documents"])
    
    # Delete single file
    res_del = client.delete("/documents/sample.txt?auto_reindex=false")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"
    
    # Re-create & clear all
    (test_data_dir / "doc1.txt").write_text("doc 1")
    (test_data_dir / "doc2.txt").write_text("doc 2")
    res_clear = client.delete("/documents")
    assert res_clear.status_code == 200
    assert len(list(test_data_dir.glob("*.txt"))) == 0

def test_query_stream_endpoint():
    """Tests SSE streaming query endpoint response headers."""
    response = client.post("/query/stream", json={"query": "Test streaming", "top_k": 2})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")



