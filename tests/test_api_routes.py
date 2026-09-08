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


def test_upload_unsupported_file_extension(tmp_path, monkeypatch):
    """Tests that uploading an unallowed file extension (e.g. .exe) returns HTTP 400."""
    import config as cfg
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)

    files = [("files", ("malicious.exe", b"binary data", "application/octet-stream"))]
    response = client.post("/upload", files=files, data={"auto_reindex": "false"})
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


def test_upload_path_traversal_prevention(tmp_path, monkeypatch):
    """Tests that directory traversal filenames (e.g. ../../secret.txt) are sanitized to secret.txt inside DATA_DIR."""
    import config as cfg
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)

    files = [("files", ("../../secret.txt", b"secret content", "text/plain"))]
    response = client.post("/upload", files=files, data={"auto_reindex": "false"})
    assert response.status_code == 200
    assert response.json()["saved_files"] == ["secret.txt"]
    assert (test_data_dir / "secret.txt").exists()
    assert not (tmp_path / "secret.txt").exists()


def test_delete_path_traversal_prevention(tmp_path, monkeypatch):
    """Tests that path traversal delete attempts are sanitized and confined to DATA_DIR."""
    import config as cfg
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)

    response = client.delete("/documents/..%2F..%2Fsome_external_file.txt?auto_reindex=false")
    assert response.status_code == 404


def test_upload_incremental_and_deduplication(tmp_path, monkeypatch):
    """Tests incremental addition of new files and SHA-256 deduplication of identical content."""
    import config as cfg
    from api.deps import get_rag_search
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    test_faiss_dir = tmp_path / "faiss_store"
    test_faiss_dir.mkdir()

    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(cfg.settings, "FAISS_STORE_DIR", test_faiss_dir)

    rag = get_rag_search()
    rag.vectorstore.persist_dir = test_faiss_dir
    rag.vectorstore.clear()

    # 1. Upload first brand new file -> Incremental add (1 doc)
    f1 = [("files", ("doc1.txt", b"First unique document text", "text/plain"))]
    res1 = client.post("/upload", files=f1, data={"auto_reindex": "true"})
    assert res1.status_code == 200
    assert res1.json()["indexed_documents_count"] == 1

    # 2. Upload duplicate content (same SHA) -> Skipped embedding (0 docs added)
    f1_dup = [("files", ("doc1.txt", b"First unique document text", "text/plain"))]
    res_dup = client.post("/upload", files=f1_dup, data={"auto_reindex": "true"})
    assert res_dup.status_code == 200
    assert "duplicate file(s) skipped" in res_dup.json()["message"]
    assert res_dup.json()["indexed_documents_count"] == 0

    # 3. Upload second brand new file -> Incremental add (1 doc added)
    f2 = [("files", ("doc2.txt", b"Second unique document text", "text/plain"))]
    res2 = client.post("/upload", files=f2, data={"auto_reindex": "true"})
    assert res2.status_code == 200
    assert res2.json()["indexed_documents_count"] == 1


def test_upload_same_filename_changed_content(tmp_path, monkeypatch):
    """Tests that uploading an existing filename with changed content triggers a full rebuild."""
    import config as cfg
    from api.deps import get_rag_search
    test_data_dir = tmp_path / "data"
    test_data_dir.mkdir()
    test_faiss_dir = tmp_path / "faiss_store"
    test_faiss_dir.mkdir()

    monkeypatch.setattr(cfg.settings, "DATA_DIR", test_data_dir)
    monkeypatch.setattr(cfg.settings, "FAISS_STORE_DIR", test_faiss_dir)

    rag = get_rag_search()
    rag.vectorstore.persist_dir = test_faiss_dir
    rag.vectorstore.clear()

    # Upload original version of resume.txt
    f_orig = [("files", ("resume.txt", b"Original resume version content", "text/plain"))]
    res_orig = client.post("/upload", files=f_orig, data={"auto_reindex": "true"})
    assert res_orig.status_code == 200
    assert res_orig.json()["indexed_documents_count"] == 1

    # Upload updated version of resume.txt (same filename, changed content) -> Full rebuild
    f_updated = [("files", ("resume.txt", b"Updated resume version content with new skills", "text/plain"))]
    res_up = client.post("/upload", files=f_updated, data={"auto_reindex": "true"})
    assert res_up.status_code == 200
    assert res_up.json()["indexed_documents_count"] == 1




