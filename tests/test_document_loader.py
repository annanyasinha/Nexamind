from core.document_loader import load_all_documents

def test_load_all_documents(tmp_data_dir):
    """Tests loading documents from sample data directory."""
    docs = load_all_documents(tmp_data_dir)
    assert len(docs) == 1
    assert "Annanya Sinha" in docs[0].page_content

def test_load_non_existent_dir(tmp_path):
    """Tests behavior when scanning a non-existent document directory."""
    non_existent = tmp_path / "does_not_exist"
    docs = load_all_documents(non_existent)
    assert docs == []


def test_load_json_document(tmp_path):
    """Tests loading JSON documents."""
    json_file = tmp_path / "sample.json"
    json_file.write_text('{"name": "Test User", "role": "Developer"}')
    docs = load_all_documents(tmp_path)
    assert len(docs) == 1
    assert "Test User" in docs[0].page_content


