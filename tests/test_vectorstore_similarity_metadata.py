import numpy as np
import faiss
import pytest
from langchain_core.documents import Document

from config import settings
from core.vectorstore import FaissVectorStore, _extract_rich_metadata
from core.prompt import build_rag_prompt, RAG_SYSTEM_INSTRUCTION, NO_CONTEXT_FOUND_MESSAGE
from core.search_engine import RAGSearch


def test_rich_metadata_extraction():
    """Tests that _extract_rich_metadata preserves full provenance metadata and computes 1-indexed page_number."""
    doc_chunk = Document(
        page_content="This is sample text on page 0.",
        metadata={
            "source": "/path/to/research_paper.pdf",
            "page": 0,
            "type": "pdf"
        }
    )
    meta = _extract_rich_metadata(doc_chunk, chunk_idx=4)
    
    assert meta["text"] == "This is sample text on page 0."
    assert meta["source"] == "/path/to/research_paper.pdf"
    assert meta["filename"] == "research_paper.pdf"
    assert meta["page"] == 0
    assert meta["page_number"] == 1  # 1-indexed page number
    assert meta["chunk_id"] == 5    # 1-indexed chunk ID
    assert meta["document_type"] == "pdf"
    assert len(meta["checksum"]) == 64  # valid sha-256


def test_index_flat_ip_l2_normalization(tmp_path):
    """Tests that FaissVectorStore uses IndexFlatIP with L2-normalized vectors to calculate Cosine Similarity."""
    vstore = FaissVectorStore(persist_dir=tmp_path / "faiss_test")
    
    # 2 dimensional test vectors
    raw_vecs = np.array([[3.0, 4.0], [1.0, 0.0]], dtype="float32")
    metadatas = [
        {"source": "vec1.txt", "text": "Vector 1"},
        {"source": "vec2.txt", "text": "Vector 2"}
    ]
    
    vstore.add_embeddings(raw_vecs, metadatas)
    assert isinstance(vstore.index, faiss.IndexFlatIP)
    
    # Query vector identical to vec1: [3.0, 4.0] -> after normalization [0.6, 0.8]
    query_vec = np.array([[3.0, 4.0]], dtype="float32")
    faiss.normalize_L2(query_vec)
    
    results = vstore.search(query_vec, top_k=2)
    assert len(results) == 2
    # Exact match dot product of normalized identical vector should be ~1.0
    assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-4)
    assert results[0]["metadata"]["source"] == "vec1.txt"


def test_similarity_filtering_and_abstention(tmp_path):
    """Tests filtering by min_similarity threshold and abstention behavior when scores are low."""
    vstore = FaissVectorStore(persist_dir=tmp_path / "faiss_thresh")
    
    # Orthogonal vectors: dot product = 0.0
    raw_vecs = np.array([[1.0, 0.0]], dtype="float32")
    metadatas = [{"source": "ortho.txt", "text": "Orthogonal text"}]
    vstore.add_embeddings(raw_vecs, metadatas)
    
    query_vec = np.array([[0.0, 1.0]], dtype="float32")
    faiss.normalize_L2(query_vec)
    
    # Search with min_similarity = 0.60
    results = vstore.search(query_vec, top_k=1, min_similarity=0.60)
    assert len(results) == 0  # Rejects chunk below 0.60 threshold


def test_prompt_system_instruction_and_context_headers():
    """Tests that prompt instructions mandate inline citations and context sections feature rich headers."""
    assert "Crucial Instruction: Cite your statements with inline context tags such as [Source 1], [Source 2]" in RAG_SYSTEM_INSTRUCTION
    
    context = "--- Context [Source 1 | resume.pdf | Page 2] ---\nCandidate has 5 years of Python experience."
    prompt = build_rag_prompt(query="What experience does candidate have?", context=context)
    
    assert "[Source 1 | resume.pdf | Page 2]" in prompt
    assert "Candidate has 5 years of Python experience." in prompt
