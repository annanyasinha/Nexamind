import pytest
import numpy as np
import faiss
from unittest.mock import MagicMock, patch

from config import settings
from core.youtube_loader import (
    extract_youtube_id,
    format_timestamp,
    chunk_youtube_transcript,
    get_or_create_youtube_vectorstore
)
from core.tools import YouTubeRAGTool


@pytest.fixture
def sample_transcript_data():
    return {
        "video_id": "J5_-l7WIO_w",
        "url": "https://www.youtube.com/watch?v=J5_-l7WIO_w",
        "full_text": "[00:00] Hello world\n[00:15] Neural networks and deep learning\n[01:10] Vector databases like FAISS",
        "raw_text": "Hello world Neural networks and deep learning Vector databases like FAISS",
        "segment_count": 3,
        "segments": [
            {"start": 0.0, "duration": 10.0, "timestamp": "00:00", "text": "Hello world welcome to machine learning"},
            {"start": 15.0, "duration": 45.0, "timestamp": "00:15", "text": "Neural networks and artificial intelligence deep learning models"},
            {"start": 135.0, "duration": 43.0, "timestamp": "02:15", "text": "Vector databases like FAISS enable fast similarity retrieval"}
        ]
    }


def test_extract_youtube_id():
    """Tests extracting 11-char ID from various URL formats."""
    assert extract_youtube_id("J5_-l7WIO_w") == "J5_-l7WIO_w"
    assert extract_youtube_id("https://www.youtube.com/watch?v=J5_-l7WIO_w") == "J5_-l7WIO_w"
    assert extract_youtube_id("https://youtu.be/J5_-l7WIO_w") == "J5_-l7WIO_w"
    assert extract_youtube_id("https://www.youtube.com/shorts/J5_-l7WIO_w") == "J5_-l7WIO_w"


def test_timestamp_formatting():
    """Tests converting seconds to formatted timestamp strings."""
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(135.5) == "02:15"
    assert format_timestamp(3665) == "01:01:05"


def test_chunk_youtube_transcript(sample_transcript_data):
    """Tests chunking segments into documents with start/end seconds and deep-link URLs."""
    chunks = chunk_youtube_transcript(sample_transcript_data, max_chunk_chars=50)
    assert len(chunks) >= 2
    
    first = chunks[0]
    assert first.metadata["video_id"] == "J5_-l7WIO_w"
    assert first.metadata["start_seconds"] == 0.0
    assert first.metadata["timestamp_start"] == "00:00"
    assert "https://www.youtube.com/watch?v=J5_-l7WIO_w&t=0s" in first.metadata["url"]
    
    third = chunks[-1]
    assert third.metadata["start_seconds"] == 135.0
    assert "https://www.youtube.com/watch?v=J5_-l7WIO_w&t=135s" in third.metadata["url"]


def test_video_isolation_and_caching(tmp_path, sample_transcript_data):
    """Tests that YouTube vector stores are cached per video_id in isolated subdirectories."""
    custom_store_dir = tmp_path / "yt_stores"
    video_id = sample_transcript_data["video_id"]
    
    # 1. Mock embeddings to avoid Gemini API network call during unit test
    mock_emb = np.array([[0.6, 0.8]], dtype="float32")
    
    with patch("core.vectorstore.EmbeddingPipeline.embed_chunks", return_value=mock_emb):
        vstore1 = get_or_create_youtube_vectorstore(
            url_or_id=video_id,
            transcript_data=sample_transcript_data,
            base_store_dir=custom_store_dir
        )
        
        expected_dir = custom_store_dir / video_id
        assert (expected_dir / "faiss.index").exists()
        assert (expected_dir / "metadata.pkl").exists()
        
        # 2. Call again with no transcript_data passed -> Should reload existing index without re-fetching
        with patch("core.youtube_loader.fetch_youtube_transcript") as mock_fetch:
            vstore2 = get_or_create_youtube_vectorstore(
                url_or_id=video_id,
                base_store_dir=custom_store_dir
            )
            mock_fetch.assert_not_called()  # Reused cached store without fetching!
            assert vstore2.index.ntotal == vstore1.index.ntotal


def test_youtube_rag_tool_semantic_search_and_abstention(tmp_path):
    """Tests YouTubeRAGTool semantic vector retrieval and low-similarity score abstention."""
    tool = YouTubeRAGTool()
    
    # Mock vector store returning high similarity match for artificial intelligence
    mock_store = MagicMock()
    mock_store.query.return_value = [
        {
            "similarity_score": 0.85,
            "metadata": {
                "text": "Deep learning models process complex representations.",
                "timestamp": "00:15 - 01:00",
                "timestamp_start": "00:15",
                "timestamp_end": "01:00",
                "start_seconds": 15.0,
                "end_seconds": 60.0,
                "url": "https://www.youtube.com/watch?v=J5_-l7WIO_w&t=15s"
            }
        }
    ]
    
    with patch("core.youtube_loader.get_or_create_youtube_vectorstore", return_value=mock_store):
        res = tool.run("What are artificial intelligence models?", url_or_id="J5_-l7WIO_w")
        assert res["tool"] == "youtube_rag"
        assert len(res["sources"]) == 1
        assert res["sources"][0]["similarity_score"] == 0.85
        assert "https://www.youtube.com/watch?v=J5_-l7WIO_w&t=15s" in res["sources"][0]["url"]
        
    # Test low similarity score abstention (all results below MIN_SIMILARITY_SCORE)
    mock_store_empty = MagicMock()
    mock_store_empty.query.return_value = []
    
    with patch("core.youtube_loader.get_or_create_youtube_vectorstore", return_value=mock_store_empty):
        res_abs = tool.run("Unrelated query about cooking recipes", url_or_id="J5_-l7WIO_w")
        assert "No relevant spoken segments found" in res_abs["output"]
        assert len(res_abs["sources"]) == 0
