"""
Unit tests for NexaMind Gemini Vision document and image loader.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain_core.documents import Document

from core.vision_loader import (
    analyze_image_bytes,
    extract_pdf_vision_documents,
    load_single_image_document
)
from core.document_loader import load_single_document


@patch("core.vision_loader._get_genai_client")
def test_analyze_image_bytes(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The graph shows revenue increasing consistently from 2022 to 2025."
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = analyze_image_bytes(
        image_bytes=b"fake_image_data",
        mime_type="image/png",
        filename="test_chart.png"
    )

    assert result is not None
    assert "revenue increasing consistently" in result
    mock_client.models.generate_content.assert_called_once()


@patch("core.document_loader.load_single_image_document")
def test_load_single_document_image(mock_load_img, tmp_path):
    img_file = tmp_path / "chart.png"
    img_file.write_bytes(b"fake_png_data")

    mock_load_img.return_value = [
        Document(
            page_content="[Gemini Vision Analysis - Image: chart.png]\nRevenue chart: 2022-2025 growth",
            metadata={"filename": "chart.png", "is_vision_extracted": True}
        )
    ]

    docs = load_single_document(img_file, enable_vision=True)
    assert len(docs) == 1
    assert docs[0].metadata["is_vision_extracted"] is True
    assert "Revenue chart" in docs[0].page_content
