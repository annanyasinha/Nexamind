"""
Unit tests for NexaMind Hybrid Tesseract + Gemini Vision loader with OCR confidence & resilience tests.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
from langchain_core.documents import Document

from core.vision_loader import (
    analyze_image_bytes_gemini,
    should_call_gemini_vision,
    extract_pdf_vision_documents,
    load_single_image_document
)
from core.document_loader import load_single_document


def test_should_call_gemini_vision_rules():
    # Plain long text with high confidence without visual indicators -> Skip Gemini Vision
    plain_text = "This is a standard text block with no visual data, containing sentence after sentence of prose text without numbers or charts." * 3
    assert should_call_gemini_vision(plain_text, ocr_confidence=90.0) is False

    # Low OCR confidence (< 65%) -> Triggers Gemini Vision fallback
    assert should_call_gemini_vision("Unclear blurry text sample", ocr_confidence=45.0) is True

    # Short OCR text -> Call Gemini Vision
    short_text = "Sales 2024"
    assert should_call_gemini_vision(short_text, ocr_confidence=95.0) is True

    # OCR text containing visual keywords -> Call Gemini Vision
    chart_ocr = "Revenue growth chart showing 2022 to 2025 data"
    assert should_call_gemini_vision(chart_ocr, ocr_confidence=90.0) is True

    # Multi-year numerical series > 60 chars -> Call Gemini Vision to interpret visual trends
    numerical_chart_ocr = "Annual Financial Results Overview: Year 2023: 40M USD | Year 2024: 65M USD | Year 2025: 90M USD total revenue"
    assert should_call_gemini_vision(numerical_chart_ocr, ocr_confidence=95.0) is True


@patch("core.vision_loader._get_genai_client")
def test_analyze_image_bytes_gemini_success(mock_get_client):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = "The chart shows revenue increasing consistently from 2022 to 2025."
    mock_client.models.generate_content.return_value = mock_response
    mock_get_client.return_value = mock_client

    result = analyze_image_bytes_gemini(
        image_bytes=b"fake_image_data",
        mime_type="image/png",
        filename="test_chart.png"
    )

    assert result is not None
    assert "revenue increasing consistently" in result
    mock_client.models.generate_content.assert_called_once()


@patch("core.vision_loader.analyze_image_bytes_gemini")
@patch("core.vision_loader.extract_text_and_confidence_from_image_bytes")
def test_gemini_failure_preserves_tesseract_ocr(mock_ocr, mock_gemini, tmp_path):
    """Tests that if Gemini Vision fails or raises an error, Tesseract OCR text is preserved without failing ingestion."""
    img_file = tmp_path / "chart.png"
    
    from PIL import Image
    from io import BytesIO
    img = Image.new("RGB", (100, 50), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    img_file.write_bytes(buf.getvalue())

    mock_ocr.return_value = ("Revenue 2023: 40M | Revenue 2024: 65M", 92.0)
    mock_gemini.return_value = None  # Gemini Vision returned None or API failed

    docs = load_single_image_document(img_file)
    assert len(docs) == 1
    assert "Revenue 2023: 40M" in docs[0].page_content
    assert docs[0].metadata["ocr_engine"] == "tesseract"
    assert docs[0].metadata["is_vision_extracted"] is False
    assert docs[0].metadata["visual_analysis"] is False


@patch("core.document_loader.load_single_image_document")
def test_load_single_document_image(mock_load_img, tmp_path):
    img_file = tmp_path / "chart.png"
    img_file.write_bytes(b"fake_png_data")

    mock_load_img.return_value = [
        Document(
            page_content="[Multimodal Image - File: chart.png]\n[Tesseract OCR Text (Conf: 88.0%)]\nRevenue 2024: 65M\n\n[Gemini Visual Analysis]\nRevenue growth chart: 2022-2025",
            metadata={"filename": "chart.png", "is_vision_extracted": True, "ocr_engine": "tesseract", "visual_analysis": True}
        )
    ]

    docs = load_single_document(img_file, enable_vision=True)
    assert len(docs) == 1
    assert docs[0].metadata["is_vision_extracted"] is True
    assert docs[0].metadata["ocr_engine"] == "tesseract"
    assert docs[0].metadata["visual_analysis"] is True
    assert "Revenue growth chart" in docs[0].page_content
