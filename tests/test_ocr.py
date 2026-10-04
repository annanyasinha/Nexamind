"""
Unit tests for core/ocr.py module with Tesseract confidence scoring and preprocessing.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image

from core.ocr import (
    extract_text_and_confidence_from_image_bytes,
    extract_text_from_image_bytes,
    is_tesseract_available,
    preprocess_image_for_ocr,
)


def test_is_tesseract_available():
    assert is_tesseract_available() is True


def test_preprocess_image_for_ocr():
    img = Image.new("RGB", (100, 50), color="white")
    processed = preprocess_image_for_ocr(img)
    assert processed.mode == "L"  # Grayscale image output


@patch("core.ocr.pytesseract.image_to_data")
def test_extract_text_and_confidence_from_image_bytes(mock_to_data):
    mock_to_data.return_value = {
        "text": ["Revenue", "2023", "40M", "Revenue", "2024", "65M"],
        "conf": ["95.0", "90.0", "85.0", "92.0", "88.0", "80.0"]
    }

    img = Image.new("RGB", (100, 50), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    png_bytes = buf.getvalue()

    text, conf = extract_text_and_confidence_from_image_bytes(png_bytes)
    assert "Revenue 2023 40M" in text
    assert conf > 80.0
    mock_to_data.assert_called_once()
