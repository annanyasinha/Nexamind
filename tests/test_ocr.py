"""
Unit tests for core/ocr.py module.
"""

from io import BytesIO
from unittest.mock import MagicMock, patch
from PIL import Image

from core.ocr import extract_text_from_image_bytes, is_tesseract_available


def test_is_tesseract_available():
    assert is_tesseract_available() is True


@patch("core.ocr.pytesseract.image_to_string")
def test_extract_text_from_image_bytes(mock_ocr):
    mock_ocr.return_value = "Revenue 2023 40M\nRevenue 2024 65M"

    # Create valid PNG bytes using PIL
    img = Image.new("RGB", (100, 50), color="white")
    buf = BytesIO()
    img.save(buf, format="PNG")
    valid_png_bytes = buf.getvalue()

    text = extract_text_from_image_bytes(valid_png_bytes)
    assert "Revenue 2023 40M" in text
    mock_ocr.assert_called_once()
