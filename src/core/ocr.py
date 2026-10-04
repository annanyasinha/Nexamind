"""
NexaMind Local OCR Module (powered by Tesseract & Pillow).
Provides fast, zero-cost text extraction from images and scanned PDF pages before Gemini Vision fallback.
"""

from io import BytesIO
from typing import Optional
from PIL import Image
import pytesseract

from utils.logger import logger


def is_tesseract_available() -> bool:
    """Checks if Tesseract OCR binary is installed and reachable by pytesseract."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """
    Extracts text from raw image bytes using Tesseract OCR.
    Returns empty string if Tesseract is unavailable or no text is detected.
    """
    if not image_bytes:
        return ""
    try:
        image = Image.open(BytesIO(image_bytes))
        # Convert image to RGB if needed (handles RGBA, Palette, Grayscale)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        logger.warning(f"Tesseract OCR extraction warning: {e}")
        return ""
