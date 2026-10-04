"""
NexaMind Local OCR Module (powered by Tesseract & Pillow).
Provides fast, zero-cost text extraction with confidence scoring and image preprocessing
for scanned PDF pages and images before Gemini Vision fallback.
"""

from io import BytesIO
from typing import Optional, Tuple
from PIL import Image, ImageEnhance, ImageOps
import pytesseract

from utils.logger import logger


def is_tesseract_available() -> bool:
    """Checks if Tesseract OCR binary is installed and reachable by pytesseract."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    """
    Applies Pillow image preprocessing (grayscale + contrast enhancement) to boost Tesseract accuracy.
    """
    try:
        # Convert to Grayscale
        gray = image.convert("L")
        # Enhance Contrast
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(1.8)
        return enhanced
    except Exception as e:
        logger.debug(f"Image preprocessing warning: {e}")
        return image


def extract_text_and_confidence_from_image_bytes(image_bytes: bytes) -> Tuple[str, float]:
    """
    Extracts text and calculates average word confidence score (0.0 to 100.0) from raw image bytes using Tesseract.
    Returns ('', 0.0) if Tesseract is unavailable, invalid image, or no text detected.
    """
    if not image_bytes:
        return "", 0.0

    try:
        raw_image = Image.open(BytesIO(image_bytes))
        processed_image = preprocess_image_for_ocr(raw_image)

        # Run pytesseract image_to_data for confidence scoring
        data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)

        valid_confs = []
        words = []

        n_boxes = len(data.get("text", []))
        for i in range(n_boxes):
            word = data["text"][i].strip()
            conf_val = float(data["conf"][i])

            if word and conf_val >= 0:
                words.append(word)
                valid_confs.append(conf_val)

        extracted_text = " ".join(words).strip()
        avg_confidence = (sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0

        return extracted_text, round(avg_confidence, 2)
    except Exception as e:
        logger.warning(f"Tesseract OCR extraction warning: {e}")
        return "", 0.0


def extract_text_from_image_bytes(image_bytes: bytes) -> str:
    """Convenience wrapper that returns extracted OCR text."""
    text, _ = extract_text_and_confidence_from_image_bytes(image_bytes)
    return text
