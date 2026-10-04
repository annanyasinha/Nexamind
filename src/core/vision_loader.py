"""
NexaMind Hybrid Multimodal Document Loader.
Combines fast local Tesseract OCR with Google Gemini Vision visual reasoning.
- Path 1: Native PDF Text (PyPDFLoader) - cheap & fast.
- Path 2: Scanned PDF Pages - PyMuPDF rendering + Tesseract OCR with confidence scoring.
- Path 3: Embedded Images & Figures - Tesseract OCR first; evaluates confidence score, visual keywords, and data density before calling Gemini Vision for complex graphs/charts.
- Resilient Fallback: If Gemini Vision fails or times out, Tesseract OCR text is preserved without failing ingestion.
"""

from io import BytesIO
from pathlib import Path

import re
from typing import Any, List, Optional, Union

import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types
from langchain_core.documents import Document

from config import settings
from core.ocr import (
    extract_text_and_confidence_from_image_bytes,
    extract_text_from_image_bytes,
    is_tesseract_available,
)
from utils.logger import logger


def _get_genai_client() -> genai.Client:
    """Returns initialized Google GenAI client."""
    if settings.GOOGLE_API_KEY:
        return genai.Client(api_key=settings.GOOGLE_API_KEY)
    return genai.Client()


FOCUSED_VISION_PROMPT = """You are an expert Document Vision Assistant for NexaMind RAG.
Analyze the provided image from document '{filename}' (Page {page_number}).

Your goal is to extract VISUAL REASONING information that OCR cannot capture:
- Charts, Graphs & Trends: Identify axes, labels, data trends, peak/lowest points, metrics (e.g., "Revenue grew from 40M in 2023 to 90M in 2025").
- Diagrams, Flowcharts & Architecture: Explain relationships between components, directional flows, and hierarchy.
- Complex Tables & Visual Structures: Summarize structural row/column insights or key takeaways.

IMPORTANT: Do NOT repeat plain OCR text word-for-word unless explaining visual relationships or chart data. Focus strictly on visual meaning and analytical takeaways.
"""

VISUAL_KEYWORDS = [
    "graph", "chart", "diagram", "figure", "flowchart", "plot", "trend",
    "architecture", "table", "revenue", "sales", "growth", "increase", "decrease",
    "profit", "margin", "quarter", "fy2", "fy1", "vs", "kpi", "workflow",
    "process", "hierarchy", "node", "legend", "axis"
]
VISUAL_REGEX = re.compile(r'\b(' + '|'.join(VISUAL_KEYWORDS) + r')\b', re.IGNORECASE)


def should_call_gemini_vision(
    ocr_text: str,
    ocr_confidence: float = 100.0,
    min_ocr_length: int = None,
    min_confidence: float = None
) -> bool:
    """
    Evaluates whether Gemini Vision is necessary for visual understanding:
    1. Low OCR Confidence (< min_confidence) -> Tesseract produced noisy/blurry OCR -> Call Gemini Vision.
    2. Short OCR text (< min_ocr_length chars) -> Likely a chart, diagram, or visual logo -> Call Gemini Vision.
    3. Contains visual/chart keywords -> Call Gemini Vision to interpret visual trends & relationships.
    4. Contains numerical data/years (e.g., '2023 40M', '2024 60M') -> Tesseract lacks visual relationship context -> Call Gemini Vision.
    5. High-confidence plain prose -> Tesseract OCR output is sufficient -> Skip Gemini Vision call to save tokens & latency.
    """
    min_len = min_ocr_length if min_ocr_length is not None else settings.OCR_MIN_TEXT_LENGTH
    min_conf = min_confidence if min_confidence is not None else settings.OCR_MIN_CONFIDENCE

    # Rule 1: Poor Tesseract confidence -> Call Gemini Vision for fallback
    if ocr_confidence < min_conf and len(ocr_text) > 0:
        logger.info(f"Low OCR confidence ({ocr_confidence}% < {min_conf}%). Triggering Gemini Vision fallback.")
        return True

    # Rule 2: Short OCR text -> Likely an image, chart, or diagram
    if len(ocr_text) < min_len:
        return True

    # Rule 3: Symbols or visual keywords matched on word boundaries
    if any(sym in ocr_text for sym in ["%", "$", "€", "£"]):
        return True

    if VISUAL_REGEX.search(ocr_text):
        return True

    # Rule 4: High digit/year density indicative of chart axis / tabular series (e.g., "2022", "2023", "2024", "2025")
    year_count = sum(1 for y in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"] if y in ocr_text)
    if year_count >= 2:
        return True

    digit_count = sum(1 for char in ocr_text if char.isdigit())
    if len(ocr_text) > 0 and (digit_count / len(ocr_text)) > 0.15:
        return True

    # Rule 5: Plain text prose with good confidence -> Tesseract OCR is sufficient
    return False


def analyze_image_bytes_gemini(
    image_bytes: bytes,
    mime_type: str,
    filename: str,
    page_number: int = 1,
    client: Optional[genai.Client] = None
) -> Optional[str]:
    """
    Sends image bytes to Gemini Vision model to extract visual understanding (charts, diagrams, flowcharts).
    Gracefully catches API exceptions so document ingestion never fails if Gemini Vision is unavailable.
    """
    if not settings.ENABLE_GEMINI_VISION:
        logger.info("Gemini Vision is disabled in application settings.")
        return None

    if not client:
        try:
            client = _get_genai_client()
        except Exception as e:
            logger.error(f"Failed to initialize GenAI client for vision analysis: {e}")
            return None

    prompt = FOCUSED_VISION_PROMPT.format(filename=filename, page_number=page_number)
    contents = [
        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        prompt
    ]

    for model in settings.LLM_MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model,
                contents=contents
            )
            if response and response.text:
                logger.info(f"Gemini Vision ({model}) analyzed visual elements for '{filename}' page {page_number}")
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini Vision attempt failed with model {model}: {e}")

    return None


def extract_pdf_vision_documents(
    pdf_path: Union[str, Path],
    min_image_size_bytes: int = None,
    max_images_per_page: int = None,
    min_scanned_page_text_len: int = None,
    render_dpi: int = None
) -> List[Document]:
    """
    Hybrid PDF Multimodal Extractor:
    1. Scanned PDF Pages: If native page text < min_scanned_page_text_len, renders page as image and runs Tesseract OCR.
    2. Embedded Images/Figures: Runs Tesseract OCR first, then evaluates confidence & decision rules before calling Gemini Vision for complex graphs/charts.
    3. Combines OCR text + Visual Reasoning into unified LangChain Document objects with rich metadata and deduplication.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        logger.warning(f"PDF file not found for multimodal extraction: {path}")
        return []

    min_img_bytes = min_image_size_bytes or settings.MIN_IMAGE_SIZE_BYTES
    max_imgs = max_images_per_page or settings.MAX_IMAGES_PER_PAGE
    min_scanned_len = min_scanned_page_text_len or settings.MIN_SCANNED_PAGE_TEXT_LEN
    dpi_val = render_dpi or settings.OCR_RENDER_DPI

    documents: List[Document] = []
    filename = path.name
    seen_contents = set()

    try:
        doc = fitz.open(str(path))
        client = None

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Check for Scanned PDF Page (Native text missing or very short)
            native_text = page.get_text().strip()
            if len(native_text) < min_scanned_len and is_tesseract_available() and settings.OCR_ENABLED:
                try:
                    pix = page.get_pixmap(dpi=dpi_val)
                    page_img_bytes = pix.tobytes("png")
                    scanned_ocr_text, scanned_conf = extract_text_and_confidence_from_image_bytes(page_img_bytes)

                    # Evaluate if scanned page contains complex visual/chart or low OCR confidence
                    run_scanned_vision = should_call_gemini_vision(scanned_ocr_text, ocr_confidence=scanned_conf)
                    scanned_vision_text = None

                    if run_scanned_vision and settings.ENABLE_GEMINI_VISION:
                        if client is None:
                            client = _get_genai_client()
                        scanned_vision_text = analyze_image_bytes_gemini(
                            image_bytes=page_img_bytes,
                            mime_type="image/png",
                            filename=filename,
                            page_number=page_num,
                            client=client
                        )

                    content_blocks = []
                    if scanned_ocr_text:
                        content_blocks.append(f"[Tesseract OCR Text (Conf: {scanned_conf}%)]\n{scanned_ocr_text}")
                    if scanned_vision_text:
                        content_blocks.append(f"[Gemini Visual Analysis]\n{scanned_vision_text}")

                    if content_blocks:
                        combo_key = f"scanned_{scanned_ocr_text}_{scanned_vision_text}"
                        if combo_key not in seen_contents:
                            seen_contents.add(combo_key)
                            doc_content = (
                                f"[Scanned PDF Page {page_num} Multimodal Analysis | Document: {filename}]\n"
                                + "\n\n".join(content_blocks)
                            )
                            documents.append(
                                Document(
                                    page_content=doc_content,
                                    metadata={
                                        "source": str(path),
                                        "filename": filename,
                                        "page": page_idx,
                                        "page_number": page_num,
                                        "document_type": "scanned_pdf_multimodal",
                                        "is_vision_extracted": bool(scanned_vision_text),
                                        "ocr_engine": "tesseract" if scanned_ocr_text else "none",
                                        "ocr_confidence": scanned_conf if scanned_ocr_text else 0.0,
                                        "visual_analysis": bool(scanned_vision_text)
                                    }
                                )
                            )
                            logger.info(f"Scanned PDF page {page_num} processed (OCR: {bool(scanned_ocr_text)}, Vision: {bool(scanned_vision_text)})")
                except Exception as e:
                    logger.warning(f"Scanned PDF page rendering/OCR error on page {page_num}: {e}")

            # 2. Extract Embedded Images / Figures
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            extracted_count = 0
            for img_index, img in enumerate(image_list):
                if extracted_count >= max_imgs:
                    break

                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"].lower()

                if len(image_bytes) < min_img_bytes:
                    continue  # Skip tiny icons/bullet graphics

                mime_type = f"image/{image_ext}" if image_ext in ["png", "jpeg", "jpg", "webp"] else "image/png"

                # Step 7: Run Tesseract OCR first with confidence scoring
                ocr_text = ""
                ocr_conf = 0.0
                if settings.OCR_ENABLED and is_tesseract_available():
                    ocr_text, ocr_conf = extract_text_and_confidence_from_image_bytes(image_bytes)

                # Deduplication check against native text
                if ocr_text and ocr_text in native_text:
                    logger.debug(f"Skipping duplicate embedded OCR text on page {page_num}")
                    continue

                # Step 8: Evaluate whether Gemini Vision is necessary
                run_vision = should_call_gemini_vision(ocr_text, ocr_confidence=ocr_conf)
                vision_text = None

                if run_vision and settings.ENABLE_GEMINI_VISION:
                    if client is None:
                        client = _get_genai_client()
                    vision_text = analyze_image_bytes_gemini(
                        image_bytes=image_bytes,
                        mime_type=mime_type,
                        filename=filename,
                        page_number=page_num,
                        client=client
                    )

                # Step 10: Combine results into unified Document
                content_blocks = []
                if ocr_text:
                    content_blocks.append(f"[Tesseract OCR Text (Conf: {ocr_conf}%)]\n{ocr_text}")
                if vision_text:
                    content_blocks.append(f"[Gemini Visual Analysis]\n{vision_text}")

                if content_blocks:
                    combo_key = f"{ocr_text}_{vision_text}"
                    if combo_key in seen_contents:
                        continue
                    seen_contents.add(combo_key)

                    extracted_count += 1
                    doc_content = (
                        f"[Multimodal Document Chunk - File: {filename} | Page {page_num} | Image {extracted_count}]\n"
                        + "\n\n".join(content_blocks)
                    )
                    documents.append(
                        Document(
                            page_content=doc_content,
                            metadata={
                                "source": str(path),
                                "filename": filename,
                                "page": page_idx,
                                "page_number": page_num,
                                "document_type": "multimodal",
                                "is_vision_extracted": bool(vision_text),
                                "ocr_engine": "tesseract" if ocr_text else "none",
                                "ocr_confidence": ocr_conf if ocr_text else 0.0,
                                "visual_analysis": bool(vision_text),
                                "image_index": img_index + 1
                            }
                        )
                    )

        doc.close()
        logger.info(f"Extracted {len(documents)} hybrid OCR + Vision document chunks from PDF '{filename}'")
    except Exception as e:
        logger.error(f"Error processing multimodal PDF '{filename}': {e}")

    return documents


def load_single_image_document(image_path: Union[str, Path]) -> List[Document]:
    """
    Loads and processes a standalone image file (.png, .jpg, .jpeg, .webp) using Hybrid Tesseract + Gemini Vision.
    """
    path = Path(image_path).resolve()
    if not path.exists() or not path.is_file():
        logger.warning(f"Image file not found: {path}")
        return []

    filename = path.name
    ext = path.suffix.lower().lstrip(".")
    mime_type = f"image/{ext}" if ext in ["png", "jpeg", "jpg", "webp"] else "image/png"

    try:
        with open(path, "rb") as f:
            image_bytes = f.read()

        ocr_text = ""
        ocr_conf = 0.0
        if settings.OCR_ENABLED and is_tesseract_available():
            ocr_text, ocr_conf = extract_text_and_confidence_from_image_bytes(image_bytes)

        run_vision = should_call_gemini_vision(ocr_text, ocr_confidence=ocr_conf)
        vision_text = None

        if run_vision and settings.ENABLE_GEMINI_VISION:
            client = _get_genai_client()
            vision_text = analyze_image_bytes_gemini(
                image_bytes=image_bytes,
                mime_type=mime_type,
                filename=filename,
                page_number=1,
                client=client
            )

        content_blocks = []
        if ocr_text:
            content_blocks.append(f"[Tesseract OCR Text (Conf: {ocr_conf}%)]\n{ocr_text}")
        if vision_text:
            content_blocks.append(f"[Gemini Visual Analysis]\n{vision_text}")

        if content_blocks:
            content = f"[Multimodal Image - File: {filename}]\n" + "\n\n".join(content_blocks)
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "filename": filename,
                        "page": 0,
                        "page_number": 1,
                        "document_type": f"image_{ext}",
                        "is_vision_extracted": bool(vision_text),
                        "ocr_engine": "tesseract" if ocr_text else "none",
                        "ocr_confidence": ocr_conf if ocr_text else 0.0,
                        "visual_analysis": bool(vision_text)
                    }
                )
            ]
    except Exception as e:
        logger.error(f"Failed to process image file '{filename}': {e}")

    return []
