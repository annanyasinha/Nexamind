"""
NexaMind Hybrid Multimodal Document Loader.
Combines fast local Tesseract OCR with Google Gemini Vision visual reasoning.
- Path 1: Native PDF Text (PyPDFLoader) - cheap & fast.
- Path 2: Scanned PDF Pages - PyMuPDF rendering + Tesseract OCR.
- Path 3: Embedded Images & Figures - Tesseract OCR first; calls Gemini Vision only when visual reasoning (charts/diagrams/trends) is required.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, List, Optional, Union

import fitz  # PyMuPDF
from PIL import Image
from google import genai
from google.genai import types
from langchain_core.documents import Document

from config import settings
from core.ocr import extract_text_from_image_bytes, is_tesseract_available
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

import re

VISUAL_KEYWORDS = [
    "graph", "chart", "diagram", "figure", "flowchart", "plot", "trend",
    "architecture", "table", "revenue", "sales", "growth", "increase", "decrease",
    "profit", "margin", "quarter", "fy2", "fy1", "vs", "kpi", "workflow",
    "process", "hierarchy", "node", "legend", "axis"
]
VISUAL_REGEX = re.compile(r'\b(' + '|'.join(VISUAL_KEYWORDS) + r')\b', re.IGNORECASE)


def should_call_gemini_vision(ocr_text: str, min_ocr_length: int = 60) -> bool:
    """
    Evaluates whether Gemini Vision is necessary for visual understanding:
    1. Short OCR text (< min_ocr_length chars) -> Likely a chart, diagram, or visual logo -> Call Gemini Vision.
    2. Contains visual/chart keywords -> Call Gemini Vision to interpret visual trends & relationships.
    3. Contains numerical data/years (e.g., '2023 40M', '2024 60M') -> Even if > 60 chars, Tesseract lacks visual relationship context -> Call Gemini Vision.
    4. Long plain prose with no visual markers -> Tesseract OCR output is sufficient -> Skip Gemini Vision call to save tokens & latency.
    """
    if len(ocr_text) < min_ocr_length:
        return True

    # Rule 2: Symbols or visual keywords matched on word boundaries
    if any(sym in ocr_text for sym in ["%", "$", "€", "£"]):
        return True

    if VISUAL_REGEX.search(ocr_text):
        return True

    # Rule 3: High digit/year density indicative of chart axis / tabular series (e.g., "2022", "2023", "2024", "2025")
    year_count = sum(1 for y in ["2020", "2021", "2022", "2023", "2024", "2025", "2026"] if y in ocr_text)
    if year_count >= 2:
        return True

    digit_count = sum(1 for char in ocr_text if char.isdigit())
    if len(ocr_text) > 0 and (digit_count / len(ocr_text)) > 0.15:
        return True

    # Rule 4: Plain text prose -> Tesseract OCR is sufficient
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
    """
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
    min_image_size_bytes: int = 5000,
    max_images_per_page: int = 3,
    min_scanned_page_text_len: int = 50
) -> List[Document]:
    """
    Hybrid PDF Multimodal Extractor:
    1. Scanned PDF Pages: If native page text < min_scanned_page_text_len, renders page as image and runs Tesseract OCR.
    2. Embedded Images/Figures: Runs Tesseract OCR first, then evaluates decision rule before calling Gemini Vision for complex graphs/charts.
    3. Combines OCR text + Visual Reasoning into unified LangChain Document objects.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        logger.warning(f"PDF file not found for multimodal extraction: {path}")
        return []

    documents: List[Document] = []
    filename = path.name

    try:
        doc = fitz.open(str(path))
        client = None

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1

            # 1. Check for Scanned PDF Page (Native text missing or very short)
            native_text = page.get_text().strip()
            if len(native_text) < min_scanned_page_text_len and is_tesseract_available():
                try:
                    pix = page.get_pixmap(dpi=150)
                    page_img_bytes = pix.tobytes("png")
                    scanned_ocr_text = extract_text_from_image_bytes(page_img_bytes)

                    if len(scanned_ocr_text) >= 20:
                        doc_content = (
                            f"[Scanned PDF Page {page_num} OCR - Tesseract | Document: {filename}]\n"
                            f"{scanned_ocr_text}"
                        )
                        documents.append(
                            Document(
                                page_content=doc_content,
                                metadata={
                                    "source": str(path),
                                    "filename": filename,
                                    "page": page_idx,
                                    "page_number": page_num,
                                    "document_type": "scanned_pdf_ocr",
                                    "is_vision_extracted": True,
                                    "ocr_engine": "tesseract",
                                    "visual_analysis": False
                                }
                            )
                        )
                        logger.info(f"Scanned PDF OCR extracted {len(scanned_ocr_text)} chars from page {page_num} of '{filename}'")
                except Exception as e:
                    logger.warning(f"Scanned PDF page rendering/OCR error on page {page_num}: {e}")

            # 2. Extract Embedded Images / Figures
            image_list = page.get_images(full=True)
            if not image_list:
                continue

            extracted_count = 0
            for img_index, img in enumerate(image_list):
                if extracted_count >= max_images_per_page:
                    break

                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"].lower()

                if len(image_bytes) < min_image_size_bytes:
                    continue  # Skip tiny icons/bullet graphics

                mime_type = f"image/{image_ext}" if image_ext in ["png", "jpeg", "jpg", "webp"] else "image/png"

                # Step 7: Run Tesseract OCR first
                ocr_text = extract_text_from_image_bytes(image_bytes)

                # Step 8: Evaluate whether Gemini Vision is necessary
                run_vision = should_call_gemini_vision(ocr_text)
                vision_text = None

                if run_vision:
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
                    content_blocks.append(f"[Tesseract OCR Text]\n{ocr_text}")
                if vision_text:
                    content_blocks.append(f"[Gemini Visual Analysis]\n{vision_text}")

                if content_blocks:
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
                                "is_vision_extracted": True,
                                "ocr_engine": "tesseract" if ocr_text else "none",
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

        ocr_text = extract_text_from_image_bytes(image_bytes)
        run_vision = should_call_gemini_vision(ocr_text)
        vision_text = None

        if run_vision:
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
            content_blocks.append(f"[Tesseract OCR Text]\n{ocr_text}")
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
                        "is_vision_extracted": True,
                        "ocr_engine": "tesseract" if ocr_text else "none",
                        "visual_analysis": bool(vision_text)
                    }
                )
            ]
    except Exception as e:
        logger.error(f"Failed to process image file '{filename}': {e}")

    return []
