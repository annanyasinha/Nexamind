"""
NexaMind Gemini Vision Multimodal Document Loader.
Extracts embedded images and visual elements from PDFs and standalone image files (.png, .jpg, .jpeg, .webp),
uses Google Gemini Vision models to generate rich OCR, diagram interpretations, and chart explanations,
and converts them into indexed document chunks for Text RAG.
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
from utils.logger import logger


def _get_genai_client() -> genai.Client:
    """Returns initialized Google GenAI client."""
    if settings.GOOGLE_API_KEY:
        return genai.Client(api_key=settings.GOOGLE_API_KEY)
    return genai.Client()


VISION_ANALYSIS_PROMPT = """You are an expert Document Vision and OCR Assistant for NexaMind RAG.
Analyze the provided image from document '{filename}' (Page {page_number}).

Provide a clear, detailed, and searchable text breakdown of this image:
1. **OCR & Raw Text**: Extract all readable text, titles, labels, legend text, and tabular values.
2. **Visual Analysis (Graphs / Charts / Diagrams)**: If there is a graph, chart, diagram, or flowchart, explain:
   - What key metric or topic it represents.
   - Specific data trends (e.g., "Revenue increases consistently from 2022 to 2025, reaching peak in 2025").
   - Relationships between entities or components in flowcharts/architectures.
3. **Summary**: A concise 2-sentence summary of the main insight.

Keep the description precise and information-dense so it can be accurately matched by a vector search query.
"""


def analyze_image_bytes(
    image_bytes: bytes,
    mime_type: str,
    filename: str,
    page_number: int = 1,
    client: Optional[genai.Client] = None
) -> Optional[str]:
    """
    Sends image bytes to Gemini Vision model to extract OCR, chart explanations, and diagram insights.
    """
    if not client:
        try:
            client = _get_genai_client()
        except Exception as e:
            logger.error(f"Failed to initialize GenAI client for vision analysis: {e}")
            return None

    prompt = VISION_ANALYSIS_PROMPT.format(filename=filename, page_number=page_number)
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
                logger.info(f"Gemini Vision ({model}) analyzed image for '{filename}' page {page_number}")
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini Vision failed with model {model}: {e}")

    return None


def extract_pdf_vision_documents(
    pdf_path: Union[str, Path],
    min_image_size_bytes: int = 5000,
    max_images_per_page: int = 3
) -> List[Document]:
    """
    Extracts embedded images from a PDF using PyMuPDF and analyzes them using Gemini Vision.
    Returns a list of LangChain Document objects containing visual summaries and OCR.
    """
    path = Path(pdf_path).resolve()
    if not path.exists():
        logger.warning(f"PDF file not found for vision extraction: {path}")
        return []

    documents: List[Document] = []
    filename = path.name

    try:
        doc = fitz.open(str(path))
        client = _get_genai_client()

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            page_num = page_idx + 1
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
                    continue  # Skip small icons/decorations

                mime_type = f"image/{image_ext}" if image_ext in ["png", "jpeg", "jpg", "webp"] else "image/png"

                vision_text = analyze_image_bytes(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    filename=filename,
                    page_number=page_num,
                    client=client
                )

                if vision_text:
                    extracted_count += 1
                    doc_content = (
                        f"[Gemini Vision Analysis - Document: {filename} | Page {page_num} | Image {extracted_count}]\n"
                        f"{vision_text}"
                    )
                    documents.append(
                        Document(
                            page_content=doc_content,
                            metadata={
                                "source": str(path),
                                "filename": filename,
                                "page": page_idx,
                                "page_number": page_num,
                                "document_type": "pdf_image_vision",
                                "is_vision_extracted": True,
                                "image_index": img_index + 1
                            }
                        )
                    )

        doc.close()
        logger.info(f"Extracted {len(documents)} Gemini Vision visual documents from PDF '{filename}'")
    except Exception as e:
        logger.error(f"Error extracting vision documents from PDF '{filename}': {e}")

    return documents


def load_single_image_document(image_path: Union[str, Path]) -> List[Document]:
    """
    Loads and analyzes a standalone image file (.png, .jpg, .jpeg, .webp) using Gemini Vision.
    Returns a list of LangChain Document objects containing image OCR and visual analysis.
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

        vision_text = analyze_image_bytes(
            image_bytes=image_bytes,
            mime_type=mime_type,
            filename=filename,
            page_number=1
        )

        if vision_text:
            content = f"[Gemini Vision Analysis - Image: {filename}]\n{vision_text}"
            return [
                Document(
                    page_content=content,
                    metadata={
                        "source": str(path),
                        "filename": filename,
                        "page": 0,
                        "page_number": 1,
                        "document_type": f"image_{ext}",
                        "is_vision_extracted": True
                    }
                )
            ]
    except Exception as e:
        logger.error(f"Failed to process standalone image file '{filename}': {e}")

    return []
