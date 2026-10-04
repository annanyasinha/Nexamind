from pathlib import Path
from typing import Any, List, Union

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

from config import settings
from core.vision_loader import extract_pdf_vision_documents, load_single_image_document
from utils.logger import logger


def load_all_documents(data_dir: Union[str, Path] = None, enable_vision: bool = None) -> List[Any]:
    """
    Load all supported files from the specified data directory and convert to LangChain document structure.
    Supported file types: PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON, PNG, JPG, JPEG, WEBP.
    Includes Gemini Vision OCR and visual chart/diagram extraction for PDFs and images.
    """
    target_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    data_path = target_dir.resolve()
    logger.info(f"Scanning document directory: {data_path}")

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    if enable_vision is None:
        enable_vision = getattr(settings, "ENABLE_GEMINI_VISION", True)

    documents = []

    # PDF files (Text + Gemini Vision for embedded diagrams/graphs)
    pdf_files = list(data_path.glob("**/*.pdf"))
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF file(s).")
        for pdf_file in pdf_files:
            try:
                loaded = PyPDFLoader(str(pdf_file)).load()
                documents.extend(loaded)
                if enable_vision:
                    vision_docs = extract_pdf_vision_documents(pdf_file)
                    documents.extend(vision_docs)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_file}: {e}")

    # Standalone Image files (Gemini Vision)
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(data_path.glob(f"**/{ext}")))

    if image_files:
        logger.info(f"Found {len(image_files)} image file(s).")
        for img_file in image_files:
            try:
                img_docs = load_single_image_document(img_file)
                documents.extend(img_docs)
            except Exception as e:
                logger.error(f"Failed to load image file {img_file}: {e}")

    # TXT files
    txt_files = list(data_path.glob("**/*.txt"))
    if txt_files:
        logger.info(f"Found {len(txt_files)} TXT file(s).")
        for txt_file in txt_files:
            try:
                loaded = TextLoader(str(txt_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load TXT {txt_file}: {e}")

    # CSV files
    csv_files = list(data_path.glob("**/*.csv"))
    if csv_files:
        logger.info(f"Found {len(csv_files)} CSV file(s).")
        for csv_file in csv_files:
            try:
                loaded = CSVLoader(str(csv_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob("**/*.xlsx"))
    if xlsx_files:
        logger.info(f"Found {len(xlsx_files)} Excel file(s).")
        for xlsx_file in xlsx_files:
            try:
                loaded = UnstructuredExcelLoader(str(xlsx_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load Excel {xlsx_file}: {e}")

    # Word files
    docx_files = list(data_path.glob("**/*.docx"))
    if docx_files:
        logger.info(f"Found {len(docx_files)} Word file(s).")
        for docx_file in docx_files:
            try:
                loaded = Docx2txtLoader(str(docx_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load Word {docx_file}: {e}")

    # JSON files
    json_files = list(data_path.glob("**/*.json"))
    if json_files:
        logger.info(f"Found {len(json_files)} JSON file(s).")
        for json_file in json_files:
            try:
                try:
                    loaded = JSONLoader(str(json_file), jq_schema=".", text_content=False).load()
                except Exception:
                    loaded = TextLoader(str(json_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load JSON {json_file}: {e}")

    logger.info(f"Total loaded document objects: {len(documents)}")
    return documents


def load_single_document(file_path: Union[str, Path], enable_vision: bool = None) -> List[Any]:
    """
    Loads a single document or image file based on its extension and returns LangChain document objects.
    Supported extensions: .pdf, .txt, .csv, .xlsx, .docx, .json, .png, .jpg, .jpeg, .webp.
    """
    path = Path(file_path).resolve()
    if not path.exists() or not path.is_file():
        logger.warning(f"File does not exist for loading: {path}")
        return []

    if enable_vision is None:
        enable_vision = getattr(settings, "ENABLE_GEMINI_VISION", True)

    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            docs = PyPDFLoader(str(path)).load()
            if enable_vision:
                v_docs = extract_pdf_vision_documents(path)
                docs.extend(v_docs)
            return docs
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            return load_single_image_document(path)
        elif ext in [".txt", ".md", ".log"]:
            return TextLoader(str(path)).load()
        elif ext == ".csv":
            return CSVLoader(str(path)).load()
        elif ext == ".xlsx":
            return UnstructuredExcelLoader(str(path)).load()
        elif ext == ".docx":
            return Docx2txtLoader(str(path)).load()
        elif ext == ".json":
            try:
                return JSONLoader(str(path), jq_schema=".", text_content=False).load()
            except Exception:
                return TextLoader(str(path)).load()
        else:
            logger.warning(f"Unsupported file extension for single document loader: {ext}")
            return []
    except Exception as e:
        logger.error(f"Failed to load single document '{path}': {e}")
        return []

