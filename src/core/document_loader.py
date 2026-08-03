from pathlib import Path
from typing import List, Any, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader, Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from utils.logger import logger
from config import settings


def load_all_documents(data_dir: Union[str, Path] = None) -> List[Any]:
    """
    Load all supported files from the specified data directory and convert to LangChain document structure.
    Supported file types: PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON.
    """
    target_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    data_path = target_dir.resolve()
    logger.info(f"Scanning document directory: {data_path}")

    if not data_path.exists():
        logger.warning(f"Data directory does not exist: {data_path}")
        return []

    documents = []

    # PDF files
    pdf_files = list(data_path.glob("**/*.pdf"))
    if pdf_files:
        logger.info(f"Found {len(pdf_files)} PDF file(s).")
        for pdf_file in pdf_files:
            try:
                loaded = PyPDFLoader(str(pdf_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load PDF {pdf_file}: {e}")

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
                    # Fallback to TextLoader if jq is missing or JSON structure requires simple text loading
                    loaded = TextLoader(str(json_file)).load()
                documents.extend(loaded)
            except Exception as e:
                logger.error(f"Failed to load JSON {json_file}: {e}")

    logger.info(f"Total loaded document objects: {len(documents)}")
    return documents
