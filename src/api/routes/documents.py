import hashlib
import os
from pathlib import Path
from typing import List
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    UploadFile,
    Request
)

from api.rate_limit import (
    limiter,
    DOCUMENT_UPLOAD_RATE_LIMIT,
    DOCUMENT_REINDEX_RATE_LIMIT,
    DOCUMENT_DELETE_RATE_LIMIT,
)
from api.deps import get_rag_search
from api.schemas import UploadResponse
from config import settings
from core.document_loader import load_single_document
from utils.logger import logger

router = APIRouter(tags=["Document Management"])

@router.get("/documents")
def list_documents():
    """Lists all uploaded files and metadata in the document knowledge dataset."""
    data_dir = settings.DATA_DIR
    if not data_dir.exists():
        return {"documents": [], "count": 0}
    
    files_info = []
    for fname in os.listdir(data_dir):
        if fname.startswith("."):
            continue
        fpath = data_dir / fname
        if fpath.is_file():
            files_info.append({
                "filename": fname,
                "size_bytes": os.path.getsize(fpath),
                "extension": os.path.splitext(fname)[1].lower()
            })
    return {"documents": files_info, "count": len(files_info)}


def sanitize_and_validate_path(filename: str, data_dir: Path) -> Path:
    """
    Sanitizes user-provided filename to prevent path traversal vulnerabilities.
    Verifies that the target path resolves strictly within data_dir.
    """
    if not filename or not filename.strip():
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    # Strip directory components (e.g. "../../secret.txt" -> "secret.txt")
    safe_name = Path(filename).name.strip()
    if not safe_name or safe_name in [".", ".."]:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    target_path = (data_dir / safe_name).resolve()
    resolved_data_dir = data_dir.resolve()

    # Ensure resolved path remains inside resolved_data_dir boundary
    if resolved_data_dir not in target_path.parents and target_path.parent != resolved_data_dir:
        raise HTTPException(status_code=400, detail="Invalid filename or path traversal detected.")

    return target_path

@router.post("/upload", response_model=UploadResponse)
@limiter.limit(DOCUMENT_UPLOAD_RATE_LIMIT)
async def upload_documents(
    request: Request,
    files: List[UploadFile] = File(...),
    auto_reindex: bool = Form(True)
):

    """
    Uploads document files securely and indexes them.
    Logic:
    - SHA-256 Checksum Match (identical content): Skip re-embedding.
    - Same Filename + Changed Content: Overwrite file & execute full index rebuild.
    - Brand New File: Save file & execute fast incremental vector addition.
    """
    data_dir = settings.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    rag = get_rag_search()

    saved_files = []
    skipped_files = []
    indexed_doc_count = 0
    needs_full_rebuild = False
    new_docs_to_index = []

    existing_checksums = rag.vectorstore.get_indexed_checksums()

    for file in files:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file missing filename.")

        safe_filename = Path(file.filename).name.strip()
        ext = Path(safe_filename).suffix.lower()

        if ext not in settings.ALLOWED_EXTENSIONS:
            allowed_list = ", ".join(sorted(settings.ALLOWED_EXTENSIONS))
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension '{ext}' for file '{safe_filename}'. Allowed extensions: {allowed_list}"
            )

        target_file = sanitize_and_validate_path(file.filename, data_dir)

        # Read contents and check size limit
        contents = await file.read()
        if len(contents) > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_MB
            raise HTTPException(
                status_code=400,
                detail=f"File '{safe_filename}' exceeds maximum allowed size limit of {max_mb}MB."
            )

        # Calculate SHA-256 checksum of uploaded content
        checksum = hashlib.sha256(contents).hexdigest()

        # Step 1: Check if content checksum is already indexed
        if checksum in existing_checksums and target_file.exists():
            logger.info(f"File '{safe_filename}' content checksum is already indexed. Skipping embedding.")
            skipped_files.append(safe_filename)
            saved_files.append(safe_filename)
            continue

        # Step 2: Check if filename already exists on disk (with different content)
        file_already_exists = target_file.exists()

        # Save/overwrite file to disk
        with open(target_file, "wb") as buffer:
            buffer.write(contents)
        saved_files.append(safe_filename)

        if auto_reindex:
            if file_already_exists:
                # Same filename + changed content -> trigger full rebuild to clear old vectors for this filename
                needs_full_rebuild = True
            else:
                # Brand new file -> load single document for fast incremental addition
                docs = load_single_document(target_file)
                new_docs_to_index.extend(docs)

    if auto_reindex:
        if needs_full_rebuild:
            logger.info("One or more existing files were updated with new content. Rebuilding vector index...")
            indexed_doc_count = rag.rebuild_index(data_dir)
        elif new_docs_to_index:
            logger.info(f"Incrementally adding {len(new_docs_to_index)} new document object(s) to FAISS index...")
            indexed_doc_count = rag.add_documents(new_docs_to_index)

    msg = f"Successfully processed {len(saved_files)} file(s)."
    if skipped_files:
        msg += f" ({len(skipped_files)} duplicate file(s) skipped re-embedding)."

    return UploadResponse(
        message=msg,
        saved_files=saved_files,
        indexed_documents_count=indexed_doc_count
    )
@router.post("/reindex")
@limiter.limit(DOCUMENT_REINDEX_RATE_LIMIT)
def reindex_vectorstore(request: Request):
    """Re-indexes all documents in the dataset and updates FAISS vector store."""
    rag = get_rag_search()
    try:
        count = rag.rebuild_index(settings.DATA_DIR)
        total_vectors = len(rag.vectorstore.metadata) if rag.vectorstore and rag.vectorstore.metadata else 0
        return {
            "status": "success",
            "message": "Successfully reindexed vector store from document dataset.",
            "loaded_documents": count,
            "total_vectors": total_vectors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {e!s}")

@router.delete("/documents/{filename}")
@limiter.limit(DOCUMENT_DELETE_RATE_LIMIT)
def delete_document(
    request: Request,
    filename: str,
    auto_reindex: bool = True
):
    """Deletes a specific document file securely and updates the FAISS vector index."""
    data_dir = settings.DATA_DIR
    target_file = sanitize_and_validate_path(filename, data_dir)

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Document '{target_file.name}' not found.")

    try:
        os.remove(target_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{target_file.name}': {e!s}")

    doc_count = 0
    if auto_reindex:
        rag = get_rag_search()
        doc_count = rag.rebuild_index(data_dir)

    return {
        "status": "success",
        "message": f"Successfully deleted document '{target_file.name}'.",
        "indexed_documents_count": doc_count
    }

@router.delete("/documents")
@limiter.limit(DOCUMENT_DELETE_RATE_LIMIT)
def clear_all_documents(request: Request):

    """Purges all files from the document dataset directory and wipes the vector index."""
    data_dir = settings.DATA_DIR
    if data_dir.exists():
        for fname in os.listdir(data_dir):
            if fname.startswith("."):
                continue
            fpath = data_dir / fname
            if fpath.is_file():
                try:
                    os.remove(fpath)
                except Exception:
                    pass
    
    rag = get_rag_search()
    rag.rebuild_index(data_dir)
    return {
        "status": "success",
        "message": "All documents removed from knowledge base dataset and vector index cleared."
    }


