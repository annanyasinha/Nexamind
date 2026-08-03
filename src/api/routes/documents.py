import os
import shutil
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from api.schemas import UploadResponse
from api.deps import get_rag_search
from config import settings

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

@router.post("/upload", response_model=UploadResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    auto_reindex: bool = Form(True)
):
    """Uploads document files and optionally rebuilds the FAISS vector index."""
    data_dir = settings.DATA_DIR
    data_dir.mkdir(parents=True, exist_ok=True)
    saved_files = []
    
    for file in files:
        file_path = data_dir / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_files.append(file.filename)
    
    doc_count = 0
    if auto_reindex:
        rag = get_rag_search()
        doc_count = rag.rebuild_index(data_dir)
        
    return UploadResponse(
        message=f"Successfully uploaded {len(saved_files)} file(s).",
        saved_files=saved_files,
        indexed_documents_count=doc_count
    )

@router.post("/reindex")
def reindex_vectorstore():
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
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")

@router.delete("/documents/{filename}")
def delete_document(filename: str, auto_reindex: bool = True):
    """Deletes a specific document file and updates the FAISS vector index."""
    data_dir = settings.DATA_DIR
    target_file = data_dir / filename
    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")
    
    try:
        os.remove(target_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete file '{filename}': {str(e)}")
    
    doc_count = 0
    if auto_reindex:
        rag = get_rag_search()
        doc_count = rag.rebuild_index(data_dir)
        
    return {
        "status": "success",
        "message": f"Successfully deleted document '{filename}'.",
        "indexed_documents_count": doc_count
    }

@router.delete("/documents")
def clear_all_documents():
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


