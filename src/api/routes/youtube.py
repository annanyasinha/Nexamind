from fastapi import APIRouter, HTTPException

from api.deps import get_rag_search
from api.schemas import YouTubeTranscriptRequest, YouTubeTranscriptResponse
from config import settings
from core.youtube_loader import fetch_youtube_transcript, save_transcript_to_dataset
from utils.logger import logger

router = APIRouter(tags=["YouTube Integration"])


@router.post("/youtube/transcript", response_model=YouTubeTranscriptResponse)
def get_youtube_transcript(req: YouTubeTranscriptRequest):
    """
    Fetches transcript for a YouTube URL/Video ID, formats timestamps,
    and optionally saves to data storage & indexes into FAISS vector store.
    """
    try:
        data = fetch_youtube_transcript(req.url)
    except Exception as e:
        logger.error(f"Failed fetching transcript for '{req.url}': {e}")
        raise HTTPException(status_code=400, detail=str(e))

    saved_filename = None
    indexed_count = 0

    if req.save_to_dataset:
        try:
            file_path = save_transcript_to_dataset(data, settings.DATA_DIR)
            saved_filename = file_path.name
        except Exception as e:
            logger.error(f"Failed to save YouTube transcript to file: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save transcript: {e!s}")

        if req.auto_reindex:
            try:
                from core.youtube_loader import get_or_create_youtube_vectorstore
                yt_vstore = get_or_create_youtube_vectorstore(data["video_id"], transcript_data=data)
                indexed_count = len(yt_vstore.metadata) if yt_vstore and yt_vstore.metadata else 0
            except Exception as e:
                logger.error(f"Per-video FAISS indexing failed after saving transcript: {e}")

    return YouTubeTranscriptResponse(
        video_id=data["video_id"],
        url=data["url"],
        full_text=data["full_text"],
        raw_text=data["raw_text"],
        segment_count=data["segment_count"],
        segments=data["segments"],
        saved_file=saved_filename,
        indexed_documents_count=indexed_count
    )
