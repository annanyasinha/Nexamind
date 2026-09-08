import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from youtube_transcript_api import (
    YouTubeTranscriptApi,
)

from config import settings
from utils.logger import logger


def extract_youtube_id(url_or_id: str) -> str:
    """
    Extracts 11-character YouTube video ID from various YouTube URL formats or raw ID strings.
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - Raw 11-character ID
    """
    if not url_or_id:
        raise ValueError("URL or Video ID cannot be empty.")

    clean_str = url_or_id.strip()

    # Direct 11-character video ID match
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", clean_str):
        return clean_str

    # Regex for URL extraction
    patterns = [
        r"(?:v=|\/embed\/|\/shorts\/|\/v\/|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, clean_str)
        if match:
            return match.group(1)

    raise ValueError(f"Could not extract a valid 11-character YouTube Video ID from: '{url_or_id}'")


def format_timestamp(seconds: float) -> str:
    """Converts seconds into HH:MM:SS or MM:SS formatted string."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fetch_youtube_transcript(url_or_id: str, languages: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Fetches transcript for a YouTube video and returns structured metadata,
    timestamped segments, and combined text.
    """
    video_id = extract_youtube_id(url_or_id)
    pref_langs = languages or ["en", "en-US", "en-GB"]

    logger.info(f"Fetching YouTube transcript for video ID: {video_id}")
    api = YouTubeTranscriptApi()

    try:
        raw_snippets = api.fetch(video_id, languages=pref_langs)
    except Exception as e:
        logger.warning(f"Primary transcript fetch failed for video {video_id}: {e}")
        try:
            transcript_list = api.list(video_id)
            found_t = None
            for t in transcript_list:
                if any(t.language_code.startswith(lang[:2]) for lang in pref_langs):
                    found_t = t
                    break
            if not found_t:
                found_t = next(iter(transcript_list), None)
            
            if found_t:
                raw_snippets = found_t.fetch()
            else:
                raise e
        except Exception as inner_e:
            err_msg = str(inner_e) if str(inner_e) else str(e)
            logger.error(f"Failed to retrieve transcript for {video_id}: {err_msg}")
            raise RuntimeError(f"Could not retrieve transcript for YouTube video ID '{video_id}'. YouTube may be restricting access or captions are unavailable. Details: {err_msg}")

    segments = []
    formatted_lines = []
    plain_text_lines = []

    for snippet in raw_snippets:
        clean_text = snippet.text.replace("\n", " ").strip()
        if not clean_text:
            continue
        
        ts_str = format_timestamp(snippet.start)
        segments.append({
            "start": snippet.start,
            "duration": snippet.duration,
            "timestamp": ts_str,
            "text": clean_text
        })
        formatted_lines.append(f"[{ts_str}] {clean_text}")
        plain_text_lines.append(clean_text)

    full_text_timestamped = "\n".join(formatted_lines)
    full_text_plain = " ".join(plain_text_lines)

    return {
        "video_id": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "full_text": full_text_timestamped,
        "raw_text": full_text_plain,
        "segments": segments,
        "segment_count": len(segments)
    }


def save_transcript_to_dataset(transcript_data: Dict[str, Any], data_dir: Optional[Path] = None) -> Path:
    """
    Saves the formatted YouTube transcript as a text file in settings.DATA_DIR.
    Filename format: youtube_{video_id}.txt
    """
    target_dir = Path(data_dir) if data_dir else settings.DATA_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    video_id = transcript_data["video_id"]
    file_path = target_dir / f"youtube_{video_id}.txt"

    content = (
        f"Source: YouTube Video ({transcript_data['url']})\n"
        f"Video ID: {video_id}\n"
        f"Type: YouTube Transcript\n"
        f"Total Segments: {transcript_data['segment_count']}\n"
        f"========================================\n\n"
        f"{transcript_data['full_text']}\n"
    )

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Saved YouTube transcript to dataset file: {file_path}")
    return file_path


def chunk_youtube_transcript(
    transcript_data: Dict[str, Any], 
    max_chunk_chars: int = 800
) -> List[Document]:
    """
    Combines raw YouTube transcript segments into text-size chunks with timestamp boundaries,
    start/end seconds, and deep-link URLs.
    """
    video_id = transcript_data["video_id"]
    base_url = f"https://www.youtube.com/watch?v={video_id}"
    segments = transcript_data.get("segments", [])
    
    if not segments:
        return []

    documents = []
    current_texts = []
    current_chars = 0
    start_seg = segments[0]
    
    for idx, seg in enumerate(segments):
        text = seg["text"].strip()
        if not text:
            continue
            
        current_texts.append(text)
        current_chars += len(text) + 1
        
        if current_chars >= max_chunk_chars or idx == len(segments) - 1:
            end_seg = seg
            chunk_content = " ".join(current_texts)
            
            start_sec = float(start_seg["start"])
            end_sec = float(end_seg["start"] + end_seg.get("duration", 0.0))
            ts_start = format_timestamp(start_sec)
            ts_end = format_timestamp(end_sec)
            deep_link = f"{base_url}&t={int(start_sec)}s"
            
            doc = Document(
                page_content=chunk_content,
                metadata={
                    "source_type": "youtube",
                    "source": f"YouTube Video ({video_id})",
                    "video_id": video_id,
                    "filename": f"youtube_{video_id}",
                    "start_seconds": start_sec,
                    "end_seconds": end_sec,
                    "timestamp_start": ts_start,
                    "timestamp_end": ts_end,
                    "timestamp": f"{ts_start} - {ts_end}",
                    "url": deep_link,
                    "document_type": "youtube_transcript"
                }
            )
            documents.append(doc)
            
            current_texts = []
            current_chars = 0
            if idx + 1 < len(segments):
                start_seg = segments[idx + 1]

    return documents


def get_or_create_youtube_vectorstore(
    url_or_id: str, 
    transcript_data: Optional[Dict[str, Any]] = None,
    base_store_dir: Optional[Path] = None
):
    """
    Retrieves existing per-video FAISS vector store or creates and indexes a new one.
    Implements per-video store isolation at youtube_faiss_store/{video_id}.
    """
    from core.vectorstore import FaissVectorStore
    video_id = extract_youtube_id(url_or_id)
    root_dir = Path(base_store_dir) if base_store_dir else settings.YOUTUBE_FAISS_STORE_DIR
    video_store_dir = root_dir / video_id
    
    faiss_path = video_store_dir / "faiss.index"
    meta_path = video_store_dir / "metadata.pkl"
    
    # Reuse cached per-video vector store if available
    if faiss_path.exists() and meta_path.exists():
        logger.info(f"Reusing cached YouTube FAISS vector store for video_id: '{video_id}' at {video_store_dir}")
        vstore = FaissVectorStore(persist_dir=video_store_dir, embedding_model=settings.EMBEDDING_MODEL)
        vstore.load()
        return vstore
    
    # Create new per-video store
    if not transcript_data:
        logger.info(f"Fetching transcript to create new YouTube vector store for video_id: '{video_id}'")
        transcript_data = fetch_youtube_transcript(video_id)
        
    chunks = chunk_youtube_transcript(transcript_data)
    vstore = FaissVectorStore(persist_dir=video_store_dir, embedding_model=settings.EMBEDDING_MODEL)
    vstore.build_from_documents(chunks)
    logger.info(f"Created and saved new YouTube FAISS vector store for video_id: '{video_id}' ({len(chunks)} chunks)")
    return vstore


def youtube_transcript_to_documents(transcript_data: Dict[str, Any]) -> List[Document]:
    """Converts transcript data into LangChain Document objects ready for embedding."""
    return chunk_youtube_transcript(transcript_data)
