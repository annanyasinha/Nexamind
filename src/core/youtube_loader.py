import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain_core.documents import Document
from config import settings
from utils.logger import logger


def extract_youtube_id(url_or_id: str) -> str:
    """
    Extracts 11-character YouTube video ID from various YouTube URL formats or raw ID string.
    Supported formats:
    - https://www.youtube.com/watch?v=VIDEO_ID
    - https://youtu.be/VIDEO_ID
    - https://www.youtube.com/embed/VIDEO_ID
    - https://www.youtube.com/shorts/VIDEO_ID
    - Raw 11-character ID (e.g. jNQXAC9IVRw)
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
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        # Fallback to listing transcripts to find auto-generated or available languages
        try:
            transcript_list = api.list(video_id)
            found_t = None
            for t in transcript_list:
                if t.language_code in pref_langs:
                    found_t = t
                    break
            if not found_t:
                found_t = next(iter(transcript_list), None)
            
            if found_t:
                raw_snippets = found_t.fetch()
            else:
                raise e
        except Exception as inner_e:
            logger.error(f"Failed to retrieve transcript for {video_id}: {inner_e}")
            raise RuntimeError(f"No usable transcript found for YouTube video ID '{video_id}'. Details: {inner_e}")
    except Exception as e:
        logger.error(f"Error fetching YouTube transcript for {video_id}: {e}")
        raise RuntimeError(f"Failed to fetch YouTube transcript: {str(e)}")

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


def youtube_transcript_to_documents(transcript_data: Dict[str, Any]) -> List[Document]:
    """Converts transcript data into LangChain Document objects ready for embedding."""
    url = transcript_data["url"]
    video_id = transcript_data["video_id"]
    full_text = transcript_data["full_text"]

    doc = Document(
        page_content=full_text,
        metadata={
            "source": f"YouTube Video ({url})",
            "video_id": video_id,
            "url": url,
            "type": "youtube_transcript"
        }
    )
    return [doc]
