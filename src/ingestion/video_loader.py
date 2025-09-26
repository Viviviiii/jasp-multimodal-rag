# src/ingestion/video_loader.py
import logging, os, subprocess, tempfile
from typing import List, Dict, Any
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

# Optional Whisper fallback
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None


def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    keep = ["video_id", "url", "title", "author", "publish_date", "duration"]
    return {k: (str(meta[k]) if meta[k] is not None else "") for k in keep if k in meta}


def fetch_video_info_and_comments(url: str, max_comments: int = 200) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Fetch metadata + top comments via yt-dlp."""
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "extractor_args": {"youtube": {"max_comments": [str(max_comments)], "comment_sort": ["top"]}},
        "noplaylist": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    if not info:
        raise RuntimeError(f"yt-dlp failed for {url}")

    meta = {
        "video_id": info.get("id"),
        "url": url,
        "title": info.get("title"),
        "description": info.get("description") or "",
        "author": info.get("uploader"),
        "publish_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "chapters": info.get("chapters") or []
    }

    comments = []
    for c in info.get("comments") or []:
        text = (c.get("text") or "").strip()
        if len(text) >= 50 or c.get("reply_count", 0) > 0:
            comments.append(c)
    return meta, comments


def fetch_transcript(video_id: str, url: str, languages: List[str] | None = None) -> List[Dict[str, Any]]:
    """Fetch transcript, fallback to Whisper if disabled."""
    if languages is None:
        languages = ["en", "en-US", "en-GB"]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
        logging.warning(f"Transcript unavailable: {e}")
        if WHISPER_AVAILABLE:
            return fetch_transcript_whisper(url)
        return []


def fetch_transcript_whisper(url: str, model_name="small") -> List[Dict[str, Any]]:
    """Download audio + transcribe with Whisper."""
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        subprocess.run(["yt-dlp", "-f", "bestaudio", "-o", audio_path, url], check=False)
        if not os.path.exists(audio_path):
            return []
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return [{"start": seg["start"], "duration": seg["end"] - seg["start"], "text": seg["text"].strip()} for seg in result["segments"]]
