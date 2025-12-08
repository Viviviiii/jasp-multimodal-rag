
"""
------------------------------------------------------------
1_video:
Video Ingestion

This module ingests JASP-related YouTube videos and prepares them for the
RAG pipeline by:

1. Reading a list of videos from `data/raw_video/video.json`
2. Fetching video metadata (title, description, uploader, chapters, etc.)
3. Downloading subtitles/transcript (JSON3) via `yt-dlp`
4. Converting transcript segments into:
   - a full-text transcript
   - chapter-level text (if uploader chapters exist)
5. Saving a structured JSON per video under `data/processed/video/`

Each output JSON contains:
  - basic video metadata (id, title, description, url, author, publish date, duration)
  - the enriched `metadata.chapters` list (with attached `text`)
  - the full transcript segments

CLI usage
---------
Run the full video ingestion pipeline for all videos defined in
`data/raw_video/video.json`:

    poetry run python -m src.ingestion.video_transcript_loader

The processed per-video JSON files will be saved into:

    data/processed/video/
------------------------------------------------------------
"""


import os
import re
import json
import requests
import yt_dlp
from loguru import logger
from pathlib import Path
from typing import Dict, Any, List
from urllib.parse import urlparse, parse_qs

# ------------------------------------------------------------
# ⚙️ CONFIGURATION
# ------------------------------------------------------------
RAW_VIDEO_PATH = Path("data/raw_video/video.json")
PROCESSED_DIR = Path("data/processed/video")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "video_transcript_loader.log"

# Ensure directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Configure Loguru
logger.remove()  # remove default handler
logger.add(
    sink=LOG_FILE,
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}",
)
logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True,
    format="<green>{time:HH:mm:ss}</green> | <level>{level}</level> | <cyan>{message}</cyan>",
)

# ------------------------------------------------------------
# 🧩 Helper: Extract YouTube video ID
# ------------------------------------------------------------
def extract_video_id(url: str) -> str:
    """
    Extract the YouTube video ID from a given URL.
    Works for both long and short YouTube links.
    """
    parsed = urlparse(url)
    if parsed.hostname in ["www.youtube.com", "youtube.com"]:
        query = parse_qs(parsed.query)
        return query.get("v", [None])[0]
    elif parsed.hostname == "youtu.be":
        return parsed.path.lstrip("/")
    else:
        return os.path.basename(url)

# ------------------------------------------------------------
# 🧩 Helper: Safe file naming
# ------------------------------------------------------------
def sanitize_filename(name: str) -> str:
    """Make video name safe for filesystem."""
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:80]

# ------------------------------------------------------------
# 🧠 Load video list
# ------------------------------------------------------------
def load_video_list(path: Path = RAW_VIDEO_PATH) -> List[Dict[str, Any]]:
    """Load list of videos from JSON file."""
    if not path.exists():
        logger.error(f"Video list not found: {path}")
        raise FileNotFoundError(f"Video list not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    videos = data.get("videos", [])
    logger.info(f"Loaded {len(videos)} video entries from {path}")
    return videos

# ------------------------------------------------------------
# 📜 Fetch video metadata
# ------------------------------------------------------------
def fetch_video_info(url: str) -> Dict[str, Any]:
    """Fetch video metadata using yt-dlp."""
    logger.info(f"Fetching video info for {url}")
    ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise RuntimeError(f"yt-dlp failed for {url}")
    except Exception as e:
        logger.exception(f"Failed to fetch video info for {url}: {e}")
        raise

    meta = {
        "video_id": info.get("id"),
        "url": url,
        "title": info.get("title"),
        "description": info.get("description") or "",
        "author": info.get("uploader"),
        "publish_date": info.get("upload_date"),
        "duration": info.get("duration"),
        "chapters": info.get("chapters") or [],
    }
    logger.success(f"Fetched metadata for '{meta['title']}' ({meta['video_id']})")
    return meta

# ------------------------------------------------------------
# 🎬 Fetch transcript (subtitles)
# ------------------------------------------------------------
def fetch_transcript(url: str, lang: str = "en") -> Dict[str, Any]:
    """Download transcript in JSON3 format using yt-dlp."""
    logger.info(f"Fetching transcript for {url} (lang={lang})")
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logger.exception(f"Failed to fetch transcript metadata for {url}: {e}")
        return {"segments": [], "full_text": ""}

    subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    tracks = subs.get(lang) or auto_subs.get(lang)
    if not tracks:
        logger.warning(f"No transcript available for {url}")
        return {"segments": [], "full_text": ""}

    sub_url = next((t["url"] for t in tracks if t["ext"] == "json3"), None)
    if not sub_url:
        logger.warning(f"No JSON3 subtitle track found for {url}")
        return {"segments": [], "full_text": ""}

    try:
        resp = requests.get(sub_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.exception(f"Failed to download captions for {url}: {e}")
        return {"segments": [], "full_text": ""}

    segments = []
    for evt in data.get("events", []):
        if "segs" in evt:
            text = "".join(seg.get("utf8", "") for seg in evt["segs"]).strip()
            if text:
                segments.append({
                    "text": text,
                    "start": evt.get("tStartMs", 0) / 1000.0,
                    "duration": evt.get("dDurationMs", 0) / 1000.0,
                })

    full_text = " ".join(seg["text"] for seg in segments)
    logger.success(f"Fetched {len(segments)} transcript segments for {url}")
    return {"segments": segments, "full_text": full_text}

# ------------------------------------------------------------
# 💾 Process and save video transcripts
# ------------------------------------------------------------
def process_and_save_videos(videos: List[Dict[str, Any]]):
    """
    Fetch metadata + transcripts for each video and save as structured JSON.

    For each entry in `videos` (typically loaded from `data/raw_video/video.json`):

      1. Fetch video metadata with `yt-dlp`:
         - video_id, title, description, author, publish_date, duration
         - uploader-defined chapters (if available)

      2. Fetch transcript/subtitles in JSON3 format (preferred language):
         - build a list of time-stamped segments: {text, start, duration}
         - join all segments into `full_text`

      3. Enrich chapters with text:
         - if uploader chapters exist, attach a `text` field to each chapter
           containing the transcript between its start and end time.
         - if no chapters exist, create a single pseudo-chapter:
           `metadata["chapters"] = [{"text": "<full transcript>"}]`.

      4. Write a JSON file per video into `data/processed/video/`, including:
         - top-level fields: video_id, name, url, category, language
         - `metadata`: enriched video metadata (including chapters with text)
         - `transcript`: original transcript segments + full_text

    Args:
        videos:
            A list of video configuration dicts, typically from
            `load_video_list()`. Each item should contain at least:
              - "url": the YouTube URL
              - "name": a human-readable name
              - optional: "language" (default "en"), "category"

    Returns:
        None. One JSON file is written per video.
    """

    def _collect_text_between(t0: float, t1: float, segs: List[Dict[str, Any]]) -> str:
        """Collect transcript text overlapping [t0, t1)."""
        parts = []
        for s in segs:
            st = float(s.get("start", 0.0))
            en = st + float(s.get("duration", 0.0))
            if en > t0 and st < t1:
                parts.append(s.get("text", ""))
        return " ".join(parts).strip()

    for v in videos:
        url = v["url"]
        lang = v.get("language", "en")
        video_name = v.get("name", "Unknown Title")

        video_id = extract_video_id(url)
        if not video_id:
            logger.warning(f"Skipping video with invalid URL: {url}")
            continue

        safe_name = sanitize_filename(video_name)
        out_path = PROCESSED_DIR / f"{safe_name}_{video_id}.json"

        if out_path.exists():
            logger.info(f"⏩ Skipping already processed video: {out_path.name}")
            continue

        logger.info(f"Processing video: {video_name} ({video_id})")

        try:
            # --- Fetch metadata and transcript ---
            meta = fetch_video_info(url)
            transcript = fetch_transcript(url, lang)

            segments = transcript.get("segments", [])
            full_text = transcript.get(
                "full_text",
                " ".join(s.get("text", "") for s in segments).strip()
            )
            chapters = meta.get("chapters") or []

            # Determine total duration
            total_duration = (
                float(meta.get("duration") or 0.0)
                if meta.get("duration") is not None
                else (segments[-1]["start"] + segments[-1].get("duration", 0.0)
                      if segments else 0.0)
            )

            # --- Enrich chapters with text ---
            if chapters:
                for i, ch in enumerate(chapters):
                    st = float(ch.get("start_time", 0.0) or 0.0)
                    if ch.get("end_time") is not None:
                        et = float(ch["end_time"])
                    else:
                        et = (
                            float(chapters[i + 1].get("start_time", st))
                            if i + 1 < len(chapters)
                            else float(total_duration or st)
                        )

                    ch["text"] = _collect_text_between(st, et, segments)
                logger.info(f"Added text to {len(chapters)} chapters for {video_id}.")
            else:
                # No uploader chapters → add one paragraph-only pseudo chapter
                meta["chapters"] = [{"text": (full_text or "").strip()}]
                logger.info(f"No chapters found; created single full-text entry for {video_id}.")

            # --- Combine results ---
            output_data = {
                "video_id": video_id,
                "name": video_name,
                "url": url,
                "category": v.get("category", ""),
                "language": lang,
                "metadata": meta,         # enriched metadata with text
                "transcript": transcript, # keep full transcript segments
            }

            # --- Save JSON output ---
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            logger.success(f"✅ Saved processed video: {out_path.name}")

        except Exception:
            logger.exception(f"❌ Failed to process video: {url}")

# ------------------------------------------------------------
# 🚀 Main Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        logger.info("🚀 Starting video transcript ingestion pipeline...")
        videos = load_video_list()
        process_and_save_videos(videos)
        logger.success("🎉 All videos processed successfully.")
    except Exception:
        logger.exception("Pipeline failed due to an unexpected error.")
