"""
------------------------------------------------------------
🎬 VIDEO SPLITTING PIPELINE (TOKEN-BASED + 10% MARGIN + READABLE TIMESTAMPS)
------------------------------------------------------------
Purpose:
    Prepare retrievable video chunks for hybrid RAG (BM25 + Chroma),
    preserving timestamps and token boundaries.

Input:
    data/processed/video/<video_name>_<video_id>.json
Output:
    data/processed/chunks/video_<video_name>_<video_id>_chunks.json

Logic:
    ✅ If chapters exist → split each chapter with SentenceSplitter (≈500 tokens)
    ✅ If no chapters → merge transcript segments until ≈500 tokens (+10% tolerance)
    ✅ start_time & end_time stored in "mm:ss" format
    ✅ Count tokens using BAAI/bge-large-en-v1.5 tokenizer
    ✅ Unified "sections" JSON schema

Usage:
    poetry run python -m src.splitting.video_split
------------------------------------------------------------
"""

import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter

# ------------------------------------------------------------
# ⚙️ CONFIG
# ------------------------------------------------------------
VIDEO_DIR = Path("data/processed/video")
CHUNK_DIR = Path("data/processed/chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 500
TOKEN_MARGIN = 0.10   # allow up to 10% overflow
OVERLAP = 50

# ------------------------------------------------------------
# 🔤 Load tokenizer + splitter
# ------------------------------------------------------------
logger.info(f"🔤 Loading embedding model tokenizer: {EMBED_MODEL}")
embedder = SentenceTransformer(EMBED_MODEL)
splitter = SentenceSplitter(chunk_size=MAX_TOKENS, chunk_overlap=OVERLAP)

# ------------------------------------------------------------
# 🧩 Helpers
# ------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Return approximate token count using embedding model tokenizer."""
    try:
        return len(embedder.tokenizer(text)["input_ids"])
    except Exception:
        return len(text.split())

def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss string (e.g., 95.3 → '1:35')."""
    if seconds is None:
        return None
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}:{secs:02d}"

def make_section(text: str, metadata: dict) -> dict:
    """Format a retrievable section."""
    return {"text": text.strip(), "metadata": metadata}

# ------------------------------------------------------------
# 🧠 Fallback grouping (token-based + 10% margin)
# ------------------------------------------------------------
def group_segments_by_tokens(segments, max_tokens=500, margin=0.1):
    """
    Merge transcript segments until token limit (~max_tokens * (1 + margin))
    is reached; preserve timestamps.
    """
    groups = []
    current_texts = []
    current_tokens = 0
    current_start = None
    prev_end = None
    token_limit = max_tokens * (1 + margin)

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        seg_tokens = count_tokens(text)
        seg_start = float(seg.get("start", 0.0))
        seg_end = seg_start + float(seg.get("duration", 0.0))

        if current_start is None:
            current_start = seg_start

        # flush if token limit exceeded
        if current_tokens + seg_tokens > token_limit and current_texts:
            groups.append({
                "text": " ".join(current_texts).strip(),
                "start_time": current_start,
                "end_time": prev_end,
            })
            current_texts = []
            current_tokens = 0
            current_start = seg_start

        current_texts.append(text)
        current_tokens += seg_tokens
        prev_end = seg_end

    # flush last group
    if current_texts:
        groups.append({
            "text": " ".join(current_texts).strip(),
            "start_time": current_start,
            "end_time": prev_end,
        })

    return groups

# ------------------------------------------------------------
# 🧱 Split one video JSON into retrievable "sections"
# ------------------------------------------------------------
def split_video_file(video_path: Path):
    logger.info(f"Splitting {video_path.name} ...")

    with open(video_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("metadata", {})
    transcript = data.get("transcript", {})

    video_id = data.get("video_id")
    title = meta.get("title", "Untitled Video")
    author = meta.get("author", "")
    url = meta.get("url", "")
    description = meta.get("description", "")
    chapters = meta.get("chapters", [])
    segments = transcript.get("segments", [])
    full_text = transcript.get("full_text", "").strip()

    base_meta = {
        "video_id": video_id,
        "video_title": title,
        "author": author,
        "url": url,
        "source_type": "video_transcript",
        "processing_date": datetime.utcnow().isoformat(),
    }

    sections = []

    # --- 1️⃣ Description ---
    if description.strip():
        desc_splits = splitter.split_text(description)
        for i, t in enumerate(desc_splits):
            metadata = {
                **base_meta,
                "source": "description",
                "section_id": f"{video_id}_desc_{i}",
                "token_length": count_tokens(t),
            }
            sections.append(make_section(t, metadata))
        logger.info(f"→ Added {len(desc_splits)} description sections.")

    # --- 2️⃣ Chapters ---
    if chapters and any("title" in ch for ch in chapters):
        for ch_idx, ch in enumerate(chapters):
            ch_text = ch.get("text", "").strip()
            if not ch_text:
                continue

            chapter_title = ch.get("title", f"Chapter {ch_idx + 1}")
            start_time = ch.get("start_time", 0.0)
            end_time = ch.get("end_time", None)

            splits = splitter.split_text(ch_text)
            for i, t in enumerate(splits):
                metadata = {
                    **base_meta,
                    "source": "chapter",
                    "chapter_title": chapter_title,
                    "start_time": format_timestamp(start_time),
                    "end_time": format_timestamp(end_time),
                    "section_id": f"{video_id}_ch{ch_idx+1}_part{i}",
                    "token_length": count_tokens(t),
                }
                sections.append(make_section(t, metadata))
            logger.info(f"→ Added {len(splits)} chapter sections for {chapter_title}")

    # --- 3️⃣ Fallback: no chapters ---
    elif segments:
        logger.info("No uploader chapters found; merging transcript segments by token count (with 10% margin).")
        groups = group_segments_by_tokens(segments, MAX_TOKENS, TOKEN_MARGIN)

        for i, g in enumerate(groups):
            metadata = {
                **base_meta,
                "source": "transcript_fallback",
                "start_time": format_timestamp(g["start_time"]),
                "end_time": format_timestamp(g["end_time"]),
                "section_id": f"{video_id}_fallback_{i}",
                "token_length": count_tokens(g["text"]),
            }
            sections.append(make_section(g["text"], metadata))

        logger.info(f"→ Added {len(groups)} fallback transcript sections with timestamps.")
    else:
        logger.warning(f"⚠️ No transcript text found for {video_path.name}")

    # --- Save ---
    out_path = CHUNK_DIR / f"video_{video_path.stem}_chunks.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"sections": sections}, f, indent=2, ensure_ascii=False)

    total_tokens = sum(s["metadata"]["token_length"] for s in sections)
    logger.success(f"✅ Saved {len(sections)} sections ({total_tokens} tokens total) → {out_path.name}")

# ------------------------------------------------------------
# 🚀 Main
# ------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Starting video chunk splitting pipeline...")
    video_files = list(VIDEO_DIR.glob("*.json"))

    if not video_files:
        logger.warning("No processed video files found in data/processed/video/")
    else:
        for vf in video_files:
            try:
                split_video_file(vf)
            except Exception:
                logger.exception(f"❌ Failed to process {vf.name}")

    logger.success("🎉 All video files converted into retrievable sections.")
