"""
poetry run python -m src.ingestion.video_loader
---------------
Utilities for ingesting YouTube videos into structured, query-ready
LangChain Document objects.

Pipeline overview:
1. Fetch video metadata (title, author, duration, chapters, description).
2. Fetch transcript (captions/subtitles) and normalize them into text segments.
3. Split description into paragraph-based chunks (fallback: length split).
4. Split transcript into semantically coherent chunks (hybrid: chapter → semantic → length).
5. Wrap description + transcript chunks into LangChain Documents.

Each Document contains:
- `page_content`: the actual text (description or transcript chunk)
- `metadata`: structured information (video id, url, timestamps, author, etc.)

Comments are intentionally excluded in this version.
"""

import logging
from typing import List, Dict, Any

import yt_dlp
import requests
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# ---------------------------
# Config
# ---------------------------
MAX_CHARS = 1200        # Maximum characters per chunk (~300 tokens)
OVERLAP_CHARS = 150     # Overlap for recursive splitter
SEMANTIC_THRESHOLD = 0.70  # Similarity threshold for semantic splitting
TEXT_MODEL_NAME = "BAAI/bge-small-en"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Load embedder once globally
embedder = SentenceTransformer(TEXT_MODEL_NAME)

# ---------------------------
# Fetch video metadata
# ---------------------------
def fetch_video_info(url: str) -> Dict[str, Any]:
    """
    Fetch metadata of a YouTube video using yt-dlp.
    Includes description and chapter info if available.
    """
    logging.info(f"📺 Fetching video info for {url}")
    ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
        if not info:
            raise RuntimeError(f"yt-dlp failed for {url}")
    except Exception as e:
        logging.error(f"❌ Failed to fetch video info: {e}")
        raise

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
    logging.info(f"✅ Metadata fetched for video {meta['video_id']} ({meta['title']})")
    return meta

# ---------------------------
# Fetch transcript (subtitles)
# ---------------------------
def fetch_transcript(url: str, lang: str = "en") -> Dict[str, Any]:
    """
    Download transcript (captions) in JSON3 format using yt-dlp.
    Returns both segment-level and full concatenated text.
    """
    logging.info(f"📝 Fetching transcript for {url} (lang={lang})")
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3"
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception as e:
        logging.error(f"❌ Failed to fetch transcript metadata: {e}")
        return {"segments": [], "full_text": ""}

    subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}
    tracks = subs.get(lang) or auto_subs.get(lang)
    if not tracks:
        logging.warning("⚠️ No transcript available for this video.")
        return {"segments": [], "full_text": ""}

    sub_url = next((t["url"] for t in tracks if t["ext"] == "json3"), None)
    if not sub_url:
        logging.warning("⚠️ No JSON3 subtitle track available.")
        return {"segments": [], "full_text": ""}

    try:
        resp = requests.get(sub_url)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logging.error(f"❌ Failed to download captions: {e}")
        return {"segments": [], "full_text": ""}

    segments = []
    for evt in data.get("events", []):
        if "segs" in evt:
            text = "".join(seg.get("utf8", "") for seg in evt["segs"]).strip()
            if text:
                segments.append({
                    "text": text,
                    "start": evt.get("tStartMs", 0) / 1000.0,
                    "duration": evt.get("dDurationMs", 0) / 1000.0
                })

    full_text = " ".join(seg["text"] for seg in segments)
    logging.info(f"✅ {len(segments)} transcript segments fetched.")
    return {"segments": segments, "full_text": full_text}

# ---------------------------
# Text splitting utilities
# ---------------------------
def length_split(text: str, meta: Dict[str, Any], uid_prefix: str) -> List[Dict[str, Any]]:
    """
    Split text into fixed-size chunks using RecursiveCharacterTextSplitter.
    Ensures maximum length and overlap.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=MAX_CHARS, chunk_overlap=OVERLAP_CHARS
    )
    chunks = []
    for i, split_text in enumerate(splitter.split_text(text)):
        chunks.append({"id": f"{uid_prefix}_len{i}", "text": split_text, "meta": meta})
    return chunks

def semantic_split(text: str, meta: Dict[str, Any], uid_prefix: str) -> List[Dict[str, Any]]:
    """
    Attempt semantic splitting for transcripts:
    - Break text into sentences
    - Use embeddings to find semantic breakpoints
    - If chunks are too long, fallback to length_split
    """
    if len(text) <= MAX_CHARS:
        return [{"id": f"{uid_prefix}_sem0", "text": text, "meta": meta}]

    sentences = text.split(". ")
    embeddings = embedder.encode(sentences, convert_to_tensor=True)
    sims = util.pytorch_cos_sim(embeddings[:-1], embeddings[1:]).diagonal().cpu().numpy()
    breakpoints = [i + 1 for i, score in enumerate(sims) if score < SEMANTIC_THRESHOLD]

    chunks, start, seg_id = [], 0, 0
    for bp in breakpoints + [len(sentences)]:
        chunk_text = ". ".join(sentences[start:bp]).strip()
        if chunk_text:
            if len(chunk_text) > MAX_CHARS:
                chunks.extend(length_split(chunk_text, meta, f"{uid_prefix}_sem{seg_id}"))
            else:
                chunks.append({"id": f"{uid_prefix}_sem{seg_id}", "text": chunk_text, "meta": meta})
            seg_id += 1
        start = bp
    return chunks

# ---------------------------
# Hybrid split: chapter → semantic → length
# ---------------------------
def hybrid_split(meta: Dict[str, Any], transcript: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Hierarchical splitting of transcript:
    1. Split transcript by chapters (if available).
    2. Within each chapter, perform semantic splitting.
    3. Enforce length limits for overly long segments.
    Returns a list of chunk dicts with metadata attached.
    """
    if not transcript["segments"]:
        logging.warning("⚠️ Transcript is empty, skipping hybrid split.")
        return []

    chapter_blocks = []
    if meta.get("chapters"):
        logging.info(f"⏩ Splitting transcript by {len(meta['chapters'])} chapters")
        chapters = meta["chapters"]
        for i, ch in enumerate(chapters):
            start = ch["start_time"]
            end = chapters[i + 1]["start_time"] if i + 1 < len(chapters) else float("inf")
            texts = [s["text"] for s in transcript["segments"] if start <= s["start"] < end]
            block_text = " ".join(texts).strip()
            if block_text:
                chapter_blocks.append({
                    "text": block_text,
                    "meta": {
                        "video_id": meta["video_id"],
                        "url": meta["url"],
                        "title": meta["title"],
                        "chapter": ch.get("title"),
                        "start_time": start,
                        "yt_link": f"{meta['url']}&t={int(start)}s",
                        "chapter_index": i
                    }
                })
    else:
        logging.info("ℹ️ No chapters found, treating transcript as single block.")
        chapter_blocks = [{
            "text": transcript["full_text"].strip(),
            "meta": {
                "video_id": meta["video_id"],
                "url": meta["url"],
                "title": meta["title"],
                "chapter": None,
                "start_time": 0,
                "yt_link": meta["url"],
                "chapter_index": 0
            }
        }]

    final_chunks = []
    for block in chapter_blocks:
        uid_prefix = f"{block['meta']['video_id']}_ch{block['meta']['chapter_index']}"
        final_chunks.extend(semantic_split(block["text"], block["meta"], uid_prefix))

    logging.info(f"✅ Produced {len(final_chunks)} transcript chunks.")
    return final_chunks

# ---------------------------
# Paragraph-first description splitter
# ---------------------------
def split_description(text: str, meta: Dict[str, Any], uid_prefix: str) -> List[Dict[str, Any]]:
    """
    Split description into paragraphs first.
    If a paragraph is too long, fallback to length_split.
    """
    if not text.strip():
        return []

    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, i = [], 0
    for p in paragraphs:
        if len(p) > MAX_CHARS:
            logging.info(f"↘️ Splitting long description paragraph {i} into smaller chunks")
            chunks.extend(length_split(p, meta, f"{uid_prefix}_p{i}"))
        else:
            chunks.append({"id": f"{uid_prefix}_p{i}", "text": p, "meta": meta})
        i += 1
    logging.info(f"✅ Produced {len(chunks)} description chunks.")
    return chunks

# ---------------------------
# Build LangChain Documents
# ---------------------------

def build_docs_from_video(meta: Dict[str, Any],
                          transcript: Dict[str, Any]) -> List[Document]:
    """
    Convert raw ingestion results into LangChain Document objects:
    - One or more Documents for description (paragraph-first split)
    - One Document per transcript chunk (from hybrid_split)
    Comments are intentionally excluded.
    """
    docs: List[Document] = []

    # --- description ---
    if meta.get("description"):
        desc_chunks = split_description(meta["description"], meta, f"{meta['video_id']}_desc")
        for ch in desc_chunks:
            desc_meta = {
                "video_id": meta["video_id"],
                "url": meta["url"],
                "title": meta["title"],
                "author": meta["author"],
                "publish_date": meta["publish_date"],
                "duration": meta["duration"],
                "type": "description"
            }
            docs.append(Document(page_content=ch["text"], metadata=desc_meta))

    # --- transcript chunks ---
    chunks = hybrid_split(meta, transcript)
    for ch in chunks:
        text = ch.get("text", "")
        if not text.strip():
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={**ch["meta"], "type": "transcript"}
            )
        )

    logging.info(
        f"📦 Built {len(docs)} docs "
        f"(desc={sum(1 for d in docs if d.metadata['type']=='description')}, "
        f"trans={sum(1 for d in docs if d.metadata['type']=='transcript')})."
    )
    return docs


# ---------------------------
# CLI entrypoint
# ---------------------------
if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=j9w7hEfeIbE"
    meta = fetch_video_info(url)
    transcript = fetch_transcript(url, lang="en")
    docs = build_docs_from_video(meta, transcript)
    print(f"✅ Built {len(docs)} documents")
    for d in docs[:3]:
        print("---")
        print("Type:", d.metadata["type"])
        print("Preview:", d.page_content[:400], "...")
