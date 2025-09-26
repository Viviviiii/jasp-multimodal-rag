import logging
import os
import subprocess
import tempfile
import hashlib
from typing import List, Dict, Any

import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, util
import chromadb
from chromadb import PersistentClient
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------
# Config
# ---------------------------
CHROMA_DIR = "data/chroma"
TEXT_COLLECTION = "jasp_text"
TEXT_MODEL_NAME = "BAAI/bge-small-en"

# splitting settings
SEMANTIC_THRESHOLD = 0.70       # lower = more splits
MAX_CHARS = 1600                # ~400 tokens
OVERLAP_CHARS = 150

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------
# Optional Whisper
# ---------------------------
try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    whisper = None

# ---------------------------
# Helpers
# ---------------------------
def clean_metadata(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure Chroma metadata only contains supported scalar types (no None)."""
    keep = ["video_id", "url", "title", "author", "publish_date", "duration"]
    cleaned = {}
    for k, v in meta.items():
        if k not in keep:
            continue
        if v is None:
            cleaned[k] = ""
        elif isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned

# ---------------------------
# Video info + comments
# ---------------------------
def fetch_video_info_and_comments(url: str, max_comments: int = 200) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
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

    # filter comments: keep long ones or with replies
    comments = []
    for c in info.get("comments") or []:
        text = (c.get("text") or "").strip()
        if len(text) >= 50 or c.get("reply_count", 0) > 0:
            comments.append(c)

    return meta, comments

# ---------------------------
# Transcript fetching
# ---------------------------
def fetch_transcript(video_id: str, url: str, languages: List[str] | None = None) -> List[Dict[str, Any]]:
    if languages is None:
        languages = ["en", "en-US", "en-GB"]
    try:
        return YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
    except (TranscriptsDisabled, NoTranscriptFound, Exception) as e:
        logging.warning(f"Transcript unavailable: {e}")
        if WHISPER_AVAILABLE:
            logging.info("🎙️ Falling back to Whisper transcription...")
            return fetch_transcript_whisper(url)
        logging.warning("⚠️ Whisper not installed. Skipping transcript.")
        return []

def fetch_transcript_whisper(url: str, model_name="small") -> List[Dict[str, Any]]:
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, "audio.mp3")
        subprocess.run(["yt-dlp", "-f", "bestaudio", "-o", audio_path, url], check=False)
        if not os.path.exists(audio_path):
            logging.error("Failed to download audio for Whisper fallback.")
            return []

        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)

        return [
            {"start": seg["start"], "duration": seg["end"] - seg["start"], "text": seg["text"].strip()}
            for seg in result["segments"]
        ]

# ---------------------------
# Splitting strategies
# ---------------------------
def split_transcript_by_chapters(transcript, chapters):
    if not chapters:
        return [{"title": "Full video", "segments": transcript}]
    results = []
    for idx, ch in enumerate(chapters):
        start = ch["start_time"]
        end = ch.get("end_time", chapters[idx+1]["start_time"] if idx+1 < len(chapters) else float("inf"))
        segs = [s for s in transcript if start <= s["start"] < end]
        results.append({"title": ch["title"], "segments": segs})
    return results

def semantic_split(transcript_segments, model, threshold=SEMANTIC_THRESHOLD):
    segs = [s for s in transcript_segments if (s.get("text") or "").strip()]
    if not segs:
        return []
    texts = [s["text"].strip() for s in segs]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    chunks, buffer = [], [texts[0]]
    prev = embs[0]
    for i in range(1, len(texts)):
        sim = float(util.cos_sim(prev, embs[i]))
        if sim < threshold:
            chunks.append(" ".join(buffer))
            buffer = [texts[i]]
        else:
            buffer.append(texts[i])
        prev = embs[i]
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks

def enforce_max_chars(chunks, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    out = []
    for ch in chunks:
        out.extend(splitter.split_text(ch))
    return [c for c in out if c.strip()]

# ---------------------------
# Build docs
# ---------------------------
def build_docs_from_video(meta: Dict[str, Any], transcript: List[Dict[str, Any]], comments: List[Dict[str, Any]], model) -> List[Document]:
    docs: List[Document] = []
    base_meta = clean_metadata(meta)

    # description as one doc
    if meta.get("description"):
        docs.append(Document(page_content=meta["description"], metadata={**base_meta, "type": "description"}))

    # transcript -> chapters -> semantic split -> enforce size
    chaptered = split_transcript_by_chapters(transcript, meta.get("chapters", []))
    for ch in chaptered:
        chapter_title = ch["title"]
        sem_blocks = semantic_split(ch["segments"], model)
        sized_blocks = enforce_max_chars(sem_blocks)

        for block in sized_blocks:
            if not block.strip():
                continue
            start = int(ch["segments"][0]["start"]) if ch["segments"] else 0
            mm, ss = divmod(start, 60)
            timestamp = f"{mm:02d}:{ss:02d}"
            yt_link = f"{meta['url']}&t={start}s"
            docs.append(Document(
                page_content=block,
                metadata={**base_meta, "type": "transcript", "chapter": chapter_title,
                          "start_time": timestamp, "yt_link": yt_link}
            ))

    # comments
    for c in comments:
        text = (c.get("text") or "").strip()
        if text:
            docs.append(Document(
                page_content=text,
                metadata={**base_meta, "type": "comment",
                          "comment_id": c.get("id"), "author_comment": c.get("author"),
                          "like_count": c.get("like_count"), "reply_count": c.get("reply_count", 0)}
            ))

    logging.info(
        f"Built {len(docs)} docs (desc={sum(1 for d in docs if d.metadata['type']=='description')}, "
        f"trans={sum(1 for d in docs if d.metadata['type']=='transcript')}, "
        f"comm={sum(1 for d in docs if d.metadata['type']=='comment')})."
    )
    return docs

# ---------------------------
# Chroma Indexer
# ---------------------------
class TextIndexer:
    def __init__(self, chroma_dir: str, collection_name: str, model_name: str):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client: PersistentClient = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def upsert_documents(self, docs: List[Document], source_prefix: str):
        if not docs:
            logging.info("No docs to embed.")
            return
        total = 0
        BATCH = 128
        for group in [docs[i:i+BATCH] for i in range(0, len(docs), BATCH)]:
            texts = [d.page_content for d in group]
            metas = [d.metadata for d in group]

            ids = []
            for d in group:
                vid = d.metadata.get("video_id", "novid")
                st = d.metadata.get("start_time", "0000").replace(":", "")
                h = hashlib.md5(d.page_content[:128].encode("utf-8")).hexdigest()[:8]
                ids.append(f"{source_prefix}:{vid}:{st}:{h}")

            embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
            self.collection.add(documents=texts, metadatas=metas, embeddings=embs, ids=ids)
            total += len(group)
        logging.info(f"Inserted {total} docs into Chroma collection '{self.collection.name}'.")

    def query(self, question: str, top_k: int = 3):
        q_emb = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()
        return self.collection.query(query_embeddings=q_emb, n_results=top_k)

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    VIDEO_URL = "https://www.youtube.com/watch?v=j9w7hEfeIbE"

    logging.info("🎥 Fetching metadata + comments via yt-dlp...")
    meta, comments = fetch_video_info_and_comments(VIDEO_URL)
    print("Metadata:", {k: meta[k] for k in ['video_id','title','author','publish_date','duration']})
    print("First 3 comments:", comments[:3])

    logging.info("📝 Fetching transcript...")
    transcript = fetch_transcript(meta["video_id"], meta["url"])
    print("First 5 transcript segments:", transcript[:5])

    logging.info("🔨 Building docs (chapter→semantic→length)...")
    model = SentenceTransformer(TEXT_MODEL_NAME)
    docs = build_docs_from_video(meta, transcript, comments, model)
    for d in docs[:2]:
        print("Doc sample:", d.page_content[:150], "...", d.metadata)

    logging.info("🔎 Embedding into Chroma...")
    indexer = TextIndexer(CHROMA_DIR, TEXT_COLLECTION, TEXT_MODEL_NAME)
    indexer.upsert_documents(docs, source_prefix="youtube")

    logging.info("🔍 Retrieval test...")
    results = indexer.query("How to run a chi-square test in JASP?", top_k=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f"\nResult {i}:\nText: {doc[:300]}...\nMetadata: {meta}")

    logging.info("✅ Video ingestion + retrieval test complete.")
