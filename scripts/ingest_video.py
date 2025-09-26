# scripts/ingest_video.py
import logging
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

from src.ingestion.video_loader import fetch_video_info_and_comments, fetch_transcript, clean_metadata
from src.retrieval.splitter import split_transcript_by_chapters, semantic_split, enforce_max_chars
from src.retrieval.index import TextIndexer

VIDEO_URL = "https://www.youtube.com/watch?v=j9w7hEfeIbE"
CHROMA_DIR = "data/chroma"
TEXT_COLLECTION = "jasp_text"
TEXT_MODEL_NAME = "BAAI/bge-small-en"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def build_docs_from_video(meta, transcript, comments, model) -> list[Document]:
    docs: list[Document] = []
    base_meta = clean_metadata(meta)

    if meta.get("description"):
        docs.append(Document(page_content=meta["description"], metadata={**base_meta, "type": "description"}))

    chaptered = split_transcript_by_chapters(transcript, meta.get("chapters", []))
    for ch in chaptered:
        sem_blocks = semantic_split(ch["segments"], model)
        sized_blocks = enforce_max_chars(sem_blocks)

        for block in sized_blocks:
            if not block.strip():
                continue
            start = int(ch["segments"][0]["start"]) if ch["segments"] else 0
            mm, ss = divmod(start, 60)
            timestamp = f"{mm:02d}:{ss:02d}"
            yt_link = f"{meta['url']}&t={start}s"
            docs.append(Document(page_content=block, metadata={**base_meta, "type": "transcript", "chapter": ch["title"], "start_time": timestamp, "yt_link": yt_link}))

    for c in comments:
        text = (c.get("text") or "").strip()
        if text:
            docs.append(Document(page_content=text, metadata={**base_meta, "type": "comment", "comment_id": c.get("id"), "author_comment": c.get("author"), "like_count": c.get("like_count"), "reply_count": c.get("reply_count", 0)}))

    logging.info(f"Built {len(docs)} docs (desc={sum(1 for d in docs if d.metadata['type']=='description')}, trans={sum(1 for d in docs if d.metadata['type']=='transcript')}, comm={sum(1 for d in docs if d.metadata['type']=='comment')}).")
    return docs


def main():
    logging.info("🎥 Fetching video data...")
    meta, comments = fetch_video_info_and_comments(VIDEO_URL)
    transcript = fetch_transcript(meta["video_id"], meta["url"])

    logging.info("🔨 Building docs...")
    model = SentenceTransformer(TEXT_MODEL_NAME)
    docs = build_docs_from_video(meta, transcript, comments, model)

    logging.info("🔎 Indexing docs...")
    indexer = TextIndexer(CHROMA_DIR, TEXT_COLLECTION, TEXT_MODEL_NAME)
    indexer.upsert_documents(docs, source_prefix="youtube")

    logging.info("✅ Video ingestion complete.")


if __name__ == "__main__":
    main()
