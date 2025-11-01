"""
-----------------------------------------------------
End-to-end pipeline to:
1. Parse PDF (text + images) → enriched documents
2. Split into semantic chunks
3. Embed and store persistently in ChromaDB

Run:
    poetry run python -m src.pipelines.build_database
-----------------------------------------------------
"""

from loguru import logger
from pathlib import Path
import time
import os

# Stage imports
from src.ingestion.pdf_load_text_images import enrich_llamaparse_with_images
from src.splitting.text_split import split_enriched_documents
from src.embedding.embedding_store import embed_and_store


# ---------------------------------------------------
# DEFAULT CONFIG
# ---------------------------------------------------
PDF_PATH = "./data/raw/Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"
CHROMA_DIR = "./data/vector_store/chroma_db"
COLLECTION_NAME = "jasp_manual_embeddings_v2"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


# ---------------------------------------------------
# UTILITY: Ensure directories exist
# ---------------------------------------------------
def ensure_directories():
    dirs = [
        "./data/processed/enriched",
        "./data/processed/chunks",
        "./data/vector_store",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.debug("📁 Ensured all output directories exist.")


# ---------------------------------------------------
# UTILITY: Timed stage decorator
# ---------------------------------------------------
def timed_stage(stage_name: str):
    """Decorator to measure and log duration of a stage."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            logger.info(f"⏳ Starting {stage_name}...")
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start
                logger.success(f"✅ {stage_name} completed in {elapsed:.2f}s")
                return result
            except Exception as e:
                logger.exception(f"❌ {stage_name} failed: {e}")
                raise
        return wrapper
    return decorator


# ---------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------
def build_database(
    pdf_path: str = PDF_PATH,
    chroma_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embed_model: str = EMBED_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict:
    """
    Run the full ingestion → splitting → embedding pipeline.
    Returns summary info for API/frontend use.
    """
    ensure_directories()
    pdf_name = Path(pdf_path).stem
    logger.info(f"🏗️ Starting full database build for {pdf_name}")

    # 1️⃣ Ingestion (text + image parsing)
    enriched_docs = _ingest_stage(pdf_path)
    if not enriched_docs:
        logger.error("❌ Ingestion failed — no enriched documents returned.")
        return {}

    # 2️⃣ Splitting
    chunks = _splitting_stage(
        enriched_docs,
        embed_model,
        chunk_size,
        chunk_overlap,
        pdf_name,
    )

    # 3️⃣ Embedding & storage
    chunks_json = Path(f"./data/processed/chunks/chunks_{pdf_name}.json")
    _embedding_stage(
        chunks_json,
        chroma_dir,
        collection_name,
        embed_model,
    )

    logger.info("🎉 Database build completed successfully!")

    # 4️⃣ Return structured summary
    return {
        "pdf_name": pdf_name,
        "num_pages": len(enriched_docs),
        "num_chunks": len(chunks),
        "chroma_dir": chroma_dir,
        "collection_name": collection_name,
        "embed_model": embed_model,
    }


# ---------------------------------------------------
# STAGE WRAPPERS WITH TIMING
# ---------------------------------------------------
@timed_stage("PDF ingestion (text + images)")
def _ingest_stage(pdf_path: str):
    return enrich_llamaparse_with_images(pdf_path)


@timed_stage("Semantic chunk splitting")
def _splitting_stage(enriched_docs, embed_model, chunk_size, chunk_overlap, pdf_name):
    chunks, _ = split_enriched_documents(
        enriched_docs=enriched_docs,
        embedding_model=embed_model,
        breakpoint_percentile_threshold=95,
        buffer_size=2,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        save_json=True,
        pdf_name=pdf_name,
    )
    return chunks


@timed_stage("Embedding and ChromaDB storage")
def _embedding_stage(chunks_json: Path, chroma_dir: str, collection_name: str, embed_model: str):
    embed_and_store(
        json_path=str(chunks_json),
        persist_dir=chroma_dir,
        collection_name=collection_name,
        embed_model_name=embed_model,
    )


# ---------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------
if __name__ == "__main__":
    summary = build_database()
    logger.success(f"📊 Build summary: {summary}")
