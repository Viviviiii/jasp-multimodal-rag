
# poetry run python -m src.pipelines.build_database_v2



import sys
import os
from pathlib import Path
from loguru import logger

# Existing pipeline steps
from src.ingestion.pdf_text_loader import run_pdf_ingestion_text_pipeline
from src.ingestion.pdf_text_add_image_description import add_image_descriptions_pipeline
from src.splitting.split_long_sections import split_json_sections_into_chunks
from src.embedding.embedding_store import embed_and_store

# -------------------------------------------------------------------------
# ⚙️ CONFIGURATION
# -------------------------------------------------------------------------
DATA_DIR = Path("/Users/ywxiu/jasp-multimodal-rag/data/raw")
MARKDOWN_DIR = Path("/Users/ywxiu/jasp-multimodal-rag/data/processed/markdown")
JSON_DIR = Path("/Users/ywxiu/jasp-multimodal-rag/data/processed")
CHUNK_DIR = JSON_DIR / "chunks"
MODEL = "llava-phi3:latest"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 500

# -------------------------------------------------------------------------
# 🚀 MAIN PIPELINE
# -------------------------------------------------------------------------
def build_multimodal_database(
    data_dir: Path = DATA_DIR,
    markdown_dir: Path = MARKDOWN_DIR,
    json_dir: Path = JSON_DIR,
    chunk_dir: Path = CHUNK_DIR,
    model: str = MODEL,
    embed_model: str = EMBED_MODEL,
    max_tokens: int = MAX_TOKENS,
):
    logger.info("🏗️ Starting full multimodal database build pipeline...")

    pdf_files = sorted(list(data_dir.glob("*.pdf")))
    if not pdf_files:
        logger.warning(f"No PDFs found in {data_dir}")
        return

    for pdf_path in pdf_files:
        logger.info(f"📄 Processing PDF: {pdf_path.name}")

        # ✅ Step 1 – text extraction
        run_pdf_ingestion_text_pipeline(
            data_dir=str(data_dir),
            markdown_dir=str(markdown_dir),
            json_dir=str(json_dir),
        )

        json_path = json_dir / f"{pdf_path.stem}.json"
        if not json_path.exists():
            logger.error(f"❌ Missing text JSON for {pdf_path.name}")
            continue

        # ✅ Step 2 – image enrichment
        add_image_descriptions_pipeline(
            pdf_path=str(pdf_path),
            text_json_path=str(json_path),
            model=model,
        )

        # ✅ Step 3 – chunk splitting
        logger.info(f"✂️ Splitting long sections for {pdf_path.stem}")
        chunk_output_path = split_json_sections_into_chunks(
            input_json=json_path,
            output_dir=chunk_dir,
            embed_model=embed_model,
            max_tokens=max_tokens,
        )
        logger.success(f"🧩 Saved chunks → {chunk_output_path}")

    # ✅ Step 4 – embed & store in Chroma
    logger.info("🧠 Generating embeddings for all chunked files...")
    embed_and_store(input_path=str(chunk_dir), embed_model_name=embed_model)
    logger.success("🎯 All embeddings stored successfully!")

    logger.success("✅ database build pipeline completed.")


if __name__ == "__main__":
    build_multimodal_database()
