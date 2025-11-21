# src/pipelines/split_long_sections.py
'''
poetry run python -m src.splitting.split_long_sections 
'''

import json
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter


# -------------------------------------------------------------------------
# ⚙️ CONFIG
# -------------------------------------------------------------------------
INPUT_JSON = Path("./data/processed/Statistical-Analysis-in-JASP-A-guide-for-students-2025_section_enriched.json")
DEFAULT_OUTPUT_DIR = Path("./data/processed/chunks")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 500
DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)



# -------------------------------------------------------------------------
# 🧠 CORE LOGIC
# -------------------------------------------------------------------------
def split_json_sections_into_chunks(
    input_json: Path=INPUT_JSON,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    embed_model: str = EMBED_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> Path:
    """
    Load a JSON file with `{"sections": [...]}`, split long sections into chunks,
    and save a new JSON with `{"sections": all_chunks}`.

    Returns the output path.
    """
    logger.info(f"📥 Loading sections from: {input_json}")
    with open(input_json, "r") as f:
        data = json.load(f)

    # Handle both possible structures
    if isinstance(data, dict):
        sections = data.get("sections", [])
    elif isinstance(data, list):
        sections = data
    else:
        raise ValueError("Unexpected JSON structure: expected list or dict with 'sections' key")

    logger.info(f"📦 Found {len(sections)} sections")

    # Tokenizer & splitter
    logger.info(f"🔤 Loading tokenizer for embed model: {embed_model}")
    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    splitter = SentenceSplitter(chunk_size=max_tokens, chunk_overlap=50)

    all_chunks = []

    for section in sections:
        text = section.get("text", "")
        meta = section.get("metadata", {}) or {}

        token_len = len(tokenizer.tokenize(text))

        # Short enough → keep as is, just update token_length
        if token_len <= max_tokens:
            meta["token_length"] = token_len
            section["metadata"] = meta
            all_chunks.append(section)
            continue

        # Too long → split into chunks
        section_id = meta.get("section_id", "sec")
        logger.info(f"✂️  Splitting section {section_id} ({token_len} tokens)")

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            new_section = section.copy()
            new_meta = meta.copy()

            new_meta["sub_section_id"] = f"{section_id}_part{i+1}"
            new_meta["token_length"] = len(tokenizer.tokenize(chunk))

            new_section["text"] = chunk
            new_section["metadata"] = new_meta
            all_chunks.append(new_section)

    # ---------------------------------------------------------------------
    # 💾 SAVE OUTPUT
    # ---------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Remove the suffix from the stem
    clean_stem = input_json.stem.replace("_section_enriched", "")

    # Save output file using the cleaned name
    output_path = output_dir / f"manual_{clean_stem}.json"

    with open(output_path, "w") as f:
        json.dump({"sections": all_chunks}, f, indent=2, ensure_ascii=False)

    logger.success(f"💾 Saved {len(all_chunks)} chunks → {output_path}")
    return output_path


# -------------------------------------------------------------------------
# 🧾 CLI ENTRYPOINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Split long sections into token-based chunks."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=str(INPUT_JSON),
        help="Path to the JSON file produced by build_database_v2.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write *_chunks.json.",
    )
    parser.add_argument(
        "--embed-model",
        type=str,
        default=EMBED_MODEL,
        help="HF model name for tokenizer (should match your embedding model).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=MAX_TOKENS,
        help="Maximum token length per chunk.",
    )

    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_dir = Path(args.output_dir)

    if not input_json.exists():
        logger.error(f"❌ Input JSON not found: {input_json}")
        raise SystemExit(1)

    split_json_sections_into_chunks(
        input_json=input_json,
        output_dir=output_dir,
        embed_model=args.embed_model,
        max_tokens=args.max_tokens,
    )
