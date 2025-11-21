# src/pipelines/split_long_sections.py
"""
Run:
    poetry run python -m src.splitting.manual_split
or:
    poetry run python -m src.splitting.manual_split --pdf Statistical-Analysis-in-JASP-A-guide-for-students-2025
"""

import json
from pathlib import Path
from loguru import logger

from transformers import AutoTokenizer
from llama_index.core.node_parser import SentenceSplitter


# -------------------------------------------------------------------------
# ⚙️ CONFIG
# -------------------------------------------------------------------------
FINAL_ENRICHED_DIR = Path("data/processed/pdf/final_enriched")
OUTPUT_DIR = Path("data/processed/chunks")   # ← ***Your requested new location***
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
MAX_TOKENS = 500


# -------------------------------------------------------------------------
# 🧠 CORE FUNCTION: split one file
# -------------------------------------------------------------------------
def split_json_sections_into_chunks(
    input_json: Path,
    output_dir: Path = OUTPUT_DIR,
    embed_model: str = EMBED_MODEL,
    max_tokens: int = MAX_TOKENS,
) -> Path:

    logger.info(f"📥 Loading sections from: {input_json}")
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        sections = data.get("sections", [])
    elif isinstance(data, list):
        sections = data
    else:
        raise ValueError("Unexpected JSON format (expected list or dict).")

    logger.info(f"📦 Found {len(sections)} sections")

    tokenizer = AutoTokenizer.from_pretrained(embed_model)
    splitter = SentenceSplitter(chunk_size=max_tokens, chunk_overlap=50)

    all_chunks = []

    for section in sections:
        text = section.get("text", "")
        meta = section.get("metadata", {}) or {}

        token_len = len(tokenizer.tokenize(text))

        if token_len <= max_tokens:
            meta["token_length"] = token_len
            section["metadata"] = meta
            all_chunks.append(section)
            continue

        # ➤ Split long section
        section_id = meta.get("section_id", "sec")
        logger.info(f"✂️ Splitting {section_id} ({token_len} tokens)")

        chunks = splitter.split_text(text)

        for i, chunk_text in enumerate(chunks):
            new_meta = meta.copy()
            new_meta["sub_section_id"] = f"{section_id}_part{i+1}"
            new_meta["token_length"] = len(tokenizer.tokenize(chunk_text))

            new_section = {
                "text": chunk_text,
                "metadata": new_meta
            }
            all_chunks.append(new_section)


    # ---------------------------------------------------------------------
    # 📊 TOKEN LENGTH SUMMARY
    # ---------------------------------------------------------------------
    token_lengths = [sec["metadata"]["token_length"] for sec in all_chunks]

    if token_lengths:
        min_len = min(token_lengths)
        max_len = max(token_lengths)
        avg_len = sum(token_lengths) / len(token_lengths)
        count = len(token_lengths)

        logger.info(
            f"📊 Token lengths — count={count}, min={min_len}, max={max_len}, avg={avg_len:.1f} tokens"
        )
    else:
        logger.warning("⚠️ No token lengths available for summary.")




    # ---------------------------------------------------------------------
    # 💾 SAVE OUTPUT (NEW DIRECTORY)
    # ---------------------------------------------------------------------
    clean_stem = input_json.stem  # e.g. Statistical-Analysis-in-JASP-...

    output_path = output_dir / f"manual_{clean_stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"sections": all_chunks}, f, indent=2, ensure_ascii=False)

    logger.success(f"💾 Saved {len(all_chunks)} chunks → {output_path}")
    return output_path


# -------------------------------------------------------------------------
# 🚀 MAIN: Process all or one
# -------------------------------------------------------------------------
def main(pdf_name: str | None):
    if pdf_name:
        pattern = pdf_name.lower()

        candidates = [
            p for p in FINAL_ENRICHED_DIR.glob("*.json")
            if pattern in p.stem.lower()
        ]

        if not candidates:
            logger.error(f"❌ No matching JSON for '{pdf_name}' found in {FINAL_ENRICHED_DIR}")
            return

        logger.info(f"🎯 Splitting only: {candidates[0].name}")
        split_json_sections_into_chunks(candidates[0])
        return

    # Otherwise process ALL enhanced JSON files
    all_files = sorted(FINAL_ENRICHED_DIR.glob("*.json"))
    logger.info(f"📘 Splitting ALL {len(all_files)} final_enriched JSONs")

    for f in all_files:
        split_json_sections_into_chunks(f)


# -------------------------------------------------------------------------
# 🧾 CLI ENTRYPOINT
# -------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Split long sections into token-based chunks.")
    parser.add_argument(
        "--pdf",
        type=str,
        default=None,
        help="Base name of a specific manual to split (without .json).",
    )

    args = parser.parse_args()
    main(args.pdf)
