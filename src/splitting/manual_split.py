
"""
2_pdf:
Token-based splitting of enriched PDF sections for embedding.

This script takes the final enriched PDF JSON files produced by
`pdf_text_add_image_description.py` (in `data/processed/pdf/final_enriched/`)
and splits any overly long sections into smaller, embedding-friendly chunks.

High-level behavior
-------------------
1. Load section-level JSON for one or all manuals from:
       data/processed/pdf/final_enriched/*.json
2. For each section:
   - If its token length (using the embedding tokenizer) is ≤ MAX_TOKENS
     → keep as-is.
   - If it is longer than MAX_TOKENS
     → split into smaller chunks using `SentenceSplitter`, with overlap.
3. Update `metadata.token_length` for each chunk.
4. Save all resulting chunks to:
       data/processed/chunks/manual_<PDF_NAME>.json

Typical usage
-------------
Split all manuals:

    poetry run python -m src.splitting.manual_split

Split a single manual (base name, without `.json`):

    poetry run python -m src.splitting.manual_split \\
        --pdf Statistical-Analysis-in-JASP-A-guide-for-students-2025
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
    """
    Split long PDF sections from a JSON file into smaller token-based chunks.

    This function is designed to work on the final enriched PDF JSONs created by
    the PDF ingestion + image-description pipeline. It ensures that every
    section fed into the embedding/indexing step stays below a configurable
    token limit.

    Behavior:
        • Load sections from `input_json` (list or {"sections": [...]})
        • For each section:
            - Compute token length using `embed_model`'s tokenizer.
            - If `token_length <= max_tokens` → keep the section unchanged.
            - If `token_length > max_tokens`:
                · Use `SentenceSplitter` to break the text into overlapping chunks.
                · For each chunk:
                    - Copy original metadata.
                    - Add `sub_section_id` (e.g. "<section_id>_part1").
                    - Recompute `token_length` for the chunk.
        • Log a summary of token length stats across all chunks.
        • Save the result as:

              output_dir / f"manual_{<input_json.stem>}.json"

          with the structure: {"sections": [ ... ]}

    Args:
        input_json:
            Path to the enriched PDF JSON file to split, typically from
            `data/processed/pdf/final_enriched/`.

        output_dir:
            Directory where the chunked JSON will be written
            (default: `data/processed/chunks/`).

        embed_model:
            Hugging Face model name used to tokenize text for length checks
            (default: `"BAAI/bge-large-en-v1.5"`).

        max_tokens:
            Maximum allowed token length for a section before splitting.

    Returns:
        Path to the output JSON file containing all (possibly split) sections.
    """

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
