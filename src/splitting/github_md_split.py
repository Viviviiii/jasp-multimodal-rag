"""
---------------------------------------------------
🧩 GitHub Markdown Splitter → Structured Chunks (Final)
---------------------------------------------------
Input:  data/raw_github/*.md
Output: data/processed/chunks/github_<filename>.json

Pipeline:
1. Parse markdown structure with LlamaIndex MarkdownNodeParser.
2. Count tokens using BAAI/bge-large-en-v1.5 tokenizer.
3. If > 500 tokens → sentence split via LlamaIndex SentenceSplitter.
4. Use first phrase (or line) of text as section_title.
5. Save as JSON with "github_" prefix.

run:
    poetry run python -m src.ingestion.github_md_loader
---------------------------------------------------
"""

import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict

from loguru import logger
from transformers import AutoTokenizer
from llama_index.core import Document
from llama_index.core.schema import TextNode
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.node_parser.text import SentenceSplitter


# =========================================================
# ⚙️ Initialize tokenizer from model
# =========================================================
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")


def count_tokens(text: str) -> int:
    """Return token length using model tokenizer."""
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def extract_section_title(text: str) -> str:
    """Use the first non-empty line or first sentence as section title."""
    stripped = text.strip()
    if not stripped:
        return None

    # Split by line or sentence boundary
    first_line = stripped.split("\n")[0].strip()
    if len(first_line) > 120:
        # if first line is too long, cut to first 15 words
        return " ".join(first_line.split()[:15]) + "..."
    return first_line


def process_markdown(
    md_path: Path,
    out_dir: Path,
    github_name: str = "jasp-stats/jaspRegression",
    token_limit: int = 500,
) -> int:
    """Split one markdown file and save as JSON with 'github_' prefix."""

    text = md_path.read_text(encoding="utf-8")
    parser = MarkdownNodeParser()
    doc = Document(text=text)
    nodes: List[TextNode] = parser.get_nodes_from_documents([doc])

    sentence_splitter = SentenceSplitter(chunk_size=token_limit, chunk_overlap=50)
    out_dir.mkdir(parents=True, exist_ok=True)

    sections = []
    processing_time = datetime.now().isoformat(timespec="seconds")

    for node in nodes:
        content = node.get_content().strip()
        if not content:
            continue

        token_count = count_tokens(content)
        section_title = extract_section_title(content)

        # --- if longer than token limit → sentence split ---
        if token_count > token_limit:
            sub_texts = sentence_splitter.split_text(content)
            for sub in sub_texts:
                sub_tok = count_tokens(sub)
                sections.append({
                    "text": sub.strip(),
                    "metadata": {
                        "doc_id": str(uuid.uuid4())[:8],
                        "source_type": "github help files",
                        "github_name": github_name,
                        "source_path": str(md_path.resolve()),
                        "markdown_file": md_path.name,
                        "section_title": extract_section_title(sub),
                        "token_length": sub_tok,
                        "processing_date": processing_time,
                    },
                })
        else:
            sections.append({
                "text": content,
                "metadata": {
                    "doc_id": str(uuid.uuid4())[:8],
                    "source_type": "github help files",
                    "github_name": github_name,
                    "source_path": str(md_path.resolve()),
                    "markdown_file": md_path.name,
                    "section_title": section_title,
                    "token_length": token_count,
                    "processing_date": processing_time,
                },
            })

    # --- Save JSON ---
    out_path = out_dir / f"github_{md_path.stem}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"sections": sections}, f, ensure_ascii=False, indent=2)

    logger.info(f"Wrote {len(sections):>4} sections → {out_path.name}")
    return len(sections)


def process_directory(
    in_dir: Path = Path("data/raw_github"),
    out_dir: Path = Path("data/processed/chunks"),
    token_limit: int = 500,
    github_name: str = "jasp-stats/jaspRegression",
):
    """Process all markdown files in directory."""
    md_files = list(in_dir.rglob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found under {in_dir}")
        return

    logger.info(f"Found {len(md_files)} markdown files under {in_dir}")
    total = 0

    for md_file in md_files:
        try:
            total += process_markdown(
                md_file, out_dir, github_name=github_name, token_limit=token_limit
            )
        except Exception as e:
            logger.exception(f"Failed to process {md_file}: {e}")

    logger.success(f"✅ Finished. Total sections written: {total}")


if __name__ == "__main__":
    process_directory(
        in_dir=Path("data/raw_github"),
        out_dir=Path("data/processed/chunks"),
        token_limit=500,
        github_name="jasp-stats/jaspRegression",
    )
