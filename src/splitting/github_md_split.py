
"""
---------------------------------------------------
2_github:
GitHub Markdown Splitter → Structured Chunks
---------------------------------------------------

This module converts raw GitHub Markdown files (downloaded by
`src/ingestion/github_md_loader.py`) into clean, token-limited chunks that
can be embedded by the RAG pipeline.

Input:
    data/raw_github/<repo_prefix>__<filename>.md

Output:
    data/processed/chunks/github_<filename>.json

What it does:
-------------
1. Parse Markdown structure using `LlamaIndex`'s `MarkdownNodeParser`.
2. Extract text nodes and compute token length using the
   BAAI/bge-large-en-v1.5 tokenizer.
3. If a node exceeds the token limit (default: 500 tokens), split it using
   `SentenceSplitter` with overlap for smoother retrieval.
4. Use the first non-empty line (or first short phrase) as the section title.
5. Attach metadata such as:
      • repo_name        (e.g., "jasp-stats/jaspRegression")
      • markdown_file    (local filename)
      • md_url           (source GitHub URL)
      • section_title
      • token_length
      • processing_date
6. Save all resulting chunks as JSON with a consistent `"github_"` prefix.

Run manually:
    poetry run python -m src.splitting.github_md_split
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
# ⚙️ a helper to add md_url metadata
# =========================================================

def load_github_md_urls(json_path="data/raw_github/github_list.json"):
    """
    Build a dictionary mapping repo name → base source_url for markdown files.
    Example:
        {
          "jasp-stats/jaspRegression": "https://github.com/jasp-stats/jaspRegression/tree/master/inst/help/"
        }
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"{json_path} not found")

    data = json.loads(json_path.read_text())
    repo_to_base_url = {}

    for entry in data.get("github", []):
        repo = entry["repo"]
        base_url = entry["source_url"].rstrip("/") + "/"   # ensure ending slash
        repo_to_base_url[repo] = base_url

    return repo_to_base_url





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
    token_limit: int = 500,
    repo_name: str = None,
    md_url: str = None,
    md_name: str = None,
):
    """
    Split one markdown file into structured JSON chunks.
    Includes:
        - repo_name
        - md_name
        - md_url
    """

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

        # --- Determine metadata to attach ---
        metadata_common = {
            "doc_id": str(uuid.uuid4())[:8],
            "source_type": "github help files",
            "repo_name": repo_name,
            "markdown_file": md_path.name,
            "md_name": md_name,  # NEW
            "md_url": md_url,    # NEW
            "source_path": str(md_path.resolve()),
            "section_title": section_title,
            "processing_date": processing_time,
        }

        # --- if longer than token limit → sentence split ---
        if token_count > token_limit:
            sub_texts = sentence_splitter.split_text(content)
            for sub in sub_texts:
                sections.append({
                    "text": sub.strip(),
                    "metadata": {
                        **metadata_common,
                        "token_length": count_tokens(sub),
                        "section_title": extract_section_title(sub),
                    },
                })
        else:
            sections.append({
                "text": content,
                "metadata": {
                    **metadata_common,
                    "token_length": token_count,
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
):
    """
    Process all markdown files stored with repo prefix:
    prefix__filename.md
    Example:
        jasp-stats_jaspRegression__RegressionLinear.md
    """

    # ----------------------------------------------------------
    # 1. Load mapping from github_list.json
    # ----------------------------------------------------------
    repo_map = load_github_md_urls()
    # repo_map example:
    #   "jasp-stats/jaspRegression": "https://github.com/.../inst/help/"
    #   "jasp-stats/jaspMixedModels": "https://github.com/.../inst/help/"

    # Build reverse lookup for safe repo prefix
    reverse_map = {
        repo.replace("/", "_"): repo
        for repo in repo_map.keys()
    }

    # ----------------------------------------------------------
    # 2. Find markdown files
    # ----------------------------------------------------------
    md_files = list(in_dir.rglob("*.md"))
    if not md_files:
        logger.warning(f"No markdown files found under {in_dir}")
        return

    logger.info(f"Found {len(md_files)} markdown files under {in_dir}")

    total = 0

    # ----------------------------------------------------------
    # 3. Determine repo + create md_url + md_name
    # ----------------------------------------------------------
    for md_file in md_files:
        fname = md_file.name

        if "__" not in fname:
            logger.warning(f"Invalid filename (missing prefix): {fname}")
            continue

        safe_repo_prefix, md_filename = fname.split("__", 1)

        # Lookup original repo_name (with /)
        if safe_repo_prefix not in reverse_map:
            logger.warning(f"Unknown repo prefix '{safe_repo_prefix}' in file: {fname}")
            continue

        repo_name = reverse_map[safe_repo_prefix]
        base_url = repo_map[repo_name]

        # e.g. https://github.com/.../inst/help/RegressionLinear.md
        md_url = f"{base_url}{md_filename}"

        # md_name = "jasp-stats/jaspRegression__RegressionLinear.md"
        md_name = f"{repo_name}__{md_filename}"

        # ------------------------------------------------------
        # 4. Process markdown chunks
        # ------------------------------------------------------
        try:
            total += process_markdown(
                md_path=md_file,
                out_dir=out_dir,
                token_limit=token_limit,
                repo_name=repo_name,
                md_url=md_url,
                md_name=md_name,
            )
        except Exception as e:
            logger.exception(f"Failed to process {md_file}: {e}")

    logger.success(f"✅ Finished. Total sections written: {total}")


if __name__ == "__main__":
    process_directory(
        in_dir=Path("data/raw_github"),
        out_dir=Path("data/processed/chunks"),
        token_limit=512
    )
