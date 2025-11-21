"""
pdf_loader.py
--------------
End-to-end PDF ingestion pipeline for JASP manual processing.

This script performs:
0. Download pdf online sources
1. Load PDF(s) using LlamaIndex's SimpleDirectoryReader
2. Clean noisy headers, merge titles, and reconstruct equations
3. Convert to structured Markdown with section headers and image placeholders
4. Split Markdown into section-level chunks
5. Compute token counts and add metadata
6. Export both Markdown and JSON outputs

Usage:
    poetry run python src/ingestion/pdf_text_loader.py
"""

import re
import os
import json
import uuid
from datetime import datetime
from typing import List
from pathlib import Path
from loguru import logger
from llama_index.core import SimpleDirectoryReader, Document
import tiktoken
import requests

# -------------------------------------------------------------------------
# download pdfs from data/raw_pdf/pdf_list.json
# -------------------------------------------------------------------------

def download_pdf(source_url: str, save_path: Path) -> bool:
    """Download a single PDF from URL and save to disk."""
    try:
        logger.info(f"⬇️ Downloading {source_url}")

        r = requests.get(source_url, stream=True, timeout=30)
        r.raise_for_status()

        with open(save_path, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)

        logger.success(f"Saved: {save_path}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download {source_url}: {e}")
        return False



def download_pdfs_from_json(json_file: str, output_dir: str) -> list[Path]:
    """
    Read PDF URLs from JSON and download them.
    
    Expected JSON structure:
    {
        "pdf": [
            {
                "name": "Statistical-Analysis-in-JASP-A-guide-for-students-2025",
                "source_url": "https://...",
                "source_name": "Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"
            }
        ]
    }
    """

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    with open(json_file, "r") as f:
        data = json.load(f)

    pdf_items = data.get("pdf", [])

    downloaded_paths = []
    metadata_list = []

    for item in pdf_items:
        name = item["name"]
        url = item["source_url"]
        filename = item.get("source_name") or (name + ".pdf")
        save_path = output / filename

        success = download_pdf(url, save_path)

        metadata_list.append({
            "name": name,
            "source_url": url,
            "local_path": str(save_path) if success else None,
            "success": success,
            "filesize_bytes": save_path.stat().st_size if success else 0,
            "timestamp": datetime.now().isoformat()
        })

        if success:
            downloaded_paths.append(save_path)

    # save summary
    summary_path = output / "downloaded_pdfs.json"
    with open(summary_path, "w") as f:
        json.dump(metadata_list, f, indent=4)

    logger.info(f"📄 PDF download summary saved → {summary_path}")

    return downloaded_paths


# -------------------------------------------------------------------------
# 🧹 Cleaning utilities
# -------------------------------------------------------------------------
def clean_page_header(text: str) -> str:
    """Remove noisy page headers or footers like '21 | P a g e' or 'JASP 0.19.3 - ...'."""
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if re.match(r"^\d+\s*\|\s*P\s*a\s*g\s*e", line):
            continue
        if "JASP 0." in line or "Professor Mark Goss-Sampson" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def merge_uppercase_titles(lines):
    """Merge consecutive ALL-CAPS lines into one composite title joined by ':'."""
    merged, buffer = [], []
    for line in lines:
        if line.isupper() and len(line.split()) > 1:
            buffer.append(line)
        else:
            if buffer:
                merged.append(": ".join(buffer))
                buffer = []
            merged.append(line)
    if buffer:
        merged.append(": ".join(buffer))
    return merged


# -------------------------------------------------------------------------
# 🧩 Section identification
# -------------------------------------------------------------------------
def convert_text_to_sections(text: str, page_number: str):
    """Split text into sections based on uppercase headers."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = merge_uppercase_titles(lines)

    sections, current_title, current_content = [], None, []
    for line in lines:
        if re.search(r"=|_|<|>", line):  # formula-like → not a header
            current_content.append(line)
            continue
        if line.isupper() and len(line.split()) > 1:
            if current_title or current_content:
                sections.append(
                    {"title": current_title, "content": "\n".join(current_content).strip()}
                )
            current_title, current_content = line, []
        else:
            current_content.append(line)

    if current_title or current_content:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})

    for s in sections:
        if s["title"]:
            s["title"] = f"{s['title']} (page {page_number})"
    return sections


# -------------------------------------------------------------------------
# 🧮 Equation and table normalization
# -------------------------------------------------------------------------
def reconstruct_equations(text: str) -> str:
    """Format typical statistical equations into LaTeX."""
    import unicodedata
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[–—−]", "-", text)

    lines, combined = [l.strip() for l in text.split("\n") if l.strip()], []
    for line in lines:
        if re.match(r"R\s*\^?\s*2\s*=", line, re.I):
            combined.append("$$R^2 = \\frac{SS_M}{SS_T}$$")
        elif re.match(r"F\s*=", line, re.I):
            combined.append("$$F = \\frac{Mean\\ SS_M}{Mean\\ SS_R}$$")
        elif re.match(r"t\s*=", line, re.I):
            combined.append(
                "$$t = \\frac{\\text{mean group 1 - mean group 2}}{\\text{standard error of mean differences}}$$"
            )
        else:
            combined.append(line)
    return "\n".join(combined)

def normalize_tables(text: str) -> str:
    """
    Remove numeric-heavy table artifacts from PDF extraction, 
    while preserving text paragraphs that include numbers or Greek symbols.
    """
    blocks = text.split("\n\n")
    cleaned_blocks = []
    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        # Count lines that contain mostly numbers (not words)
        numeric_lines = sum(1 for l in lines if re.match(r"^[\d\s.,%]+$", l))
        repetition_ratio = 1 - len(set(lines)) / len(lines)

        # Drop only if it's clearly a table: many numeric-only lines, repetitive structure
        if len(lines) > 6 and numeric_lines / len(lines) > 0.6 and repetition_ratio > 0.3:
            continue

        cleaned_blocks.append(block)

    return "\n\n".join(cleaned_blocks)


# -------------------------------------------------------------------------
# 🖼️ Placeholder helpers
# -------------------------------------------------------------------------
def make_placeholders(tag: str) -> str:
    """
    Create a single standardized image placeholder for a section.
    
    Args:
        tag (str): A normalized identifier such as SECTION_NAME_PAGE_6
    
    Returns:
        str: A single markdown-compatible placeholder string.
    
    Example:
        >>> make_placeholder("USING_THE_JASP_ENVIRONMENT_PAGE_6")
        '[IMAGE_PLACEHOLDER_USING_THE_JASP_ENVIRONMENT_PAGE_6]'
    """
    return f"[IMAGE_PLACEHOLDER_{tag}]"




# -------------------------------------------------------------------------
# 🧠 PDF → Markdown conversion
# -------------------------------------------------------------------------

def load_and_convert_to_markdown(
    data_dir: str,
    output_dir: str,
    target_pdf: str | None = None
) -> List[Path]:
    """Load PDFs, clean text, and save as Markdown files."""

    # -------------------------------------------------------------------------
    # 🔒 Step 0 — Only include .pdf files
    # -------------------------------------------------------------------------
    all_pdfs = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if target_pdf:
        # normalize for safety
        target_pdf = target_pdf.lower()
        pdf_files = [
            str(Path(data_dir) / f)
            for f in all_pdfs
            if f.lower() == target_pdf
        ]

        if not pdf_files:
            logger.error(f"❌ Target PDF '{target_pdf}' not found in folder: {data_dir}")
            return []

        logger.info(f"🎯 Processing ONLY target PDF: {target_pdf}")

    else:
        # original behavior → process all PDFs
        pdf_files = [
            str(Path(data_dir) / f)
            for f in all_pdfs
        ]
        logger.info(f"📂 Found {len(pdf_files)} PDF(s) to process.")

    # load only the filtered files
    docs = SimpleDirectoryReader(
        input_files=pdf_files,
        recursive=False
    ).load_data()

    os.makedirs(output_dir, exist_ok=True)
    output_paths = []


    grouped = {}
    for d in docs:
        fname = Path(d.metadata["file_name"]).stem
        grouped.setdefault(fname, []).append(d)

    for fname, doc_list in grouped.items():
        all_sections = []
        logger.info(f"📄 Processing {fname} ({len(doc_list)} pages)")

        for d in doc_list:
            page_num = d.metadata.get("page_label", "Unknown")
            text = clean_page_header(d.text)
            sections = convert_text_to_sections(text, page_num)

            for s in sections:
                content = reconstruct_equations(s["content"])
                content = normalize_tables(content)
                s["content"] = content
            all_sections.extend(sections)

        md_lines = []
        for sec in all_sections:
            if sec["title"]:
                md_lines.append(f"# {sec['title']}\n")
            if sec["content"]:
                md_lines.append(sec["content"])
        md_text = "\n\n".join(md_lines).strip()

        # Insert placeholders before headers
        previous_tag, previous_title = None, None

        def insert_before_header(match):
            nonlocal previous_tag, previous_title
            header_line = match.group(0)
            insert_text = ""
            if previous_tag:
                image_title = f"{previous_title}-IMAGE DESCRIPTIONS"
                insert_text = f"# {image_title}\n{make_placeholders(previous_tag)}\n\n"
            clean_header = header_line.replace("#", "").strip()
            new_tag = re.sub(r"[^A-Za-z0-9]+", "_", clean_header).strip("_").upper()
            previous_tag, previous_title = new_tag, clean_header
            return f"{insert_text}{header_line}"

        md_text = re.sub(r"(?m)^# .+", insert_before_header, md_text)

        if previous_tag:
            final_title = f"{previous_title}-IMAGE DESCRIPTIONS"
            md_text += f"\n\n# {final_title}\n{make_placeholders(previous_tag)}"

        START_MARKER = "# USING THE JASP ENVIRONMENT (page 6)"
        match = re.search(re.escape(START_MARKER), md_text)
        if match:
            md_text = md_text[match.start():]
            logger.info(f"✂️ Trimmed before start marker: {START_MARKER}")

        out_path = Path(output_dir) / f"{fname}.md"
        out_path.write_text(md_text, encoding="utf-8")
        output_paths.append(out_path)
        logger.success(f"📘 Markdown created → {out_path}")

    return output_paths





# -------------------------------------------------------------------------
# ✂️ Split Markdown into section-level docs
# -------------------------------------------------------------------------
def split_markdown_by_sections(document_text: str) -> List[dict]:
    pattern = r"^# (.+?) \(page (\d+)\)"
    matches = list(re.finditer(pattern, document_text, flags=re.MULTILINE))
    sections = []
    for i, match in enumerate(matches):
        title, page = match.group(1), match.group(2)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document_text)
        section_text = document_text[start:end].strip()
        sections.append({
            "text": section_text,
            "metadata": {
                "section_title": title,
                "page_start": page,
                "page_end": page,
            },
        })
    return sections


# -------------------------------------------------------------------------
# 🧮 Token counting and section document building
# -------------------------------------------------------------------------

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def build_section_documents(docs: List[Document]) -> List[Document]:
    documents = []
    for doc in docs:
        for sec in split_markdown_by_sections(doc.text):
            section_doc = Document(
                text=sec["text"],
                metadata={
                    "pdf_name": doc.metadata.get("file_name"),
                    **sec["metadata"],
                },
            )
            section_doc.metadata["token_length"] = count_tokens(sec["text"])
            documents.append(section_doc)
    logger.info(f"✅ Built {len(documents)} section-level documents.")
    return documents


# -------------------------------------------------------------------------
# 💾 Save as JSON for downstream RAG
# -------------------------------------------------------------------------
def save_sections_to_json(
    documents: List[Document],
    output_path: str,
    pdf_name: str,
    source_path: str,
):
    placeholder_pattern = re.compile(r"\[image_placeholder_[a-z0-9_]+\]", re.IGNORECASE)

    def clean_none(d: dict):
        return {k: v for k, v in d.items() if v not in [None, ""]}

    records = []
    for doc in documents:
        text = doc.text
        meta = doc.metadata.copy()
        section_title = meta.get("section_title", "")
        page_start = str(meta.get("page_start", ""))
        section_id = (
            re.sub(r"[^A-Za-z0-9]+", "_", section_title)
            .strip("_")
            .upper()
            + f"_PAGE_{page_start}"
        )
        placeholders = [p.upper() for p in placeholder_pattern.findall(text)]
        record_meta = clean_none({
            "doc_id": str(uuid.uuid4())[:8],
            "pdf_name": pdf_name,
            "source_path": source_path,
            "markdown_file": Path(source_path).name,
            "section_id": section_id,
            "section_title": section_title,
            "page_start": page_start,
            "image_placeholders": placeholders,
            "token_length": meta.get("token_length", count_tokens(text)),
            "processing_date": datetime.now().isoformat(timespec="seconds"),
        })
        records.append({"text": text, "metadata": record_meta})

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(f"💾 Saved {len(records)} section-level documents → {output_path}")

# -------------------------------------------------------------------------
# 🚀 Unified pipeline entry point (CLI + importable)
# -------------------------------------------------------------------------
def run_pdf_ingestion_text_pipeline(
    pdf_list_json: str,
    data_dir: str,
    markdown_dir: str,
    json_dir: str,
    target_pdf: str | None = None,
) -> list[Path]:

    logger.info("🚀 Starting full PDF ingestion pipeline...")

    # ----------------------------------------------------------
    # 0️⃣ STEP 0 — Download PDFs based on a JSON URL list
    # ----------------------------------------------------------
    logger.info("📥 Step 0: Downloading PDFs from online sources...")
    downloaded_pdfs = download_pdfs_from_json(pdf_list_json, data_dir)

    if not downloaded_pdfs:
        logger.error("❌ No PDFs downloaded. Aborting pipeline.")
        return []

    logger.success(f"📥 Downloaded {len(downloaded_pdfs)} PDFs.")


    # ----------------------------------------------------------
    # 1️⃣ Convert PDFs → Markdown
    # ----------------------------------------------------------
    logger.info("📝 Step 1: Converting PDFs → Markdown...")
    

    markdown_paths = load_and_convert_to_markdown(
        data_dir=data_dir,
        output_dir=markdown_dir,
        target_pdf=target_pdf
    )
    # ----------------------------------------------------------
    # 2️⃣ Convert Markdown → Section Documents
    # 3️⃣ Export Section Documents → JSON
    # ----------------------------------------------------------
    logger.info("📦 Step 2–3: Converting Markdown to JSON...")

    generated_jsons = []

    for md_path in markdown_paths:
        pdf_name = f"{md_path.stem}.pdf"

        docs = [
            Document(
                text=md_path.read_text(encoding="utf-8"),
                metadata={"file_name": pdf_name}
            )
        ]

        section_docs = build_section_documents(docs)

        json_out = Path(json_dir) / f"{md_path.stem}.json"
        save_sections_to_json(section_docs, str(json_out), pdf_name, str(md_path))
        generated_jsons.append(json_out)


    logger.success(f"✅ Completed ingestion for {len(generated_jsons)} PDFs.")
    return generated_jsons


# -------------------------------------------------------------------------
# 📦 CLI execution wrapper
# -------------------------------------------------------------------------

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    PDF_LIST_JSON = PROJECT_ROOT / "data/raw_pdf/pdf_list.json"
    RAW_PDF_DIR   = PROJECT_ROOT / "data/raw_pdf"
    MARKDOWN_DIR  = PROJECT_ROOT / "data/processed/pdf/markdown"
    JSON_DIR      = PROJECT_ROOT / "data/processed/pdf/text"
    target_pdf    = "Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf" # if no, run the whole folder

    run_pdf_ingestion_text_pipeline(
        str(PDF_LIST_JSON),
        str(RAW_PDF_DIR),
        str(MARKDOWN_DIR),
        str(JSON_DIR),
        target_pdf=target_pdf,   # <-- Named for clarity
    )
