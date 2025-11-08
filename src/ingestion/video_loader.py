"""
pdf_loader.py
--------------
End-to-end pipeline using your original logic:

1. Load PDF(s) with LlamaIndex SimpleDirectoryReader
2. Clean text, extract sections, reconstruct equations, normalize tables
3. Create a Markdown file with image placeholders
4. Load the Markdown back, split into section-level chunks
5. Count tokens and save chunks + metadata into a JSON file

Run:
    poetry run python pdf_loader.py

Edit the CONFIG section at the top to point to your PDF, markdown, and JSON paths.
"""

import re
import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger
from llama_index.core import SimpleDirectoryReader, Document
import tiktoken


# ----------------------------
# CONFIG – EDIT THESE PATHS
# ----------------------------
PDF_DIR = "/Users/ywxiu/jasp-multimodal-rag/data/raw"
MARKDOWN_OUTPUT_DIR = "/Users/ywxiu/jasp-multimodal-rag/data/processed/markdown"
JSON_OUTPUT_PATH = "/Users/ywxiu/jasp-multimodal-rag/data/processed/test_only.json"

# If you like, also keep these explicit names (as in your original example)
PDF_NAME = "Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"
MARKDOWN_FILE = (
    "/Users/ywxiu/jasp-multimodal-rag/data/processed/markdown/"
    "Statistical-Analysis-in-JASP-A-guide-for-students-2025.md"
)


# ----------------------------
# YOUR ORIGINAL FUNCTIONS
# (kept the same as you sent)
# ----------------------------

def clean_page_header(text: str) -> str:
    """Remove noisy headers/footers like '21 | P a g e' or 'JASP 0.19.3 - ...'."""
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
    """Merge consecutive ALL-CAPS lines into a single title joined by ':'."""
    merged = []
    buffer = []
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


def convert_text_to_sections(text: str, page_number: str):
    """Split text into sections based on uppercase headers — but skip mathematical expressions."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    lines = merge_uppercase_titles(lines)

    sections = []
    current_title = None
    current_content = []

    for line in lines:
        # Detect formula-like line (should NOT be treated as header)
        if re.search(r"=|_|\|^|<|>", line):
            current_content.append(line)
            continue

        # Real uppercase title (not formula)
        if line.isupper() and len(line.split()) > 1:
            # Save previous section
            if current_title or current_content:
                sections.append(
                    {"title": current_title, "content": "\n".join(current_content).strip()}
                )
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    if current_title or current_content:
        sections.append({"title": current_title, "content": "\n".join(current_content).strip()})

    # Add page numbers to section titles
    for s in sections:
        if s["title"]:
            s["title"] = f"{s['title']} (page {page_number})"

    return sections


def reconstruct_equations(text: str) -> str:
    """
    Detect and format specific statistical formulas:
      1. R^2 = SS_M / SS_T
      2. F = Mean SS_M / Mean SS_R
      3. t = mean group 1 - mean group 2 / standard error of mean differences
         (also handles missing dash between group 1 and group 2)
      4. t = (X1 - X2) / sqrt((S1)^2/n1 + (S2)^2/n2)
    Handles:
      - Hyphen variants (–, —, −, --)
      - Unicode math italic letters (𝒎, 𝒆, 𝟏, etc.)
    """
    import re
    import unicodedata

    # Normalize Unicode math letters to ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalize all dash types
    text = re.sub(r"[–—−]", "-", text)
    text = text.replace("--", "-")

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    combined = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # ---------- Case 1: R² ----------
        if re.match(r"R\s*\^?\s*2\s*=", line, re.I):
            if i + 2 < len(lines):
                n1, n2 = lines[i + 1], lines[i + 2]
                if re.search(r"SSM?", n1, re.I) and re.search(r"SST?", n2, re.I):
                    combined.append("$$R^2 = \\frac{SS_M}{SS_T}$$")
                    i += 3
                    continue

        # ---------- Case 2: F-ratio ----------
        if re.match(r"F\s*=", line, re.I):
            if i + 2 < len(lines):
                n1, n2 = lines[i + 1], lines[i + 2]
                if re.search(r"Mean\s+SSM?", n1, re.I) and re.search(r"Mean\s+SSR?", n2, re.I):
                    combined.append("$$F = \\frac{Mean\\ SS_M}{Mean\\ SS_R}$$")
                    i += 3
                    continue

        # ---------- Case 3: Independent t-statistic ----------
        if re.match(r"t\s*=", line, re.I):
            if i + 2 < len(lines):
                num, den = lines[i + 1], lines[i + 2]
                num = re.sub(r"[–—−]", "-", num)
                num = num.replace("--", "-")
                den = re.sub(r"[–—−]", "-", den)

                # If missing dash but has both group 1 & 2 → insert it
                if re.search(r"mean\s+group\s*1", num, re.I) and re.search(r"mean\s+group\s*2", num, re.I):
                    if "-" not in num:
                        num = re.sub(r"(mean\s+group\s*1)\s+(mean\s+group\s*2)", r"\1 - \2", num, flags=re.I)

                if re.search(r"mean\s+group\s*1.*-.*mean\s+group\s*2", num, re.I) and re.search(
                    r"standard\s+error", den, re.I
                ):
                    combined.append(
                        "$$t = \\frac{\\text{"
                        + num.strip()
                        + "}}{\\text{"
                        + den.strip()
                        + "}}$$"
                    )
                    i += 3
                    continue

        # ---------- Case 4: (X1 - X2) type ----------
        if re.match(r"t\s*=", line, re.I):
            if i + 1 < len(lines):
                den = lines[i + 1]
                if re.search(r"\(.*X1.*-.*X2.*\)", line, re.I) and re.search(r"S1", den, re.I):
                    combined.append(
                        "$$t = \\frac{(X_1 - X_2)}{\\sqrt{\\frac{(S_1)^2}{n_1} + \\frac{(S_2)^2}{n_2}}}$$"
                    )
                    i += 2
                    continue

        # ---------- Default ----------
        combined.append(line)
        i += 1

    return "\n".join(combined)


def normalize_tables(text: str) -> str:
    """
    Detect and remove any flattened table-like structures (numeric-heavy or repetitive short lines).
    """
    blocks = text.split("\n\n")
    cleaned_blocks = []

    for block in blocks:
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        if not lines:
            continue

        # Heuristic: numeric-heavy or repeated short lines
        num_lines = sum(1 for l in lines if re.search(r"\d", l) or "<" in l or ">" in l)
        short_lines = sum(1 for l in lines if len(l.split()) <= 4)
        repetition_ratio = 1 - len(set(lines)) / len(lines)

        if (
            len(lines) > 4
            and (num_lines / len(lines) > 0.4 or short_lines / len(lines) > 0.6)
            and repetition_ratio > 0.1
        ):
            logger.debug(f"🧹 Removed flattened table ({len(lines)} lines)")
            continue

        cleaned_blocks.append(block)

    return "\n\n".join(cleaned_blocks)


def make_placeholders(tag: str) -> str:
    """Return 4 standardized placeholders."""
    return "\n".join(f"[IMAGE_PLACEHOLDER_{tag}_{i}]" for i in range(1, 5))


def load_and_convert_to_markdown(data_dir: str, output_dir: str):
    """Load PDFs, clean text, and produce Markdown with placeholders using previous section names."""
    docs = SimpleDirectoryReader(data_dir).load_data()
    os.makedirs(output_dir, exist_ok=True)

    grouped = {}
    for d in docs:
        fname = Path(d.metadata["file_name"]).stem
        grouped.setdefault(fname, []).append(d)

    for fname, doc_list in grouped.items():
        all_sections = []

        for d in doc_list:
            page_num = d.metadata.get("page_label", "Unknown")

            # Step 1: basic cleaning only (remove headers/footers)
            text = clean_page_header(d.text)

            # Step 2: detect sections early — while titles are still intact
            sections = convert_text_to_sections(text, page_num)

            # Step 3: process content inside each section, leave titles untouched
            for s in sections:
                content = s["content"]
                content = reconstruct_equations(content)
                # ⚠️ NOTE: normalize_equations is assumed to exist in your project,
                # since your original script runs fine with it.
                # We keep this call exactly as you had it.
                content = normalize_equations(content)
                content = normalize_tables(content)
                s["content"] = content

            all_sections.extend(sections)

        # build markdown text
        md_lines = []
        for sec in all_sections:
            if sec["title"]:
                md_lines.append(f"# {sec['title']}\n")
            if sec["content"]:
                md_lines.append(sec["content"])

        md_text = "\n\n".join(md_lines).strip()

        # ✅ Insert placeholders before each header — tag = previous header’s name
        previous_tag = None
        previous_title = None

        def insert_before_header(match):
            nonlocal previous_tag, previous_title
            header_line = match.group(0)

            insert_text = ""
            # Only insert if it's not the first header
            if previous_tag:
                image_title = f"{previous_title}-IMAGE DESCRIPTIONS"
                insert_text = f"# {image_title}\n" + make_placeholders(previous_tag) + "\n\n"

            # derive new tag and title for the *next* section
            clean_header = header_line.replace("#", "").strip()
            new_tag = re.sub(r"[^A-Za-z0-9]+", "_", clean_header).strip("_").upper()
            previous_tag = new_tag
            previous_title = clean_header

            return f"{insert_text}{header_line}"

        md_text = re.sub(r"(?m)^# .+", insert_before_header, md_text)

        # ✅ Add final placeholders with last section’s tag and title
        if previous_tag:
            final_title = f"{previous_title}-IMAGE DESCRIPTIONS"
            md_text += f"\n\n# {final_title}\n" + make_placeholders(previous_tag)

        # ----------------------------
        # CONFIG
        # ----------------------------
        START_MARKER = "# USING THE JASP ENVIRONMENT (page 6)"

        # ✅ Remove everything before the configured start marker
        if START_MARKER:
            pattern = re.escape(START_MARKER)
            match = re.search(pattern, md_text)
            if match:
                start_idx = match.start()
                md_text = md_text[start_idx:]
                logger.info(f"✂️ Trimmed content before start marker: {START_MARKER}")
            else:
                logger.warning(f"⚠️ Start marker not found in {fname}, keeping full text.")

        out_path = Path(output_dir) / f"{fname}.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md_text + "\n")

        logger.success(
            f"📘 Markdown created: placeholders named after previous sections → {out_path}"
        )


def split_markdown_by_sections(document_text):
    """
    Split markdown into section chunks by '# TITLE (page N)' pattern.
    Returns a list of dicts with 'text' and 'metadata'.
    """
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


def save_test_only_json(
    documents: List[Document],
    output_path: str,
    pdf_name: str,
    source_path: str,
):
    """
    Save section-level LlamaIndex Documents to a JSON file
    with minimal, ChromaDB-safe metadata.
    Handles case-insensitive placeholder extraction.
    """
    encoding = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(encoding.encode(text))

    def clean_none_values(d: dict) -> dict:
        """Remove keys with None or empty strings (Chroma-safe)."""
        return {k: v for k, v in d.items() if v not in [None, ""]}

    # --- Pattern: match image placeholders, case-insensitive ---
    placeholder_pattern = re.compile(r"\[image_placeholder_[a-z0-9_]+\]", re.IGNORECASE)

    records = []
    for doc in documents:
        text = doc.text
        meta = doc.metadata.copy()

        # --- Parse section title and page number ---
        section_title = meta.get("section_title", "")
        page_start = str(meta.get("page_start", ""))
        section_id = (
            re.sub(r"[^A-Za-z0-9]+", "_", section_title)
            .strip("_")
            .upper()
            + f"_PAGE_{page_start}"
        )

        # --- Extract image placeholders (case-insensitive) ---
        placeholders = placeholder_pattern.findall(text)
        # Normalize placeholders to uppercase for consistency
        placeholders = [p.upper() for p in placeholders]

        # --- Build metadata dict ---
        record_meta = {
            "doc_id": str(uuid.uuid4())[:8],
            "pdf_name": pdf_name,
            "source_path": source_path,
            "markdown_file": Path(source_path).name,
            "section_id": section_id,
            "section_title": section_title,
            "page_start": page_start,
            "image_placeholders": placeholders,
            "token_length": count_tokens(text),
            "processing_date": datetime.now().isoformat(timespec="seconds"),
        }

        # --- Remove null/empty keys ---
        record_meta = clean_none_values(record_meta)

        # --- Append record ---
        records.append({"text": text, "metadata": record_meta})

    # --- Save to JSON ---
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved {len(records)} section-level documents → {output_path}")
    if len(records) > 1:
        print(f"📸 Example placeholders from second doc: {records[1]['metadata'].get('image_placeholders', [])}")


# ----------------------------
# MAIN – wire everything together
# ----------------------------

def main():
    # 1) PDF → Markdown (your original workflow)
    load_and_convert_to_markdown(PDF_DIR, MARKDOWN_OUTPUT_DIR)

    # 2) Load the produced markdown back with SimpleDirectoryReader
    docs = SimpleDirectoryReader(
        input_dir=MARKDOWN_OUTPUT_DIR,
        recursive=False
    ).load_data()

    print(len(docs), docs[0].metadata)
    print(docs[0].text[:500])

    # 3) Split markdown into section-level "documents"
    documents = []
    for doc in docs:
        for sec in split_markdown_by_sections(doc.text):
            documents.append(
                Document(
                    text=sec["text"],
                    metadata={
                        "pdf_name": doc.metadata.get("file_name"),
                        **sec["metadata"],
                    }
                )
            )

    print(f"✅ Created {len(documents)} section-level documents")

    # 4) Save as JSON exactly like your original example
    save_test_only_json(
        documents=documents,
        output_path=JSON_OUTPUT_PATH,
        pdf_name=PDF_NAME,
        source_path=MARKDOWN_FILE,
    )


if __name__ == "__main__":
    logger.info("🚀 Starting full PDF → Markdown → JSON pipeline (original logic)...")
    main()
    logger.info("✅ Pipeline completed successfully!")
