
"""
1_pdf_b:
Image extraction and description pipeline for JASP PDF sections.

This module enriches section-level text JSON (from `pdf_text_loader.py`) with image paths and short, model-generated descriptions. 
It is designed for JASP manuals and similar PDF documentation.

High-level steps
----------------
1. Extract images from each PDF page using PyMuPDF (fitz), skipping tiny
   or boilerplate images (logos, icons).
2. Summarize each image with an Ollama/LLaVA vision-language model.
3. Save page-level image summaries to JSON for audit and reuse.
4. Match page-level images to section-level placeholders produced by
   `pdf_text_loader.py` and append image descriptions to the section text.
5. Recompute token lengths for each enriched section and save a final
   JSON file ready for RAG spilliting,embedding and indexing.

Typical CLI usage
-----------------
Process all PDFs in the default folder:

    poetry run python -m src.ingestion.pdf_text_add_image_description

Or only a single PDF:

    poetry run python -m src.ingestion.pdf_text_add_image_description \\
        --pdf "Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"

Typical Python usage
--------------------
    from src.ingestion.pdf_text_add_image_description import (
        add_image_descriptions_pipeline,
        run_pdf_add_image_descriptions_batch,
    )

    # Single PDF
    final_json = add_image_descriptions_pipeline(
        pdf_path="data/raw_pdf/Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf",
        text_json_path="data/processed/pdf/text/Statistical-Analysis-in-JASP-A-guide-for-students-2025.json",
    )

    # Batch over a folder
    outputs = run_pdf_add_image_descriptions_batch(
        pdf_dir="data/raw_pdf",
        text_json_dir="data/processed/pdf/text",
        target_pdf=None,
    )
"""


import os
import re
import json
import base64
import fitz  # PyMuPDF
import requests

from time import sleep
from tqdm import tqdm
from pathlib import Path
from loguru import logger
from typing import Dict, List, Optional, Any,Tuple


# -------------------------------------------------------------------------
# ⚙️ GLOBAL CONFIGURATION
# -------------------------------------------------------------------------
MIN_W_DEFAULT = 150
MIN_H_DEFAULT = 150
OLLAMA_URL_DEFAULT = "http://localhost:11434/api/generate"
MODEL_DEFAULT = "llava-phi3:latest"
MAX_PER_PLACEHOLDER = 2

# -------------------------------------------------------------------------
# 📘 STEP 1 — Full-PDF image extraction (your version)
# -------------------------------------------------------------------------
def extract_images_from_pdf(
    pdf_path: str,
    output_root: str = "./data/processed/pdf/images_extracted",
    min_width: int = MIN_W_DEFAULT,
    min_height: int = MIN_H_DEFAULT,
    overwrite: bool = False,
    max_pages: Optional[int] = None,
    skip_first_n_images_per_page: int = 2,
) -> Dict[int, List[str]]:

    """
    Extract raster images from a PDF and save them to disk.

    This function scans each page of a PDF, retrieves all embedded images,
    applies basic filtering (minimum width/height), and writes the valid
    images to an output folder. The function returns a mapping from page
    numbers to lists of extracted image file paths.

    Parameters
    ----------
    pdf_path : str
        Path to the source PDF file.

    output_root : str
        Directory where extracted images will be stored. A subfolder
        named after the PDF file (without extension) will be created.

    min_width : int
        Minimum image width (in pixels) required for extraction.
        Smaller images (e.g., icons or UI fragments) are ignored.

    min_height : int
        Minimum image height (in pixels) required for extraction.

    overwrite : bool
        If False (default), existing extracted image files are reused.
        If True, images are re-extracted and overwritten on disk.

    max_pages : Optional[int]
        Maximum number of pages to process. If None (default), all pages 
        are scanned.

    skip_first_n_images_per_page : int
        Number of images at the beginning of each page's image list to skip.
        This is useful for ignoring boilerplate images such as logos or
        decorative UI elements. If set to a very large value (e.g., 100),
        the function effectively skips all images on most pages.

    Returns
    -------
    Dict[int, List[str]]
        A dictionary mapping page numbers (1-based) to lists of extracted 
        image file paths. Pages without valid images are omitted.

    Notes
    -----
    - Skipping many images does not break the pipeline; it simply results 
      in empty pages downstream, meaning no image summaries will be added.
    - Extraction uses PyMuPDF (fitz), which preserves original image data
      and formats (e.g., PNG, JPEG, etc.).
    - This function does not perform any summarization; it only extracts
      and filters images. Subsequent steps handle summarization and dispatch.
    """
    if not os.path.isfile(pdf_path):
        logger.error(f"❌ PDF not found: {pdf_path}")
        return {}

    pdf_name = Path(pdf_path).stem
    output_dir = Path(output_root) / pdf_name
    
    output_dir.mkdir(parents=True, exist_ok=True)

    images_by_page: Dict[int, List[str]] = {}

    try:
        with fitz.open(pdf_path) as doc:
            total_pages = len(doc)
            max_pages = min(max_pages or total_pages, total_pages)
            logger.info(f"📘 Extracting images from {pdf_name} ({max_pages}/{total_pages} pages)")

            for page_index in tqdm(range(max_pages), desc=f"Extracting {pdf_name}"):
                page = doc[page_index]
                image_list = page.get_images(full=True)
                if not image_list:
                    continue

                if skip_first_n_images_per_page > 0:
                    original_count = len(image_list)
                    image_list = image_list[skip_first_n_images_per_page:]
                    skipped = original_count - len(image_list)
                    if skipped > 0:
                        logger.debug(
                            f"Skipped {skipped} images on page {page_index+1} "
                            f"(keeping {len(image_list)})"
                        )

                page_imgs = []
                for img_idx, img_info in enumerate(
                    image_list, start=skip_first_n_images_per_page
                ):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)
                    w, h = img_data.get("width", 0), img_data.get("height", 0)
                    if w < min_width or h < min_height:
                        logger.debug(f"Skipping small image ({w}x{h}) on page {page_index+1}")
                        continue

                    ext = img_data.get("ext", "png")
                    img_path = output_dir / f"p{page_index+1:03d}_i{img_idx:02d}.{ext}"

                    if img_path.exists() and not overwrite:
                        page_imgs.append(str(img_path))
                        continue

                    with open(img_path, "wb") as f:
                        f.write(img_data["image"])
                    page_imgs.append(str(img_path))

                if page_imgs:
                    images_by_page[page_index + 1] = page_imgs

        total_imgs = sum(len(v) for v in images_by_page.values())
        logger.info(f"✅ Extracted {total_imgs} images from {len(images_by_page)} pages.")
        return images_by_page

    except Exception as e:
        logger.exception(f"⚠️ Failed to extract images: {e}")
        return {}

# -------------------------------------------------------------------------
# 🤖 STEP 2 — Summarize images (same as before)
# -------------------------------------------------------------------------

def summarize_image_with_llava(
    image_path: str,
    model: str = MODEL_DEFAULT,
    ollama_url: str = OLLAMA_URL_DEFAULT
) -> str:
    """Summarize a single image via Ollama."""
    if not os.path.exists(image_path):
        return ""

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": model,
        "prompt": (
            "Identify the type of JASP-related image (plot, graph, table, or interface). "
            "Describe it briefly in a statistician-like manner, starting with: "
            "'The X plot (or graph, or table, or JASP interface) shows…'. "
            "Focus on the key conclusion or highlight what is selected, checked, or chosen "
            "in the interface. If the image type cannot be recognized, skip the description."
        ),
        "images": [img_b64],
        "stream": False
    }

    try:
        resp = requests.post(ollama_url, json=payload, timeout=200)
        resp.raise_for_status()
        summary = resp.json().get("response", "").strip()
        preview = (summary[:80] + "...") if len(summary) > 80 else summary
        logger.info(f"🖼️ {Path(image_path).name}: {preview}")
        return summary

    except Exception as e:
        logger.warning(f"⚠️ Summarization failed for {image_path}: {e}")
        return "(no summary)"




def summarize_all_images(images_by_page: Dict[int, List[str]], model: str = MODEL_DEFAULT, output_path: str = None) -> Dict[int, Dict[str, List[str]]]:
    """Summarize all extracted images and optionally save JSON."""
    results = {}
    for page, imgs in tqdm(images_by_page.items(), desc="Summarizing images"):
        summaries = [summarize_image_with_llava(img, model=model) for img in imgs]
        results[page] = {"images": imgs, "summaries": summaries}

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.success(f"💾 Saved image summaries → {output_path}")

    return results

# -------------------------------------------------------------------------
# 🧩 STEP 3 — Dispatch (unchanged from working version)
# -------------------------------------------------------------------------
def dispatch_images_to_placeholders(
    text_json_path: str,
    image_data: Dict[int | str, Dict[str, List[str]]],
    source_metadata: dict,
    output_path: Optional[str] = None,
) -> str:

    """
    Assign image descriptions to placeholders, allowing fallback to nearby pages.

    Behavior:
      1. Removed `max_per_placeholder` (assign all available images).
      2. If no exact page match:
         - Check 1 page before
         - Then 2 pages before
         - Then 3 pages before
         - Continue until a previous page with images is found
         - If none found in previous pages, check pages ahead (1, 2, 3...).
      3. Each placeholder gets all available images from the matched page.

    Args:
        text_json_path: Path to section-level JSON from pdf_text_loader
        image_data: Dict like {1: {"images": [...], "summaries": [...]}, ...}
        output_path: Optional path for enriched JSON

    Returns:
        str: Path to enriched JSON file
    """
    text_json_path = Path(text_json_path)
    if not text_json_path.exists():
        raise FileNotFoundError(f"Text JSON not found: {text_json_path}")

    with open(text_json_path, "r", encoding="utf-8") as f:
        sections: List[Dict[str, Any]] = json.load(f)

    # Normalize image_data keys to ints
    page_queues: Dict[int, List[Tuple[str, str]]] = {
        int(k): list(zip(v.get("images", []), v.get("summaries", [])))
        for k, v in image_data.items()
    }
    available_pages = sorted(page_queues.keys())

    logger.info(
        f"📘 Loaded {len(sections)} text sections and {len(page_queues)} pages with images."
    )

    placeholder_pattern = re.compile(r"PAGE_(\d+)")
    updated_count = 0
    appended_count = 0

    def find_nearest_page(page_num: int) -> Optional[int]:
        """Find nearest page with available images.
        Searches backward first (1, 2, 3, ... pages before),
        and only if none found, searches forward (1, 2, 3, ... pages after).
        """
        # 1️⃣ Exact match first
        if page_num in page_queues and page_queues[page_num]:
            return page_num

        # 2️⃣ Look backward step by step
        max_page_gap = max(available_pages) if available_pages else 0
        for gap in range(1, max_page_gap + 1):
            prev_page = page_num - gap
            if prev_page in page_queues and page_queues[prev_page]:
                return prev_page

        # 3️⃣ Look forward if no earlier page found
        for gap in range(1, max_page_gap + 1):
            next_page = page_num + gap
            if next_page in page_queues and page_queues[next_page]:
                return next_page

        return None

    # Iterate through sections and assign images
    for sec in sections:
        meta = sec.get("metadata", {})
        # Add PDF-level metadata
        meta.update(source_metadata)

        placeholders = meta.get("image_placeholders") or []

        if not placeholders:
            meta.pop("image_paths", None)
            meta.pop("image_summaries", None)
            continue

        ph = placeholders[0]
        m = placeholder_pattern.search(ph)
        if not m:
            logger.debug(f"Skipping malformed placeholder: {ph}")
            meta.pop("image_paths", None)
            meta.pop("image_summaries", None)
            continue

        page_num = int(m.group(1))
        nearest_page = find_nearest_page(page_num)

        if nearest_page is None:
            logger.debug(f"No images found near page {page_num} (section_id={meta.get('section_id')})")
            meta["image_paths"] = []
            meta["image_summaries"] = []
            continue

        queue = page_queues.get(nearest_page, [])
        if not queue:
            meta["image_paths"] = []
            meta["image_summaries"] = []
            continue

        # Assign all available images from that page
        paths, summaries = zip(*queue) if queue else ([], [])
        meta["image_paths"] = list(paths)
        meta["image_summaries"] = list(summaries)
        updated_count += bool(queue)

         # ⭐ NEW: append image summaries into the section text (for RAG / LlamaIndex)
        if summaries:
            sec_text = sec.get("text", "")
            sec["text"] = sec_text + "\n\nAssociated image descriptions:\n" + "\n".join(summaries)
            appended_count += 1

        # Empty this queue so the same images are not reused
        page_queues[nearest_page] = []

        logger.debug(
            f"✅ Assigned {len(paths)} images from page {nearest_page} → section {meta.get('section_id')}"
        )




    # Save output
    # Save enriched JSON into text_with_images/<PDF_NAME>.json
    out_dir = Path("data/processed/pdf/text_with_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean filename (no suffix)
    pdf_name = Path(text_json_path).stem
    out_path = out_dir / f"{pdf_name}.json"


    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sections, f, indent=2, ensure_ascii=False)

    logger.success(f"💾 Dispatched {updated_count} sections with images → {out_path}")
    return str(out_path)






import tiktoken  # lightweight, efficient tokenizer used by OpenAI models


# -------------------------------------------------------------------------
# 🧮 Helper — Update token length per section
# -------------------------------------------------------------------------
import json
import re
from pathlib import Path
from loguru import logger

from transformers import AutoTokenizer

# -------------------------------------------------------------------------
# 🧮 Tokenize using your real embedding tokenizer + log oversize sections
# -------------------------------------------------------------------------

EMBED_MAX_TOKENS = 512   # your embedding model limit
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(EMBED_MODEL)

def simple_tokenize(text: str) -> int:
    """Tokenize text using your embedding model tokenizer."""
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


 

def update_token_length(json_path: str) -> str:
    """
    Updates the `metadata.token_length` for each section in the given JSON file,
    using a simple open-source-friendly tokenizer (no paid API / proprietary deps).

    Args:
        json_path: Path to the JSON file (each element is a section dict).

    Returns:
        Path to the updated JSON file.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for sec in data:
        text = sec.get("text", "")
        meta = sec.get("metadata", {})
        tokens = simple_tokenize(text)
        meta["token_length"] = tokens
        sec["metadata"] = meta

   
   # Count sections exceeding embedding model token limit
    oversized = [
        sec for sec in data
        if sec.get("metadata", {}).get("token_length", 0) > EMBED_MAX_TOKENS
    ]

    if oversized:
        logger.warning(f"⚠️ {len(oversized)} sections exceed {EMBED_MAX_TOKENS} tokens!")

        # Log detail: section_id & length
        for sec in oversized:
            m = sec["metadata"]
            logger.warning(
                f"    - {m.get('section_id')} : {m.get('token_length')} tokens"
            )
    else:
        logger.info("✅ No sections exceed the embedding token limit.")

    # Save final enriched JSON into final_enriched/<PDF_NAME>.json
    out_dir = Path("data/processed/pdf/final_enriched")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove folder suffixes from name
    pdf_name = Path(json_path).stem  # Already clean now
    updated_path = out_dir / f"{pdf_name}.json"


    with open(updated_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.success(f"🔢 Updated token_length for {len(data)} sections → {updated_path}")
    return str(updated_path)



def load_pdf_source_metadata(pdf_name: str) -> dict:
    """
    Load PDF metadata (source_url, etc.) from data/raw_pdf/pdf_list.json
    based on the PDF name.
    """
    meta_file = Path("data/raw_pdf/pdf_list.json")
    if not meta_file.exists():
        logger.warning("⚠️ pdf_list.json not found — no source_url added.")
        return {}

    with open(meta_file, "r", encoding="utf-8") as f:
        data = json.load(f).get("pdf", [])

    for item in data:
        # match by the base name, without ".pdf"
        if item.get("name") == pdf_name:
            return {
                "source_type": "pdf",
                "source_url": item.get("source_url"),
            }

    logger.warning(f"⚠️ No metadata found in pdf_list.json for '{pdf_name}'")
    return {}




# -------------------------------------------------------------------------
# 🚀 STEP 4 — Unified pipeline
# -------------------------------------------------------------------------
def add_image_descriptions_pipeline(
    pdf_path: str,
    text_json_path: str,
    model: str = MODEL_DEFAULT,
):
    """
    Enrich a single PDF's section JSON with image paths and descriptions.

    This function runs the full image pipeline for one PDF:

        1. Extract images from all pages of `pdf_path` (if no summary JSON exists yet).
        2. Summarize each image using an Ollama/LLaVA vision-language model.
        3. Save a page-level image JSON under `data/processed/pdf/image_description/`.
        4. Dispatch images and summaries into the section-level JSON produced by
           `pdf_text_loader.py`, based on page numbers and placeholders.
        5. Append image descriptions to the section text (for better RAG context).
        6. Recompute token lengths for each section and save a final enriched JSON
           under `data/processed/pdf/final_enriched/`.

    if a page-level image-summary JSON already exists for this PDF, extraction and summarization
    are skipped and the existing summaries are reused.

    Args:
        pdf_path:
            Path to the source PDF file (e.g. `data/raw_pdf/...pdf`).

        text_json_path:
            Path to the section-level JSON produced by `run_pdf_ingestion_text_pipeline`,
            typically under `data/processed/pdf/text/`.

        model:
            Name of the Ollama vision-language model to use (default: `llava-phi3:latest`).

    Returns:
        Path (as a string) to the final enriched JSON file containing:
            - section text + appended image descriptions
            - metadata including image paths, image summaries, and token_length.
    """

    pdf_name = Path(pdf_path).stem

    # Load metadata
    pdf_source_meta = load_pdf_source_metadata(pdf_name)

    # Where the summary JSON should be
    image_json_dir = Path("data/processed/pdf/image_description")
    image_json_dir.mkdir(parents=True, exist_ok=True)
    image_json_path = image_json_dir / f"{pdf_name}.json"

    # ------------------------------------------------------
    # ✅ EARLY SKIP: If image summary exists, skip extraction & summarization
    # ------------------------------------------------------
    if image_json_path.exists():
        logger.info(f"⏭️ Found existing image summary: {image_json_path}")
        logger.info("⏭️ Skipping image extraction & summarization.")

        with open(image_json_path, "r", encoding="utf-8") as f:
            image_data = json.load(f)

    else:
        # ------------------------------------------------------
        # Step 1: Extract images
        # ------------------------------------------------------
        images_by_page = extract_images_from_pdf(pdf_path, overwrite=False)

        if not images_by_page:
            logger.warning(f"⚠️ No extractable images found in {pdf_name}, but continuing with empty image data.")
            image_data = {}
        else:
            # ------------------------------------------------------
            # Step 2: Summarize
            # ------------------------------------------------------
            image_data = summarize_all_images(
                images_by_page,
                model=model,
                output_path=image_json_path
            )

    # ------------------------------------------------------
    # Step 3: Dispatch into section-level JSON
    # ------------------------------------------------------
    enriched_path = dispatch_images_to_placeholders(
        text_json_path,
        image_data=image_data,
        source_metadata=pdf_source_meta,
    )

    # ------------------------------------------------------
    # Step 4: Update token lengths
    # ------------------------------------------------------
    final_path = update_token_length(enriched_path)

    logger.success(f"🏁 Pipeline complete → Final enriched JSON created: {final_path}")
    return final_path



def run_pdf_add_image_descriptions_batch(
    pdf_dir: str,
    text_json_dir: str,
    target_pdf: str | None = None,
    model: str = MODEL_DEFAULT,
):

    """
    Run the image-description pipeline for one or many PDFs in a folder.

    For each PDF in `pdf_dir`:
        - Find the corresponding section JSON in `text_json_dir`
          (same stem, `.json` extension).
        - Call `add_image_descriptions_pipeline` to enrich sections with
          image paths, image summaries, and updated token lengths.

    Args:
        pdf_dir:
            Directory containing the raw PDFs (e.g. `data/raw_pdf`).

        text_json_dir:
            Directory containing section-level JSON files generated by
            `run_pdf_ingestion_text_pipeline` (e.g. `data/processed/pdf/text`).

        target_pdf:
            Optional PDF filename to restrict processing to a single file
            (case-insensitive). If `None`, all `*.pdf` files in `pdf_dir`
            are processed.

        model:
            Name of the Ollama vision-language model to use for summarization.

    Returns:
        A list of paths (as strings) to the final enriched JSON files,
        one per successfully processed PDF.
    """
    pdf_dir = Path(pdf_dir)
    text_json_dir = Path(text_json_dir)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if target_pdf:
        target_pdf = target_pdf.lower()
        pdf_files = [p for p in pdf_files if p.name.lower() == target_pdf]

        if not pdf_files:
            logger.error(f"❌ Target PDF '{target_pdf}' not found in {pdf_dir}")
            return []

        logger.info(f"🎯 Processing ONLY: {target_pdf}")
    else:
        logger.info(f"📘 Processing all {len(pdf_files)} PDFs found in {pdf_dir}")

    outputs = []

    for pdf_path in pdf_files:
        pdf_name = pdf_path.stem
        text_json_path = text_json_dir / f"{pdf_name}.json"

        if not text_json_path.exists():
            logger.warning(f"⚠️ JSON not found for {pdf_name}, skipping.")
            continue

        logger.info(f"🚀 Running pipeline for: {pdf_name}")
        out = add_image_descriptions_pipeline(
            pdf_path=str(pdf_path),
            text_json_path=str(text_json_path),
            model=model,
        )
        outputs.append(out)

    logger.success(f"🏁 Completed image description pipeline for {len(outputs)} PDF(s).")
    return outputs


# -------------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    RAW_PDF_DIR = PROJECT_ROOT / "data/raw_pdf"
    TEXT_JSON_DIR = PROJECT_ROOT / "data/processed/pdf/text"


    # Set None to process all PDFs
    target_pdf = None
    # Or:
    # target_pdf = "Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"

    run_pdf_add_image_descriptions_batch(
        pdf_dir=str(RAW_PDF_DIR),
        text_json_dir=str(TEXT_JSON_DIR),
        target_pdf=target_pdf,
        model=MODEL_DEFAULT,
    )
