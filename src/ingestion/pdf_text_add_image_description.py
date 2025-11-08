"""
add_image_description.py
------------------------
Unified section-level image extraction, summarization, and dispatching
for JASP manual PDFs.

This pipeline performs:
1️⃣ Extract images from each relevant PDF page
2️⃣ Summarize each image using an Ollama/LLaVA model
3️⃣ Save a page-level JSON of image summaries (for audit or reuse)
4️⃣ Dispatch summarized images into section placeholders from text JSON
   → up to `max_per_placeholder` images per placeholder

Run:
    poetry run python src/ingestion/pdf_text_add_image_description.py
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
    output_root: str = "./data/page_images",
    min_width: int = MIN_W_DEFAULT,
    min_height: int = MIN_H_DEFAULT,
    overwrite: bool = False,
    max_pages: Optional[int] = None,
    skip_first_n_images_per_page: int = 2,
) -> Dict[int, List[str]]:
    """Extract all valid images from a PDF."""
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
    out_path = output_path or text_json_path.with_name(
        text_json_path.stem + "_with_images.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
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


def simple_tokenize(text: str) -> int:
    """
    Very simple, open-source-friendly tokenizer.
    Counts 'tokens' as sequences of non-whitespace characters.
    This is not model-specific, but good enough for relative length.
    """
    if not text:
        return 0
    # Split on any whitespace; each non-empty piece is a 'token'
    return len(re.findall(r"\S+", text))
    

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

    updated_path = Path("data/processed") / f"{Path(json_path).stem.replace('_with_images', '')}_section_enriched.json"

    with open(updated_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.success(f"🔢 Updated token_length for {len(data)} sections → {updated_path}")
    return str(updated_path)


# -------------------------------------------------------------------------
# 🚀 STEP 4 — Unified pipeline
# -------------------------------------------------------------------------
def add_image_descriptions_pipeline(pdf_path: str, text_json_path: str, model: str = MODEL_DEFAULT):
    """
    Full pipeline:
      1. Extract images from the PDF.
      2. Summarize images with LLaVA.
      3. Dispatch summaries into the text JSON.
      4. Update token length for enriched sections.
    """
    pdf_name = Path(pdf_path).stem
    image_json_path = Path(f"./data/processed/image_summaries/{pdf_name}_summaries.json")

    # ✅ Step 1: Extract all pages
    images_by_page = extract_images_from_pdf(pdf_path, overwrite=False)
    if not images_by_page:
        logger.warning("No images found.")
        return

    # ✅ Step 2: Summarize
    image_data = summarize_all_images(images_by_page, model=model, output_path=image_json_path)

    # ✅ Step 3: Dispatch (creates *_with_images.json)
    enriched_path = dispatch_images_to_placeholders(text_json_path, image_data=image_data)

    # ✅ Step 4: Update token length for enriched data
    final_path = update_token_length(enriched_path)

    logger.success(f"🏁 Pipeline complete → Final enriched JSON with tokens: {final_path}")
    return final_path





# -------------------------------------------------------------------------
# CLI entry
# -------------------------------------------------------------------------
if __name__ == "__main__":
    PDF_PATH = "/Users/ywxiu/jasp-multimodal-rag/data/data/raw/Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"
    TEXT_JSON = "/Users/ywxiu/jasp-multimodal-rag/data/processed/Statistical-Analysis-in-JASP-A-guide-for-students-2025.json"
    add_image_descriptions_pipeline(PDF_PATH, TEXT_JSON, model="llava-phi3:latest")
