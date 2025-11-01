
'''
The PDF ingestion process involved extracting textual and visual information from the JASP manual 
to prepare data for downstream retrieval-augmented generation (RAG) tasks. Using LlamaParse, 
the PDFs were parsed into structured text documents, 
while relevant figures were extracted and summarized through LLaVA to capture visual context. 
The textual and visual content were then merged into unified, page-level representations and
stored as structured JSON files. This format preserves both semantic and contextual information,
 enabling efficient reuse for later stages such as text chunking, embedding generation, 
 and vector database indexing.


run it:
poetry env activate
poetry run python src/ingestion/pdf_load_text_images.py

 
 '''








import uuid
import os
import sys
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger
from llama_parse import LlamaParse
from llama_index.core import Document
import base64
import requests
from time import sleep

# -----------------------------
# ENVIRONMENT SETUP
# -----------------------------
env_path = Path.cwd() / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ .env file loaded from: {env_path}")
else:
    print("⚠️ .env file not found. Using system environment variables.")

# -----------------------------
# LOGGING CONFIGURATION (Loguru Only)
# -----------------------------
os.makedirs("logs", exist_ok=True)

logger.remove()  # remove default handler
logger.add(sys.stderr, level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
logger.add("logs/pdf_load_text_images.log", rotation="1 MB", level="DEBUG")

logger.info("🔧 Logging initialized (Loguru)")



# -----------------------------
# MAIN FUNCTION
# -----------------------------
def load_pdf_with_llamaparse(
    file_path: str,
    result_type: str = "text",
    api_key: Optional[str] = None,
    save_output: bool = False,
    output_dir: Optional[str] = None,
) -> List[Document]:
    """
    Parse and load a PDF document into structured LlamaIndex Documents using LlamaParse.

    Args:
        file_path (str): Path to the PDF file.
        result_type (str): 'text', 'markdown', or 'json'. Default = 'text'.
        api_key (Optional[str]): Llama Cloud API key. Defaults to .env variable.
        save_output (bool): Whether to save parsed text to disk for inspection.
        output_dir (Optional[str]): Directory to save outputs. Defaults to PROCESSED_DIR in .env or './data/processed'.

    Returns:
        List[Document]: List of Documents with text + metadata.
    """
    logger.info(f"📄 Starting LlamaParse for {os.path.basename(file_path)}")

    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    api_key = api_key or os.getenv("LLAMA_CLOUD_API_KEY")
    if not api_key:
        raise ValueError("❌ Missing LLAMA_CLOUD_API_KEY in environment variables or function argument.")

    output_dir = output_dir or os.getenv("PROCESSED_DIR", "./data/processed")
    os.makedirs(output_dir, exist_ok=True)

    parser = LlamaParse(api_key=api_key, result_type=result_type)

    try:
        raw_docs = parser.load_data(file_path)
        if not raw_docs:
            logger.warning("⚠️ No content extracted from PDF.")
            return []

        documents = [
            Document(
                text=d.text.strip(),
                metadata={"source": os.path.basename(file_path), "page": i + 1}
            )
            for i, d in enumerate(raw_docs)
        ]

        logger.info(f"✅ Parsed {len(documents)} pages successfully.")

        if save_output:
            pdf_name = os.path.splitext(os.path.basename(file_path))[0]
            out_file = os.path.join(output_dir, f"{pdf_name}_text.txt")
            with open(out_file, "w", encoding="utf-8") as f:
                for i, doc in enumerate(documents):
                    f.write(f"\n--- Page {i+1} ---\n{doc.text}\n")
            logger.info(f"💾 Saved parsed text to {out_file}")

        return documents

    except Exception as e:
        logger.exception(f"Error parsing PDF: {e}")
        raise e



MIN_W_DEFAULT = 150
MIN_H_DEFAULT = 150


def extract_images_from_pdf(
    pdf_path: str,
    output_root: str = "./data/page_images",
    min_width: int = MIN_W_DEFAULT,
    min_height: int = MIN_H_DEFAULT,
    overwrite: bool = False,
    max_pages: Optional[int] = None,
    skip_first_n_images_per_page: int = 2,  # ✅ new parameter
) -> Dict[int, List[str]]:
    """
    Extract all valid images from a PDF into structured folders, while skipping
    the first N images on each page (e.g., logos or repeated headers).

    Args:
        pdf_path (str): Path to the PDF file.
        output_root (str): Base directory to store extracted images.
        min_width (int): Minimum image width to include.
        min_height (int): Minimum image height to include.
        overwrite (bool): If True, overwrite existing extracted images.
        max_pages (Optional[int]): Limit number of pages to process.
        skip_first_n_images_per_page (int): Number of first images to skip on
            each page (useful to ignore logos or decorative headers).

    Returns:
        Dict[int, List[str]]: Mapping of page numbers to lists of saved image paths.
            Example: {1: ["./data/page_images/file/p001_i02.png", ...], 2: [...], ...}
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

                # ✅ Skip the first N images per page (e.g., logos)
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
                # enumerate starting from the skipped index for consistent naming
                for img_idx, img_info in enumerate(
                    image_list, start=skip_first_n_images_per_page
                ):
                    xref = img_info[0]
                    img_data = doc.extract_image(xref)
                    w, h = img_data.get("width", 0), img_data.get("height", 0)

                    # ✅ Skip too-small images
                    if w < min_width or h < min_height:
                        logger.debug(f"Skipping small image ({w}x{h}) on page {page_index+1}")
                        continue

                    ext = img_data.get("ext", "png")
                    img_path = output_dir / f"p{page_index+1:03d}_i{img_idx:02d}.{ext}"

                    # ✅ Avoid re-writing if file exists and overwrite=False
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


def summarize_image_with_llava(
    image_path: str,
    model: str,
    prompt: str = "Summarize this image in a concise, academic tone relevant to a statistical software(JASP) interface screenshots or outputs.",
    ollama_url: str = "http://localhost:11434/api/generate",
    max_retries: int = 2,
) -> str:
    """
    Summarizes an image using a vision-language model served by Ollama (e.g. LLaVA).

    Args:
        image_path (str): Path to image.
        model (str): Ollama model name.
        prompt (str): Prompt for image summarization.
        ollama_url (str): Ollama API endpoint.
        max_retries (int): Retry attempts on timeout/failure.

    Returns:
        str: Summary text (or empty string if failed).
    """
    if not os.path.exists(image_path):
        logger.error(f"Image not found: {image_path}")
        return ""

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {"model": model, "prompt": prompt, "images": [img_b64], "stream": False}

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(ollama_url, json=payload, timeout=90)
            response.raise_for_status()
            summary = response.json().get("response", "").strip()

            if summary:
                preview = summary[:80] + ("..." if len(summary) > 80 else "")
                logger.info(f"🖼️ {os.path.basename(image_path)} → {preview}")
                return summary
            else:
                logger.warning(f"Empty summary from model for {image_path}")
                return ""

        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {e}")
            sleep(2)

    logger.error(f"❌ Failed to summarize {image_path} after {max_retries} retries.")
    return ""


def extract_and_summarize_images(
    pdf_path: str,
    output_root: str = "./data/page_images",
    model: str = "llava-phi3:latest",
    save_json: bool = True,
    overwrite: bool = False,
    max_pages: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[int, Dict[str, List[str]]]:
    """
    Full multimodal image pipeline:
    1. Extract images from PDF.
    2. Summarize each image via Ollama (LLaVA).
    3. Save JSON results for later enrichment.
    """
    pdf_name = Path(pdf_path).stem
    logger.info(f"🚀 Starting image pipeline for {pdf_name}")

    # Step 1: extract
    images_by_page = extract_images_from_pdf(pdf_path, output_root, overwrite=overwrite, max_pages=max_pages)
    if not images_by_page:
        logger.warning(f"No images extracted from {pdf_name}")
        return {}

    # Step 2: summarize
    enriched_data = {}
    for page_num, img_paths in tqdm(images_by_page.items(), desc="Summarizing images"):
        summaries = []
        for img_path in img_paths:
            if dry_run:
                summaries.append("(skipped - dry run)")
                continue

            summary = summarize_image_with_llava(img_path, model=model)
            summaries.append(summary or "(no summary)")

        enriched_data[page_num] = {"images": img_paths, "summaries": summaries}

    logger.info(f"✅ Completed summarization for {len(enriched_data)} pages.")

    # Step 3: save JSON
    if save_json:
        out_dir = Path("./data/processed/image_summaries")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{pdf_name}_summaries.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched_data, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Saved summaries to {out_path}")

    return enriched_data


def enrich_llamaparse_with_images(
    pdf_path: str,
    result_type: str = "text",
    save_output: bool = True,
    output_dir: str = "./data/processed/enriched",
) -> List[Document]:
    """
    Enriches text parsed from PDF using image summaries from LLaVA.
    
    Args:
        pdf_path (str): Path to the PDF file.
        result_type (str): Output format from LlamaParse ("text" / "markdown" / "json").
        save_output (bool): Whether to save merged results as JSON.
        output_dir (str): Directory for enriched output.

    Returns:
        List[Document]: Enriched LlamaIndex Document objects.
    """

    pdf_name = Path(pdf_path).stem
    logger.info(f"🚀 Starting multimodal enrichment for '{pdf_name}'")

    # 1️⃣ Load text from PDF via LlamaParse
    text_docs = load_pdf_with_llamaparse(pdf_path, result_type=result_type, save_output=False)
    if not text_docs:
        logger.warning("⚠️ No text documents parsed. Skipping enrichment.")
        return []

    logger.info(f"📝 Loaded {len(text_docs)} text pages from LlamaParse")

    # 2️⃣ Extract and summarize images via LLaVA
    image_summaries = extract_and_summarize_images(
        pdf_path=pdf_path,
        save_json=False,
        overwrite=False
    )

    if not image_summaries:
        logger.warning("⚠️ No images found — proceeding with text only enrichment.")
    else:
        logger.info(f"🖼️ Loaded image summaries for {len(image_summaries)} pages.")

    # 3️⃣ Merge text + image summaries by page
    enriched_docs: List[Document] = []
    for doc in text_docs:
        page_num = doc.metadata.get("page", 0)
        image_data = image_summaries.get(page_num, {})
        image_paths = image_data.get("images", [])
        image_summary_texts = image_data.get("summaries", [])

        merged_text = doc.text.strip()

        if image_summary_texts:
            merged_text += "\n\n[Image Summaries on this page:]\n"
            for i, summary in enumerate(image_summary_texts, 1):
                merged_text += f"({i}) {summary}\n"

        enriched_doc = Document(
            text=merged_text.strip(),

            metadata = {
                "data_source": "jasp_manual",                  # which collection/domain this PDF belongs to
                "document_name": Path(pdf_path).stem,          # clean PDF name without extension
                "source": Path(pdf_path).name,                 # PDF filename with .pdf
                "page": page_num,
                "images": image_paths,
                "image_summaries": image_summary_texts,
                "uuid": str(uuid.uuid4()),                     # unique identifier
                "char_length": len(merged_text),               # useful for filtering later
            }

        )
        enriched_docs.append(enriched_doc)

    logger.info(f"✅ Enriched {len(enriched_docs)} pages with image context")

    # 4️⃣ Save as JSON (optional)
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        out_path = Path(output_dir) / f"{pdf_name}_enriched.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [d.to_dict() for d in enriched_docs],
                f, indent=2, ensure_ascii=False
            )
        logger.info(f"💾 Enriched output saved to: {out_path}")

    return enriched_docs



# -----------------------------
# TEST RUN (if executed directly)
# -----------------------------
if __name__ == "__main__":
    sample_pdf = "./data/raw/test_pages25-28.pdf"
    enriched_docs = enrich_llamaparse_with_images(sample_pdf)
    print(f"\nProcessed {len(enriched_docs)} enriched documents ✅")

    # ✅ Show the first 2 enriched texts (text + image summaries)
    for i, doc in enumerate(enriched_docs[:2], start=1):
        print(f"\n{'='*80}")
        print(f"🧾 ENRICHED PAGE {i} (Page {doc.metadata.get('page')})")
        print(f"{'='*80}\n")
        print(doc.text[:5000])  # Limit to 5000 chars for readability
        print("\n" + "-"*80)


