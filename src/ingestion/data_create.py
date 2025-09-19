import logging
import os
import time
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import fitz  # PyMuPDF for image extraction
from PIL import Image
import torch
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from transformers import BlipProcessor, BlipForConditionalGeneration
import random
import re






# =========================
# 1. LOAD PDF (Texts+images)
# =========================
PDF_PATH = "data/Statistical-Analysis-in-JASP-A-guide-for-students-2025.pdf"
OUTPUT_IMG_DIR = "data/images"
CHROMA_DIR = "data/chroma"  # persistent dir which can inspect in VSCode


# Image extraction rules
SKIP_IMAGES = 2 # skip the page heanders
MIN_WIDTH = 100 # skip small logos
MIN_HEIGHT = 100


# Preview range (manual pages)
# Reproducibility (where possible)
random.seed(42)
torch.manual_seed(42)
PAGE_OFFSET = 4  # manual_page = pdf_page - PAGE_OFFSET
DEFAULT_PREVIEW_RANGE = (1, 3)  # user's friendly page numbers
PREVIEW_SAMPLES = 2 # peek at chunk examples

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def manual_page_from_pdf_page(pdf_page: int) -> int:
    """Convert 1-based pdf_page to 1-based manual_page (pages <1 are 'front matter' and excluded)."""
    return pdf_page - PAGE_OFFSET

def valid_manual_page(manual_page: int) -> bool:
    return manual_page >= 1


#image captioning helper. 
# It takes an image file and tries to turn it into a short descriptive sentence 
# using BLIP (Bootstrapping Language-Image Pretraining), a pretrained vision–language model.
def describe_image(path: str) -> str:
    """Generate caption via BLIP. If BLIP unavailable, return a placeholder."""
    if blip_processor is None or blip_model is None:
        return "Uncaptioned image"
    try:
        image = Image.open(path).convert("RGB")
        inputs = blip_processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = blip_model.generate(**inputs, max_new_tokens=50)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        logging.error(f"Failed to caption image {path}: {e}")
        return "Uncaptioned image"




def load_pdf_with_images(
    path: str,
    output_img_dir: str = OUTPUT_IMG_DIR,
    preview_range: Tuple[int, int] = None,
    skip_first: int = SKIP_IMAGES,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT
) -> Tuple[List[Document], Dict[int, List[str]]]:
    """
    Load a PDF into LangChain Document objects with text (per page) and extract images.
    Pages with manual_page < 1 are dropped entirely (they’re front matter).
    Returns:
        docs: list[Document] with metadata {pdf_page, page, source, type="page"}
        images_by_page: dict[manual_page] -> [image paths]
    """
    t0 = time.time()
    filename = os.path.basename(path)

    # ---- Load text pages ----
    raw_docs = PyPDFLoader(path).load()  # one Document per PDF page
    docs: List[Document] = []

    for i, doc in enumerate(raw_docs):
        pdf_page = i + 1
        manual_page = manual_page_from_pdf_page(pdf_page)
        if not valid_manual_page(manual_page):
            continue  # drop front matter

        # normalize metadata
        doc.metadata = {
            "source": filename,
            "pdf_page": pdf_page,
            "page": manual_page,
            "type": "page",
        }
        docs.append(doc)

    # ---- Extract images ----
    os.makedirs(output_img_dir, exist_ok=True)
    pdf_doc = fitz.open(path)
    images_by_page: Dict[int, List[str]] = {}

    for page_index in range(len(pdf_doc)):
        pdf_page = page_index + 1
        manual_page = manual_page_from_pdf_page(pdf_page)
        if not valid_manual_page(manual_page):
            continue  # ignore images on front matter

        page = pdf_doc[page_index]
        image_infos = page.get_images(full=True)
        if not image_infos:
            continue

        saved_paths = []
        for img_idx, img_tuple in enumerate(image_infos[skip_first:], start=skip_first):
            xref = img_tuple[0]
            base_image = pdf_doc.extract_image(xref)
            width = base_image.get("width", 0)
            height = base_image.get("height", 0)
            if width < min_width or height < min_height:
                continue

            image_bytes = base_image["image"]
            image_ext = base_image.get("ext", "png")
            # deterministic, readable filename
            short_prefix = Path(filename).stem[:10]  # first 10 chars
            fname = f"{short_prefix}_p{manual_page:03d}_i{img_idx:02d}.{image_ext}"
            fpath = os.path.join(output_img_dir, fname)

            with open(fpath, "wb") as f:
                f.write(image_bytes)

            saved_paths.append(fpath)

        if saved_paths:
            images_by_page[manual_page] = saved_paths

    elapsed = time.time() - t0
    logging.info(
        f"Loaded {len(docs)} manual pages and extracted "
        f"{sum(len(v) for v in images_by_page.values())} images in {elapsed:.2f}s."
    )

    # ---- Log short page previews ----
    logging.info("Listing pages (first 80 chars):")
    for doc in docs:
        page_num = doc.metadata["page"]
        snippet = doc.page_content[:80].replace("\n", " ")
        logging.info(f"Page {page_num}: {snippet}...")

    # ---- Console preview of selected manual pages ----
    if preview_range is None:
        preview_range = DEFAULT_PREVIEW_RANGE
    start, end = preview_range
    logging.info(f"Previewing pages {start}..{end}")
    for doc in docs:
        page_num = doc.metadata["page"]
        if start <= page_num <= end:
            logging.info("=" * 48)
            logging.info(f"📄 Source: {doc.metadata['source']} | Page {page_num}")
            logging.info("=" * 48)
            content = doc.page_content[:1000]
            logging.info(content)
            if page_num in images_by_page:
                logging.info(f"🖼️ {len(images_by_page[page_num])} image(s): {images_by_page[page_num]}")

    return docs, images_by_page











# =========================
# 2. SPLITTING
# =========================

# Splitter defaults
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

TEXT_MODEL_NAME = "BAAI/bge-small-en"
text_model = SentenceTransformer(TEXT_MODEL_NAME)

# CLIP (open_clip): for image pixel based emdedding
import open_clip
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = clip_model.to(device)

# BLIP (captioning)—load once, degrade gracefully if it fails: for image's textual description
def _load_blip():
    try:
        proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        mdl = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        logging.info("BLIP captioning model loaded.")
        return proc, mdl
    except Exception as e:
        logging.warning(f"BLIP failed to load; continuing without captions. Error: {e}")
        return None, None

blip_processor, blip_model = _load_blip()








def build_page_docs(docs, images_by_page):
    page_docs = []
    for doc in docs:
        page_num = doc.metadata["page"]
        text = doc.page_content
        captions = []

        if page_num in images_by_page:
            for path in images_by_page[page_num]:
                caption = describe_image(path)
                captions.append(f"[Figure: {caption}]")

        full_text = text + "\n\n" + "\n".join(captions)
        page_docs.append(Document(page_content=full_text, metadata=doc.metadata))
    return page_docs



def hybrid_split(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
    preview_samples: int = PREVIEW_SAMPLES
) -> List[Document]:
    """
    Hybrid splitting:
      - Keep obvious headers/tables/figures as standalone.
      - Otherwise recurse with a character splitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    header_pattern = re.compile(
        r"^(CHAPTER\b|\d+\.\s+|##|###|Table\s+\d+|Figure\s+\d+)",
        re.IGNORECASE
    )

    chunks: List[Document] = []
    for doc in docs:
        text = doc.page_content
        meta = {**doc.metadata}  # copy

        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for para in paragraphs:
            if header_pattern.match(para):
                chunks.append(
                    Document(page_content=para, metadata={**meta, "split_type": "header_or_table"})
                )
                continue

            for sub in splitter.split_text(para):
                chunks.append(
                    Document(page_content=sub, metadata={**meta, "split_type": "body"})
                )

    total = len(chunks)
    header_count = sum(1 for c in chunks if c.metadata["split_type"] == "header_or_table")
    body_count = total - header_count
    logging.info(f"Hybrid split → {total} chunks "
                 f"({header_count} headers/tables, {body_count} body).")

    # Preview N random chunks
    if total:
        sample = random.sample(chunks, min(preview_samples, total))
        for i, c in enumerate(sample, 1):
            snippet = c.page_content[:300].replace("\n", " ")
            logging.info(f"--- Chunk sample {i} ---")
            logging.info(f"Type: {c.metadata['split_type']} | Page: {c.metadata.get('page')}")
            logging.info(f"{snippet}...")

    return chunks






# =========================
# 3. EMBEDDING → CHROMA
# =========================

# =========================
# CHROMA (Persistent): for embeddings storage and quering
# =========================
os.makedirs(CHROMA_DIR, exist_ok=True)
chroma_client: PersistentClient = chromadb.PersistentClient(path=CHROMA_DIR)
text_collection = chroma_client.get_or_create_collection("jasp_text")
image_collection = chroma_client.get_or_create_collection("jasp_images")

# drop the collection entirely before re-populating
chroma_client.delete_collection("jasp_text")
chroma_client.delete_collection("jasp_images")

text_collection = chroma_client.get_or_create_collection("jasp_text")
image_collection = chroma_client.get_or_create_collection("jasp_images")


def _batch(iterable, n):
    """Yield lists of length n (last may be shorter)."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def embed_texts(chunks: List[Document], source: str):
    """
    Embed text chunks and upsert into Chroma in batches.
    """
    if not chunks:
        logging.info("No text chunks to embed.")
        return

    BATCH = 128
    total = 0
    for group in _batch(list(enumerate(chunks)), BATCH):
        docs_batch, embeds_batch, metas_batch, ids_batch = [], [], [], []
        for idx, chunk in group:
            text = chunk.page_content
            meta = {**chunk.metadata, "type": "text_chunk"}
            emb = text_model.encode(text, normalize_embeddings=True).tolist()

            page = meta.get("page", -1)
            ids_batch.append(f"{source}:p{page}:t{idx}")
            docs_batch.append(text)
            embeds_batch.append(emb)
            metas_batch.append(meta)

        text_collection.add(
            documents=docs_batch,
            embeddings=embeds_batch,
            metadatas=metas_batch,
            ids=ids_batch
        )
        total += len(group)

    logging.info(f"Inserted {total} text chunks into Chroma (jasp_text).")

def embed_images(images_by_page: Dict[int, List[str]], source: str):
    """
    For each image:
      - CLIP embedding → image_collection
      - BLIP caption (if available) → stored as the document text in image_collection
      - Also insert the caption as a separate text record in text_collection for text-only recall
    """
    if not images_by_page:
        logging.info("No images to embed.")
        return

    total = 0
    BATCH = 64

    items = []
    for page, paths in images_by_page.items():
        for i, path in enumerate(paths):
            items.append((page, i, path))

    for group in _batch(items, BATCH):
        img_docs, img_embeds, img_metas, img_ids = [], [], [], []
        cap_docs, cap_embeds, cap_metas, cap_ids = [], [], [], []

        for page, i, path in group:
            # CLIP
            image = Image.open(path).convert("RGB")
            tensor = clip_preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = clip_model.encode_image(tensor)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb.cpu().numpy().flatten().tolist()

            # BLIP caption
            caption = describe_image(path)

            meta_img = {
                "type": "image",
                "source": source,
                "page": page,
                "path": path,
                "caption": caption,
            }

            img_docs.append(caption if caption else f"Image from page {page}")
            img_embeds.append(emb)
            img_metas.append(meta_img)
            img_ids.append(f"{source}:p{page}:i{i}")

            # ALSO store caption as text for text-side recall
            text_emb = text_model.encode(caption, normalize_embeddings=True).tolist()
            cap_docs.append(caption)
            cap_embeds.append(text_emb)
            cap_metas.append({**meta_img, "type": "image_caption"})
            cap_ids.append(f"{source}:p{page}:ic{i}")

        image_collection.add(
            documents=img_docs, embeddings=img_embeds, metadatas=img_metas, ids=img_ids
        )
        text_collection.add(
            documents=cap_docs, embeddings=cap_embeds, metadatas=cap_metas, ids=cap_ids
        )
        total += len(group)

    logging.info(f"Inserted {total} images (+captions) into Chroma (images + text).")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    logging.info("🚀 Starting JASP multimodal RAG pipeline")

    # Fresh image dir for this run (optional—keep if you need a clean slate)
    shutil.rmtree(OUTPUT_IMG_DIR, ignore_errors=True)

    source_name = Path(PDF_PATH).name

    # 1) Load PDF text + images (front matter dropped)
    docs, images_by_page = load_pdf_with_images(PDF_PATH)

    # 2) Split text
    chunks = hybrid_split(docs)

    # 3) Embed
    embed_texts(chunks, source=source_name)
    embed_images(images_by_page, source=source_name)

    logging.info("✅ Multimodal RAG pipeline finished successfully.")
