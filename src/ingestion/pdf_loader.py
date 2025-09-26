import os, time, re
from pathlib import Path
from typing import Dict, List, Tuple
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

PAGE_OFFSET_DEFAULT = 4  # pdf_page - PAGE_OFFSET = manual_page
SKIP_IMAGES_DEFAULT = 2
MIN_W_DEFAULT, MIN_H_DEFAULT = 100, 100

def manual_page_from_pdf_page(pdf_page: int, page_offset: int) -> int:
    """1-based pdf_page -> 1-based manual_page (front matter => <1)."""
    return pdf_page - page_offset

def valid_manual_page(manual_page: int) -> bool:
    return manual_page >= 1

def load_pdf_pages(path: str, page_offset: int = PAGE_OFFSET_DEFAULT) -> List[Document]:
    """One Document per manual page; drops front matter."""
    filename = os.path.basename(path)
    raw_docs = PyPDFLoader(path).load()
    docs: List[Document] = []
    for i, doc in enumerate(raw_docs):
        pdf_page = i + 1
        manual_page = manual_page_from_pdf_page(pdf_page, page_offset)
        if not valid_manual_page(manual_page):
            continue
        doc.metadata = {
            "source": filename,
            "pdf_page": pdf_page,
            "page": manual_page,
            "type": "page",
        }
        docs.append(doc)
    return docs

def extract_images(
    path: str,
    output_img_dir: str,
    page_offset: int = PAGE_OFFSET_DEFAULT,
    skip_first: int = SKIP_IMAGES_DEFAULT,
    min_width: int = MIN_W_DEFAULT,
    min_height: int = MIN_H_DEFAULT,
) -> Dict[int, List[str]]:
    """
    Extract images per manual page; skip front matter & tiny assets.
    Returns: {manual_page: [image_paths]}
    """
    os.makedirs(output_img_dir, exist_ok=True)
    pdf_doc = fitz.open(path)
    filename = Path(path).name
    short_prefix = Path(filename).stem[:10]

    images_by_page: Dict[int, List[str]] = {}
    for page_index in range(len(pdf_doc)):
        pdf_page = page_index + 1
        manual_page = manual_page_from_pdf_page(pdf_page, page_offset)
        if not valid_manual_page(manual_page):
            continue

        page = pdf_doc[page_index]
        infos = page.get_images(full=True)
        if not infos:
            continue

        saved = []
        for img_idx, info in enumerate(infos[skip_first:], start=skip_first):
            xref = info[0]
            base = pdf_doc.extract_image(xref)
            w, h = base.get("width", 0), base.get("height", 0)
            if w < min_width or h < min_height:
                continue
            ext = base.get("ext", "png")
            fpath = os.path.join(
                output_img_dir, f"{short_prefix}_p{manual_page:03d}_i{img_idx:02d}.{ext}"
            )
            with open(fpath, "wb") as f:
                f.write(base["image"])
            saved.append(fpath)

        if saved:
            images_by_page[manual_page] = saved

    return images_by_page

def load_pdf_with_images(
    path: str,
    output_img_dir: str,
    page_offset: int = PAGE_OFFSET_DEFAULT,
    skip_first: int = SKIP_IMAGES_DEFAULT,
    min_width: int = MIN_W_DEFAULT,
    min_height: int = MIN_H_DEFAULT,
) -> Tuple[List[Document], Dict[int, List[str]]]:
    """Convenience wrapper: returns (docs, images_by_page)."""
    docs = load_pdf_pages(path, page_offset=page_offset)
    images = extract_images(
        path,
        output_img_dir=output_img_dir,
        page_offset=page_offset,
        skip_first=skip_first,
        min_width=min_width,
        min_height=min_height,
    )
    return docs, images
