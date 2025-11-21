"""
---------------------------------------------------
🌐 FASTAPI BACKEND for Local RAG + Ollama Generation
---------------------------------------------------

Exposes REST endpoints for:
    - Hybrid retrieval → RRF → BGE reranking
    - Page-level PDF preview rendering
    - LLM answer generation with retrieved docs (optional)

Test interactively:
    👉 http://127.0.0.1:8000/redoc
Run:
    poetry run uvicorn backend_api.main:app --reload --port 8000
---------------------------------------------------
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

import fitz  # PyMuPDF

from src.retrieval.retrieval import retrieve_clean
from src.generation.generation import run_generation_pipeline


# =============================================================
# 🌐 FASTAPI APP SETUP
# =============================================================

app = FastAPI(title="JASP–RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# 📁 PATH CONFIG
# =============================================================

ROOT = Path(__file__).resolve().parents[1]       # backend_api/
PROJECT_ROOT = ROOT.parents[0]                   # project root

PDF_ROOT = PROJECT_ROOT / "data/raw_pdf"         # raw PDF files (unchunked)
PREVIEW_ROOT = PROJECT_ROOT / "static/previews"  # cached high-res page renders

PDF_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

# Serve original PDFs directly:
app.mount("/docs", StaticFiles(directory=str(PDF_ROOT)), name="docs")


# =============================================================
# 🔎 /retrieve → returns RAG-metadata for Streamlit UI
# =============================================================
@app.post("/retrieve")
def retrieve_docs(payload: dict):
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field.")

    try:
        results = retrieve_clean(query)  # your RAG pipeline
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return {"results": results}


# =============================================================
# 🤖 /generate → optional LLM answer generation
# =============================================================
@app.post("/generate")
def generate_answer(payload: dict):
    query = payload.get("query")
    model = payload.get("model", "mistral:7b")

    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field.")

    try:
        output = run_generation_pipeline(query, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    return output


# =============================================================
# 🖼  PDF PAGE PREVIEW HELPERS
# =============================================================

def render_pdf_page(pdf_path: Path, out_path: Path, page: int):
    """
    Render a single PDF page into a PNG file.
    Uses 2× zoom matrix for clarity.
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    if page < 1 or page > total_pages:
        raise HTTPException(
            status_code=400,
            detail=f"Page {page} out of range (1–{total_pages})."
        )

    page_obj = doc.load_page(page - 1)
    pix = page_obj.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2× zoom for quality

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pix.save(out_path)
    return out_path


# =============================================================
# 🖼 /preview/pdf/{pdf_id}/{page}.png → high-res preview
# =============================================================
@app.get("/preview/pdf/{pdf_id}/{page}")
def pdf_page_preview(pdf_id: str, page: int):

    pdf_path = PDF_ROOT / f"{pdf_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_id}")

    out_path = PREVIEW_ROOT / pdf_id / f"{page}.png"

    # Generate on-demand & cache
    if not out_path.exists():
        try:
            render_pdf_page(pdf_path, out_path, page)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF render error: {e}")

    return FileResponse(out_path, media_type="image/png")


# ============================
# Root endpoint
# ============================
@app.get("/")
def hello():
    return {"status": "ok", "message": "JASP RAG Backend Running"}
