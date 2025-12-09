

"""
---------------------------------------------------
6:
 FastAPI backend for local JASP RAG + Ollama
---------------------------------------------------

This service exposes a simple HTTP API for the JASP RAG system. It connects:

  • retrieval  →  src.retrieval.retrieval.retrieve_clean
  • generation →  src.generation.generation.run_generation_pipeline
  • PDF preview rendering → PyMuPDF (fitz)

Exposed endpoints
-----------------
1. POST /retrieve
   - Runs the retrieval pipeline (BM25 / vector / hybrid).
   - Returns cleaned, normalized metadata for:
       • PDFs (manual pages)
       • GitHub markdown help files
       • YouTube videos (with timestamps)
   - Used by the frontend to show “docs that may help”.

2. POST /generate
   - Runs the full RAG pipeline: retrieval + LLM generation (via Ollama).
   - Uses the same retrieval modes as `/retrieve`.
   - Returns:
       • answer (markdown)
       • sources actually used in the prompt.

3. GET /preview/pdf/{pdf_id}/{page}
   - On-demand, cached rendering of a single PDF page as PNG.
   - Allows the frontend to show high-res page previews.

4. GET /config/retrieval-modes
   - Lists all supported retrieval modes and the default.
   - Useful for building dropdowns / toggles in the UI.

5. GET /
   - Simple health-check endpoint (`{"status": "ok"}`).

Interactive docs
----------------
Swagger UI (interactive) 👉  http://127.0.0.1:8000/docs  
ReDoc (read-only)       👉  http://127.0.0.1:8000/redoc

Run locally:

    poetry run uvicorn backend_api.main:app --reload --port 8000

---------------------------------------------------
"""


import os
from pathlib import Path
from typing import List, Literal

import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.retrieval.retrieval import retrieve_clean
from src.generation.generation import run_generation_pipeline


# =============================================================
# 🔧 RETRIEVAL MODES (shared contract with frontend)
# =============================================================

RetrievalMode = Literal[
    "bm25",
    "vector",
    #"bm25_vector",
    "bm25_vector_fusion",
    "bm25_vector_fusion_rerank",
]

RETRIEVAL_MODES: List[RetrievalMode] = [
    "bm25",
    "vector",
   # "bm25_vector",
    "bm25_vector_fusion",
    "bm25_vector_fusion_rerank",
]

DEFAULT_RETRIEVAL_MODE: RetrievalMode = "bm25_vector_fusion_rerank"


# =============================================================
# 🌐 FASTAPI APP SETUP
# =============================================================

app = FastAPI(
    title="JASP–RAG Backend",
    version="1.1.0",
    description=(
        "Backend service for the JASP RAG chatbot.\n\n"
        "Provides retrieval and generation endpoints with multiple retrieval modes "
        "plus PDF page previews for the UI."
    ),
)

# Allow local dev from any origin (Streamlit, JASP UI, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================
# 📁 PATH CONFIG
# =============================================================

# Note: keep the same path logic you already use in the project.
ROOT = Path(__file__).resolve().parents[1]       # (kept as-is for compatibility)
PROJECT_ROOT = ROOT.parents[0]

PDF_ROOT = PROJECT_ROOT / "data/raw_pdf"         # raw PDF files (unchunked)
PREVIEW_ROOT = PROJECT_ROOT / "static/previews"  # cached high-res page renders

PDF_ROOT.mkdir(parents=True, exist_ok=True)
PREVIEW_ROOT.mkdir(parents=True, exist_ok=True)

# Serve original PDFs directly:
app.mount("/docs", StaticFiles(directory=str(PDF_ROOT)), name="docs")


# =============================================================
# 📦 REQUEST MODELS
# =============================================================

class RetrieveRequest(BaseModel):
    query: str = Field(..., description="User's natural language question.")
    mode: RetrievalMode = Field(
        DEFAULT_RETRIEVAL_MODE,
        description=(
            "Retrieval mode to use.\n"
            "Options: bm25 | vector | bm25_vector | bm25_vector_fusion | bm25_vector_fusion_rerank"
        ),
    )


class GenerateRequest(BaseModel):
    query: str = Field(..., description="User's natural language question.")
    model: str = Field(
        "mistral:7b",
        description="Ollama model name to use for generation (e.g. mistral:7b, llama3:8b, phi3:mini).",
    )
    mode: RetrievalMode = Field(
        DEFAULT_RETRIEVAL_MODE,
        description=(
            "Retrieval mode used before generation.\n"
            "Options: bm25 | vector | bm25_vector | bm25_vector_fusion | bm25_vector_fusion_rerank"
        ),
    )


# =============================================================
# 🔎 /retrieve → returns RAG-metadata for Streamlit/UI
# =============================================================

@app.post(
    "/retrieve",
    summary="Retrieve relevant documents",
    tags=["retrieval"],
)
def retrieve_docs(payload: RetrieveRequest):
    """
    Run the retrieval pipeline (BM25, vector, or hybrid) for a user query.

    - Uses `retrieve_clean(query, mode=...)` from `src.retrieval.retrieval`.
    - Returns a list of clean metadata dicts:
        • PDFs: pdf_id, page_number, total_pages, title, source_url, score, content, ...
        • Markdown: markdown_file, section_title, source_url, score, content, ...
        • Videos: video_title, chapter_title/section, timestamp, video_link, score, ...

    The frontend can:
        - Display these as “Docs that may help”
        - Use `pdf_id` + `page_number` to call `/preview/pdf/{pdf_id}/{page}`
        - Directly link to `source_url` / `video_link`.
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        results = retrieve_clean(query, mode=payload.mode)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return {
        "query": query,
        "mode": payload.mode,
        "results": results,
    }


# =============================================================
# 🤖 /generate → optional LLM answer generation
# =============================================================

@app.post(
    "/generate",
    summary="Generate answer using RAG (retrieval + LLM)",
    tags=["generation"],
)
def generate_answer(payload: GenerateRequest):
    """
    Run the full RAG pipeline for a user query:

        1. Retrieval with the selected mode (same as /retrieve).
        2. Answer generation via Ollama using the retrieved context.

    Returns:
        - query: original question
        - model: Ollama model used
        - retrieval_mode: chosen retrieval mode
        - answer: markdown answer string
        - sources: list of source dicts (same shape as in /retrieve, plus rank/text)
    """
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    try:
        output = run_generation_pipeline(
            query=query,
            model=payload.model,
            mode=payload.mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    return output


# =============================================================
# ⚙️ /config/retrieval-modes → list modes for the UI
# =============================================================

@app.get(
    "/config/retrieval-modes",
    summary="List available retrieval modes",
    tags=["config"],
)
def get_retrieval_modes():
    """
    Small helper endpoint for the frontend.

    Allows the UI to:
        - Discover which retrieval modes are supported by the backend.
        - Know which mode is the default.

    Useful for building dropdowns or toggles in Streamlit / JASP.
    """
    return {
        "modes": RETRIEVAL_MODES,
        "default": DEFAULT_RETRIEVAL_MODE,
    }


# =============================================================
# 🖼  PDF PAGE PREVIEW HELPERS
# =============================================================

def render_pdf_page(pdf_path: Path, out_path: Path, page: int) -> Path:
    """
    Render a single PDF page into a PNG file using PyMuPDF.

    - `page` is 1-based (page=1 is the first page).
    - Uses a 2× zoom matrix for better readability in the UI.
    - Writes the PNG to `out_path` and returns that path.
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

@app.get(
    "/preview/pdf/{pdf_id}/{page}",
    summary="Render a single PDF page as PNG",
    tags=["previews"],
)
def pdf_page_preview(pdf_id: str, page: int):
    """
    Return a high-resolution PNG preview of a specific PDF page.

    - `pdf_id` should match the `pdf_id` used in retrieval metadata
      (usually the filename without extension).
    - `page` is 1-based (page=1 is the first page).

    The image is rendered on-demand once and then cached under:
        static/previews/{pdf_id}/{page}.png
    """
    pdf_path = PDF_ROOT / f"{pdf_id}.pdf"
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_id}")

    out_path = PREVIEW_ROOT / pdf_id / f"{page}.png"

    # Generate on-demand & cache
    if not out_path.exists():
        try:
            render_pdf_page(pdf_path, out_path, page)
        except HTTPException:
            # Re-raise HTTP errors (e.g. page out of range)
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"PDF render error: {e}")

    return FileResponse(out_path, media_type="image/png")


# ============================
# Root endpoint
# ============================

@app.get(
    "/",
    summary="Health check",
    tags=["meta"],
)
def hello():
    """
    Simple health-check endpoint to verify that the backend is running.
    """
    return {"status": "ok", "message": "JASP RAG Backend Running"}
