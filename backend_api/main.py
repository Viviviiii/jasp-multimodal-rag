"""
---------------------------------------------------
🌐 FASTAPI BACKEND for Local RAG + Ollama Generation
---------------------------------------------------

Exposes REST endpoints for:
    - Hybrid retrieval + RRF + BGE reranking
    - Contextual answer generation using local Ollama models

Test interactively:
    👉 http://127.0.0.1:8000/docs

Run:
    poetry run uvicorn backend_api.main:app --reload --port 8000
---------------------------------------------------
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import List, Dict, Any

# ✅ Import from your pipeline (new unified entrypoint)
from src.pipelines.generate_answer import run_generation_pipeline

# ---------- FastAPI App ----------
app = FastAPI(title="JASP RAG Backend", version="1.0")

# ---------- Request / Response Schemas ----------
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral:7b"

class DocumentItem(BaseModel):
    rank: int
    source: str
    page: str | int | None
    chunk_id: str | None
    score: float | None

class GenerationResponse(BaseModel):
    query: str
    model: str
    answer: str
    sources: List[DocumentItem]

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"message": "✅ JASP RAG API is running", "docs": "/docs"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_endpoint(request: QueryRequest):
    """
    Generate an AI-assisted answer from the JASP RAG pipeline.
    """
    try:
        logger.info(f"📩 Query received: {request.query} | Model: {request.model}")

        # ✅ Run your unified pipeline
        result = run_generation_pipeline(request.query, model=request.model)

        return GenerationResponse(
            query=result["query"],
            model=result["model"],
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        logger.exception(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
