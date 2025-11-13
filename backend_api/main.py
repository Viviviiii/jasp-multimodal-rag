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
from src.generation.generation import run_generation_pipeline

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
    section: str | None = None   # ✅ NEW
    text: str | None = None      # ✅ NEW


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

        # ✅ Include section + text content in API response

        clean_sources = []
        for i, src in enumerate(result["sources"], 1):
            meta = src.get("metadata", {})
            source_type = meta.get("source_type", "document")

            if source_type == "video_transcript":
                video_title = meta.get("video_title", "Untitled Video")
                chapter_title = meta.get("chapter_title", None)
                video_author = meta.get("author", "Unknown")
                video_url = meta.get("url", "")
                start_time = meta.get("start_time", "")
                end_time = meta.get("end_time", "")
                time_range = meta.get("time_range") or f"{start_time}–{end_time}" if start_time else "N/A"

                # Construct deep link to specific timestamp
                if video_url and start_time:
                    try:
                        mins, secs = start_time.split(":")
                        timestamp = int(mins) * 60 + int(secs)
                        video_link = f"{video_url}&t={timestamp}s"
                    except Exception:
                        video_link = video_url
                else:
                    video_link = video_url

                # Display readable section title
                section_label = f"🎥 {chapter_title}" if chapter_title else f"🎥 Video Segment {time_range}"

                source_display = f"{video_title} ({video_author})"

                clean_sources.append({
                    "rank": i,
                    "source": source_display,
                    "page": time_range,
                    "chunk_id": meta.get("section_id") or meta.get("doc_id"),
                    "score": src.get("score"),
                    "section": section_label,
                    "text": src.get("text", "")[:1200],
                    "source_type": source_type,
                    "video_link": video_link,
                    "video_url": video_url,  # ✅ Base YouTube URL
                })

            else:
                # Text-based source (PDF/Markdown)
                clean_sources.append({
                    "rank": i,
                    "source": meta.get("pdf_name") or meta.get("markdown_file") or meta.get("source") or "Unknown",
                    "page": str(meta.get("page_start") or meta.get("page") or "?"),
                    "chunk_id": meta.get("section_id") or meta.get("doc_id"),
                    "score": src.get("score"),
                    "section": meta.get("section_title") or "N/A",
                    "text": src.get("text", "")[:1200],
                    "source_type": source_type,
                    "video_link": None,
                    "video_url": None,
                })







        return GenerationResponse(
            query=result["query"],
            model=result["model"],
            answer=result["answer"],
            sources=clean_sources,
        )

    except Exception as e:
        logger.exception(f"❌ Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
