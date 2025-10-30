"""
---------------------------------------------------
🌐 FASTAPI BACKEND for Local RAG + Ollama Generation
open → http://127.0.0.1:8000/docs
 to test interactively.
---------------------------------------------------
Run:
    poetry run uvicorn backend_api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from loguru import logger
from typing import List, Dict, Any

from src.generation.generation import answer_query

app = FastAPI(title="JASP RAG Backend", version="1.0")

# ---------- Request/Response Schemas ----------
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral:7b"

class GenerationResponse(BaseModel):
    answer: str
    documents: List[Dict[str, Any]]

# ---------- Routes ----------
@app.get("/")
async def root():
    return {"message": "JASP RAG API is running 🚀"}

@app.post("/generate", response_model=GenerationResponse)
async def generate_answer(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query} | model={request.model}")
        answer, nodes = answer_query(request.query)

        docs = []
        for n in nodes:
            md = n.node.metadata or {}
            docs.append({
                "source": md.get("source", "N/A"),
                "page": md.get("page", "N/A"),
                "chapter": md.get("chapter", "N/A"),
                "chunk_id": md.get("chunk_id", "N/A"),
                "text": n.node.get_content()
            })

        return GenerationResponse(answer=answer, documents=docs)

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
