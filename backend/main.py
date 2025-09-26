"""
FastAPI backend for the JASP multimodal RAG chatbot.

- Provides an `/ask` endpoint for querying the JASP manual.
- Wraps the retrieval + generation pipeline into a simple API.

Usage:
    poetry run uvicorn backend.main:app --reload

Swagger docs:
    http://127.0.0.1:8000/docs

Example query:
    curl -X POST "http://127.0.0.1:8000/ask" \
         -H "Content-Type: application/json" \
         -d '{"query":"How do I run ANOVA in JASP?","model":"mistral:7b-instruct","topk":2}'


Stop the backend:
       Press CTRL+C in the terminal
"""





from fastapi import FastAPI
from pydantic import BaseModel
import logging

from src.retrieval.query import build_rag_chain

app = FastAPI(title="JASP Multimodal RAG API", version="0.1.0")
logging.basicConfig(level=logging.INFO)


# ✅ Request schema
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral:7b-instruct"
    topk: int = 1


# ✅ Response schema
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.post("/ask", response_model=QueryResponse)
async def ask_rag(query: QueryRequest):
    """
    API endpoint to query the JASP manual via RAG.
    Defaults: mistral:7b-instruct and topk=1.
    """
    try:
        chain = build_rag_chain(query.model, query.topk)
        result = chain.invoke({"query": query.query})

        answer = result.get("result", "No answer produced.")
        sources = []

        if result.get("source_documents"):
            seen = set()
            for doc in result["source_documents"]:
                page = doc.metadata.get("page")
                if doc.metadata.get("type") == "image_caption":
                    src = f"Image caption (Page {page})"
                else:
                    src = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")

                if src not in seen:
                    sources.append(src)
                    seen.add(src)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logging.exception("Error during RAG query")
        return QueryResponse(answer="Error occurred", sources=[str(e)])
"""
FastAPI backend for the JASP multimodal RAG chatbot.

Features:
- Provides `/ask` endpoint to query the ingested JASP manual.
- Wraps the retrieval + generation pipeline into a REST API.
- Returns both the answer and the source documents (manual pages or image captions).

Usage:
    poetry run uvicorn backend.main:app --reload

Docs:
    Swagger UI: http://127.0.0.1:8000/docs
    ReDoc:      http://127.0.0.1:8000/redoc

Example:
    curl -X POST "http://127.0.0.1:8000/ask" \
         -H "Content-Type: application/json" \
         -d '{"query":"How do I run ANOVA in JASP?","model":"mistral:7b-instruct","topk":2}'
"""

from fastapi import FastAPI
from pydantic import BaseModel
import logging

from src.retrieval.query import build_rag_chain

# Create app
app = FastAPI(title="JASP Multimodal RAG API", version="0.1.0")
logging.basicConfig(level=logging.INFO)


# ✅ Request schema
class QueryRequest(BaseModel):
    query: str
    model: str = "mistral:7b-instruct"
    topk: int = 1


# ✅ Response schema
class QueryResponse(BaseModel):
    answer: str
    sources: list[str]


@app.get("/")
async def root():
    """Root endpoint — shows basic info."""
    return {"message": "JASP RAG API is running. See /docs for interactive Swagger UI."}


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
async def ask_rag(query: QueryRequest):
    """
    Query the JASP manual via RAG.
    - Defaults: mistral:7b-instruct and topk=1.
    - Returns both answer and sources.
    """
    try:
        chain = build_rag_chain(query.model, query.topk)
        result = chain.invoke({"query": query.query})

        answer = result.get("result", "No answer produced.")
        sources = []

        if result.get("source_documents"):
            seen = set()
            for doc in result["source_documents"]:
                page = doc.metadata.get("page")
                if doc.metadata.get("type") == "image_caption":
                    src = f"Image caption (Page {page})"
                else:
                    src = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")

                if src not in seen:
                    sources.append(src)
                    seen.add(src)

        return QueryResponse(answer=answer, sources=sources)

    except Exception as e:
        logging.exception("Error during RAG query")
        return QueryResponse(answer="Error occurred", sources=[str(e)])
