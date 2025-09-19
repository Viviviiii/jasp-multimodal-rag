# main.py
from fastapi import FastAPI
from pydantic import BaseModel
import logging
from manual_query import build_rag_chain
from src.pipelines.query import build_rag_chain




app = FastAPI()
logging.basicConfig(level=logging.INFO)

# ✅ Fixed defaults: mistral:7b-instruct and topk=1
class Query(BaseModel):
    query: str
    model: str = "mistral:7b-instruct"
    topk: int = 1

@app.post("/ask")
async def ask_rag(query: Query):
    """
    API endpoint to query the JASP manual via RAG.
    Always defaults to mistral:7b-instruct with topk=1 unless explicitly overridden.
    """
    try:
        chain = build_rag_chain(query.model, query.topk)
        result = chain.invoke({"query": query.query})

        answer = result.get("result", "No answer produced.")
        sources = []

        if result.get("source_documents"):
            for doc in result["source_documents"]:
                page = doc.metadata.get("page")
                src = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")
                sources.append(src)

        return {"answer": answer, "sources": list(set(sources))}

    except Exception as e:
        logging.exception("Error during RAG query")
        return {"error": str(e)}
