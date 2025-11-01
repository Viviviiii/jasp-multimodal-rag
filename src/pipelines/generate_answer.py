"""
---------------------------------------------------
🤖 GENERATION PIPELINE MODULE (RAG END-TO-END)
---------------------------------------------------

Orchestrates:
    1. Hybrid retrieval + RRF fusion + cross-encoder reranking
    2. LLM-based answer generation via Ollama

Designed for FastAPI / Streamlit integration.

Run:
    poetry run python -m src.pipelines.generate_answer --q "How to split data files in JASP?"

---------------------------------------------------
"""

from __future__ import annotations
from typing import Dict
from loguru import logger

# ✅ Import from your existing modules
from src.retrieval.retrieval import retrieve_top_k, NodeWithScore
from src.generation.generation import generate_answer


def run_generation_pipeline(query: str, model: str = "mistral:7b") -> Dict[str, object]:
    """
    Execute the full RAG pipeline: retrieve → rerank → generate.
    Returns a structured dict for frontend rendering.
    """
    logger.info(f"🔍 Running RAG generation pipeline for query: {query!r}")

    # 1️⃣ Retrieve & rerank
    reranked_nodes = retrieve_top_k(query)
    logger.success(f"✅ Retrieved and reranked top-{len(reranked_nodes)} nodes.")

    # 2️⃣ Generate answer
    answer_text = generate_answer(query, reranked_nodes, model=model)

    # 3️⃣ Prepare sources for display
    sources = []
    for i, node in enumerate(reranked_nodes, start=1):
        meta = getattr(node.node, "metadata", {})
        sources.append({
            "rank": i,
            "source": meta.get("source", "N/A"),
            "page": meta.get("page", "?"),
            "chunk_id": meta.get("chunk_id", "N/A"),
            "score": getattr(node, "score", None)
        })

    return {
        "query": query,
        "model": model,
        "answer": answer_text.strip(),
        "sources": sources
    }


# ---------- Local test ----------
if __name__ == "__main__":
    result = run_generation_pipeline("How to split data files in JASP?", model="mistral:7b")

    print("\n" + "="*80)
    print("🧠 FINAL ANSWER\n" + "-"*80)
    print(result["answer"])
    print("\n📚 SOURCES")
    for s in result["sources"]:
        print(f"[{s['rank']}] {s['source']} p.{s['page']} (score={s['score']})")
