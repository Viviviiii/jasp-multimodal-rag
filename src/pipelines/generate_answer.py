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

# --------------------------------------------------
# 🧩 METADATA UNPACKING
# --------------------------------------------------
def unpack_source(node_item, rank: int):
    node = getattr(node_item, "node", node_item)
    meta = getattr(node, "metadata", {}) or {}
    text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")

    return {
        "rank": rank,
        "source": meta.get("pdf_name") or meta.get("markdown_file") or meta.get("source") or "Unknown",
        "page": meta.get("page_start") or meta.get("page") or "?",
        "chunk_id": meta.get("section_id") or meta.get("doc_id") or f"chunk_{rank}",
        "score": getattr(node_item, "score", None),
        "section": meta.get("section_title") or "Unknown section",
        "text": text[:1200],
        "metadata": meta,  # ✅ pass full metadata to FastAPI response
    }


# --------------------------------------------------
# 🧩 MAIN PIPELINE ENTRYPOINT (for backend + CLI)
# --------------------------------------------------
def run_generation_pipeline(query: str, model: str = "mistral:7b"):
    """
    Unified pipeline used by both CLI and FastAPI backend.
    Returns structured response with flattened metadata.
    """
    logger.info(f"🎯 Running generation pipeline for query: {query}")

    # 1️⃣ Retrieve top chunks
    reranked_nodes = retrieve_top_k(query)
    logger.info(f"Retrieved {len(reranked_nodes)} chunks")

    # 2️⃣ Normalize metadata for frontend
    clean_sources = [unpack_source(item, rank=i+1) for i, item in enumerate(reranked_nodes)]

    # 3️⃣ Generate contextual answer
    final_answer = generate_answer(query, reranked_nodes, model=model)

    # 4️⃣ Return final structured result
    return {
        "query": query,
        "model": model,
        "answer": final_answer,
        "sources": clean_sources,
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
