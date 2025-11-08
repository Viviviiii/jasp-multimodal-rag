"""
---------------------------------------------------
💬 GENERATION PIPELINE (RAG FINAL STAGE)
---------------------------------------------------
"""

from __future__ import annotations
import sys
import argparse
from typing import List, Tuple
from loguru import logger
from ollama import Client

from src.retrieval.retrieval import retrieve_top_k, NodeWithScore


PROMPT_TEMPLATE = """
You are a documentation assistant for JASP.
Use the context below to answer the user's question clearly and concisely.

Question:
{query}

Context:
{context}

Answer in markdown, include source references when available.
"""


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
# 🧠 GENERATE ANSWER WITH OLLAMA
# --------------------------------------------------
def generate_answer(
    query: str,
    top_nodes: List[NodeWithScore],
    model: str = "mistral:7b",
    max_chars_per_doc: int = 800,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    """Generate answer from retrieved context."""
    client = Client()
    context_blocks = []

    for i, node in enumerate(top_nodes, 1):
        meta = getattr(node.node, "metadata", {})
        text = node.node.get_content()[:max_chars_per_doc].strip()

        page = (
            meta.get("page")
            or meta.get("page_start")
            or meta.get("page_number")
            or "?"
        )
        document_name = (
            meta.get("markdown_file")
            or meta.get("pdf_name")
            or meta.get("source")
            or "N/A"
        )
        section = (
            meta.get("section_title")
            or meta.get("sub_section_id")
            or "N/A"
        )

        header = f"[Doc {i}] {document_name} (p.{page} | sec:{section})"
        context_blocks.append(f"{header}\n{text}")

    prompt = prompt_template.format(
        query=query, context="\n\n".join(context_blocks)
    )

    logger.info(f"🚀 Generating answer with model: {model}")
    stream = client.generate(model=model, prompt=prompt, stream=True)

    full_output = ""
    for chunk in stream:
        token = chunk.get("response", "")
        sys.stdout.write(token)
        sys.stdout.flush()
        full_output += token

    logger.success("\n✅ Generation completed.")
    return full_output


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

# --------------------------------------------------
# 🧪 CLI MODE (for debugging)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run RAG generation pipeline")
    parser.add_argument("--q", "--query", dest="query", type=str, default="How to split data files in JASP?")
    parser.add_argument("--model", type=str, default="mistral:7b")
    args = parser.parse_args()

    result = run_generation_pipeline(args.query, args.model)

    print("\n" + "="*80)
    print("🧠 FINAL ANSWER\n" + "-"*80)
    print(result["answer"])

    print("\n📚 SOURCES")
    for s in result["sources"]:
        print(f"[{s['rank']}] {s['source']} p.{s['page']} | sec:{s['section']}")


if __name__ == "__main__":
    main()
