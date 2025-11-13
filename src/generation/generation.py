"""
---------------------------------------------------
💬 GENERATION PIPELINE (RAG FINAL STAGE)


poetry run python -m src.generation.generation --q "How to run Bayesian Linear Mixed Models in JASP?"
---------------------------------------------------
"""

from __future__ import annotations
import sys
import argparse
from typing import List, Tuple
from loguru import logger
from ollama import Client
from typing import Dict





from src.retrieval.retrieval import retrieve_top_k, NodeWithScore


PROMPT_TEMPLATE = """
You are a documentation assistant for JASP.
Use the context below to answer the user's question clearly and concisely.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Question:
{query}

Context:
{context}

Answer in markdown, include source references when available.
"""



# --------------------------------------------------
# 🧩 METADATA UNPACKING (robust multi-source version)
# --------------------------------------------------
def unpack_source(node_item, rank: int) -> Dict:
    """
    Normalize metadata across PDFs, GitHub markdown, and videos.
    Guarantees consistent keys: source_type, source, section, page.
    """
    node = getattr(node_item, "node", node_item)
    meta = getattr(node, "metadata", {}) or {}
    text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")

    source_type = meta.get("source_type", "document").lower()
    score = getattr(node_item, "score", None)

    # ---------- 📘 PDF / Document ----------
    if "pdf" in source_type or meta.get("pdf_name"):
        source = meta.get("pdf_name", "Unknown PDF")
        page = meta.get("page_start") or meta.get("page") or "?"
        section = meta.get("section_title") or "Document section"
        display_source = f"📘 {source}"

    # ---------- 🧩 GitHub Markdown ----------
    elif "github" in source_type or meta.get("markdown_file"):
        source = meta.get("markdown_file", "GitHub help file")
        github_repo = meta.get("github_name", "JASP GitHub")
        section = meta.get("section_title") or "GitHub section"
        page = meta.get("section_title") or "–"
        display_source = f"🐙 {github_repo}/{source}"

    # ---------- 🎥 Video Transcript ----------
    elif "video" in source_type or meta.get("video_title"):
        video_title = meta.get("video_title", "Untitled Video")
        chapter = meta.get("chapter_title")
        start = meta.get("start_time")
        end = meta.get("end_time")
        time_range = f"{start}–{end}" if start and end else start or "N/A"
        section = chapter or f"Segment {time_range}"
        page = time_range
        display_source = f"🎥 {video_title}"

    # ---------- Default fallback ----------
    else:
        source = meta.get("source") or "Unknown source"
        section = meta.get("section_title") or "General section"
        page = meta.get("page") or "?"
        display_source = source

    return {
        "rank": rank,
        "source_type": source_type,
        "source": display_source,
        "page": page,
        "section": section,
        "chunk_id": meta.get("doc_id") or f"chunk_{rank}",
        "score": score,
        "text": text[:1200],
        "metadata": meta,
    }


# --------------------------------------------------
# 🧠 GENERATE ANSWER (with explicit no-doc warning)
# --------------------------------------------------
def generate_answer(
    query: str,
    top_nodes: List[NodeWithScore],
    model: str = "mistral:7b",
    max_chars_per_doc: int = 800,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    """Generate an answer; include explicit warning when no supporting documents are found."""

    client = Client()

    # ⚠️ Case 1: No retrieved context — fallback generation
    if not top_nodes:
        logger.warning("⚠️ No documents retrieved — generating answer without database support.")

        warning_banner = (
            "🐝 **Warning:**  *This answer was generated without support from the present database.* "
            "*The response is based solely on general model knowledge.*  \n\n"
            "🌶️*More support can be found:https://jasp-stats.org/jasp-materials/* \n\n"
            "---\n\n"
        )

        prompt = (
            f"No relevant context was found in the database for the following question:\n"
            f"'{query}'\n\n"
            "Please provide a helpful general answer, "
            "but clearly mention that the response is *not supported by the JASP documentation*."
        )

        stream = client.generate(model=model, prompt=prompt, stream=True)
        full_output = ""
        for chunk in stream:
            token = chunk.get("response", "")
            sys.stdout.write(token)
            sys.stdout.flush()
            full_output += token

        logger.success("\n✅ Fallback generation completed (no documents).")
        return warning_banner + full_output.strip()

    # ✅ Case 2: Normal RAG generation with retrieved context
    context_blocks = []
    for i, node in enumerate(top_nodes, 1):
        meta = getattr(node.node, "metadata", {})
        text = node.node.get_content()[:max_chars_per_doc].strip()

        page = meta.get("page") or meta.get("page_start") or meta.get("page_number") or "?"
        document_name = meta.get("markdown_file") or meta.get("pdf_name") or meta.get("source") or "N/A"
        section = meta.get("section_title") or meta.get("sub_section_id") or "N/A"

        src_info = unpack_source(node, i)
        header = f"[{src_info['rank']}] {src_info['source']} (sec:{src_info['section']} | p:{src_info['page']})"

        context_blocks.append(f"{header}\n{text}")

    prompt = prompt_template.format(query=query, context="\n\n".join(context_blocks))

    logger.info(f"🚀 Generating answer with model: {model}")
    stream = client.generate(model=model, prompt=prompt, stream=True)

    full_output = ""
    for chunk in stream:
        token = chunk.get("response", "")
        sys.stdout.write(token)
        sys.stdout.flush()
        full_output += token

    logger.success("\n✅ Generation completed.")
    return full_output.strip()



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





