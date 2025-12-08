
"""
---------------------------------------------------
5:
 RAG generation pipeline (final answer stage)
---------------------------------------------------

This module is the final step of the RAG system: it takes a user query,
retrieves relevant chunks (PDF, GitHub, video), and then asks an Ollama model 
to generate a JASP-focused answer grounded in that context.

High-level flow
---------------
1. Call `retrieve_clean(...)` to get top-k normalized results for the query
   (defaulted mode: BM25 / vectors / hybrid + rerank).
2. Build a concise context block from the retrieved sources, including
   their titles, sections, page numbers, or timestamps.
3. Insert the query and context into a prompt template tailored for JASP
   documentation help.
4. Stream the answer from an Ollama model (default: `mistral:7b`).
5. Return both:
   - the final answer (markdown)
   - the list of sources used to generate it.

If no documents are retrieved, the pipeline explicitly warns that the
answer is not supported by the local JASP documentation and falls back
to general model knowledge.

CLI usage
---------
    poetry run python -m src.generation.generation \\
        --q "How to run Bayesian Linear Mixed Models in JASP?" \\
        --mode bm25_vector_fusion_rerank

Main programmatic entrypoint:
    from src.generation.generation import run_generation_pipeline

    result = run_generation_pipeline(
        query="How to run repeated measures ANOVA in JASP?",
        model="mistral:7b",
        mode="bm25_vector_fusion_rerank",
    )

    print(result["answer"])
    print(result["sources"])

---------------------------------------------------
"""


from __future__ import annotations

import sys
import argparse
from typing import List, Dict

from loguru import logger
from ollama import Client

from src.retrieval.retrieval import retrieve_clean


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
# 🧠 GENERATE ANSWER (Ollama, with/without docs)
# --------------------------------------------------
def generate_answer_from_docs(
    query: str,
    docs: List[Dict],
    model: str = "mistral:7b",
    max_chars_per_doc: int = 800,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    """
    Generate an answer using Ollama, given a list of doc dicts
    from `retrieve_clean`, each already containing metadata.

    Expected doc fields (depending on source type):
        - source_type: "pdf" | "markdown" | "video" | ...
        - title, pdf_id, markdown_file, video_title
        - section, section_title
        - page_number, page, timestamp
        - content or text
        - score
    """

    client = Client()

    # ⚠️ Case 1: No retrieved context — fallback generation
    if not docs:
        logger.warning("⚠️ No documents retrieved — generating answer without database support.")

        warning_banner = (
            "🐝 **Warning:**  *This answer was generated without support from the present database.* "
            "*The response is based solely on general model knowledge.*  \n\n"
            "🌶️ *More support can be found: https://jasp-stats.org/jasp-materials/* \n\n"
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
    for src_info in docs:
        text = (src_info.get("text") or src_info.get("content") or "").strip()
        if max_chars_per_doc:
            text = text[:max_chars_per_doc]

        rank = src_info.get("rank", "?")

        # Prefer source-specific titles, but never invent anything
        title = (
            src_info.get("title")
            or src_info.get("pdf_id")
            or src_info.get("markdown_file")
            or src_info.get("video_title")
            or src_info.get("source")
            or "Source"
        )

        section = (
            src_info.get("section")
            or src_info.get("section_title")
            or "-"
        )

        # Page can be page_number, page, or timestamp (for video)
        page = (
            src_info.get("page_number")
            or src_info.get("page")
            or src_info.get("timestamp")
            or "-"
        )

        header = f"[{rank}] {title} (sec:{section} | p:{page})"
        context_blocks.append(f"{header}\n{text}")

    prompt = prompt_template.format(
        query=query,
        context="\n\n".join(context_blocks),
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
    logger.info("🧪 DEBUG — FINAL PROMPT SENT TO OLLAMA:\n" + prompt[:3000])

    return full_output.strip()


# --------------------------------------------------
# 🧩 MAIN PIPELINE (for backend + CLI)
# --------------------------------------------------
def run_generation_pipeline(
    query: str,
    model: str = "mistral:7b",
    mode: str = "bm25_vector_fusion_rerank",
):
    """
    End-to-end RAG pipeline used by both CLI and backend.

    Steps:
        1. Retrieve top-k relevant chunks via `retrieve_clean(query, mode=...)`.
        2. Normalize them into a list of source dicts, each with:
             - source_type (pdf / markdown / video / ...)
             - title / section / page_number / timestamp
             - text/content and score
        3. Call `generate_answer_from_docs(...)` to build a JASP-specific
           prompt and stream an answer from the Ollama model.
        4. Return a structured dict containing:
             - "query"
             - "model"
             - "retrieval_mode"
             - "answer"   (markdown string)
             - "sources"  (the list of used context chunks)

    Args:
        query:
            User's natural language question.

        model:
            Ollama model name to use for answer generation
            (e.g. "mistral:7b", "llama3.2:latest", ...).

        mode:
            Retrieval mode to use before generation, consistent with
            `src.retrieval.retrieval` (bm25, vector, bm25_vector,
            bm25_vector_fusion, bm25_vector_fusion_rerank).

    Returns:
        A dict with the generated answer and the supporting sources,
        suitable for direct use in the API or UI.
    """


    logger.info(f"🎯 Running generation pipeline for query: {query} (mode={mode})")

    # 1️⃣ Retrieve top chunks with selected mode
    try:
        raw_docs = retrieve_clean(query, mode=mode)  # already "clean" dicts
    except Exception as e:
        logger.error(f"❌ Retrieval failed in generation pipeline: {e}")
        raw_docs = []

    logger.info(f"Retrieved {len(raw_docs)} chunks in mode={mode}")

    # 2️⃣ Just add rank + unified text, do NOT reinterpret metadata
    sources: List[Dict] = []
    for i, d in enumerate(raw_docs, 1):
        doc = dict(d)  # shallow copy
        doc["rank"] = i
        doc["text"] = doc.get("text") or doc.get("content") or ""
        sources.append(doc)

    # 3️⃣ Generate contextual answer with Ollama
    final_answer = generate_answer_from_docs(query, sources, model=model)

    # 4️⃣ Return final structured result
    return {
        "query": query,
        "model": model,
        "retrieval_mode": mode,
        "answer": final_answer,
        "sources": sources,
    }


# --------------------------------------------------
# 🧪 CLI MODE (for debugging)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run RAG generation pipeline")
    parser.add_argument(
        "--q", "--query",
        dest="query",
        type=str,
        default="How to split data files in JASP?",
    )
    parser.add_argument("--model", type=str, default="mistral:7b")
    parser.add_argument(
        "--mode",
        type=str,
        default="bm25_vector_fusion_rerank",
        choices=[
            "bm25",
            "vector",
            "bm25_vector",
            "bm25_vector_fusion",
            "bm25_vector_fusion_rerank",
        ],
        help="Retrieval mode used before generation.",
    )
    args = parser.parse_args()

    result = run_generation_pipeline(args.query, args.model, mode=args.mode)

    print("\n" + "=" * 80)
    print("🧠 FINAL ANSWER\n" + "-" * 80)
    print(result["answer"])

    print("\n📚 SOURCES (mode: {})".format(result.get("retrieval_mode", args.mode)))
    for s in result["sources"]:
        source_type = str(s.get("source_type", "text")).lower()

        print("\n----------------------------------------")
        print(f"[{s.get('rank', '?')}] ({source_type})")

        # ---------- PDF ----------
        if source_type == "pdf":
            pdf_id = s.get("pdf_id") or s.get("title") or "PDF Document"
            page = s.get("page_number") or s.get("page") or "?"
            total_pages = s.get("total_pages", "?")
            section = s.get("title") or s.get("section") or "(no section title)"
            url = s.get("source_url") or "N/A"
            score = s.get("score", "N/A")

            print(f" 📄 PDF      : {pdf_id}")
            print(f" 🧩 Section  : {section}")
            print(f" 📑 Page     : {page}/{total_pages}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- VIDEO ----------
        elif source_type == "video":
            title = s.get("title") or s.get("video_title") or "(no video title)"
            section = s.get("section") or s.get("chapter_title") or "Video segment"
            timestamp = s.get("timestamp") or s.get("page") or "N/A"
            url = s.get("video_link") or s.get("source_url") or "N/A"
            score = s.get("score", "N/A")

            print(f" 🎥 Video    : {title}")
            print(f" 🧩 Section  : {section}")
            print(f" ⏱️ Time     : {timestamp}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- MARKDOWN / GITHUB ----------
        elif source_type == "markdown":
            title = s.get("title") or s.get("markdown_file") or "(no file name)"
            section = s.get("section") or s.get("section_title") or "(no section title)"
            url = s.get("source_url") or "N/A"
            score = s.get("score", "N/A")

            print(f" 📘 Markdown : {title}")
            print(f" 🧩 Section  : {section}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- Fallback ----------
        else:
            title = (
                s.get("title")
                or s.get("source")
                or s.get("pdf_id")
                or s.get("markdown_file")
                or "(no title)"
            )
            section = s.get("section") or s.get("section_title") or "N/A"
            page = s.get("page") or s.get("page_number") or s.get("timestamp") or "?"
            url = s.get("source_url") or "N/A"
            score = s.get("score", "N/A")

            print(f" 📄 Source   : {title}")
            print(f" 🧩 Section  : {section}")
            print(f" 📑 Page     : {page}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # Optional: short snippet
        snippet = (s.get("text") or s.get("content") or "").strip()
        if snippet:
            if len(snippet) > 300:
                snippet = snippet[:300] + "..."
            print(f" 🗒️ Text     : {snippet}")

    print("\n")


if __name__ == "__main__":
    main()
