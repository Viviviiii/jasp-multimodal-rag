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





from src.retrieval.retrieval import retrieve_top_k, NodeWithScore,retrieve_clean


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

def unpack_source(doc, rank: int) -> Dict:
    """
    Robust metadata normalization that preserves your original
    PDF / GitHub / Video logic, but also handles dicts safely.

    Accepts:
        - NodeWithScore objects (node.metadata)
        - dicts from retrieve_clean()

    Returns a consistent structure:
        source_type, source, source_url, page, section, score, text, metadata
    """

    # ============================================================
    # 1) SUPPORT BOTH NodeWithScore OBJECTS AND CLEAN DICTIONARIES
    # ============================================================

    # CASE A – NodeWithScore from retrieve_top_k()
    if hasattr(doc, "node"):
        node = doc.node
        meta = getattr(node, "metadata", {}) or {}
        text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", "")
        score = getattr(doc, "score", None)

    # CASE B – Dictionary from retrieve_clean()
    elif isinstance(doc, dict):
        # FIX: merge ALL top-level fields into metadata
        meta = {}

        # copy all fields except text/content/score into meta
        for k, v in doc.items():
            if k not in ("text", "content", "score"):
                meta[k] = v

        text = doc.get("text") or doc.get("content") or ""
        score = doc.get("score")





    # CASE C – Unexpected type
    else:
        logger.warning(f"unpack_source received unexpected type at rank {rank}: {type(doc)}")
        return {
            "rank": rank,
            "source_type": "unknown",
            "source": "Unknown",
            "source_url": None,
            "page": "?",
            "section": "N/A",
            "chunk_id": f"chunk_{rank}",
            "score": None,
            "text": "",
            "metadata": {},
        }

    # Ensure meta is always a dict
    if not isinstance(meta, dict):
        meta = {}

    # Safely extract basic values
    source_type_raw = (
        meta.get("source_type")
        or (doc.get("source_type") if isinstance(doc, dict) else None)
        or ""
    )
    source_type_raw = str(source_type_raw).lower()


    # ============================================================
    # 📄 PDF METADATA (fixed)
    # ============================================================
    if source_type_raw == "pdf" or meta.get("pdf_id"):
        return {
            "rank": rank,
            "source_type": "pdf",

            # FIX: use pdf_id or fallback to title
            "source": (
                meta.get("pdf_id")
                or meta.get("pdf_name")
                or meta.get("source")
                or "PDF Document"
            ),

            "source_url": meta.get("source_url"),

            # FIX: use page_number correctly
            "page": (
                meta.get("page_number")
                or meta.get("page_start")
                or meta.get("page")
                or "?"
            ),

            # FIX: use title from retrieve_clean
            "section": (
                meta.get("title")
                or meta.get("section_title")
                or "PDF section"
            ),

            "chunk_id": meta.get("chunk_id") or f"chunk_{rank}",
            "score": score,
            "text": text[:1200],
            "metadata": meta,
        }



    # ============================================================
    # 3) GITHUB MARKDOWN BLOCK
    # ============================================================
    if (
        "github" in source_type_raw
        or meta.get("markdown_file")
        or (isinstance(doc, dict) and doc.get("markdown_file"))
    ):
        return {
            "rank": rank,
            "source_type": "github",
            "source": meta.get("markdown_file") or doc.get("markdown_file"),
            "source_url": meta.get("source_url") or doc.get("source_url"),
            "page": meta.get("section_title") or doc.get("section_title") or "-",
            "section": meta.get("section_title") or "GitHub section",
            "chunk_id": meta.get("doc_id") or f"chunk_{rank}",
            "score": score,
            "text": text[:1200],
            "metadata": meta,
        }

    # ============================================================
    # 4) VIDEO BLOCK
    # ============================================================
    if (
        "video" in source_type_raw
        or meta.get("video_title")
        or (isinstance(doc, dict) and doc.get("video_title"))
    ):
        return {
            "rank": rank,
            "source_type": "video",
            "source": meta.get("video_title") or doc.get("video_title"),
            "source_url": meta.get("video_url") or meta.get("source_url")
                           or doc.get("source_url"),
            "page": meta.get("start_time") or doc.get("timestamp") or None,
            "section": meta.get("chapter_title") or "Video segment",
            "chunk_id": meta.get("doc_id") or f"chunk_{rank}",
            "score": score,
            "text": text[:1200],
            "metadata": meta,
        }

    # ============================================================
    # 5) FALLBACK (unknown category)
    # ============================================================
    return {
        "rank": rank,
        "source_type": source_type_raw or "unknown",
        "source": meta.get("source")
                   or (doc.get("source") if isinstance(doc, dict) else None)
                   or "Unknown",
        "source_url": meta.get("source_url") or (doc.get("source_url") if isinstance(doc, dict) else None),
        "page": meta.get("page") or doc.get("page") if isinstance(doc, dict) else "?",
        "section": meta.get("section_title") or "N/A",
        "chunk_id": meta.get("doc_id") or f"chunk_{rank}",
        "score": score,
        "text": text[:1200] if text else "",
        "metadata": meta,
    }
      
    



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
    Generate an answer using Ollama, given a list of flattened doc dicts
    from `retrieve_clean`.
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
    for src_info in docs:  # docs is already clean_sources
        text = (src_info.get("text") or "").strip()
        if max_chars_per_doc:
            text = text[:max_chars_per_doc]

        header = (
            f"[{src_info['rank']}] {src_info['source']} "
            f"(sec:{src_info['section']} | p:{src_info['page']})"
        )

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
# 🧩 MAIN PIPELINE ENTRYPOINT (for backend + CLI)
# --------------------------------------------------
def run_generation_pipeline(query: str, model: str = "mistral:7b"):
    """
    Unified pipeline used by both CLI and FastAPI backend.
    Uses the same retrieval (`retrieve_clean`) as the /retrieve endpoint,
    then calls Ollama to generate an answer.
    """
    logger.info(f"🎯 Running generation pipeline for query: {query}")

    # 1️⃣ Retrieve top chunks (same as /retrieve, so no HF/meta-tensor issues)
    try:
        raw_docs = retrieve_clean(query)
    except Exception as e:
        logger.error(f"❌ Retrieval failed in generation pipeline: {e}")
        raw_docs = []

    logger.info(f"Retrieved {len(raw_docs)} chunks")

    # 2️⃣ Normalize metadata for frontend
    clean_sources = [unpack_source(doc, rank=i + 1) for i, doc in enumerate(raw_docs)]
    

    # 3️⃣ Generate contextual answer with Ollama
    final_answer = generate_answer_from_docs(query, clean_sources, model=model)

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
    parser.add_argument(
        "--q", "--query",
        dest="query",
        type=str,
        default="How to split data files in JASP?",
    )
    parser.add_argument("--model", type=str, default="mistral:7b")
    args = parser.parse_args()

    result = run_generation_pipeline(args.query, args.model)

    print("\n" + "=" * 80)
    print("🧠 FINAL ANSWER\n" + "-" * 80)
    print(result["answer"])

    print("\n📚 SOURCES")
    for s in result["sources"]:
        print(f"[{s['rank']}] {s['source']} p.{s['page']} | sec:{s['section']}")


if __name__ == "__main__":
    main()






