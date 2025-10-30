"""
---------------------------------------------------
💬 GENERATION PIPELINE (RAG FINAL STAGE)
Combines: Hybrid Retrieval → RRF → Cross-Encoder → Ollama LLM

- Works with local Ollama models (e.g., Mistral, Llama 3.2, Phi-3)
- Can be imported in FastAPI or Streamlit frontends
- PROMPT_TEMPLATE is modular and editable

Run:
    poetry run python -m src.generation.generation --q "How to split data files in JASP?"
---------------------------------------------------
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Tuple
from loguru import logger
from ollama import Client

# ---------- Imports from retrieval pipeline ----------
from src.retrieval.retrieval import (
    load_docs_for_bm25,
    build_vector_retriever,
    build_bm25_retriever,
    rrf_fuse,
    rerank_cross_encoder,
    NodeWithScore
)

 
# ---------- CONFIG ----------

CHROMA_DIR = "/Users/ywxiu/jasp-multimodal-rag/data/vector_store/chroma_db"
COLLECTION_NAME = "jasp_manual_embeddings_v2"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNKS_JSON = "/Users/ywxiu/jasp-multimodal-rag/data/processed/chunks/chunks_test_pages25-28.json"

# Retrieval sizes
K_SEMANTIC = 10       # top-N from vector retriever
K_BM25 = 10           # top-N from BM25 retriever
RRF_K = 60             # RRF constant (larger => dampens rank differences)
TOP_AFTER_RRF = 10     # shortlist after RRF, before cross-encoder
TOP_FINAL = 3          # final K after cross-encoder rerank

# Cross-encoder models
BGE_RERANKER = "BAAI/bge-reranker-large"
ST_CE_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Ollama model
OLLAMA_MODEL = "mistral:7b"





# ---------- GLOBAL PROMPT TEMPLATE ----------
PROMPT_TEMPLATE = """
You are a documentation assistant that helps users find accurate answers from official sources.

Use the provided context to answer the question.
Whenever you include a fact, cite the corresponding source by its document number,
and show key metadata (page, chapter, or link) for traceability.

Question:
{query}

Context:
{context}

Instructions:
- Answer concisely and accurately based on the context.
- Include references in the format:
  [Doc 1 – Source: test_pages25-28.pdf, Page: 3, Chapter: 'Descriptive Stats', Link: example.com]
- Summarize which documents were most relevant.
- If multiple docs mention the same concept, combine them logically.

Answer:
"""

# ---------- GENERATION CORE ----------
def generate_answer(
    query: str,
    top_nodes: List[NodeWithScore],
    model: str = "mistral:7b",
    max_chars_per_doc: int = 1200,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    """
    Generate an answer with Ollama LLM based on reranked context.
    Returns the full text output.
    """

    client = Client()
    context_blocks = []
    structured_refs = []

    # ---------- Build Context ----------
    for i, node in enumerate(top_nodes, 1):
        meta = getattr(node.node, "metadata", {})
        source = meta.get("source", "N/A")
        page = meta.get("page", "N/A")
        chapter = meta.get("chapter", "N/A")
        link = meta.get("link", None)
        chunk_id = meta.get("chunk_id", i)
        text = node.node.get_content()[:max_chars_per_doc]

        context_blocks.append(
            f"[Document {i}] "
            f"(Source: {source}, Page: {page}, Chapter: {chapter}, Chunk: {chunk_id}"
            + (f", Link: {link}" if link else "")
            + f")\n{text}"
        )

        structured_refs.append({
            "doc_id": i,
            "source": source,
            "page": page,
            "chapter": chapter,
            "link": link,
            "chunk_id": chunk_id
        })

    context = "\n\n".join(context_blocks)
    logger.info(f"🧩 Total context length: {len(context)} characters")

    # ---------- Prepare Prompt ----------
    prompt = prompt_template.format(query=query, context=context)

    # ---------- Stream Generation ----------
    logger.info(f"🚀 Generating with model: {model}")
    stream = client.generate(model=model, prompt=prompt, stream=True)

    full_output = ""
    for chunk in stream:
        token = chunk.get("response", "")
        sys.stdout.write(token)
        sys.stdout.flush()
        full_output += token

    logger.success("\n✅ Generation completed.")
    logger.info("\n📚 References used in context:")
    for ref in structured_refs:
        logger.info(
            f" - Doc {ref['doc_id']}: {ref['source']} "
            f"(Page {ref['page']}, Chapter {ref['chapter']}, Link: {ref['link']})"
        )

    return full_output


# ---------- END-TO-END PIPELINE ----------
def answer_query(query: str) -> Tuple[str, List[NodeWithScore]]:
    """Full RAG pipeline: retrieval → fusion → rerank → generation"""
    docs = load_docs_for_bm25(CHUNKS_JSON)
    sem_retriever = build_vector_retriever()
    bm25_retriever = build_bm25_retriever(docs)

    logger.info("🔎 Running hybrid retrieval...")
    sem_results = sem_retriever.retrieve(query)
    bm25_results = bm25_retriever.retrieve(query)
    logger.success(f"Semantic: {len(sem_results)}, BM25: {len(bm25_results)}")

    fused = rrf_fuse(sem_results, bm25_results, k=RRF_K, top_n=TOP_AFTER_RRF)
    reranked = rerank_cross_encoder(fused, query, top_k=TOP_FINAL)

    logger.success(f"Final reranked K={len(reranked)}. Starting generation...")
    answer = generate_answer(query, reranked)
    return answer, reranked


# ---------- CLI ENTRY ----------
def main():
    parser = argparse.ArgumentParser(description="Run end-to-end RAG generation with Ollama")
    parser.add_argument("--q", "--query", dest="query", type=str, required=False,
                        default="How to split data files in JASP?")
    parser.add_argument("--model", type=str, default="mistral:7b", help="Ollama model name")
    args = parser.parse_args()

    ans, nodes = answer_query(args.query)

    print("\n" + "="*80)
    print("🧠 FINAL ANSWER\n" + "-"*80)
    print(ans)
    print("\n" + "-"*80)
    print("📚 SOURCES")
    for i, n in enumerate(nodes, start=1):
        md = n.node.metadata or {}
        src = md.get("source") or md.get("file") or md.get("doc_id") or "unknown"
        page = md.get("page")
        print(f"[{i}] {src}" + (f":p{page}" if page is not None else ""))


if __name__ == "__main__":
    main()
