"""
---------------------------------------------------
🔎 HYBRID → RRF → CROSS-ENCODER RERANK PIPELINE
(LlamaIndex + Chroma + Ollama, optional LangChain)
1.Hybrid retrieval = BM25 (lexical) + Chroma (semantic).

2.Reciprocal Rank Fusion (RRF) to merge the two result lists.

3.Cross-encoder rerank with BAAI/bge-reranker-large (or SentenceTransformer fallback).

---------------------------------------------------
Run:
    poetry run python -m src.retrieval.retrieval --q "How to split data files in JASP?"
---------------------------------------------------
"""


from __future__ import annotations
import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

from loguru import logger
import chromadb

# ---------- LlamaIndex core ----------
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore

# BM25 (import path differs by version)
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:
    # 0.10+ sometimes moves things
    try:
        from llama_index.core.retrievers import BM25Retriever  # type: ignore
    except Exception as e:
        BM25Retriever = None
        logger.warning(f"BM25Retriever not found; install `rank-bm25` and LlamaIndex with bm25 extras. Error: {e}")

# Rerankers (FlagEmbedding BGE reranker preferred)
FlagEmbeddingReranker = None
SentenceTransformerRerank = None
try:
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker  # older path
except Exception:
    try:
        from llama_index.core.postprocessor import FlagEmbeddingReranker  # newer path
    except Exception:
        pass

if FlagEmbeddingReranker is None:
    # Fallback to SentenceTransformer cross-encoder
    try:
        from llama_index.postprocessor import SentenceTransformerRerank  # older path
    except Exception:
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank  # newer path
        except Exception:
            pass

# Ollama LLM (LlamaIndex wrapper)
try:
    from llama_index.llms.ollama import Ollama
except Exception:
    Ollama = None
    logger.warning("Ollama wrapper not available; install llama-index-llms-ollama or pin a compatible version.")







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
#OLLAMA_MODEL = "llama3.1:8b"
OLLAMA_MODEL = "mistral:7b"


# ---------- UTIL: Load documents for BM25 ----------
def load_docs_for_bm25(json_path: str) -> List[Document]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Chunk JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        md = item.get("metadata", {})
        flat_md = {}
        for k, v in md.items():
            if isinstance(v, list):
                flat_md[k] = ", ".join(map(str, v))
            elif isinstance(v, dict):
                flat_md[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat_md[k] = v
        # Keep doc id if present to help dedup/fusion
        if "id" in item:
            flat_md["doc_id"] = item["id"]
        docs.append(Document(text=item["text"], metadata=flat_md))
    logger.info(f"BM25 corpus: {len(docs)} documents")
    return docs


# ---------- Build retrievers ----------

# --------------------------------------------------
# Build Semantic (Vector) Retriever
# --------------------------------------------------
def build_vector_retriever(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embed_model_name: str = EMBED_MODEL,
):
    """Rebuild a semantic retriever from existing Chroma vector store."""
    import chromadb
    from llama_index.vector_stores.chroma import ChromaVectorStore
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core import VectorStoreIndex, StorageContext

    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_col = chroma_client.get_or_create_collection(name=collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_col)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
        embed_model=embed_model,
    )

    retriever = index.as_retriever(similarity_top_k=K_SEMANTIC)
    return retriever


# --------------------------------------------------
# Build BM25 Retriever
# --------------------------------------------------
def build_bm25_retriever(docs: List[Document]):
    """
    Build a BM25 retriever compatible with llama-index-retrievers-bm25>=0.6.0.
    """
    try:
        from llama_index.retrievers.bm25 import BM25Retriever
    except Exception as e:
        raise RuntimeError(f"BM25Retriever import failed: {e}")

    logger.info(f"Initializing BM25 retriever with {len(docs)} documents...")
    retriever = BM25Retriever(docs, similarity_top_k=K_BM25)
    return retriever


# ---------- RRF (Reciprocal Rank Fusion) ----------
import hashlib
import json

def stable_node_key(n: NodeWithScore) -> str:
    """Stable deduplication key for RRF fusion."""
    node = n.node
    text = node.get_content().strip()
    meta = node.metadata or {}

    # Use a stable subset of metadata that identifies a chunk (e.g., doc_id, page)
    id_hint = meta.get("doc_id") or meta.get("source") or meta.get("file") or ""
    key_data = f"{id_hint}::{text}"

    # MD5 is fine for deduplication
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()

def rrf_fuse(results_a, results_b, k=RRF_K, top_n=TOP_AFTER_RRF):
    def scores(rr):
        s = {}
        for rank, n in enumerate(rr, start=1):
            s[stable_node_key(n)] = 1.0 / (k + rank)
        return s

    s_a = scores(results_a)
    s_b = scores(results_b)

    lookup = {}
    for lst in (results_a, results_b):
        for n in lst:
            lookup.setdefault(stable_node_key(n), n)

    fused = []
    for key in set(list(s_a.keys()) + list(s_b.keys())):
        fused.append((key, s_a.get(key, 0.0) + s_b.get(key, 0.0)))

    fused_sorted = sorted(fused, key=lambda x: x[1], reverse=True)[:top_n]
    return [lookup[k] for k, _ in fused_sorted]



# ---------- Cross-encoder rerank ----------
def rerank_cross_encoder(nodes: List[NodeWithScore], query: str, top_k: int = TOP_FINAL) -> List[NodeWithScore]:
    if FlagEmbeddingReranker is not None:
        logger.info(f"Cross-encoder rerank with BGE: {BGE_RERANKER}")
        reranker = FlagEmbeddingReranker(model=BGE_RERANKER, top_n=top_k)
        return reranker.postprocess_nodes(nodes, query_str=query)
    elif SentenceTransformerRerank is not None:
        logger.info(f"Cross-encoder rerank with SentenceTransformer: {ST_CE_FALLBACK}")
        reranker = SentenceTransformerRerank(model=ST_CE_FALLBACK, top_n=top_k)
        return reranker.postprocess_nodes(nodes, query_str=query)
    else:
        logger.warning("No cross-encoder available; skipping rerank.")
        return nodes[:top_k]


# ---------- MAIN PIPELINE ----------
def main():
    parser = argparse.ArgumentParser(description="🔎 Hybrid → RRF → Cross-encoder Rerank Pipeline")
    parser.add_argument("--q", "--query", dest="query", type=str, required=True, help="User query string")
    args = parser.parse_args()
    query = args.query.strip()
    logger.info(f"🔍 Query: {query}")

    # 1️⃣ Load corpus for BM25
    docs = load_docs_for_bm25(CHUNKS_JSON)

    # 2️⃣ Build retrievers
    semantic_retriever = build_vector_retriever()
    bm25_retriever = build_bm25_retriever(docs)

    # 3️⃣ Retrieve candidates
    logger.info("⚙️ Retrieving from BM25 and semantic retrievers...")
    bm25_results = bm25_retriever.retrieve(query)
    sem_results = semantic_retriever.retrieve(query)
    logger.success(f"BM25: {len(bm25_results)} results | Semantic: {len(sem_results)} results")

    # 4️⃣ Fuse via RRF
    fused = rrf_fuse(sem_results, bm25_results)
    logger.success(f"🔗 Fused results (RRF): {len(fused)}")

    # 5️⃣ Cross-encoder rerank
    reranked = rerank_cross_encoder(fused, query, top_k=TOP_FINAL)
    logger.success(f"🏁 Reranked top-{TOP_FINAL} results ready")



        # 6️⃣ Display results neatly
    print(f"\n🏁 Reranked top-{TOP_FINAL}: {len(reranked)}\n{'='*60}")
    for i, item in enumerate(reranked, 1):
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", {})
        text = getattr(node, "text", str(node))
        doc_length = len(text)

        print(f"\n⭐ Final Rank {i}")
        print(f"   📄 Source : {metadata.get('source', 'N/A')}")
        print(f"   📑 Page   : {metadata.get('page', 'N/A')}")
        print(f"   🔢 New Score : {getattr(item, 'score', 'N/A')}")
        print(f"   🧩 Chunk  : {metadata.get('chunk_id', 'N/A')}")
        print(f"   📏 Length : {doc_length} characters")
        print(f"   🗒️ Text   : {text}")
        print("-" * 60)

    logger.info("✅ Retrieval pipeline completed successfully.")


# ---------- ENTRY POINT ----------
if __name__ == "__main__":
    main()
