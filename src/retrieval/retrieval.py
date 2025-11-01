"""
---------------------------------------------------
🔎 HYBRID → RRF → CROSS-ENCODER RERANK PIPELINE
(LlamaIndex + Chroma + Ollama, optional LangChain)

1. Hybrid retrieval = BM25 (lexical) + Chroma (semantic).
2. Reciprocal Rank Fusion (RRF) to merge the two result lists.
3. Cross-encoder rerank with BAAI/bge-reranker-large (or SentenceTransformer fallback).

---------------------------------------------------
Run:
    poetry run python -m src.retrieval.retrieval --q "How to split data files in JASP?"
  
---------------------------------------------------
"""




from __future__ import annotations
import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict
from loguru import logger
import chromadb

# ---------- LlamaIndex core ----------
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore

# ---------- BM25 Retriever ----------
try:
    from llama_index.retrievers.bm25 import BM25Retriever
except Exception:
    try:
        from llama_index.core.retrievers import BM25Retriever  # type: ignore
    except Exception as e:
        BM25Retriever = None
        logger.warning(f"BM25Retriever not found; install rank-bm25 and LlamaIndex with bm25 extras. Error: {e}")



# ---------- Rerankers (FlagEmbedding BGE reranker preferred) ----------
FlagEmbeddingReranker = None
SentenceTransformerRerank = None

try:
    # ✅ For modular installs (LlamaIndex >= 0.13)
    from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
    from loguru import logger
    logger.info("✅ Loaded FlagEmbeddingReranker from llama_index.postprocessor.flag_embedding_reranker")
except ImportError:
    try:
        # 🧩 For legacy or core-only installs (LlamaIndex 0.10–0.12)
        from llama_index.core.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
        logger.info("✅ Loaded FlagEmbeddingReranker from llama_index.core.postprocessor.flag_embedding_reranker")
    except ImportError:
        logger.warning("⚠️ FlagEmbeddingReranker not available. Falling back to SentenceTransformer.")

# ---------- Fallback Cross-Encoder (MiniLM) ----------
if FlagEmbeddingReranker is None:
    try:
        from llama_index.postprocessor import SentenceTransformerRerank
        logger.info("✅ Loaded SentenceTransformerRerank fallback.")
    except ImportError:
        try:
            from llama_index.core.postprocessor import SentenceTransformerRerank
            logger.info("✅ Loaded SentenceTransformerRerank fallback (core path).")
        except ImportError:
            logger.warning("⚠️ No cross-encoder reranker available.")





# ---------- Ollama LLM Wrapper ----------
try:
    from llama_index.llms.ollama import Ollama
except Exception:
    Ollama = None
    logger.warning("Ollama wrapper not available; install llama-index-llms-ollama or pin a compatible version.")

# ---------- CONFIG ----------
CHROMA_DIR = "/Users/ywxiu/jasp-multimodal-rag/data/vector_store/chroma_db"
COLLECTION_NAME = "jasp_manual_embeddings_v2"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
CHUNKS_JSON = "/Users/ywxiu/jasp-multimodal-rag/data/processed/chunks/"

K_SEMANTIC = 10
K_BM25 = 10
RRF_K = 60
TOP_AFTER_RRF = 10
TOP_FINAL = 3

BGE_RERANKER = "BAAI/bge-reranker-large"
ST_CE_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "mistral:7b"


# --------------------------------------------------
# LOAD DOCUMENTS FOR BM25
# --------------------------------------------------
def load_docs_for_bm25_onlyone(json_path: str) -> List[Document]:
    """Load chunks and prepare for BM25 retriever."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Chunk JSON not found: {json_path}")

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

        if "id" in item:
            flat_md["doc_id"] = item["id"]

        docs.append(Document(text=item["text"], metadata=flat_md))

    logger.info(f"BM25 corpus loaded: {len(docs)} documents")
    return docs


def load_docs_for_bm25(input_path: Union[str, list[str]]) -> List[Document]:
    """
    Load one or more JSON chunk files for BM25 retriever.
    - input_path can be:
        • a single JSON file
        • a folder containing multiple *.json files
        • a list of JSON paths
    """
    json_files = []

    if isinstance(input_path, list):
        json_files = [Path(p) for p in input_path]
    elif os.path.isdir(input_path):
        json_files = list(Path(input_path).glob("*.json"))
    elif os.path.isfile(input_path):
        json_files = [Path(input_path)]
    else:
        raise FileNotFoundError(f"❌ Invalid input path: {input_path}")

    if not json_files:
        raise FileNotFoundError(f"❌ No JSON files found in {input_path}")

    all_docs = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_name = Path(json_path).stem
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

            flat_md["source"] = source_name
            if "id" in item:
                flat_md["doc_id"] = item["id"]

            all_docs.append(Document(text=item["text"], metadata=flat_md))

        logger.info(f"📄 Loaded {len(data)} chunks from {json_path}")

    logger.success(f"✅ Total BM25 corpus size: {len(all_docs)} chunks from {len(json_files)} files.")
    return all_docs



# --------------------------------------------------
# LOAD DOCUMENTS FOR BM25 (Supports folder or list)
# --------------------------------------------------


# --------------------------------------------------
# BUILD RETRIEVERS
# --------------------------------------------------


def build_bm25_retriever(docs: List[Document]):
    """Build a BM25 retriever."""
    if BM25Retriever is None:
        raise RuntimeError("BM25Retriever not available. Install llama-index with bm25 extras.")
    logger.info(f"Initializing BM25 retriever with {len(docs)} documents...")
    return BM25Retriever(docs, similarity_top_k=K_BM25)




def build_vector_retriever(
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embed_model_name: str = EMBED_MODEL,
):
    """Rebuild a semantic retriever from existing Chroma vector store."""
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
# RECIPROCAL RANK FUSION (RRF)
# --------------------------------------------------
def stable_node_key(n: NodeWithScore) -> str:
    """Generate a stable deduplication key for RRF fusion."""
    node = n.node
    text = node.get_content().strip()
    meta = node.metadata or {}
    id_hint = meta.get("doc_id") or meta.get("source") or meta.get("file") or ""
    key_data = f"{id_hint}::{text}"
    return hashlib.md5(key_data.encode("utf-8")).hexdigest()


def rrf_fuse(results_a, results_b, k=RRF_K, top_n=TOP_AFTER_RRF):
    """Fuse two ranked lists using Reciprocal Rank Fusion (RRF)."""
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


# --------------------------------------------------
# CROSS-ENCODER RERANK
# --------------------------------------------------


def rerank_cross_encoder(nodes: List[NodeWithScore], query: str, top_k: int = TOP_FINAL) -> List[NodeWithScore]:
    """Apply cross-encoder reranking (BGE preferred)."""

    if FlagEmbeddingReranker is not None:
        logger.info(f"⚙️ Cross-encoder rerank with BGE model: {BGE_RERANKER}")
        reranker = FlagEmbeddingReranker(
            model=BGE_RERANKER,
            top_n=top_k,
            use_fp16=True,  # ✅ faster on MPS or GPU
        )
        results = reranker.postprocess_nodes(nodes, query_str=query)
        logger.success("✅ Reranking completed with FlagEmbeddingReranker")
        return results

    elif SentenceTransformerRerank is not None:
        logger.info(f"⚙️ Cross-encoder rerank with SentenceTransformer fallback: {ST_CE_FALLBACK}")
        reranker = SentenceTransformerRerank(model=ST_CE_FALLBACK, top_n=top_k)
        return reranker.postprocess_nodes(nodes, query_str=query)

    else:
        logger.warning("⚠️ No cross-encoder available — skipping rerank.")
        return nodes[:top_k]






# --------------------------------------------------
# RETRIEVAL WRAPPER FUNCTION
# --------------------------------------------------
def retrieve_top_k(query: str, top_k: int = TOP_FINAL):
    """
    Run the full hybrid retrieval pipeline:
      1. Load BM25 corpus
      2. Build semantic + BM25 retrievers
      3. Retrieve from both
      4. Fuse via RRF
      5. Rerank via cross-encoder

    Returns:
        List[NodeWithScore]: Final reranked top_k results
    """
    logger.info(f"🔍 Starting retrieval pipeline for query: '{query}'")

    # 1️⃣ Load corpus for BM25
    docs = load_docs_for_bm25(CHUNKS_JSON)

    # 2️⃣ Build retrievers
    semantic_retriever = build_vector_retriever()
    bm25_retriever = build_bm25_retriever(docs)

    # 3️⃣ Retrieve candidates
    logger.info("⚙️ Retrieving from BM25 and semantic retrievers...")
    bm25_results = bm25_retriever.retrieve(query)
    sem_results = semantic_retriever.retrieve(query)
    logger.success(f"BM25: {len(bm25_results)} | Semantic: {len(sem_results)}")

    # 4️⃣ Fuse via RRF
    fused = rrf_fuse(sem_results, bm25_results)
    logger.success(f"🔗 Fused results (RRF): {len(fused)}")

    # 5️⃣ Cross-encoder rerank
    reranked = rerank_cross_encoder(fused, query, top_k=top_k)
    logger.success(f"🏁 Reranked top-{top_k} results ready")

    return reranked


# --------------------------------------------------
# MAIN PIPELINE (SIMPLIFIED)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="🔎 Hybrid → RRF → Cross-encoder Rerank Pipeline")
    parser.add_argument("--q", "--query", dest="query", type=str, required=True, help="User query string")
    args = parser.parse_args()

    query = args.query.strip()
    results = retrieve_top_k(query, top_k=TOP_FINAL)

    print(f"\n🏁 Reranked top-{TOP_FINAL}: {len(results)}\n{'='*60}")
    for i, item in enumerate(results, 1):
        node = getattr(item, "node", item)
        metadata = getattr(node, "metadata", {})
        text = getattr(node, "text", str(node))

        print(f"\n⭐ Final Rank {i}")
        print(f" 📄 Source : {metadata.get('source', 'N/A')}")
        print(f" 📑 Page   : {metadata.get('page', 'N/A')}")
        print(f" 🔢 Score  : {getattr(item, 'score', 'N/A')}")
        print(f" 🧩 Chunk  : {metadata.get('chunk_id', 'N/A')}")
        print(f" 🗒️ Text   : {text[:400]}...")
        print("-" * 60)

    logger.info("✅ Retrieval pipeline completed successfully.")


if __name__ == "__main__":
    main()
