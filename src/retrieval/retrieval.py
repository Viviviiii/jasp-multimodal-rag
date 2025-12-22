
"""
---------------------------------------------------
4:
 Hybrid retrieval pipeline (BM25 + vectors + rerank)
---------------------------------------------------

This module implements the core retrieval logic for the JASP RAG system.
It combines:

  • BM25 keyword search over all chunked documents
  • semantic vector retrieval from the ChromaDB store
  • Reciprocal Rank Fusion (RRF) for hybrid ranking
  • optional cross-encoder reranking (BGE or MiniLM)

Data sources (all unified as "sections" JSON → chunks):
  • PDF manuals (page-level sections)
  • GitHub Markdown help files
  • YouTube video transcripts (with timestamps)


Supported retrieval modes
-------------------------
1. `bm25`
   Pure BM25 keyword retrieval (no vectors, no fusion, no rerank).

2. `vector`
   Pure semantic / vector retrieval from ChromaDB.

3. `bm25_vector`
   Simple mix: k/2 from BM25 + k/2 from vector, merged and deduplicated.

4. `bm25_vector_fusion`
   BM25 (with metadata boosting) + vector → fused with RRF (no cross-encoder).

5. `bm25_vector_fusion_rerank`  (default, full pipeline)
   Same as (4), then reranked by a cross-encoder (BGE or MiniLM fallback).

CLI usage
---------
Quick manual test from the terminal:

    poetry run python -m src.retrieval.retrieval \\
        --q "How to run repeated measures ANOVA in JASP?" \\
        --mode bm25_vector_fusion_rerank

In code, the main entrypoints are:

    from src.retrieval.retrieval import retrieve_top_k, retrieve_clean

    nodes = retrieve_top_k("your query", mode="bm25_vector_fusion_rerank")
    clean = retrieve_clean("your query", mode="bm25_vector_fusion_rerank")

---------------------------------------------------
"""


from __future__ import annotations
import os
import json
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict, Union
from loguru import logger
import chromadb
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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

K_SEMANTIC = 20
K_BM25 = 20
BOOST_WEIGHT = 5
RRF_K = 120
TOP_AFTER_RRF = 10
TOP_FINAL = 5
SCORE_THRESHOLD = -3

BGE_RERANKER = "BAAI/bge-reranker-large"
ST_CE_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "mistral:7b"

# --------------------------------------------------
# Helper: retrieval modes
# --------------------------------------------------

RETRIEVAL_MODES = {
    "bm25",
    "vector",
    "bm25_vector",
    "bm25_vector_fusion",
    "bm25_vector_fusion_rerank",
}


# --------------------------------------------------
# LOAD DOCUMENTS FOR BM25
# --------------------------------------------------

def get_node_metadata(node) -> Dict:
    """
    Robustly extract and normalize metadata from a LlamaIndex Node / Document / NodeWithScore.
    Ensures PDF metadata includes: source_type, source_url, page.
    """
    if hasattr(node, "node"):
        node = node.node

    meta = getattr(node, "metadata", None)
    if meta:
        meta = dict(meta)
    else:
        meta = {}

    # extra_info is deprecated in newer LlamaIndex; most modern nodes use metadata only.
    extra = getattr(node, "extra_info", None)
    if extra and not meta:
        # Only fall back if metadata is empty, to reduce noise
        meta.update(extra)

    source_node = getattr(node, "source_node", None)
    if source_node is not None:
        src_meta = getattr(source_node, "metadata", {}) or {}
        meta.update(src_meta)

    if meta.get("pdf_name"):
        meta["source_type"] = "pdf"
    else:
        meta["source_type"] = meta.get("source_type", "document")

    meta["page"] = (
        meta.get("page_start")
        or meta.get("page")
        or meta.get("page_number")
        or None
    )

    if "source_url" in meta:
        meta["source_url"] = meta["source_url"]

    return meta

# -----------------------------------------------------------
# METADATA NORMALIZATION
# -----------------------------------------------------------

def normalize_metadata(node_item: NodeWithScore, rank: int) -> Dict:
    """
    Unify metadata into a frontend-friendly structure.
    """
    node = getattr(node_item, "node", node_item)
    meta = getattr(node, "metadata", {}) or {}
    text = node.get_content()[:1000]
    score = getattr(node_item, "score", None)

    source_type = meta.get("source_type", "").lower()

    # ---- PDF ----
    if "pdf" in source_type:
        raw_page = meta.get("page_start") or meta.get("page") or 1
        try:
            page_number = int(raw_page)
        except Exception:
            page_number = 1

        raw_total = meta.get("total_pages") or 1
        try:
            total_pages = int(raw_total)
        except Exception:
            total_pages = 1

        return {
            "source_url": meta.get("source_url"),
            "rank": rank,
            "source_type": "pdf",
            "pdf_id": meta.get("pdf_name", "").replace(".pdf", ""),
            "page_number": page_number,
            "total_pages": total_pages,
            "title": meta.get("section_title"),
            "score": score,
            "content": text,
        }

    # ---- VIDEO ----
    if "video" in source_type:

        start_time_str = meta.get("start_time")

        def convert_to_seconds(ts):
            if not ts:
                return None
            parts = ts.split(":")
            parts = list(map(int, parts))

            if len(parts) == 3:
                h, m, s = parts
            elif len(parts) == 2:
                h = 0
                m, s = parts
            elif len(parts) == 1:
                h = 0
                m = 0
                s = parts[0]
            else:
                return None

            return h * 3600 + m * 60 + s

        second_offset = convert_to_seconds(start_time_str)

        return {
            "rank": rank,
            "source_type": "video",
            "title": meta.get("video_title"),
            "section": meta.get("chapter_title") or meta.get("start_time") or "video description",
            "video_link": meta.get("url"),
            "timestamp": start_time_str,
            "second_offset": second_offset,
            "score": score,
            "content": text,
        }

    # ---- MARKDOWN / GITHUB ----
    if "markdown" in source_type or "github" in source_type:
        return {
            "rank": rank,
            "source_type": "markdown",
            "title": meta.get("markdown_file") or meta.get("md_name"),
            "section": meta.get("section_title"),
            "source_url": meta.get("md_url"),
            "score": score,
            "content": text,
        }

    # fallback
    return {
        "source_url": meta.get("source_url"),
        "rank": rank,
        "source_type": "text",
        "title": meta.get("section_title"),
        "score": score,
        "content": text,
    }

# --------------------------------------------------
# LOAD DOCUMENTS FOR BM25 (Supports folder or list)
# --------------------------------------------------

def load_docs_for_bm25(input_path: Union[str, list[str]]) -> List[Document]:
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

    all_docs: List[Document] = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "sections" in data:
            data = data["sections"]

        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            data = [json.loads(x) for x in data]

        source_name = Path(json_path).stem
        for item in data:
            if not isinstance(item, dict) or "text" not in item:
                logger.warning(f"⚠️ Skipping malformed item in {json_path}")
                continue

            md = item.get("metadata", {})
            flat_md: Dict[str, str] = {}
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

            flat_md["pdf_name"] = md.get("pdf_name") or source_name
            flat_md["page_start"] = md.get("page_start") or md.get("page") or None
            flat_md["section_title"] = md.get("section_title") or md.get("title") or "Unknown section"
            flat_md["section_id"] = md.get("section_id") or f"{source_name}_chunk_{item.get('id', len(all_docs))}"

            flat_md["start_time"] = md.get("start_time")
            flat_md["end_time"] = md.get("end_time")
            flat_md["video_title"] = md.get("video_title")
            flat_md["chapter_title"] = md.get("chapter_title")
            flat_md["url"] = md.get("url")
            flat_md["source_type"] = md.get("source_type")

            all_docs.append(Document(text=item["text"], metadata=flat_md))

        logger.info(f"📄 Loaded {len(all_docs)} chunks from {json_path}")

    logger.success(f"✅ Total BM25 corpus size: {len(all_docs)} chunks from {len(json_files)} files.")
    return all_docs

# --------------------------------------------------
# BUILD RETRIEVERS
# --------------------------------------------------

class BoostedBM25Retriever:
    """BM25 retriever with metadata-based score boosting."""
    def __init__(self, retriever: BM25Retriever, boost_weight: float = 0.2):
        self.retriever = retriever
        self.boost_weight = boost_weight

    def retrieve(self, query: str):
        results = self.retriever.retrieve(query)
        boosted_results = []

        for node in results:
            meta = node.metadata or {}
            section_title = str(meta.get("section_title", "")).lower()
            video_title = str(meta.get("title", "")).lower()
            q_lower = query.lower()

            if any(kw in section_title or kw in video_title for kw in q_lower.split()):
                node.score += self.boost_weight
                logger.debug(f"⚡ Boosted score by {self.boost_weight} for match in '{section_title or video_title}'")

            boosted_results.append(node)

        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results

def build_bm25_retriever(docs: List[Document], use_boost: bool = True):
    """
    Build a BM25 retriever, optionally wrapped with metadata-based boosting.

    use_boost = False → pure BM25 (for baseline evaluation)
    use_boost = True  → BM25 + soft filtering via metadata boosting
    """
    if BM25Retriever is None:
        raise RuntimeError("BM25Retriever not available. Install llama-index with bm25 extras.")

    logger.info(f"Initializing BM25 retriever with {len(docs)} documents...")

    base_retriever = BM25Retriever(docs, similarity_top_k=K_BM25)
    if not use_boost:
        logger.info("🔹 Using pure BM25 (no metadata boosting).")
        return base_retriever

    logger.info(f"🔹 Using BoostedBM25Retriever with boost_weight={BOOST_WEIGHT}.")
    boosted_retriever = BoostedBM25Retriever(base_retriever, boost_weight=BOOST_WEIGHT)
    return boosted_retriever

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
    """Generate a stable deduplication key for RRF fusion / unions."""
    node = n.node
    text = node.get_content().strip()
    meta = node.metadata or {}
    id_hint = meta.get("doc_id") or meta.get("source") or meta.get("file") or ""
    key_data = f"{id_hint}::{text}"
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

    fused_nodes = []
    for key, fused_score in fused_sorted:
        n = lookup[key]
        # 🔹 overwrite score with RRF score so downstream prints & clean output see it
        n.score = fused_score
        fused_nodes.append(n)

    return fused_nodes

# --------------------------------------------------
# CROSS-ENCODER RERANK
# --------------------------------------------------
#version 1:filtered
def rerank_cross_encoder_filtered(nodes: List[NodeWithScore], query: str, top_k: int = TOP_FINAL) -> List[NodeWithScore]:
    """Apply cross-encoder reranking (BGE preferred) with threshold filtering."""
    if not nodes:
        logger.warning("⚠️ No retrieved nodes to rerank.")
        return []

    results: List[NodeWithScore] = []
    if FlagEmbeddingReranker is not None:
        logger.info(f"⚙️ Cross-encoder rerank with BGE model: {BGE_RERANKER}")
        reranker = FlagEmbeddingReranker(
            model=BGE_RERANKER,
            top_n=top_k,
            use_fp16=True,
        )
        results = reranker.postprocess_nodes(nodes, query_str=query)
        logger.success("✅ Reranking completed with FlagEmbeddingReranker")

    elif SentenceTransformerRerank is not None:
        logger.info(f"⚙️ Cross-encoder rerank with SentenceTransformer fallback: {ST_CE_FALLBACK}")
        reranker = SentenceTransformerRerank(model=ST_CE_FALLBACK, top_n=top_k)
        results = reranker.postprocess_nodes(nodes, query_str=query)

    else:
        logger.warning("⚠️ No cross-encoder available — skipping rerank.")
        results = nodes[:top_k]

    filtered = [n for n in results if getattr(n, "score", -99) > SCORE_THRESHOLD]
    logger.info(
        f"📊 Retained {len(filtered)}/{len(results)} nodes after score filtering (score > {SCORE_THRESHOLD})"
    )

    return filtered

#version 2:no filter, used in protocol
def rerank_cross_encoder(nodes: List[NodeWithScore], query: str, top_k: int = TOP_FINAL) -> List[NodeWithScore]:
    """Apply cross-encoder reranking (BGE preferred). Always return top_k results (no threshold filtering)."""
    if not nodes:
        logger.warning("⚠️ No retrieved nodes to rerank.")
        return []

    results: List[NodeWithScore] = []
    if FlagEmbeddingReranker is not None:
        logger.info(f"⚙️ Cross-encoder rerank with BGE model: {BGE_RERANKER}")
        reranker = FlagEmbeddingReranker(
            model=BGE_RERANKER,
            top_n=top_k,
            use_fp16=True,
        )
        results = reranker.postprocess_nodes(nodes, query_str=query)
        logger.success("✅ Reranking completed with FlagEmbeddingReranker")

    elif SentenceTransformerRerank is not None:
        logger.info(f"⚙️ Cross-encoder rerank with SentenceTransformer fallback: {ST_CE_FALLBACK}")
        reranker = SentenceTransformerRerank(model=ST_CE_FALLBACK, top_n=top_k)
        results = reranker.postprocess_nodes(nodes, query_str=query)

    else:
        logger.warning("⚠️ No cross-encoder available — skipping rerank.")
        results = nodes[:top_k]

    # --- minimal change: remove score threshold filtering ---
    logger.info(f"📊 Returning top {len(results)} nodes (no score threshold applied)")

    return results[:top_k]   # ensure still exactly top_k


# helper for FastAPI backend to avoid rebuilding everything every request.
from functools import lru_cache

@lru_cache(maxsize=1)
def get_bm25_docs():
    return tuple(load_docs_for_bm25(CHUNKS_JSON))

@lru_cache(maxsize=1)
def get_bm25_retriever(use_boost: bool):
    docs = list(get_bm25_docs())  # convert back to list
    return build_bm25_retriever(docs, use_boost=use_boost)

@lru_cache(maxsize=1)
def get_vector_retriever():
    return build_vector_retriever()

# --------------------------------------------------
# RETRIEVAL WRAPPER FUNCTION WITH MODES
# --------------------------------------------------
def retrieve_top_k(
    query: str,
    top_k: int = TOP_FINAL,
    mode: str = "bm25_vector_fusion_rerank",
):
    """
    Run retrieval in one of the supported hybrid modes and return raw Nodes.

    Modes:
        "bm25"
            • Pure BM25 keyword retrieval.
            • Uses the BM25 corpus loaded from chunked JSON files.
            • No vector search, no fusion, no reranker.

        "vector"
            • Pure semantic/vector retrieval.
            • Uses embeddings stored in the ChromaDB collection.
            • No BM25, no fusion, no reranker.

        "bm25_vector"
            • Simple combination: k/2 results from BM25 + k/2 from vector.
            • Both lists are deduplicated based on text/ID.
            • No RRF fusion, no cross-encoder.

        "bm25_vector_fusion"
            • Hybrid retrieval:
                - BM25 (with optional metadata-based boosting)
                - semantic/vector retriever
              → fused via Reciprocal Rank Fusion (RRF).
            • No cross-encoder reranking.

        "bm25_vector_fusion_rerank"
            • Full pipeline (default):
                1) BM25 (+ boosting) + semantic retriever
                2) RRF fusion of both ranked lists
                3) Cross-encoder reranking (BGE reranker if available,
                   otherwise MiniLM fallback)
            

    Args:
        query:
            Natural language query string.

        top_k:
            Number of final results to return (after fusion/rerank if applicable).

        mode:
            Retrieval mode name. If an unknown mode is passed, it falls back
            to `"bm25_vector_fusion_rerank"`.

    Returns:
        A list of `NodeWithScore` objects (LlamaIndex), which can be passed
        to `normalize_metadata(...)` or to `retrieve_clean(...)` to obtain
        a frontend-friendly dict representation.
    """

    mode = mode.lower().strip()
    if mode not in RETRIEVAL_MODES:
        logger.warning(f"Unknown mode '{mode}', falling back to 'bm25_vector_fusion_rerank'.")
        mode = "bm25_vector_fusion_rerank"

    logger.info(f"🔍 Starting retrieval pipeline for query: '{query}' (mode={mode})")

    # Special-case: vector-only mode (no BM25 corpus load needed)
    if mode == "vector":
        logger.info("🚩 Mode 'vector': pure semantic/vector retrieval only.")
        semantic_retriever = get_vector_retriever()
        sem_results = semantic_retriever.retrieve(query)
        logger.success(f"Semantic: {len(sem_results)} results")
        final = sem_results[:top_k]
        for i, n in enumerate(final):
            logger.debug(f"📌 Vector-only {i+1}: {n.node.metadata}")
        return final

    # 1️⃣ For all other modes, decide boosting + get retrievers
    use_boost = mode in {"bm25_vector_fusion", "bm25_vector_fusion_rerank"}

    bm25_retriever = get_bm25_retriever(use_boost=use_boost)
    semantic_retriever = get_vector_retriever()




    # ---------- MODE 1: pure BM25 ----------
    if mode == "bm25":
        logger.info("🚩 Mode 'bm25': pure BM25 only.")
        bm25_results = bm25_retriever.retrieve(query)
        logger.success(f"BM25: {len(bm25_results)} results")
        final = bm25_results[:top_k]
        for i, n in enumerate(final):
            logger.debug(f"📌 BM25-only {i+1}: {n.node.metadata}")
        return final

    # From here on, we need semantic retriever as well
    semantic_retriever = build_vector_retriever()

    logger.info("⚙️ Retrieving from BM25 and semantic retrievers...")

    bm25_results = bm25_retriever.retrieve(query)
    for n in bm25_results:
        n.node.metadata["retriever"] = "bm25"
    sem_results = semantic_retriever.retrieve(query)
    for n in sem_results:
        n.node.metadata["retriever"] = "vector"
    logger.success(f"BM25: {len(bm25_results)} | Semantic: {len(sem_results)}")


    # ---------- MODE 3: BM25 + vectors (balanced k/2 from each, de-duplicated) ----------
    if mode == "bm25_vector":
        logger.info("🚩 Mode 'bm25_vector': k/2 from BM25 + k/2 from vector (no fusion, no rerank).")

        # how many from each
        k_bm25 = math.ceil(top_k / 2)
        k_vec = top_k - k_bm25  # handles odd k

        # cap by what we actually have
        bm25_results = bm25_retriever.retrieve(query)
        sem_results = semantic_retriever.retrieve(query)

        bm25_slice = bm25_results[: min(k_bm25, len(bm25_results))]
        sem_slice = sem_results[: min(k_vec, len(sem_results))]

        # optional: tag origin (nice for debugging / thesis)
        for n in bm25_slice:
            n.node.metadata["retriever"] = "bm25"
        for n in sem_slice:
            n.node.metadata["retriever"] = "vector"

        # union with de-duplication
        seen = set()
        combined: List[NodeWithScore] = []

        def add_unique(lst):
            for n in lst:
                key = stable_node_key(n)
                if key not in seen:
                    seen.add(key)
                    combined.append(n)

        # BM25 first, then vector
        add_unique(bm25_slice)
        add_unique(sem_slice)

        final = combined[:top_k]

        logger.success(
            f"✅ bm25_vector: requested top_k={top_k} → "
            f"{len(bm25_slice)} from BM25, {len(sem_slice)} from vector, "
            f"{len(final)} after de-dup."
        )

        for i, n in enumerate(final, start=1):
            logger.debug(
                f"📌 BM25+vector {i}: "
                f"retriever={n.node.metadata.get('retriever')} | meta={n.node.metadata}"
            )

        return final




    # ---------- MODE 4 & 5: Hybrid with RRF fusion ----------
    fused = rrf_fuse(sem_results, bm25_results)
    logger.success(f"🔗 Fused results (RRF): {len(fused)}")

    if mode == "bm25_vector_fusion":
        logger.info("🚩 Mode 'bm25_vector_fusion': RRF fusion only (no cross-encoder).")
        final = fused[:top_k]
        for i, n in enumerate(final):
            logger.debug(f"📌 Fused-only {i+1}: {n.node.metadata}")
        return final

    # ---------- MODE 5: Hybrid + RRF + cross-encoder ----------
    logger.info("🚩 Mode 'bm25_vector_fusion_rerank': RRF + cross-encoder rerank.")
    reranked = rerank_cross_encoder(fused, query, top_k=top_k)
    logger.success(f"🏁 Reranked top-{top_k} results ready")
    
    for i, n in enumerate(reranked):
        logger.debug(f"📌 RAW {i+1}: {n.node.metadata}")
    return reranked

def retrieve_clean(
    query: str,
    top_k: int = TOP_FINAL,
    mode: str = "bm25_vector_fusion_rerank",
) -> List[Dict]:
    nodes = retrieve_top_k(query, top_k=top_k, mode=mode)

    clean_list: List[Dict] = []
    for i, node in enumerate(nodes, start=1):
        clean_list.append(normalize_metadata(node, i))
        logger.debug(f"📦 retrieve_clean[{i}]: {clean_list[-1]}")

    return clean_list



# --------------------------------------------------
# MAIN PIPELINE (SIMPLIFIED)
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="🔎 Hybrid / BM25 / Vector retrieval CLI")
    parser.add_argument("--q", "--query", dest="query", type=str, required=True, help="User query string")
    parser.add_argument(
        "--mode",
        type=str,
        default="bm25_vector_fusion_rerank",
        choices=sorted(RETRIEVAL_MODES),
        help="Retrieval mode: bm25 | vector | bm25_vector | bm25_vector_fusion | bm25_vector_fusion_rerank",
    )
    args = parser.parse_args()

    query = args.query.strip()
    mode = args.mode.strip()

    results = retrieve_top_k(query, top_k=TOP_FINAL, mode=mode)

    print(f"\n=== Retrieval mode: {mode} ===")
    for i, item in enumerate(results, 1):
        clean = normalize_metadata(item, rank=i)
        source_type = clean.get("source_type", "text")

        print(f"\n⭐ Rank {i} [{source_type}]")

        # ---------- PDF ----------
        if source_type == "pdf":
            pdf_id = clean.get("pdf_id") or "Unknown PDF"
            page = clean.get("page_number", "?")
            total_pages = clean.get("total_pages", "?")
            title = clean.get("title") or "(no section title)"
            url = clean.get("source_url") or "N/A"
            score = clean.get("score", "N/A")

            print(f" 📄 PDF      : {pdf_id}")
            print(f" 🧩 Section  : {title}")
            print(f" 📑 Page     : {page}/{total_pages}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- VIDEO ----------
        elif source_type == "video":
            title = clean.get("title") or "(no video title)"
            section = clean.get("section") or "(no chapter)"
            timestamp = clean.get("timestamp") or "N/A"
            link = clean.get("video_link") or "N/A"
            score = clean.get("score", "N/A")

            print(f" 🎥 Video    : {title}")
            print(f" 🧩 Section  : {section}")
            print(f" ⏱️ Time     : {timestamp}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {link}")

        # ---------- MARKDOWN / GITHUB ----------
        elif source_type == "markdown":
            title = clean.get("title") or "(no file name)"
            section = clean.get("section") or "(no section title)"
            url = clean.get("source_url") or "N/A"
            score = clean.get("score", "N/A")

            print(f" 📘 Markdown : {title}")
            print(f" 🧩 Section  : {section}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- Fallback ----------
        else:
            title = clean.get("title") or "(no title)"
            url = clean.get("source_url") or "N/A"
            score = clean.get("score", "N/A")

            print(f" 📄 Source   : {title}")
            print(f" 🔢 Score    : {score}")
            print(f" 🔗 URL      : {url}")

        # ---------- Common: text preview ----------
        snippet = clean.get("content", "")
        snippet = (snippet[:400] + "...") if len(snippet) > 400 else snippet
        print(f" 🗒️ Text     : {snippet}")
        print("-" * 60)

    logger.info("✅ Retrieval pipeline completed successfully.")

if __name__ == "__main__":
    main()
