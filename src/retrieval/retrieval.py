"""
---------------------------------------------------
🔎 HYBRID → RRF → CROSS-ENCODER RERANK PIPELINE
(LlamaIndex + Chroma + Ollama, optional LangChain)

1. Hybrid retrieval = BM25 (lexical) + Chroma (semantic).
2. Reciprocal Rank Fusion (RRF) to merge the two result lists.
3. Cross-encoder rerank with BAAI/bge-reranker-large (or SentenceTransformer fallback).

---------------------------------------------------
Run:
    poetry run python -m src.retrieval.retrieval --q "How to run repeated measures ANOVA in JASP?"
  
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
BOOST_WEIGHT = 5
RRF_K = 120
TOP_AFTER_RRF = 10
TOP_FINAL = 5
SCORE_THRESHOLD = -3 

BGE_RERANKER = "BAAI/bge-reranker-large"
ST_CE_FALLBACK = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL = "mistral:7b"


# --------------------------------------------------
# LOAD DOCUMENTS FOR BM25
# --------------------------------------------------

def get_node_metadata(node) -> Dict:
    """
    Robustly extract and normalize metadata from a LlamaIndex Node / Document / NodeWithScore.
    Ensures PDF metadata includes: source_type, source_url, page.
    """

    # ------------------------------
    # 1. Unwrap NodeWithScore
    # ------------------------------
    if hasattr(node, "node"):
        node = node.node

    # ------------------------------
    # 2. Extract metadata from Node
    # ------------------------------
    meta = getattr(node, "metadata", None)
    if meta:
        meta = dict(meta)
    else:
        meta = {}

    # fallback for older versions (extra_info)
    extra = getattr(node, "extra_info", None)
    if extra:
        meta.update(extra)

    # fallback for nodes with source_node metadata
    source_node = getattr(node, "source_node", None)
    if source_node is not None:
        src_meta = getattr(source_node, "metadata", {}) or {}
        meta.update(src_meta)

    # ------------------------------
    # 3. Normalize metadata fields
    # ------------------------------

    # Ensure a source type exists
    if meta.get("pdf_name"):
        meta["source_type"] = "pdf"
    else:
        meta["source_type"] = meta.get("source_type", "document")

    # Page number normalization
    meta["page"] = (
        meta.get("page_start")
        or meta.get("page")
        or meta.get("page_number")
        or None
    )

    # Ensure PDF URL is preserved
    if "source_url" in meta:
        meta["source_url"] = meta["source_url"]

    # Always return a flat dict
    return meta




# -----------------------------------------------------------
# METADATA NORMALIZATION
# -----------------------------------------------------------
def normalize_metadata(node_item: NodeWithScore, rank: int) -> Dict:
    """
    Unify metadata into a frontend-friendly structure:
    
    Returns keys:
      - source_type: pdf | video | markdown
      - pdf_id, page_number, total_pages
      - video_link, second_offset
      - markdown_file, repo_url
      - text (chunk)
      - score
    """

    node = getattr(node_item, "node", node_item)
    meta = getattr(node, "metadata", {}) or {}
    text = node.get_content()[:1000]
    score = getattr(node_item, "score", None)

    source_type = meta.get("source_type", "").lower()


    # ---- PDF ----
    if "pdf" in source_type:
        # Convert page_number safely
        raw_page = meta.get("page_start") or meta.get("page") or 1
        try:
            page_number = int(raw_page)
        except:
            page_number = 1  # fallback: page 1

        # Convert total_pages safely
        raw_total = meta.get("total_pages") or 1
        try:
            total_pages = int(raw_total)
        except:
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

        # Extract raw start_time from metadata (e.g. "0:20", "1:23", "00:02:15")
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
            "section": meta.get("chapter_title") or meta.get("start_time") or meta.get("source_type"),
            "video_link": meta.get("url"),
            "timestamp": start_time_str,         # Human friendly "0:20"
            "second_offset": second_offset,      # Numeric 20
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


# -----------------------------------------------------------
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

    all_docs = []

    for json_path in json_files:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # ✅ Handle dict-wrapped structure
        if isinstance(data, dict) and "sections" in data:
            data = data["sections"]

        # ✅ Handle stringified dicts (double-encoded)
        if isinstance(data, list) and all(isinstance(x, str) for x in data):
            data = [json.loads(x) for x in data]

        source_name = Path(json_path).stem
        for item in data:
            if not isinstance(item, dict) or "text" not in item:
                logger.warning(f"⚠️ Skipping malformed item in {json_path}")
                continue

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

            # ✅ Add robust metadata mapping
            flat_md["pdf_name"] = md.get("pdf_name") or source_name
            flat_md["page_start"] = md.get("page_start") or md.get("page") or None
            flat_md["section_title"] = md.get("section_title") or md.get("title") or "Unknown section"
            flat_md["section_id"] = md.get("section_id") or f"{source_name}_chunk_{item.get('id', len(all_docs))}"

            # Ensure video metadata is preserved
            flat_md["start_time"] = md.get("start_time")
            flat_md["end_time"] = md.get("end_time")
            flat_md["video_title"] = md.get("video_title")
            flat_md["chapter_title"] = md.get("chapter_title")
            flat_md["url"] = md.get("url")   # important!!
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

            # Simple keyword-based boosting
            if any(kw in section_title or kw in video_title for kw in q_lower.split()):
                node.score += self.boost_weight
                logger.debug(f"⚡ Boosted score by {self.boost_weight} for match in '{section_title or video_title}'")

            boosted_results.append(node)

        # Re-sort by new scores
        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results



def build_bm25_retriever(docs: List[Document]):
    """Build a BM25 retriever with optional metadata boost."""
    if BM25Retriever is None:
        raise RuntimeError("BM25Retriever not available. Install llama-index with bm25 extras.")
    logger.info(f"Initializing BM25 retriever with {len(docs)} documents...")

    base_retriever = BM25Retriever(docs, similarity_top_k=K_BM25)
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
    """Apply cross-encoder reranking (BGE preferred) with positive-score filtering."""

    if not nodes:
        logger.warning("⚠️ No retrieved nodes to rerank.")
        return []

    results = []
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

    # ✅ Filter out nodes with non-positive scores
    filtered = [n for n in results if getattr(n, "score", -99) > SCORE_THRESHOLD]
    logger.info(
        f"📊 Retained {len(filtered)}/{len(results)} nodes after score filtering (score > {SCORE_THRESHOLD})"
    )

    return filtered





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
    for i, n in enumerate(reranked):
        logger.debug(f"📌 RAW {i+1}: {n.node.metadata}")


    return reranked


def retrieve_clean(query: str, top_k: int = TOP_FINAL) -> List[Dict]:
    nodes = retrieve_top_k(query, top_k)

    clean_list = []
    for i, node in enumerate(nodes, start=1):
        clean_list.append(normalize_metadata(node, i))
        logger.debug(f"📦 retrieve_clean[{i}]: {clean_list[-1]}")

    return clean_list

# --------------------------------------------------
# MAIN PIPELINE (SIMPLIFIED)
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="🔎 Hybrid → RRF → Cross-encoder Rerank Pipeline")
    parser.add_argument("--q", "--query", dest="query", type=str, required=True, help="User query string")
    args = parser.parse_args()

    query = args.query.strip()
    results = retrieve_top_k(query, top_k=TOP_FINAL)

    for i, item in enumerate(results, 1):
        node = getattr(item, "node", item)
        metadata = get_node_metadata(node)
        
        text = node.get_content() if hasattr(node, "get_content") else getattr(node, "text", str(node))

        # robust page extraction
        page = (
            metadata.get("page")
            or metadata.get("page_start")
            or metadata.get("page_number")
            or (
                metadata.get("section_id", "").split("_PAGE_")[-1]
                if "PAGE_" in metadata.get("section_id", "")
                else None
            )
            or "?"
        )

        print(f"\n⭐ Final Rank {i}")
        print(f" 📄 Source : {metadata.get('source', metadata.get('pdf_name', 'N/A'))}")
        print(f" 📑 Page   : {page}")
        print(f" 🔢 Score  : {getattr(item, 'score', 'N/A')}")
        print(f" 🧩 Chunk  : {metadata.get('chunk_id', metadata.get('section_id', 'N/A'))}")
        print(f" 🗒️ Text   : {text[:400]}...")
        print("-" * 60)
    logger.info("✅ Retrieval pipeline completed successfully.")


if __name__ == "__main__":
    main()
