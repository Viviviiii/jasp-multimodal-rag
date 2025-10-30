"""
---------------------------------------------------
🧠 EMBEDDING STORE PIPELINE v2 (LLAMAINDEX + CHROMA)
---------------------------------------------------
Implements the modern, modular embedding and storage
workflow recommended by LlamaIndex. Supports metadata
filtering and hybrid retrieval integration.
---------------------------------------------------
Run:
    poetry run python -m src.embedding.embedding_store
---------------------------------------------------
"""

import os
import json
from loguru import logger
import chromadb
from pathlib import Path

# LlamaIndex core imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters


# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
CHROMA_DIR = "data/vector_store/chroma_db"
COLLECTION_NAME = "jasp_manual_embeddings_v2"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_JSON_PATH = "data/processed/chunks/chunks_test_pages25-28.json"


# ---------------------------------------------------
# LOAD & FLATTEN METADATA
# ---------------------------------------------------
def load_chunks(json_path: str = DEFAULT_JSON_PATH) -> list[Document]:
    """Load enriched chunked data and prepare flat metadata."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Chunk JSON file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for item in data:
        metadata = item.get("metadata", {})
        flat_metadata = {}

        # ✅ Flatten metadata for Chroma compatibility
        for k, v in metadata.items():
            if isinstance(v, list):
                flat_metadata[k] = ", ".join(map(str, v))
            elif isinstance(v, dict):
                flat_metadata[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat_metadata[k] = v

        docs.append(Document(text=item["text"], metadata=flat_metadata))

    logger.info(f"📄 Loaded {len(docs)} enriched chunks with flattened metadata")
    return docs


# ---------------------------------------------------
# EMBEDDING + STORAGE PIPELINE
# ---------------------------------------------------
def embed_and_store(
    docs: list[Document],
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embed_model_name: str = EMBED_MODEL,
):
    """Embed documents and store persistently in ChromaDB."""
    logger.info("🚀 Initializing embedding model and Chroma vector store...")

    # 1️⃣ Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # 2️⃣ Persistent local Chroma client and collection
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    # 3️⃣ Wrap collection into a LlamaIndex-compatible vector store
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # 4️⃣ Build and persist index
    index = VectorStoreIndex.from_documents(
        docs,
        embed_model=embed_model,
        storage_context=storage_context,
    )

    storage_context.persist(persist_dir=persist_dir)
    logger.success(f"✅ Stored embeddings in ChromaDB at {persist_dir}")

    # 5️⃣ Verify retrieval with metadata-based filtering (version-safe)
    filters = MetadataFilters(
        filters=[MetadataFilter(key="source", value="test_pages25-28.pdf")]
    )
    retriever = index.as_retriever(filters=filters)

    sample_query = "How to split data files in JASP?"
    results = retriever.retrieve(sample_query)

    logger.info(f"🔍 Retrieved {len(results)} results for sample query")
    for r in results[:3]:
        logger.info(f"- Score: {r.score:.4f} | Metadata: {r.metadata}")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------
def main():
    docs = load_chunks()
    embed_and_store(docs)
    logger.info("🎯 Embedding store v2 completed successfully.")


if __name__ == "__main__":
    main()
