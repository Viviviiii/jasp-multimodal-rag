
"""
---------------------------------------------------
3:
Embedding store pipeline (LlamaIndex + ChromaDB)
---------------------------------------------------

This module is the final step before retrieval: it takes all chunked JSON
documents (from `data/processed/chunks/`), embeds their text with a Hugging
Face model, and stores the vectors in a persistent ChromaDB collection.

Input:
    • One or more JSON chunk files, each with:
        { "sections": [ { "text": "...", "metadata": {...} }, ... ] }
      Typically located in:
        data/processed/chunks/

Output:
    • A ChromaDB collection on disk at:
        data/vector_store/chroma_db/
      with all documents embedded and ready for vector retrieval.

Key features
------------
✅ Supports multiple JSON sources (single file, folder, or list of paths)  
✅ Stable document IDs via MD5 hash (prevents duplicates on re-run)  
✅ Skips already-embedded documents by checking existing IDs in Chroma  
✅ Uses `BAAI/bge-large-en-v1.5` (by default) via `HuggingFaceEmbedding`  
✅ Exposes a single `embed_and_store(...)` function and a CLI entrypoint

Run standalone:
    poetry run python -m src.embedding.embedding_store

---------------------------------------------------
"""


import os
import json
import hashlib
from pathlib import Path
from typing import List, Union
from loguru import logger
import chromadb

# LlamaIndex imports
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



# ---------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------
CHROMA_DIR = "data/vector_store/chroma_db"
COLLECTION_NAME = "jasp_manual_embeddings_v2"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_JSON_PATH = "data/processed/chunks"


# ---------------------------------------------------
# HELPERS
# ---------------------------------------------------

def flatten_metadata(metadata: dict) -> dict:
    """Flatten nested metadata structures for Chroma and remove None values."""
    flat = {}
    for k, v in metadata.items():
        if v is None:
            continue  # 🚫 Skip None values entirely
        if isinstance(v, list):
            flat[k] = ", ".join(map(str, v))
        elif isinstance(v, dict):
            flat[k] = json.dumps(v, ensure_ascii=False)
        else:
            flat[k] = str(v)  # ✅ Always store as string for Chroma safety
    return flat




def make_doc_id(text: str, source: str, chunk_id: int) -> str:
    """Generate a reproducible, unique document ID."""
    raw = f"{source}_{chunk_id}_{text}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def load_chunks_from_json(json_path: str) -> List[Document]:
    """Load one JSON file and convert to LlamaIndex Documents with IDs."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ File not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ✅ Handle structure like {"sections": [...]}
    if isinstance(data, dict) and "sections" in data:
        data = data["sections"]

    if not isinstance(data, list):
        raise ValueError(f"Expected a list of chunks in {json_path}, got {type(data)}")

    source_name = Path(json_path).stem
    docs = []
    for idx, item in enumerate(data):
        if isinstance(item, str):
            # fallback safety if a string accidentally appears
            item = {"text": item, "metadata": {}}

        text = item.get("text", "")
        metadata = flatten_metadata(item.get("metadata", {}))
        metadata.update({"source": source_name, "chunk_id": str(idx)})

        doc_id = make_doc_id(text, source_name, idx)
        docs.append(Document(id_=doc_id, text=text, metadata=metadata))

    logger.info(f"📄 Loaded {len(docs)} chunks from {json_path}")
    return docs



def load_all_chunks(input_path: Union[str, list[str]]) -> List[Document]:
    """Load all JSON chunk files from a directory or list of paths."""
    all_docs = []

    if isinstance(input_path, list):
        json_files = input_path
    elif os.path.isdir(input_path):
        json_files = list(Path(input_path).glob("*.json"))
    else:
        json_files = [Path(input_path)]

    if not json_files:
        raise FileNotFoundError(f"❌ No JSON files found in {input_path}")

    for jf in json_files:
        all_docs.extend(load_chunks_from_json(str(jf)))

    logger.success(f"✅ Loaded total {len(all_docs)} chunks from {len(json_files)} source files.")
    return all_docs


# ---------------------------------------------------
# EMBEDDING + STORAGE PIPELINE
# ---------------------------------------------------
def embed_and_store(
    input_path: Union[str, list[str], List[Document]] = DEFAULT_JSON_PATH,
    persist_dir: str = CHROMA_DIR,
    collection_name: str = COLLECTION_NAME,
    embed_model_name: str = EMBED_MODEL,
):
    """
    Incremental-safe embedding + storage pipeline for ChromaDB.

    This function takes chunked documents (from JSON or preloaded
    `Document` objects), embeds their text using a Hugging Face model,
    and stores them in a persistent ChromaDB collection. It is designed
    to be safe to re-run: existing document IDs are detected and skipped,
    so no duplicate embeddings are created.

    Behavior:
        1. Load all chunks as `Document` objects:
             - If `input_path` is a directory → load all `*.json` files.
             - If `input_path` is a file path → load that file only.
             - If `input_path` is a list of paths → load each file.
             - If `input_path` is a list of `Document` objects →
               use them directly.

        2. Initialize the embedding model:
             - Default: `BAAI/bge-large-en-v1.5` via `HuggingFaceEmbedding`.

        3. Connect to (or create) a ChromaDB collection at `persist_dir`
           with name `collection_name`.

        4. Retrieve existing document IDs from Chroma and filter out
           any `Document` whose `id_` is already present.

        5. Embed only the new documents and add:
             - embeddings
             - raw text
             - flattened metadata
             - IDs

        6. Optionally run a small retrieval sanity check to ensure
           the collection is usable by LlamaIndex.

    Args:
        input_path:
            Either:
              - a directory containing chunked JSON files,
              - a single JSON file path,
              - a list of JSON file paths, or
              - a list of pre-constructed `Document` objects.

        persist_dir:
            Filesystem path where the ChromaDB database is stored.

        collection_name:
            Name of the Chroma collection (e.g. `"jasp_manual_embeddings_v2"`).

        embed_model_name:
            Hugging Face model name to use for text embeddings.

    Returns:
        None. Embeddings are written to the ChromaDB collection on disk.
    """


    logger.info("🚀 Initializing embedding model and Chroma vector store...")

    # 1️⃣ Load documents
    if isinstance(input_path, list) and all(isinstance(x, Document) for x in input_path):
        docs = input_path
        logger.info(f"📦 Using {len(docs)} preloaded Document objects.")
    else:
        docs = load_all_chunks(input_path)

    # 2️⃣ Initialize embedding model
    embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

    # 3️⃣ Connect to persistent Chroma store
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    chroma_collection = chroma_client.get_or_create_collection(name=collection_name)

    existing_count = chroma_collection.count()
    logger.info(f"📊 Existing embeddings in collection: {existing_count}")

    # 4️⃣ Get existing IDs directly from Chroma
    try:
        existing_data = chroma_collection.get(include=[])
        existing_ids = set(existing_data["ids"]) if existing_data and "ids" in existing_data else set()
        logger.info(f"🧾 Retrieved {len(existing_ids)} existing IDs from Chroma.")
    except Exception as e:
        logger.warning(f"⚠️ Could not load existing IDs: {e}")
        existing_ids = set()

    # 5️⃣ Filter new docs
    new_docs = [doc for doc in docs if doc.id_ not in existing_ids]
    logger.info(f"🧩 {len(new_docs)} new documents to embed (skipped {len(docs) - len(new_docs)} duplicates).")

    if not new_docs:
        logger.success("🍒 No new documents to embed — all up to date!")
        return

    # 6️⃣ Embed and add to Chroma
    logger.info(f"🔢 Embedding {len(new_docs)} new documents with {embed_model_name}...")

    texts = [doc.text for doc in new_docs]
    metadatas = [doc.metadata for doc in new_docs]
    ids = [doc.id_ for doc in new_docs]

    embeddings = []
    for i, text in enumerate(texts):
        emb = embed_model.get_text_embedding(text)
        embeddings.append(emb)
        if (i + 1) % 50 == 0:
            logger.debug(f"🪶 Embedded {i+1}/{len(new_docs)} chunks...")

    chroma_collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids,
    )

    new_count = chroma_collection.count()
    added = new_count - existing_count

    logger.success(f"✅ Added {added} new embeddings. Total now: {new_count}")

    # 7️⃣ Quick retrieval sanity check
    try:
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core import StorageContext, VectorStoreIndex

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

        sample_query = "How to split data files in JASP?"
        retriever = index.as_retriever()
        results = retriever.retrieve(sample_query)
        logger.info(f"🔍 Test query retrieved {len(results)} results.")
    except Exception as e:
        logger.warning(f"⚠️ Retrieval verification skipped: {e}")



# ---------------------------------------------------
# MAIN (Standalone)
# ---------------------------------------------------
def main():
    logger.info("🏗️ Running standalone embedding pipeline...")
    embed_and_store(DEFAULT_JSON_PATH)
    logger.success("🎯 Embedding store pipeline completed successfully.")


if __name__ == "__main__":
    main()
