import os
from typing import List, Dict
import chromadb
from chromadb import PersistentClient

class ChromaIndex:
    def __init__(self, persist_dir: str, batch_size_text: int = 128, batch_size_images: int = 64):
        os.makedirs(persist_dir, exist_ok=True)
        self.client: PersistentClient = chromadb.PersistentClient(path=persist_dir)
        self.text = self.client.get_or_create_collection("jasp_text")
        self.images = self.client.get_or_create_collection("jasp_images")
        self.batch_size_text = batch_size_text
        self.batch_size_images = batch_size_images

    def reset_collections(self):
        """Delete + recreate collections for a clean slate."""
        for name in ("jasp_text", "jasp_images"):
            try:
                self.client.delete_collection(name)
            except Exception:
                pass
        self.text = self.client.get_or_create_collection("jasp_text")
        self.images = self.client.get_or_create_collection("jasp_images")

    def _batch_iter(self, items: List, batch_size: int):
        """Yield slices of size batch_size (last may be shorter)."""
        for i in range(0, len(items), batch_size):
            yield items[i:i + batch_size]

    def upsert_text_chunks(self, docs: List[str], embs: List[List[float]], metas: List[Dict], ids: List[str]):
        """Upsert text chunks into Chroma in batches (default=128)."""
        for i, batch in enumerate(self._batch_iter(list(zip(docs, embs, metas, ids)), self.batch_size_text), start=1):
            d, e, m, idx = zip(*batch)
            self.text.add(documents=list(d), embeddings=list(e), metadatas=list(m), ids=list(idx))

    def upsert_images(self, docs: List[str], embs: List[List[float]], metas: List[Dict], ids: List[str]):
        """Upsert image embeddings into Chroma in batches (default=64)."""
        for i, batch in enumerate(self._batch_iter(list(zip(docs, embs, metas, ids)), self.batch_size_images), start=1):
            d, e, m, idx = zip(*batch)
            self.images.add(documents=list(d), embeddings=list(e), metadatas=list(m), ids=list(idx))






import os, hashlib, json, logging
import chromadb
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer
from langchain.schema import Document

class TextIndexer:
    """
    Handles embedding and upserting text Documents into a Chroma collection.
    Reused by PDF, video, and other ingestion pipelines.
    """

    def __init__(self, chroma_dir: str, collection_name: str, model_name: str):
        os.makedirs(chroma_dir, exist_ok=True)
        self.client: PersistentClient = chromadb.PersistentClient(path=chroma_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.model = SentenceTransformer(model_name)

    def _make_id(self, doc: Document, prefix: str) -> str:
        """Stable ID based on content + metadata hash."""
        uid_str = doc.page_content + json.dumps(doc.metadata, sort_keys=True)
        h = hashlib.md5(uid_str.encode("utf-8")).hexdigest()[:12]
        return f"{prefix}:{h}"

    def upsert_documents(self, docs: list[Document], source_prefix: str):
        """Embed and insert docs into Chroma, generating unique IDs."""
        if not docs:
            logging.info("No docs to embed.")
            return

        total = 0
        BATCH = 128
        for group in [docs[i:i + BATCH] for i in range(0, len(docs), BATCH)]:
            texts = [d.page_content for d in group]
            metas = [d.metadata for d in group]
            ids = [self._make_id(d, source_prefix) for d in group]

            embs = self.model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

            self.collection.add(documents=texts, metadatas=metas, embeddings=embs, ids=ids)
            total += len(group)

        logging.info(f"✅ Inserted {total} docs into Chroma collection '{self.collection.name}'.")

    def query(self, question: str, top_k: int = 3):
        """Query collection with a natural language question."""
        q_emb = self.model.encode([question], normalize_embeddings=True, show_progress_bar=False).tolist()
        return self.collection.query(query_embeddings=q_emb, n_results=top_k)


    def as_retriever(self, search_kwargs=None):
        """
        Expose this Chroma collection as a LangChain retriever.
        Useful for hybrid retrieval (dense + BM25).
        """
        try:
            from langchain_chroma import Chroma   # ✅ new package
        except ImportError:
            from langchain.vectorstores import Chroma  # fallback for old LangChain

        store = Chroma(
            client=self.client,
            collection_name=self.collection.name,
            embedding_function=lambda x: self.model.encode(
                x, normalize_embeddings=True
            ).tolist()
        )
        return store.as_retriever(search_kwargs=search_kwargs or {"k": 5})
