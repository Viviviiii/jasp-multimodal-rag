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
