import re, random
from typing import Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from ..ingestion.captioner import BLIPCaptioner

def build_page_docs(docs: List[Document], images_by_page: Dict[int, List[str]], captioner: BLIPCaptioner) -> List[Document]:
    """Append [Figure: caption] lines for each page image."""
    out = []
    for doc in docs:
        pg = doc.metadata["page"]
        text = doc.page_content
        captions = []
        if pg in images_by_page:
            for path in images_by_page[pg]:
                cap = captioner.describe(path)
                captions.append(f"[Figure: {cap}]")
        full = text + ("\n\n" + "\n".join(captions) if captions else "")
        out.append(Document(page_content=full, metadata=doc.metadata))
    return out

def hybrid_split(
    docs: List[Document],
    chunk_size: int = 1000,
    overlap: int = 100,
    preview_samples: int = 2,
) -> List[Document]:
    """Keep headers/tables as single chunks; split the rest recursively."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    header_pattern = re.compile(r"^(CHAPTER\b|\d+\.\s+|##|###|Table\s+\d+|Figure\s+\d+)", re.IGNORECASE)

    chunks: List[Document] = []
    for doc in docs:
        meta = {**doc.metadata}
        paragraphs = [p.strip() for p in doc.page_content.split("\n\n") if p.strip()]
        for para in paragraphs:
            if header_pattern.match(para):
                chunks.append(Document(page_content=para, metadata={**meta, "split_type": "header_or_table"}))
            else:
                for sub in splitter.split_text(para):
                    chunks.append(Document(page_content=sub, metadata={**meta, "split_type": "body"}))
    # (Optional) preview handled by caller/logger
    return chunks




# video splitter
from typing import List
from sentence_transformers import SentenceTransformer, util
from langchain_text_splitters import RecursiveCharacterTextSplitter

SEMANTIC_THRESHOLD = 0.70
MAX_CHARS = 1600
OVERLAP_CHARS = 150


def split_transcript_by_chapters(transcript, chapters):
    if not chapters:
        return [{"title": "Full video", "segments": transcript}]
    results = []
    for idx, ch in enumerate(chapters):
        start = ch["start_time"]
        end = ch.get("end_time", chapters[idx+1]["start_time"] if idx+1 < len(chapters) else float("inf"))
        segs = [s for s in transcript if start <= s["start"] < end]
        results.append({"title": ch["title"], "segments": segs})
    return results


def semantic_split(transcript_segments, model: SentenceTransformer, threshold=SEMANTIC_THRESHOLD):
    segs = [s for s in transcript_segments if (s.get("text") or "").strip()]
    if not segs:
        return []
    texts = [s["text"].strip() for s in segs]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    chunks, buffer = [], [texts[0]]
    prev = embs[0]
    for i in range(1, len(texts)):
        sim = float(util.cos_sim(prev, embs[i]))
        if sim < threshold:
            chunks.append(" ".join(buffer))
            buffer = [texts[i]]
        else:
            buffer.append(texts[i])
        prev = embs[i]
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks


def enforce_max_chars(chunks, max_chars=MAX_CHARS, overlap=OVERLAP_CHARS):
    splitter = RecursiveCharacterTextSplitter(chunk_size=max_chars, chunk_overlap=overlap)
    out = []
    for ch in chunks:
        out.extend(splitter.split_text(ch))
    return [c for c in out if c.strip()]
