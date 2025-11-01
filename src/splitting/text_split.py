
"""
poetry env activate
poetry run python -m src.splitting.text_split

"""






# ---------------------------------------------------
# 📘 ENRICHED PDF HYBRID SPLITTING PIPELINE (IMPROVED)
# ---------------------------------------------------

import os
import json
import uuid
from pathlib import Path
from loguru import logger
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser, SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def split_enriched_documents(
    enriched_docs: list[Document],
    embedding_model: str = "BAAI/bge-large-en-v1.5",
    breakpoint_percentile_threshold: int = 95,
    buffer_size: int = 2,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 200,       # 🆕 avoid very short chunks
    save_json: bool = True,
    output_dir: str = "./data/processed/chunks",
    pdf_name: str | None = None,
) -> tuple[list[Document], HuggingFaceEmbedding]:
    """Improved hybrid splitting for enriched PDF documents."""

    logger.info("🚀 Starting improved hybrid splitting of enriched PDF documents")

    # 1️⃣ Initialize embedding model for semantic segmentation
    embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # 2️⃣ Semantic segmentation
    semantic_splitter = SemanticSplitterNodeParser(
        embed_model=embed_model,
        breakpoint_percentile_threshold=breakpoint_percentile_threshold,
        buffer_size=buffer_size,
    )
    semantic_nodes = semantic_splitter.get_nodes_from_documents(enriched_docs)
    logger.info(f"🧠 Semantic segmentation produced {len(semantic_nodes)} nodes")

    # 3️⃣ Sentence refinement
    sentence_splitter = SentenceSplitter.from_defaults(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        include_metadata=True,
        include_prev_next_rel=True,
    )

    final_chunks: list[Document] = []
    global_idx = 0

    for node_idx, node in enumerate(semantic_nodes):
        sentences = sentence_splitter.split_text(node.get_content())

        # Merge small ones to prevent tiny fragments
        merged_sentences = []
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) < min_chunk_chars:
                buffer += " " + sent
            else:
                if buffer.strip():
                    merged_sentences.append(buffer.strip())
                buffer = sent
        if buffer.strip():
            merged_sentences.append(buffer.strip())

        for sent_idx, sentence_text in enumerate(merged_sentences):
            chunk_id = f"{node.metadata.get('page', 'NA')}_{node_idx}_{sent_idx}"

        # ✅ Derive document and data source safely (outside Document())
            source_file = node.metadata.get("source", "unknown.pdf")
            document_name = Path(source_file).stem

            final_chunks.append(
                Document(
                    text=sentence_text,

                    metadata={
                        "data_source": "jasp_manual",
                        "document_name": document_name,
                        "source": node.metadata.get("source"),
                        "page": node.metadata.get("page"),
                        "images": list(set(node.metadata.get("images", []))),  # dedup
                        "semantic_id": node_idx,
                        "chunk_id": chunk_id,
                        "uuid": str(uuid.uuid4()),  # 🔑 unique id
                        "split_method": "semantic+sentence",
                        "char_length": len(sentence_text),
                    },
                )
            )
            global_idx += 1

    logger.info(f"✂️ Total chunks after cleanup: {len(final_chunks)}")

    # 4️⃣ Save JSON output
    if save_json:
        os.makedirs(output_dir, exist_ok=True)
        pdf_name = pdf_name or "unnamed_doc"
        out_path = Path(output_dir) / f"chunks_{pdf_name}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                [{"text": d.text, "metadata": d.metadata} for d in final_chunks],
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info(f"💾 Chunked output saved to: {out_path}")

    logger.success("✅ Hybrid splitting completed successfully")
    return final_chunks, embed_model


# ---------------------------------------------------
# 🧪 MAIN ENTRYPOINT -test splitting
# ---------------------------------------------------

def load_enriched_json(json_path: str) -> list[Document]:
    """Load enriched JSON (from enrich_llamaparse_with_images) as Document objects."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"❌ Enriched JSON not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [Document(text=item["text"], metadata=item.get("metadata", {})) for item in data]
    return docs


if __name__ == "__main__":
    enriched_json_path = "data/processed/enriched/test_pages25-28_enriched.json"
    pdf_name = Path(enriched_json_path).stem.replace("_enriched", "")

    logger.info(f"🚀 Loading enriched JSON: {enriched_json_path}")
    enriched_docs = load_enriched_json(enriched_json_path)
    logger.info(f"✅ Loaded {len(enriched_docs)} enriched documents")

    # Run the hybrid splitting
    chunks, embed_model = split_enriched_documents(
        enriched_docs=enriched_docs,
        embedding_model="BAAI/bge-large-en-v1.5",
        breakpoint_percentile_threshold=95,
        buffer_size=2,
        chunk_size=400,
        chunk_overlap=50,
        save_json=True,
        pdf_name=pdf_name,
    )

    logger.success(f"✂️ Splitting completed: {len(chunks)} chunks for {pdf_name}")

    # Display samples
    for i, doc in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(doc.text[:300], "...\n")  # optional truncate
