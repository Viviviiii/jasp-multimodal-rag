import sys
import logging
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

# ==============================
# CONFIG
# ==============================
CHROMA_PATH = "data/indexes/chroma"   # updated to match ingestion
TEXT_MODEL_NAME = "BAAI/bge-small-en"
DEFAULT_MODEL = "llama3"
DEFAULT_TOPK = 3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ==============================
# EMBEDDINGS WRAPPER
# ==============================
class BGEEmbeddings:
    """Wrapper so SentenceTransformer can be used as Chroma embeddings."""

    def __init__(self, model_name: str = TEXT_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].tolist()

# ==============================
# PROMPT TEMPLATE
# ==============================
PROMPT_TEMPLATE = """
You are a helpful assistant for the JASP manual.

Context (from the manual):
{context}

Question:
{question}

Instructions:
- ONLY use the information in the provided context. Do not make up or guess.
- Provide clear, structured instructions (use bullet points or steps if relevant).

Answer:
"""

# ==============================
# BUILD RETRIEVAL-BASED QA CHAIN
# ==============================
def build_rag_chain(llm_model: str = DEFAULT_MODEL, k: int = DEFAULT_TOPK) -> RetrievalQA:
    """Builds and returns a RetrievalQA chain."""
    if not Path(CHROMA_PATH).exists():
        logging.error("Chroma DB not found at '%s'. Run the ingestion script first.", CHROMA_PATH)
        sys.exit(1)

    embeddings = BGEEmbeddings(TEXT_MODEL_NAME)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name="jasp_text",
    )

    retriever = db.as_retriever(search_kwargs={"k": k})
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    try:
        llm = Ollama(model=llm_model)
    except Exception as e:
        logging.error("Could not connect to Ollama: %s", e)
        logging.error("Ensure Ollama is running and the model is pulled (e.g., `ollama pull %s`).", llm_model)
        sys.exit(1)

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

# ==============================
# PUBLIC API
# ==============================
def run_query(query_text: str, llm_model: str = DEFAULT_MODEL, k: int = DEFAULT_TOPK) -> dict:
    """
    Run a query against the JASP manual using the RAG pipeline.
    Returns {"answer": str, "sources": list}.
    """
    chain = build_rag_chain(llm_model, k)
    result = chain.invoke({"query": query_text})

    sources = []
    if result.get("source_documents"):
        seen = set()
        for doc in result["source_documents"]:
            page = doc.metadata.get("page")
            source = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")
            if source not in seen:
                sources.append(source)
                seen.add(source)

    return {"answer": result["result"], "sources": sources}
