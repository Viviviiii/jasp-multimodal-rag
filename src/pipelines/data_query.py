"""
data_query.py — Ask questions against the JASP manual (multimodal RAG pipeline)

This script connects to the Chroma vector database you built during ingestion
and lets a user query it with natural language. Retrieval is based on
SentenceTransformer (BGE small) embeddings, ensuring consistency with the
indexing script.

Usage:
  
python manual_query.py "How do I run multiple regression in JASP?" --model mistral:7b-instruct



Requirements:
    - Ollama running locally with a model like llama3 installed
    - Chroma DB created beforehand (via your ingestion pipeline)
"""

import argparse
import sys
import logging
from pathlib import Path

from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

# ==============================
# CONFIG
# ==============================
CHROMA_PATH = "data/chroma"  # must match ingestion script
TEXT_MODEL_NAME = "BAAI/bge-small-en"  # same as ingestion
DEFAULT_MODEL = "llama3"  # Ollama LLM to use
DEFAULT_TOPK = 1  # number of chunks to retrieve

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ==============================
# EMBEDDINGS (BGE WRAPPER)
# ==============================
class BGEEmbeddings:
    """
    Wrapper to make SentenceTransformer work as an embedding function for Chroma.
    Provides both document- and query-level embeddings.
    """

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
# HELPER — BUILD CHAIN
# ==============================
def build_rag_chain(llm_model: str, k: int = DEFAULT_TOPK) -> RetrievalQA:
    """Builds and returns a RetrievalQA chain with the given LLM model and top-k chunks."""
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
# PUBLIC API — RUN QUERY
# ==============================
def run_query(query_text: str, llm_model: str = DEFAULT_MODEL, k: int = DEFAULT_TOPK) -> str:
    """
    Run a query against the JASP manual using the RAG pipeline.
    Returns only the answer string.
    """
    chain = build_rag_chain(llm_model, k)
    result = chain.invoke({"query": query_text})
    return result["result"]

# ==============================
# CLI ENTRYPOINT
# ==============================
def main():
    parser = argparse.ArgumentParser(description="Query the JASP manual via RAG.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use (default: llama3)")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    chain = build_rag_chain(args.model, args.topk)
    result = chain.invoke({"query": args.query_text})

    print("\n--- Response ---\n")
    print(result["result"].strip())

    print("\n--- Sources ---")
    if result.get("source_documents"):
        seen = set()
        for doc in result["source_documents"]:
            page = doc.metadata.get("page")
            source = f"Page {page}" if page else doc.metadata.get("path", "Unknown source")
            if source not in seen:  # avoid duplicates
                print(f"- {source}")
                seen.add(source)
    else:
        print("No source documents retrieved.")


if __name__ == "__main__":
    main()
