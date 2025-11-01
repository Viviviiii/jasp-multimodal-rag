
"""
---------------------------------------------------
💬 GENERATION PIPELINE (RAG FINAL STAGE)
🧩 Pipeline Overview:
1. Retrieve the top-ranked document chunks from the hybrid retrieval pipeline.
2. Format those chunks into a structured prompt with metadata context.
3. Stream a generated answer from a local Ollama model (e.g., Mistral, Llama 3, Phi-3).
4. Display the final answer and its document sources.

⚙️ Supported Ollama Models:
- mistral:7b       → balanced, accurate, local-friendly
- llama3.2:3b      → fast and lightweight
- phi3:mini        → compact and efficient for small contexts

Run:
    poetry run python -m src.generation.generation --q "How to split data files in JASP?"
---------------------------------------------------
"""





from __future__ import annotations
import sys
import argparse
from typing import List, Tuple
from loguru import logger
from ollama import Client

from src.retrieval.retrieval import retrieve_top_k, NodeWithScore


PROMPT_TEMPLATE = """
You are a documentation assistant for JASP.
Use the context below to answer the user's question clearly and concisely.

Question:
{query}

Context:
{context}

Answer in markdown, include source references when available.
"""


def generate_answer(
    query: str,
    top_nodes: List[NodeWithScore],
    model: str = "mistral:7b",
    max_chars_per_doc: int = 1200,
    prompt_template: str = PROMPT_TEMPLATE,
) -> str:
    client = Client()
    context_blocks = []

    for i, node in enumerate(top_nodes, 1):
        meta = getattr(node.node, "metadata", {})
        text = node.node.get_content()[:max_chars_per_doc]
        context_blocks.append(f"[Doc {i}] ({meta.get('source', 'N/A')} p.{meta.get('page', '?')})\n{text}")

    prompt = prompt_template.format(query=query, context="\n\n".join(context_blocks))

    logger.info(f"🚀 Generating answer using {model}")
    stream = client.generate(model=model, prompt=prompt, stream=True)

    full_output = ""
    for chunk in stream:
        token = chunk.get("response", "")
        sys.stdout.write(token)
        sys.stdout.flush()
        full_output += token

    logger.success("\n✅ Generation completed.")
    return full_output


def answer_query(query: str, model: str = "mistral:7b") -> Tuple[str, List[NodeWithScore]]:
    reranked_nodes = retrieve_top_k(query)
    answer = generate_answer(query, reranked_nodes, model=model)
    return answer, reranked_nodes


def main():
    parser = argparse.ArgumentParser(description="Run RAG generation pipeline")
    parser.add_argument("--q", "--query", dest="query", type=str, default="How to split data files in JASP?")
    parser.add_argument("--model", type=str, default="mistral:7b")
    args = parser.parse_args()

    ans, nodes = answer_query(args.query, args.model)
    print("\n" + "="*80)
    print("🧠 FINAL ANSWER\n" + "-"*80)
    print(ans)
    print("\n📚 SOURCES")
    for i, n in enumerate(nodes, 1):
        md = n.node.metadata or {}
        print(f"[{i}] {md.get('source', 'unknown')} p.{md.get('page', '?')}")

if __name__ == "__main__":
    main()
