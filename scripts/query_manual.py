
"""
CLI script to query the JASP manual via RAG.

Features:
- Retrieves chunks from Chroma DB.
- Runs generation with an Ollama model (e.g., LLaMA, Mistral).
- Prints both answer and sources.

Usage:
    poetry run python -m scripts.query_manual "How do I run ANOVA in JASP?" --model mistral:7b-instruct --topk 2

"""




import argparse
from src.retrieval.query import run_query, DEFAULT_MODEL, DEFAULT_TOPK

def main():
    parser = argparse.ArgumentParser(description="Query the JASP manual via RAG.")
    parser.add_argument("query_text", type=str, help="The query text.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model to use (default: llama3)")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK, help="Number of chunks to retrieve (default: 3)")
    args = parser.parse_args()

    result = run_query(args.query_text, args.model, args.topk)

    print("\n--- Response ---\n")
    print(result["answer"].strip())

    print("\n--- Sources ---")
    if result["sources"]:
        for s in result["sources"]:
            print(f"- {s}")
    else:
        print("No source documents retrieved.")

if __name__ == "__main__":
    main()
