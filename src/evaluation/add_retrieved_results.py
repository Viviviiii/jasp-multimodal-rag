"""
-------------------------------------------------------------------------------
Typical usage:

    poetry run python -m src.evaluation.add_retrieved_results


— Run Retrieval and Save Evaluation Results (JSONL)
-------------------------------------------------------------------------------

This script runs the retrieval component of the multimodal JASP RAG system on a
prepared QA test set, computes simple retrieval metrics, and saves the results
to a JSONL file.

Input:
    • A JSON file containing the annotated QA test set
      (e.g., data/test_QA/QA_template.json), where each entry has:
          {
              "id": "q_1",
              "query": "...",
              "answerable": true/false/null,
              "ground_truth_answer": "...",      # optional for this script
              "relevant_chunk_ids": ["id1", ...] # used for evaluation
          }


Output:
    • A JSONL file at:
        data/test_QA/retrieval_results.jsonl

      Each line is a JSON object of the form:
          {
              "id": "q_1",
              "query": "...",
              "answerable": true,
              "retrieved_ids": ["doc_a", "doc_b", "doc_c"],
              "success_at_k": true,
              "top1_relevant": false
          }

This file is intended to be consumed by later analysis notebooks or scripts
to compute aggregate retrieval metrics, inspect failure cases, and compare
different retrieval configurations.



-------------------------------------------------------------------------------
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List
import loguru
from src.retrieval.retrieval import retrieve_top_k


logger = loguru.logger


# ------------------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------------------

def load_test_set(test_json_path: Path) -> List[Dict[str, Any]]:
    """
    Load the QA test set from a JSON file.

    The JSON file is expected to contain a list of question objects, each with
    at least 'id', 'query', and 'relevant_chunk_ids' fields.

    Args:
        test_json_path (Path):
            Path to the test set JSON file.

    Returns:
        List[Dict[str, Any]]: Parsed list of test items.
    """
    if not test_json_path.exists():
        logger.error(f"Test set JSON file not found at: {test_json_path}")
        return []

    logger.info(f"Loading test set from: {test_json_path}")

    try:
        with open(test_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file: {e}")
        return []

    logger.info(f"Loaded {len(data)} test questions.")
    return data


def run_retrieval_and_evaluation(
    test_set: List[Dict[str, Any]],
    retrieve_fn: Callable[[str, int], List[Any]],
    k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Run retrieval for each query in the test set and compute simple metrics.

    """
    retrieval_results: List[Dict[str, Any]] = []

    logger.info(f"Starting retrieval for {len(test_set)} questions with K={k}.")

    for idx, item in enumerate(test_set):
        q = item.get("query", "")
        qid = item.get("id", f"q_{idx + 1}")
        relevant_ids = set(item.get("relevant_chunk_ids", []))
        answerable = item.get("answerable", None)

        if not q:
            logger.warning(f"Skipping item with empty query (id={qid}).")
            continue

        # Run retrieval
        retrieved = retrieve_fn(q, k)

        # Extract doc/section IDs from the retrieved chunks
        retrieved_ids: List[str] = []
        for c in retrieved:
            metadata = getattr(c, "metadata", {}) or {}
            rid = metadata.get("doc_id") or metadata.get("section_id")
            if rid is not None:
                retrieved_ids.append(rid)

        # Evaluate success@k and top1_relevant
        success_at_k = bool(relevant_ids) and any(
            rid in relevant_ids for rid in retrieved_ids[:k]
        )
        top1_relevant = bool(relevant_ids) and bool(retrieved_ids) and (
            retrieved_ids[0] in relevant_ids
        )

        retrieval_results.append(
            {
                "id": qid,
                "query": q,
                "answerable": answerable,
                "retrieved_ids": retrieved_ids,
                "success_at_k": success_at_k,
                "top1_relevant": top1_relevant,
            }
        )

    logger.info(
        f"Finished retrieval for all questions. Generated "
        f"{len(retrieval_results)} result entries."
    )

    return retrieval_results


def save_results_jsonl(results: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Save retrieval results into a JSONL file.

    Each element in `results` is written as one JSON object per line.

    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving retrieval results to: {output_path}")

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for item in results:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + "\n")

        logger.info(f"Successfully wrote {len(results)} lines to {output_path}")
    except Exception as e:
        logger.error(f"Failed to write retrieval results JSONL: {e}")


def log_aggregate_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Compute and log simple aggregate retrieval metrics.

    Metrics:
        • success@k rate
        • top1_relevant rate


    """
    if not results:
        logger.warning("No results provided for aggregate metrics.")
        return

    n = len(results)
    success_count = sum(1 for r in results if r.get("success_at_k"))
    top1_count = sum(1 for r in results if r.get("top1_relevant"))

    success_rate = success_count / n
    top1_rate = top1_count / n

    logger.info(
        f"Aggregate metrics over {n} queries:"
        f"\n  success@k:     {success_count}/{n} ({success_rate:.3f})"
        f"\n  top1_relevant: {top1_count}/{n} ({top1_rate:.3f})"
    )


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

def main() -> None:
    """
    Main entry point for running retrieval and saving evaluation results.

    Steps:
        1. Define the input test JSON path and output JSONL path.
        2. Load the test set.
        3. Run retrieval and compute per-query metrics.
        4. Save per-query results to a JSONL file.
        5. Log aggregate success@k and top1_relevant metrics.
    """
    # ------------------------------------------------------------------
    # Adjust these paths to your environment if needed
    # ------------------------------------------------------------------
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / "data" / "test_QA"

    # If you still use a custom name like "questions_template copy.json",
    # you can change this filename accordingly.
    TEST_JSON = data_dir / "QA_filled_1.json"

    OUTPUT_JSONL = data_dir / "retrieval_result.jsonl"

    # Top-K cutoff for success@k
    K = 3

    # 1. Load test set
    test_set = load_test_set(TEST_JSON)
    if not test_set:
        logger.warning("Test set is empty or failed to load. Exiting.")
        return

    # 2. Run retrieval and evaluation
    results = run_retrieval_and_evaluation(
        test_set=test_set,
        retrieve_fn=lambda q, top_k: retrieve_top_k(q, top_k=top_k),
        k=K,
    )

    # 3. Save results
    save_results_jsonl(results, OUTPUT_JSONL)

    # 4. Log aggregate metrics
    log_aggregate_metrics(results)


if __name__ == "__main__":
    main()
