"""
-------------------------------------------------------------------------------
Typical usage:
    python src/evaluation/create_Q_temp.py

— Generate Clean QA Templates for Retrieval Evaluation
-------------------------------------------------------------------------------

This script automates the creation of a standardized **QA template** used in the
evaluation of the multimodal JASP RAG system.

The pipeline performs two core tasks:

1. Extract questions from a raw text file (e.g., data/test_QA/Qs.txt)
   - Loads a plain-text file with mixed content (questions, headers, notes)
   - Removes emoji-based section headers and other non-question lines
   - Keeps only lines that end with a question mark ("?")

2. Generate a structured JSON QA template
   - Builds a list of dictionaries, each representing one evaluation sample:
        {
            "id": "q_1",
            "query": "How do I run ANOVA in JASP?",
            "answerable": null,
            "ground_truth_answer": null,
            "relevant_chunk_ids": []
        }

-------------------------------------------------------------------------------
"""

import json
import loguru  # Using loguru for rich logging
from pathlib import Path
from typing import List, Dict, Any

# Initialize the logger instance using loguru
logger = loguru.logger


def load_and_filter_questions(input_file_path: Path) -> List[str]:
    """
    Load raw text lines from a questions file and extract only valid questions.

    This function performs several cleaning steps:
        • Verifies that the input file exists
        • Strips whitespace from each line
        • Skips empty lines
        • Skips emoji-based section headers (e.g., "🧪 Test questions")
        • Keeps only lines that end with '?' to ensure consistent question format

    This ensures that the resulting question list is directly usable for
    evaluation-template creation without manual preprocessing.

    Args:
        input_file_path (Path):
            Path to the input .txt file containing raw question lines.

    Returns:
        List[str]: A list of cleaned questions ready for template generation.
    """
    questions: List[str] = []

    if not input_file_path.exists():
        logger.error(f"Input file not found at: {input_file_path}")
        return []

    logger.info(f"Loading questions from: {input_file_path}")

    with open(input_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip emoji section headers (as in original script)
            if line.startswith(("🧪", "📚", "⚙️", "💡", "❓")):
                continue

            # CRITICAL FILTER: Only include lines that end with a question mark
            if line.endswith("?"):
                questions.append(line)

    logger.info(f"Successfully loaded and filtered {len(questions)} questions.")
    return questions


def create_template_and_save(question_list: List[str], output_file_path: Path) -> None:
    """
    Convert a list of plain-text questions into a structured JSON QA template.

    For every question, this function builds a JSON entry containing:
        - "id": unique question ID (e.g., "q_1", "q_2", ...)
        - "query": the original question string
        - "answerable": placeholder (manual annotation: True/False)
        - "ground_truth_answer": placeholder for the gold-standard textual answer
        - "relevant_chunk_ids": empty list to be filled with relevant chunk IDs

    The resulting JSON file is:
        • Human-readable (indented formatting)
        • Ready for manual annotation
        • Directly consumable by evaluation scripts

    Args:
        question_list (List[str]):
            Cleaned list of questions produced by `load_and_filter_questions`.

        output_file_path (Path):
            Output path where QA_template.json will be written.
    """
    qa_template: List[Dict[str, Any]] = []

    for i, line in enumerate(question_list):
        # Generate question ID (e.g., "q_1", "q_2", ...)
        qid = f"q_{i + 1}"

        qa_template.append({
            "id": qid,
            "query": line,
            "answerable": None,           # Manual fill: True/False
            "ground_truth_answer": None,  # Manual fill: Text answer
            "relevant_chunk_ids": []      # Manual fill: List of IDs
        })

    logger.info(f"Saving {len(qa_template)} structured entries to: {output_file_path}")

    try:
        with open(output_file_path, "w", encoding="utf-8") as outfile:
            # Use indent=4 for a nicely formatted and human-readable JSON file
            json.dump(qa_template, outfile, indent=4)
        logger.info("Template saved successfully.")
    except Exception as e:
        logger.error(f"Failed to write JSON file: {e}")


def main() -> None:
    """
    Main entry point for running the question template generation pipeline.

    Workflow:
        1. Define input and output paths:
            - Input:  Qs.txt (raw questions file)
            - Output: QA_template.json (structured evaluation template)
        2. Load and filter questions from the input file.
        3. If at least one question is found, generate and save the JSON template.
           Otherwise, emit a warning and exit without writing a file.

    This function is typically run once per evaluation batch when the content of
    `Qs.txt` changes or when a new set of forum-inspired questions is defined.
    """
    # Define file paths based on the project structure
    INPUT_FILE_PATH = Path("/Users/ywxiu/jasp-multimodal-rag/data/test_QA/Qs.txt")
    OUTPUT_FILE_PATH = INPUT_FILE_PATH.parent / "QA_template.json"

    # 1. Load and filter questions
    filtered_questions = load_and_filter_questions(INPUT_FILE_PATH)

    # 2. Create template structure and save to JSON
    if filtered_questions:
        create_template_and_save(filtered_questions, OUTPUT_FILE_PATH)
    else:
        logger.warning("No questions were loaded. Skipping template generation.")


# Script entry point
if __name__ == "__main__":
    main()
