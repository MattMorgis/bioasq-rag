import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def extract_pubmed_id(url: str) -> Optional[str]:
    """
    Extract PubMed ID from a PubMed URL.

    Args:
        url: PubMed URL

    Returns:
        PubMed ID or None if extraction fails
    """
    match = re.search(r"pubmed/(\d+)", url)
    if match:
        return match.group(1)
    return None


def process_question(question: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Process a single BioASQ question and transform it into the desired format.

    Args:
        question: Dictionary containing question data

    Returns:
        Dictionary with processed question data or None if processing fails
    """
    try:
        # Check for required fields
        if (
            "id" not in question
            or "body" not in question
            or "documents" not in question
            or "ideal_answer" not in question
        ):
            logger.warning(
                f"Skipping question with missing required fields: {question.get('id', 'unknown')}"
            )
            return None

        # Extract relevant passage IDs from document URLs
        relevant_passage_ids = []
        for doc_url in question.get("documents", []):
            pubmed_id = extract_pubmed_id(doc_url)
            if pubmed_id:
                relevant_passage_ids.append(pubmed_id)

        # Get the ideal answer (may be a list or a string)
        ideal_answer = question.get("ideal_answer", [])
        if isinstance(ideal_answer, list) and len(ideal_answer) > 0:
            ideal_answer = ideal_answer[0]

        # Create question entry
        question_entry = {
            "question_id": question.get("id", ""),
            "question": question.get("body", ""),
            "answer": ideal_answer,
            "relevant_passage_ids": relevant_passage_ids,
            "type": question.get("type", ""),
            "snippets": question.get("snippets", []),
        }

        return question_entry
    except Exception as e:
        logger.error(f"Error processing question {question.get('id', 'unknown')}: {e}")
        return None


def process_question_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a BioASQ question file and extract all questions.

    Args:
        file_path: Path to the question file

    Returns:
        List of processed questions
    """
    processed_questions = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for question in data.get("questions", []):
            processed_question = process_question(question)
            if processed_question:
                processed_questions.append(processed_question)
    except Exception as e:
        logger.error(f"Error processing question file {file_path}: {e}")

    return processed_questions


def create_question_datasets(
    training_file: str, goldset_dir: str, dev_output_path: str, test_output_path: str
) -> tuple:
    """
    Process BioASQ questions and create dev and test datasets.

    Args:
        training_file: Path to the training file
        goldset_dir: Directory containing goldset files
        dev_output_path: Path to write the dev JSONL file
        test_output_path: Path to write the test JSONL file

    Returns:
        Tuple with the number of dev and test questions processed
    """
    # Use training file for dev dataset
    dev_questions = process_question_file(training_file)

    # Use goldset files for test dataset
    test_questions = []
    goldset_dir_path = Path(goldset_dir)

    for goldset_file in goldset_dir_path.glob("*.json"):
        test_questions.extend(process_question_file(str(goldset_file)))

    # Write dev questions to JSONL file
    os.makedirs(os.path.dirname(dev_output_path), exist_ok=True)
    with open(dev_output_path, "w", encoding="utf-8") as f:
        for question in dev_questions:
            f.write(json.dumps(question) + "\n")

    # Write test questions to JSONL file
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    with open(test_output_path, "w", encoding="utf-8") as f:
        for question in test_questions:
            f.write(json.dumps(question) + "\n")

    logger.info(
        f"Dev dataset created with {len(dev_questions)} questions at {dev_output_path}"
    )
    logger.info(
        f"Test dataset created with {len(test_questions)} questions at {test_output_path}"
    )

    return len(dev_questions), len(test_questions)
