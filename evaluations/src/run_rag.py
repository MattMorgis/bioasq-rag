#!/usr/bin/env python3
import concurrent.futures
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import requests


def process_eval_data(eval_file_path: str, output_file_path: str, max_workers: int = 2):
    """
    Process each question in the eval.jsonl file, send it to the RAG endpoint,
    and save the combined results to a new file.
    Uses concurrent processing to handle multiple questions at once.
    """
    # Ensure output directory exists
    output_dir = Path(output_file_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read all questions from the eval file
    eval_data_list = []
    with open(eval_file_path, "r") as eval_file:
        for line_number, line in enumerate(eval_file, 1):
            try:
                eval_data = json.loads(line)
                eval_data["line_number"] = line_number
                eval_data_list.append(eval_data)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_number}: {e}")

    # Process questions concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for eval_data in eval_data_list:
            futures.append(executor.submit(process_single_question, eval_data))

        # Write results to output file as they complete
        with open(output_file_path, "w") as output_file:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        output_file.write(json.dumps(result) + "\n")
                except Exception as e:
                    print(f"Error processing question: {e}")


def process_single_question(eval_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Process a single question and return the combined result."""
    line_number = eval_data.get("line_number", 0)
    question = eval_data.get("question", "")

    if not question:
        print(f"Warning: Line {line_number} has no question field. Skipping.")
        return None

    # Send the question to the RAG endpoint
    print(f"Processing question {line_number}: {question[:50]}...")
    rag_response = query_rag_endpoint(question)

    # Extract only the chunk texts from rag_sources
    rag_sources_chunks = [
        src.get("chunk", "") for src in rag_response.get("sources", [])
    ]

    # Combine the eval data with the RAG response
    result = {
        "question_id": eval_data.get("question_id", ""),
        "question": question,
        "ground_truth_answer": eval_data.get("answer", ""),
        "relevant_passage_ids": eval_data.get("relevant_passage_ids", []),
        "type": eval_data.get("type", ""),
        "snippets": eval_data.get("snippets", []),
        "rag_answer": rag_response.get("answer", ""),
        "rag_sources": rag_sources_chunks,
    }

    return result


def query_rag_endpoint(query: str) -> Dict[str, Any]:
    """
    Send a query to the RAG endpoint and return the response.
    """
    url = "http://localhost:3000/rag/query"

    try:
        response = requests.post(url, json={"query": query}, timeout=30)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.RequestException as e:
        print(f"Error querying RAG endpoint: {e}")
        # Return an empty response in case of error
        return {"answer": "", "sources": []}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "Usage: python evaluate_rag.py <eval_file_path> <output_file_path> [max_workers]"
        )
        print(
            "Example: python evaluate_rag.py data/bioasq-12b-rag-dataset/data/eval.jsonl results/rag_evaluation.jsonl 2"
        )
        sys.exit(1)

    eval_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    # Allow setting max_workers from command line
    max_workers = 2
    if len(sys.argv) > 3:
        try:
            max_workers = int(sys.argv[3])
        except ValueError:
            print(f"Invalid max_workers value: {sys.argv[3]}. Using default: 2")

    print(f"Processing evaluation data from {eval_file_path}")
    print(f"Saving results to {output_file_path}")
    print(f"Using {max_workers} workers for parallel processing")

    process_eval_data(eval_file_path, output_file_path, max_workers)

    print("Evaluation complete!")
