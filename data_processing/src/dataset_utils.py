import json
import logging
import os

logger = logging.getLogger(__name__)


def create_dataset_metadata(
    output_dir: str, corpus_count: int, dev_count: int, test_count: int
) -> None:
    """
    Create a dataset-info.json file with metadata about the dataset.

    Args:
        output_dir: Base directory for the dataset
        corpus_count: Number of documents in the corpus
        dev_count: Number of questions in the dev set
        test_count: Number of questions in the test set
    """
    metadata = {
        "name": "bioasq-12b-rag-dataset",
        "version": "1.0.0",
        "description": "BioASQ 12B dataset processed for RAG applications",
        "splits": {
            "corpus": {"num_examples": corpus_count, "file": "data/corpus.jsonl"},
            "dev": {"num_examples": dev_count, "file": "data/dev.jsonl"},
            "test": {"num_examples": test_count, "file": "data/test.jsonl"},
        },
        "features": {
            "corpus": [
                "id",
                "title",
                "text",
                "url",
                "publication_date",
                "journal",
                "authors",
                "doi",
                "keywords",
                "mesh_terms",
            ],
            "questions": [
                "question_id",
                "question",
                "answer",
                "relevant_passage_ids",
                "type",
                "snippets",
            ],
        },
    }

    metadata_path = os.path.join(output_dir, "dataset-info.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Dataset metadata created at {metadata_path}")


def validate_dataset(dataset_dir: str) -> bool:
    """
    Validate the dataset files to ensure they exist and are properly formatted.

    Args:
        dataset_dir: Base directory for the dataset

    Returns:
        True if validation passes, False otherwise
    """
    required_files = [
        "data/corpus.jsonl",
        "data/dev.jsonl",
        "data/test.jsonl",
        "dataset-info.json",
        "README.md",
    ]

    # Check if all required files exist
    for file in required_files:
        file_path = os.path.join(dataset_dir, file)
        if not os.path.exists(file_path):
            logger.error(f"Missing required file: {file}")
            return False

    # Validate JSONL files
    for jsonl_file in ["data/corpus.jsonl", "data/dev.jsonl", "data/test.jsonl"]:
        file_path = os.path.join(dataset_dir, jsonl_file)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
                if not line:
                    logger.error(f"Empty file: {jsonl_file}")
                    return False

                # Try to parse the first line as JSON
                json.loads(line)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {jsonl_file}")
            return False
        except Exception as e:
            logger.error(f"Error validating {jsonl_file}: {e}")
            return False

    logger.info("Dataset validation successful")
    return True
