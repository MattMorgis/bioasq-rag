import json
import logging
import os

logger = logging.getLogger(__name__)


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
