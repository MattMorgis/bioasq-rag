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


def create_dataset_readme(output_dir: str) -> None:
    """
    Create a README.md file for the dataset.

    Args:
        output_dir: Base directory for the dataset
    """
    readme_content = """# BioASQ 12B RAG Dataset

A processed version of the BioASQ 12B dataset optimized for Retrieval-Augmented Generation (RAG) applications in biomedical question answering.

## Dataset Structure

The dataset contains three main components:

1. **Corpus** (`data/corpus.jsonl`): A collection of PubMed abstracts including metadata.
   - Each line is a JSON object containing:
     - `id`: PubMed ID
     - `title`: Title of the paper
     - `text`: Abstract text
     - `url`: PubMed URL
     - `publication_date`: Publication date
     - `journal`: Journal name
     - `authors`: List of authors
     - `doi`: Digital Object Identifier (if available)
     - `keywords`: Keywords
     - `mesh_terms`: MeSH terms

2. **Dev Questions** (`data/dev.jsonl`): Development set of biomedical questions.
   - Each line is a JSON object containing:
     - `question_id`: Unique identifier for the question
     - `question`: The question text
     - `answer`: Ideal answer
     - `relevant_passage_ids`: List of PubMed IDs for relevant abstracts
     - `type`: Question type (e.g., factoid, list, yes/no, summary)
     - `snippets`: Relevant snippets from abstracts

3. **Test Questions** (`data/test.jsonl`): Test set of biomedical questions.
   - Same structure as dev questions

## Usage

This dataset is designed for training and evaluating RAG systems for biomedical question answering.

## Source

This dataset is derived from the [BioASQ Challenge](http://bioasq.org/) task 12b dataset.

## License

The dataset follows the original BioASQ data license.
"""

    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    logger.info(f"Dataset README created at {readme_path}")


def prepare_huggingface_structure(output_dir: str) -> None:
    """
    Create the necessary files for Hugging Face dataset publishing.

    Args:
        output_dir: Base directory for the dataset
    """
    # Create .gitattributes file for large files
    gitattributes_content = """*.jsonl filter=lfs diff=lfs merge=lfs -text
"""

    gitattributes_path = os.path.join(output_dir, ".gitattributes")
    with open(gitattributes_path, "w", encoding="utf-8") as f:
        f.write(gitattributes_content)

    logger.info(f"Created .gitattributes at {gitattributes_path}")


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
