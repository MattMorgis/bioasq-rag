import argparse
import logging
import os

from src.corpus_processor import create_corpus
from src.dataset_utils import (
    create_dataset_metadata,
    create_dataset_readme,
    prepare_huggingface_structure,
    validate_dataset,
)
from src.question_processor import create_question_datasets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dataset_creation.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


def main():
    """
    Main function to create the BioASQ RAG dataset.
    """
    parser = argparse.ArgumentParser(
        description="Process BioASQ data for RAG applications"
    )

    parser.add_argument(
        "--abstracts_dir",
        default="data/abstracts",
        help="Directory containing PubMed abstract JSON files",
    )

    parser.add_argument(
        "--training_file",
        default="data/BioASQ-12b/training/training12b_new.json",
        help="Path to BioASQ training file",
    )

    parser.add_argument(
        "--goldset_dir",
        default="data/BioASQ-12b/goldset",
        help="Directory containing BioASQ goldset files",
    )

    parser.add_argument(
        "--output_dir",
        default="data/bioasq-12b-rag-dataset",
        help="Output directory for the processed dataset",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)

    logger.info("Starting BioASQ RAG dataset creation")

    # Process and create corpus
    corpus_path = os.path.join(args.output_dir, "data/corpus.jsonl")
    logger.info(f"Creating corpus from {args.abstracts_dir}")
    corpus_count = create_corpus(args.abstracts_dir, corpus_path)

    # Process and create question datasets
    dev_path = os.path.join(args.output_dir, "data/dev.jsonl")
    test_path = os.path.join(args.output_dir, "data/test.jsonl")
    logger.info(
        f"Creating question datasets from {args.training_file} and {args.goldset_dir}"
    )
    dev_count, test_count = create_question_datasets(
        args.training_file, args.goldset_dir, dev_path, test_path
    )

    # Create dataset metadata
    logger.info("Creating dataset metadata")
    create_dataset_metadata(args.output_dir, corpus_count, dev_count, test_count)

    # Create dataset README
    logger.info("Creating dataset README")
    create_dataset_readme(args.output_dir)

    # Prepare Hugging Face structure
    logger.info("Preparing Hugging Face structure")
    prepare_huggingface_structure(args.output_dir)

    # Validate dataset
    logger.info("Validating dataset")
    if validate_dataset(args.output_dir):
        logger.info(f"Dataset successfully created at {args.output_dir}")
        logger.info(f"Corpus: {corpus_count} abstracts")
        logger.info(f"Dev questions: {dev_count} questions")
        logger.info(f"Test questions: {test_count} questions")
    else:
        logger.error("Dataset validation failed")


if __name__ == "__main__":
    main()
