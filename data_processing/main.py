import argparse
import logging
import os

from src.corpus_processor import create_corpus
from src.dataset_utils import (
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
    eval_path = os.path.join(args.output_dir, "data/eval.jsonl")
    logger.info(
        f"Creating question datasets from {args.training_file} and {args.goldset_dir}"
    )
    dev_count, eval_count = create_question_datasets(
        args.training_file, args.goldset_dir, dev_path, eval_path
    )

    # Validate dataset
    logger.info("Validating dataset")
    if validate_dataset(args.output_dir):
        logger.info(f"Dataset successfully created at {args.output_dir}")
        logger.info(f"Corpus: {corpus_count} abstracts")
        logger.info(f"Dev questions: {dev_count} questions")
        logger.info(f"Eval questions: {eval_count} questions")
    else:
        logger.error("Dataset validation failed")


if __name__ == "__main__":
    main()
