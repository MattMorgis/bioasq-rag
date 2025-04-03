import json
import logging
from pathlib import Path
from typing import Set

from src.data_collection.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


class PubMedURLCollector:
    """
    Class to collect and deduplicate PubMed URLs from BioASQ dataset files.

    This class processes both the BioASQ training and golden set files to extract
    all unique PubMed URLs from the 'documents' field of each question.
    """

    def __init__(self, data_dir: str = "data"):
        """
        Initialize the PubMedURLCollector with the path to the data directory.

        Args:
            data_dir: Path to the directory containing BioASQ datasets
        """
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "BioASQ-12b" / "training"
        self.goldset_dir = self.data_dir / "BioASQ-12b" / "goldset"
        self.unique_urls: Set[str] = set()

    def _extract_urls_from_file(self, file_path: Path) -> Set[str]:
        """
        Extract PubMed URLs from a single BioASQ dataset file.

        Args:
            file_path: Path to the BioASQ dataset file

        Returns:
            A set of unique PubMed URLs found in the file
        """
        logger.info(f"Processing file: {file_path}")
        urls = set()

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Extract URLs from the 'documents' field of each question
            for question in data.get("questions", []):
                if "documents" in question:
                    urls.update(question["documents"])
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")

        logger.info(f"Found {len(urls)} URLs in {file_path.name}")
        return urls

    def collect_urls(self) -> Set[str]:
        """
        Collect all unique PubMed URLs from all BioASQ dataset files.

        Returns:
            A set of all unique PubMed URLs
        """
        # Process training files
        training_files = list(self.training_dir.glob("*.json"))
        logger.info(f"Found {len(training_files)} training files")

        for file_path in training_files:
            self.unique_urls.update(self._extract_urls_from_file(file_path))

        # Process goldset files
        goldset_files = list(self.goldset_dir.glob("*.json"))
        logger.info(f"Found {len(goldset_files)} goldset files")

        for file_path in goldset_files:
            self.unique_urls.update(self._extract_urls_from_file(file_path))

        logger.info(f"Total unique PubMed URLs: {len(self.unique_urls)}")
        return self.unique_urls

    def save_urls_to_file(
        self, output_path: str = "data/unique_pubmed_urls.txt"
    ) -> None:
        """
        Save all collected unique PubMed URLs to a file.

        Args:
            output_path: Path to save the URLs
        """
        if not self.unique_urls:
            self.collect_urls()

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            for url in sorted(self.unique_urls):
                f.write(f"{url}\n")

        logger.info(f"Saved {len(self.unique_urls)} unique URLs to {output_path}")


if __name__ == "__main__":
    # Set up logging to both console and file
    setup_logging(log_level="INFO", log_file="url_collector.log")

    collector = PubMedURLCollector()
    collector.collect_urls()
    collector.save_urls_to_file()
