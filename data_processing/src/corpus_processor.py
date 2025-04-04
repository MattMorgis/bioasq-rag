import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def process_abstract(file_path: Path) -> Optional[Dict[str, Any]]:
    """
    Process a single PubMed abstract file and transform it into the desired format for the corpus.

    Args:
        file_path: Path to the JSON file containing the abstract

    Returns:
        Dictionary with processed abstract data or None if processing fails
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            abstract_data = json.load(f)

        # Extract PubMed ID from filename
        pubmed_id = file_path.stem

        # Create URL
        pubmed_url = f"http://www.ncbi.nlm.nih.gov/pubmed/{pubmed_id}"

        # Create corpus entry
        corpus_entry = {
            "id": pubmed_id,
            "title": abstract_data.get("title", ""),
            "text": abstract_data.get("abstract", ""),
            "url": pubmed_url,
            "publication_date": abstract_data.get("publication_date", ""),
            "journal": abstract_data.get("journal", ""),
            "authors": abstract_data.get("authors", []),
            "doi": abstract_data.get("doi"),
            "keywords": abstract_data.get("keywords", []),
            "mesh_terms": abstract_data.get("mesh_terms", []),
        }

        return corpus_entry
    except Exception as e:
        logger.error(f"Error processing abstract {file_path}: {e}")
        return None


def create_corpus(abstracts_dir: str, output_path: str) -> int:
    """
    Process all abstracts in the given directory and create the corpus JSONL file.

    Args:
        abstracts_dir: Directory containing abstract JSON files
        output_path: Path to write the corpus JSONL file

    Returns:
        Number of abstracts processed
    """
    abstracts_dir_path = Path(abstracts_dir)
    corpus_entries = []
    count = 0

    # Process each JSON file in the abstracts directory
    for abstract_file in abstracts_dir_path.glob("*.json"):
        # Skip any empty or invalid files
        if abstract_file.stat().st_size == 0:
            continue

        corpus_entry = process_abstract(abstract_file)
        if corpus_entry:
            corpus_entries.append(corpus_entry)
            count += 1

        # Log progress every 1000 abstracts
        if count % 1000 == 0 and count > 0:
            logger.info(f"Processed {count} abstracts...")

    # Write corpus to JSONL file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in corpus_entries:
            f.write(json.dumps(entry) + "\n")

    logger.info(f"Corpus created with {count} abstracts at {output_path}")
    return count
