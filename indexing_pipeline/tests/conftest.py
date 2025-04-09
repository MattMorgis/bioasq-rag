import json
import os
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from src.models.document import Document


@pytest.fixture
def sample_pubmed_abstract() -> Document:
    """Return a sample PubMedAbstract for testing."""
    return Document(
        id="123456",
        title="Sample Medical Abstract for Testing",
        text="This is a sample abstract about a medical topic.",
        url="https://www.ncbi.nlm.nih.gov/pubmed/123456",
        publication_date="2023-01-15",
        journal="Journal of Medical Testing",
        authors=["Smith, J", "Johnson, A"],
        keywords=["testing", "embeddings"],
        mesh_terms=["Testing", "Embeddings"],
        doi="10.1234/test.5678",
    )


@pytest.fixture
def temp_corpus_file(sample_pubmed_abstract) -> Generator[Path, None, None]:
    """Create a temporary corpus file with sample abstracts for testing."""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".jsonl") as f:
        # Write 5 sample abstracts to the temp file
        for i in range(5):
            abstract_dict = {
                "id": f"{123456 + i}",
                "title": f"Sample Medical Abstract {i + 1}",
                "text": f"This is sample abstract {i + 1} about a medical topic.",
                "url": f"https://www.ncbi.nlm.nih.gov/pubmed/{123456 + i}",
                "publication_date": "2023-01-15",
                "journal": "Journal of Medical Testing",
                "authors": ["Smith, J", "Johnson, A"],
                "keywords": ["testing", "embeddings"],
                "mesh_terms": ["Testing", "Embeddings"],
                "doi": f"10.1234/test.{5678 + i}",
            }
            f.write(json.dumps(abstract_dict) + "\n")

        temp_file_path = Path(f.name)

    yield temp_file_path

    # Cleanup the temp file after the test
    if os.path.exists(temp_file_path):
        os.unlink(temp_file_path)
