"""Shared pytest fixtures for all tests."""

from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.data_pipelines.clients.pubmed_client import PubMedClient


@pytest.fixture
def mock_pubmed_client():
    """Return a mock PubMedClient for testing."""
    client = MagicMock(spec=PubMedClient)
    client.get_abstract_by_id = AsyncMock()
    client.get_abstracts_by_ids = AsyncMock()
    return client


@pytest.fixture
def mock_pubmed_abstract() -> Dict[str, Any]:
    """Return a mock PubMed abstract for testing."""
    return {
        "id": "15858239",
        "title": "The role of ret gene in the pathogenesis of Hirschsprung disease",
        "abstract": "This is a test abstract about Hirschsprung disease and the RET gene.",
        "authors": ["Smigiel R", "Patkowski D", "Slezak R", "Czernik J", "Sasiadek M"],
        "publication_date": "2004 Jul-Sep",
        "journal": "Med Wieku Rozwoj",
        "doi": "10.1000/test.12345",
        "keywords": ["Hirschsprung Disease", "Genetics", "RET Gene"],
        "mesh_terms": [
            "Chromosome Aberrations",
            "Hirschsprung Disease/genetics",
            "Humans",
        ],
    }
