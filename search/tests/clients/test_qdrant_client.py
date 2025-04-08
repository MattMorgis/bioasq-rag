import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.clients.qdrant_client import QdrantSearchClient


@pytest.fixture
def mock_qdrant_client():
    with patch("src.clients.qdrant_client.OriginalQdrantClient") as mock_client:
        yield mock_client


@pytest.fixture
def mock_sentence_transformer():
    with patch("src.clients.qdrant_client.SentenceTransformer") as mock_transformer:
        # Use numpy array instead of list to match the real output
        mock_transformer.return_value.encode.return_value = np.array([0.1, 0.2, 0.3])
        yield mock_transformer


@pytest.fixture
def mock_search_result():
    mock_result = MagicMock()
    mock_result.payload = {
        "abstract_id": "123456",
        "title": "Test Abstract",
        "text": "This is a test abstract about biomedical research.",
        "url": "https://pubmed.ncbi.nlm.nih.gov/123456/",
        "publication_date": "2022-01-01",
        "journal": "Test Journal",
        "authors": ["Author One", "Author Two"],
    }
    mock_result.score = 0.95
    return mock_result


def test_qdrant_client_init():
    """Test that the client initializes with default values."""
    # Override environment variables for testing
    with patch.dict(os.environ, {"QDRANT_HOST": "test-host", "QDRANT_PORT": "1234"}):
        client = QdrantSearchClient()
        assert client.host == "test-host"
        assert client.port == 1234


def test_qdrant_client_search(
    mock_qdrant_client, mock_sentence_transformer, mock_search_result
):
    """Test that the search method returns expected results."""
    # Set up the mock to return our test result
    mock_qdrant_client.return_value.search.return_value = [mock_search_result]

    # Create the client
    client = QdrantSearchClient()

    # Call the search method
    results = client.search("test query", limit=5)

    # Verify the results
    assert len(results) == 1
    assert results[0].abstract_id == "123456"
    assert results[0].title == "Test Abstract"
    assert results[0].score == 0.95

    # Verify that the client was called with expected parameters
    mock_qdrant_client.return_value.search.assert_called_once_with(
        collection_name=client.collection_name,
        query_vector=[0.1, 0.2, 0.3],
        limit=5,
        with_payload=True,
    )


def test_qdrant_client_empty_results(mock_qdrant_client, mock_sentence_transformer):
    """Test that the search method handles empty results."""
    # Set up the mock to return an empty list
    mock_qdrant_client.return_value.search.return_value = []

    # Create the client
    client = QdrantSearchClient()

    # Call the search method
    results = client.search("test query")

    # Verify the results
    assert len(results) == 0


def test_qdrant_client_none_payload(
    mock_qdrant_client, mock_sentence_transformer, mock_search_result
):
    """Test that the search method handles None payloads."""
    # Set up the mock to return a result with None payload
    mock_search_result.payload = None
    mock_qdrant_client.return_value.search.return_value = [mock_search_result]

    # Create the client
    client = QdrantSearchClient()

    # Call the search method
    results = client.search("test query")

    # Verify the results
    assert len(results) == 0
