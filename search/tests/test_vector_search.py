from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from main import app
from src.clients.qdrant_client import QdrantConnectionError
from src.models.models import SearchResult

# Create a test client
client = TestClient(app)

# Create a test search result
search_result = SearchResult(
    abstract_id="123456",
    title="Test Abstract",
    text="This is a test abstract about biomedical research.",
    score=0.95,
    url="https://pubmed.ncbi.nlm.nih.gov/123456/",
    publication_date="2022-01-01",
    journal="Test Journal",
    authors=["Author One", "Author Two"],
)


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies for all tests."""
    # Create a mock for search_client
    mock_client = MagicMock()
    mock_client.search.return_value = [search_result]

    # Apply the patch for the duration of the test
    with patch("src.routes.search_client", mock_client):
        yield


def test_vector_search_endpoint():
    """Test the search endpoint with mocked search client."""
    # Make a request to the search endpoint
    response = client.get("/search/vector?query=test+query&limit=5")

    # Check response status code and structure
    assert response.status_code == 200
    data = response.json()

    # Verify the response structure
    assert "results" in data
    assert "query" in data
    assert "total_results" in data

    # Verify query parameter was passed correctly
    assert data["query"] == "test query"

    # Verify that we have results (don't check exact count)
    assert len(data["results"]) > 0
    assert data["total_results"] > 0

    # If we have actual results, check one of them
    if len(data["results"]) > 0:
        result = data["results"][0]
        assert "abstract_id" in result
        assert "title" in result
        assert "score" in result
        assert "journal" in result


def test_vector_search_error_handling():
    """Test error handling in the search endpoint."""
    # Override the mock for this specific test
    with patch("src.routes.search_client.search", side_effect=Exception("Test error")):
        response = client.get("/search/vector?query=error+test")

        # Check that we get a 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]


def test_qdrant_connection_error_handling():
    """Test handling of Qdrant connection errors."""
    # Override the mock for this specific test
    with patch(
        "src.routes.search_client.search",
        side_effect=QdrantConnectionError("Qdrant server unavailable"),
    ):
        response = client.get("/search/vector?query=connection+error+test")

        # Check that we get a 503 error
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "unavailable" in data["detail"].lower()
