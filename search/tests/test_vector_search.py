from unittest.mock import patch

from fastapi.testclient import TestClient
from src.models.models import SearchResult

# Now we'll patch our search client instead of the QdrantClient directly
with patch("src.routes.search_client") as mock_search_client:
    # Create an actual SearchResult instance for the mock to return
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

    # Set up the mock search client
    mock_search_client.search.return_value = [search_result]

    # Now import the app
    from main import app

client = TestClient(app)


def test_vector_search_endpoint():
    """Test the search endpoint with mocked search client."""
    # Make a request to the search endpoint
    response = client.get("/search/vector?query=test+query&limit=5")

    # Print error response for debugging
    if response.status_code != 200:
        print(f"Error response: {response.json()}")

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
    # Mock an exception in the search client
    with patch("src.routes.search_client.search", side_effect=Exception("Test error")):
        response = client.get("/search/vector?query=error+test")

        # Check that we get a 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]
