from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

# Only mock QdrantClient to avoid dependency on a running Qdrant server
with patch("qdrant_client.QdrantClient") as MockQdrantClient:
    # Set up mocks
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
    MockQdrantClient.return_value.search.return_value = [mock_result]

    # Now import the app
    from main import app

client = TestClient(app)


def test_vector_search_endpoint():
    """Test the search endpoint with mocked Qdrant client but real transformer."""
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

    # Verify result content
    assert len(data["results"]) == 1
    assert data["total_results"] == 1

    result = data["results"][0]
    assert result["abstract_id"] == "123456"
    assert result["title"] == "Test Abstract"
    assert result["score"] == 0.95
    assert result["journal"] == "Test Journal"


def test_vector_search_error_handling():
    """Test error handling in the search endpoint."""
    # Mock an exception in the search
    with patch("src.routes.qdrant_client.search", side_effect=Exception("Test error")):
        response = client.get("/search/vector?query=error+test")

        # Check that we get a 500 error
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Test error" in data["detail"]
