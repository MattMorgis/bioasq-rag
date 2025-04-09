"""Unit tests for vector search client."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from src.search.models import SearchResponse, SearchResult

# Import the classes to be tested - using relative imports per project convention
from src.search.vector_client import VectorSearchClient, search_documents


@pytest.fixture
def mock_search_response():
    """Return a mock search response for testing."""
    return {
        "results": [
            {
                "abstract_id": "12345678",
                "title": "Test Article",
                "text": "This is a test abstract about biomedical information.",
                "score": 0.92,
                "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
                "publication_date": "2021-01-01",
                "journal": "Test Journal of Medicine",
                "authors": ["Smith, John", "Doe, Jane"],
            },
            {
                "abstract_id": "87654321",
                "title": "Another Test Article",
                "text": "This is another test abstract.",
                "score": 0.85,
                "url": "https://pubmed.ncbi.nlm.nih.gov/87654321/",
                "publication_date": "2020-12-15",
                "journal": "Medical Research Journal",
                "authors": ["Johnson, Robert"],
            },
        ],
        "query": "test query",
        "total_results": 2,
    }


@pytest.mark.asyncio
async def test_vector_search_client_init():
    """Test vector search client initialization."""
    # Test with default URL
    client = VectorSearchClient()
    assert client.base_url == "http://localhost:8000"

    # Test with custom URL
    custom_url = "https://custom-search-api.example.com"
    client = VectorSearchClient(base_url=custom_url)
    assert client.base_url == custom_url

    # Test with environment variable
    with patch.dict(
        os.environ, {"SEARCH_API_URL": "https://env-search-api.example.com"}
    ):
        client = VectorSearchClient()
        assert client.base_url == "https://env-search-api.example.com"


@pytest.mark.asyncio
async def test_search_success(mock_search_response):
    """Test successful search query."""
    client = VectorSearchClient()

    # Mock the httpx AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Use MagicMock instead of AsyncMock for json() to return the data directly
    mock_response.json.return_value = mock_search_response
    # Use a synchronous mock for raise_for_status to avoid coroutine issues
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.return_value = mock_response

        # Call the search method
        result = await client.search("test query")

        # Verify the request was made correctly
        mock_client_instance.get.assert_called_once_with(
            "http://localhost:8000/search/vector",
            params={"query": "test query", "limit": 10},
            timeout=30.0,
        )

        # Verify the response was processed correctly
        assert isinstance(result, SearchResponse)
        assert result.query == "test query"
        assert result.total_results == 2
        assert len(result.results) == 2
        assert result.results[0].abstract_id == "12345678"
        assert result.results[0].title == "Test Article"
        assert result.results[0].score == 0.92


@pytest.mark.asyncio
async def test_search_http_error():
    """Test handling of HTTP errors during search."""
    client = VectorSearchClient()

    # Mock the httpx AsyncClient to raise an HTTPStatusError
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    # Create the error
    mock_error = httpx.HTTPStatusError(
        "Server error", request=MagicMock(), response=mock_response
    )
    # Make raise_for_status raise the error synchronously
    mock_response.raise_for_status.side_effect = mock_error

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.return_value = mock_response

        # Call the search method and expect an exception
        with pytest.raises(httpx.HTTPStatusError):
            await client.search("test query")


@pytest.mark.asyncio
async def test_search_connection_error():
    """Test handling of connection errors during search."""
    client = VectorSearchClient()

    # Mock the httpx AsyncClient to raise a ConnectError
    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.side_effect = httpx.ConnectError("Failed to connect")

        # Call the search method and expect a ConnectionError
        with pytest.raises(ConnectionError):
            await client.search("test query")


@pytest.mark.asyncio
async def test_search_validation_error(mock_search_response):
    """Test handling of validation errors when parsing response."""
    client = VectorSearchClient()

    # Create an invalid response (missing required fields)
    invalid_response = {
        "results": [{"abstract_id": "12345678"}],  # Missing required fields
        "query": "test query",
        # Missing total_results field
    }

    # Mock the httpx AsyncClient
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = invalid_response
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient") as mock_client:
        mock_client_instance = AsyncMock()
        mock_client.return_value.__aenter__.return_value = mock_client_instance
        mock_client_instance.get.return_value = mock_response

        # Call the search method and expect a ValidationError
        with pytest.raises(ValidationError):
            await client.search("test query")


@pytest.mark.asyncio
async def test_search_documents_helper(mock_search_response):
    """Test the search_documents helper function."""
    # Mock the VectorSearchClient.search method
    with patch("src.search.vector_client.VectorSearchClient") as mock_client_cls:
        mock_client = mock_client_cls.return_value

        # Create a SearchResponse object for the mock return value
        search_response = SearchResponse(**mock_search_response)
        # Set up the mock to return the actual SearchResponse object, not a coroutine
        mock_client.search = AsyncMock(return_value=search_response)

        # Call the helper function
        results = await search_documents("test query", limit=5)

        # Verify the client was called correctly
        mock_client.search.assert_called_once_with(query="test query", limit=5)

        # Verify the results
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].abstract_id == "12345678"
        assert results[1].abstract_id == "87654321"
