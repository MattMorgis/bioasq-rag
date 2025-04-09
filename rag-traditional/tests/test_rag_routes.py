from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from src.search.models import SearchResponse, SearchResult


# Mock the main app for testing
@pytest.fixture
def test_client():
    from fastapi import FastAPI
    from src.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_search_response():
    """Create a mock search response with dummy data"""
    return SearchResponse(
        results=[
            SearchResult(
                abstract_id="123456",
                title="Test Medical Paper",
                text="This is a test abstract with medical information.",
                score=0.95,
                url="https://example.com/paper/123456",
                publication_date="2023-01-01",
                journal="Journal of Medical Tests",
                authors=["Dr. Test", "Prof. Example"],
            )
        ],
        query="test query",
        total_results=1,
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response"""
    return "Based on the provided abstract, the answer to the query is..."


@patch("src.routes.VectorSearchClient")
@patch("src.routes.OpenAILLM")
def test_rag_query_endpoint(
    mock_llm_class,
    mock_search_client_class,
    test_client,
    mock_search_response,
    mock_llm_response,
):
    """Test the RAG query endpoint with mocked dependencies"""
    # Setup mocks
    mock_search_client = AsyncMock()
    mock_search_client.search.return_value = mock_search_response
    mock_search_client_class.return_value = mock_search_client

    mock_llm = AsyncMock()
    mock_llm.prompt.return_value = mock_llm_response
    mock_llm_class.return_value = mock_llm

    # Test data
    test_request = {
        "query": "What is the treatment for disease X?",
        "max_results": 3,
        "temperature": 0.5,
    }

    # Make request
    response = test_client.post("/rag/query", json=test_request)

    # Assertions
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == test_request["query"]
    assert data["answer"] == mock_llm_response
    assert len(data["sources"]) == 1

    # Verify mock calls
    mock_search_client.search.assert_called_once_with(
        query=test_request["query"], limit=test_request["max_results"]
    )

    # Check that LLM was called with appropriate arguments
    mock_llm.prompt.assert_called_once()
    call_args = mock_llm.prompt.call_args[1]
    assert call_args["temperature"] == test_request["temperature"]
    assert test_request["query"] in call_args["prompt"]
