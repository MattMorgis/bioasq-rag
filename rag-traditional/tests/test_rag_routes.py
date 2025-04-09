from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from src.search.models import SearchResponse, SearchResult


@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app"""
    from fastapi import FastAPI
    from src.routes import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def mock_search_result():
    """Create a mock search result"""
    return SearchResult(
        abstract_id="123456",
        title="Test Title",
        text="This is a test abstract about biomedical research.",
        score=0.95,
        url="https://pubmed.ncbi.nlm.nih.gov/123456/",
        publication_date="2023-01-01",
        journal="Test Journal",
        authors=["Author One", "Author Two"],
    )


@pytest.fixture
def mock_search_response(mock_search_result):
    """Create a mock search response with one result"""
    return SearchResponse(
        results=[mock_search_result],
        query="test query",
        total_results=1,
    )


@pytest.fixture
def mock_llm_response():
    """Create a mock LLM response"""
    return "Based on the provided abstract, the answer to the query is..."


@patch("src.routes.save_prompt_to_file")
@patch("src.routes.VectorSearchClient")
@patch("src.routes.OpenAILLM")
def test_rag_query_endpoint(
    mock_llm_class,
    mock_search_client_class,
    mock_save_prompt,
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

    mock_save_prompt.return_value = "mock/path/to/prompt.txt"

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

    # Verify save_prompt was not called (default log_prompt=False)
    mock_save_prompt.assert_not_called()


@patch("src.routes.save_prompt_to_file")
@patch("src.routes.VectorSearchClient")
@patch("src.routes.OpenAILLM")
def test_rag_query_endpoint_with_prompt_logging(
    mock_llm_class,
    mock_search_client_class,
    mock_save_prompt,
    test_client,
    mock_search_response,
    mock_llm_response,
):
    """Test the RAG query endpoint with prompt logging enabled"""
    # Setup mocks
    mock_search_client = AsyncMock()
    mock_search_client.search.return_value = mock_search_response
    mock_search_client_class.return_value = mock_search_client

    mock_llm = AsyncMock()
    mock_llm.prompt.return_value = mock_llm_response
    mock_llm_class.return_value = mock_llm

    mock_save_prompt.return_value = "mock/path/to/prompt.txt"

    # Test data with log_prompt enabled
    test_request = {
        "query": "What is the treatment for disease X?",
        "max_results": 3,
        "temperature": 0.5,
        "log_prompt": True,
    }

    # Make request
    response = test_client.post("/rag/query", json=test_request)

    # Assertions
    assert response.status_code == 200

    # Verify save_prompt was called
    mock_save_prompt.assert_called_once()
    call_args = mock_save_prompt.call_args[0]
    assert test_request["query"] == call_args[0]  # query
    assert isinstance(call_args[1], str)  # prompt
    assert isinstance(call_args[2], str)  # system_message


@patch("src.routes.save_prompt_to_file")
@patch("src.routes.VectorSearchClient")
@patch("src.routes.OpenAILLM")
def test_rag_query_stream_endpoint(
    mock_llm_class,
    mock_search_client_class,
    mock_save_prompt,
    test_client,
    mock_search_response,
):
    """Test the streaming RAG query endpoint with mocked dependencies"""
    # Setup mocks
    mock_search_client = AsyncMock()
    mock_search_client.search.return_value = mock_search_response
    mock_search_client_class.return_value = mock_search_client

    # Setup streaming mock
    mock_llm = AsyncMock()

    # Create a proper async generator for testing that accepts the same parameters as prompt_stream
    async def mock_stream_generator(
        prompt, system_message=None, temperature=0.7, max_tokens=None
    ):
        chunks = ["This", " is", " a", " streaming", " response"]
        for chunk in chunks:
            yield chunk

    # Use the proper async generator
    mock_llm.prompt_stream = mock_stream_generator
    mock_llm_class.return_value = mock_llm

    mock_save_prompt.return_value = "mock/path/to/prompt.txt"

    # Test data
    test_request = {
        "query": "What is the treatment for disease X?",
        "max_results": 3,
        "temperature": 0.5,
    }

    # Make request
    response = test_client.post("/rag/query/stream", json=test_request)

    # Assertions
    assert response.status_code == 200

    # Parse the streaming response as plain text
    content = response.content.decode()
    assert "This is a streaming response" in content
    assert "--- Sources ---" in content

    # Check that the source information is included
    assert "Test Title" in content
    assert "(2023-01-01)" in content

    # Verify mock calls
    mock_search_client.search.assert_called_once_with(
        query=test_request["query"], limit=test_request["max_results"]
    )

    # Verify save_prompt was not called (default log_prompt=False)
    mock_save_prompt.assert_not_called()


@patch("src.routes.save_prompt_to_file")
@patch("src.routes.VectorSearchClient")
@patch("src.routes.OpenAILLM")
def test_rag_query_stream_endpoint_with_prompt_logging(
    mock_llm_class,
    mock_search_client_class,
    mock_save_prompt,
    test_client,
    mock_search_response,
):
    """Test the streaming RAG query endpoint with prompt logging enabled"""
    # Setup mocks
    mock_search_client = AsyncMock()
    mock_search_client.search.return_value = mock_search_response
    mock_search_client_class.return_value = mock_search_client

    # Setup streaming mock
    mock_llm = AsyncMock()

    # Create a proper async generator for testing that accepts the same parameters as prompt_stream
    async def mock_stream_generator(
        prompt, system_message=None, temperature=0.7, max_tokens=None
    ):
        chunks = ["This", " is", " a", " streaming", " response"]
        for chunk in chunks:
            yield chunk

    # Use the proper async generator
    mock_llm.prompt_stream = mock_stream_generator
    mock_llm_class.return_value = mock_llm

    mock_save_prompt.return_value = "mock/path/to/prompt.txt"

    # Test data with log_prompt enabled
    test_request = {
        "query": "What is the treatment for disease X?",
        "max_results": 3,
        "temperature": 0.5,
        "log_prompt": True,
    }

    # Make request
    response = test_client.post("/rag/query/stream", json=test_request)

    # Assertions
    assert response.status_code == 200

    # Verify save_prompt was called
    mock_save_prompt.assert_called_once()
    call_args = mock_save_prompt.call_args[0]
    assert test_request["query"] == call_args[0]  # query
    assert isinstance(call_args[1], str)  # prompt
    assert isinstance(call_args[2], str)  # system_message
