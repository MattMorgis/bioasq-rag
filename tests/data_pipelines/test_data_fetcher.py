from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_pipelines.data_fetcher import DataFetcher
from src.data_pipelines.pubmed_client import PubMedClient, PubMedClientError


@pytest.fixture
def mock_pubmed_client():
    """Return a mock PubMedClient."""
    client = MagicMock(spec=PubMedClient)
    client.get_abstract_by_id = AsyncMock()
    client.get_abstracts_by_ids = AsyncMock()
    return client


@pytest.fixture
def mock_pubmed_abstract() -> Dict[str, Any]:
    """Return a mock abstract."""
    return {
        "id": "15858239",
        "title": "The role of ret gene in the pathogenesis of Hirschsprung disease",
        "abstract": "This is a test abstract about Hirschsprung disease.",
        "authors": ["Smigiel R", "Patkowski D", "Slezak R"],
        "publication_date": "2004 Jul-Sep",
        "journal": "Med Wieku Rozwoj",
        "doi": "10.1000/test.12345",
        "keywords": ["Hirschsprung Disease", "Genetics", "RET Gene"],
    }


@pytest.fixture
def data_fetcher(mock_pubmed_client):
    """Return a DataFetcher with a mock PubMedClient."""
    return DataFetcher(mock_pubmed_client)


@pytest.mark.asyncio
async def test_extract_pubmed_id(data_fetcher):
    """Test extracting PubMed ID from URL."""
    url = "http://www.ncbi.nlm.nih.gov/pubmed/15858239"
    pubmed_id = data_fetcher._extract_pubmed_id(url)
    assert pubmed_id == "15858239"


@pytest.mark.asyncio
async def test_fetch_single_abstract_success(
    data_fetcher, mock_pubmed_client, mock_pubmed_abstract
):
    """Test successful fetching of a single abstract."""
    # Configure the mock client
    mock_pubmed_client.get_abstract_by_id.return_value = mock_pubmed_abstract

    # Call the method
    url = "http://www.ncbi.nlm.nih.gov/pubmed/15858239"
    result = await data_fetcher.fetch_single_abstract(url)

    # Verify the result
    assert result is not None
    assert result["id"] == "15858239"
    assert (
        result["title"]
        == "The role of ret gene in the pathogenesis of Hirschsprung disease"
    )

    # Verify the client was called with the correct ID
    mock_pubmed_client.get_abstract_by_id.assert_called_once_with("15858239")


@pytest.mark.asyncio
async def test_fetch_single_abstract_error(data_fetcher, mock_pubmed_client):
    """Test error handling during single abstract fetching."""
    # Configure the mock client to raise an exception
    mock_pubmed_client.get_abstract_by_id.side_effect = PubMedClientError("Test error")

    # Call the method
    url = "http://www.ncbi.nlm.nih.gov/pubmed/15858239"
    result = await data_fetcher.fetch_single_abstract(url)

    # Verify None result due to error
    assert result is None

    # Verify the client was called
    mock_pubmed_client.get_abstract_by_id.assert_called_once_with("15858239")


@pytest.mark.asyncio
async def test_run_success(data_fetcher, mock_pubmed_client, mock_pubmed_abstract):
    """Test successful run method."""
    # Configure the mock client
    mock_pubmed_client.get_abstract_by_id.return_value = mock_pubmed_abstract

    # Call the run method
    with patch("builtins.print") as mock_print:
        result = await data_fetcher.run()

    # Verify the result
    assert result is not None
    assert result["id"] == "15858239"
    assert (
        result["title"]
        == "The role of ret gene in the pathogenesis of Hirschsprung disease"
    )

    # Verify the client was called with the correct ID
    mock_pubmed_client.get_abstract_by_id.assert_called_once_with("15858239")

    # Verify print was called
    assert mock_print.call_count > 0


@pytest.mark.asyncio
async def test_run_error(data_fetcher, mock_pubmed_client):
    """Test error handling during run method."""
    # Configure the mock client to raise an exception
    mock_pubmed_client.get_abstract_by_id.side_effect = PubMedClientError("Test error")

    # Call the run method
    with patch("builtins.print") as mock_print:
        result = await data_fetcher.run()

    # Verify None result due to error
    assert result is None

    # Verify the client was called
    mock_pubmed_client.get_abstract_by_id.assert_called_once_with("15858239")

    # Verify print was called with error message
    mock_print.assert_called_with(
        f"\nFailed to fetch abstract from {data_fetcher.pubmed_url}"
    )
