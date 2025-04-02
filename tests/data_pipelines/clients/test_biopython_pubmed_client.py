from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from src.data_pipelines.clients.biopython_pubmed_client import BioPythonPubMedClient
from src.data_pipelines.clients.pubmed_client import PubMedClientError


@pytest.fixture
def mock_record() -> Dict[str, Any]:
    """Return a mock PubMed record."""
    return {
        "PMID": "12345",
        "TI": "Test Article Title",
        "AB": "This is a test abstract for the BioPython PubMed client.",
        "AU": ["Smith, John", "Doe, Jane"],
        "DP": "2024 Jan 15",
        "JT": "Journal of Testing",
        "LID": "10.1234/test.12345 [doi]",
        "MH": ["Bioinformatics", "Testing", "Python"],
    }


@pytest.fixture
def mock_handle():
    """Create a mock handle for Entrez.efetch."""
    mock = MagicMock()
    mock.close = MagicMock()
    return mock


@pytest.fixture
def biopython_pubmed_client():
    """Return a BioPython PubMed client with a test email."""
    return BioPythonPubMedClient(email="test@example.com")


@pytest.mark.asyncio
async def test_get_abstract_by_id_success(
    biopython_pubmed_client, mock_record, mock_handle
):
    """Test successful retrieval of a single abstract."""
    with (
        patch("Bio.Entrez.efetch", return_value=mock_handle),
        patch("Bio.Medline.parse") as mock_parse,
    ):
        # Configure the mock to return our test record
        mock_parse.return_value = iter([mock_record])

        # Call the method
        result = await biopython_pubmed_client.get_abstract_by_id("12345")

        # Verify the result
        assert result["id"] == "12345"
        assert result["title"] == "Test Article Title"
        assert (
            result["abstract"]
            == "This is a test abstract for the BioPython PubMed client."
        )
        assert len(result["authors"]) == 2
        assert result["authors"][0] == "Smith, John"
        assert result["publication_date"] == "2024 Jan 15"
        assert result["journal"] == "Journal of Testing"
        assert result["doi"] == "10.1234/test.12345"
        assert "Bioinformatics" in result["keywords"]


@pytest.mark.asyncio
async def test_get_abstract_by_id_no_record(biopython_pubmed_client, mock_handle):
    """Test error handling when no record is found."""
    with (
        patch("Bio.Entrez.efetch", return_value=mock_handle),
        patch("Bio.Medline.parse") as mock_parse,
    ):
        # Configure the mock to return an empty iterator
        mock_parse.return_value = iter([])

        # Verify that an exception is raised
        with pytest.raises(PubMedClientError) as exc_info:
            await biopython_pubmed_client.get_abstract_by_id("99999")

        assert "Failed to retrieve abstract" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_abstract_by_id_api_error(biopython_pubmed_client):
    """Test error handling when the NCBI API returns an error."""
    with patch("Bio.Entrez.efetch", side_effect=Exception("API error")):
        # Verify that an exception is raised
        with pytest.raises(PubMedClientError) as exc_info:
            await biopython_pubmed_client.get_abstract_by_id("12345")

        assert "Failed to retrieve abstract" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_success(
    biopython_pubmed_client, mock_record, mock_handle
):
    """Test successful retrieval of multiple abstracts."""
    records = [
        mock_record,
        {**mock_record, "PMID": "67890", "TI": "Second Test Article"},
    ]

    with (
        patch("Bio.Entrez.efetch", return_value=mock_handle),
        patch("Bio.Medline.parse") as mock_parse,
    ):
        # Configure the mock to return our test records
        mock_parse.return_value = records

        # Call the method
        results = await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        # Verify the results
        assert len(results) == 2
        assert results[0]["id"] == "12345"
        assert results[1]["id"] == "67890"
        assert results[1]["title"] == "Second Test Article"


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_missing_records(
    biopython_pubmed_client, mock_handle
):
    """Test error handling when not all requested records are found."""
    with (
        patch("Bio.Entrez.efetch", return_value=mock_handle),
        patch("Bio.Medline.parse") as mock_parse,
    ):
        # Configure the mock to return fewer records than requested
        mock_parse.return_value = []

        # Verify that an exception is raised
        with pytest.raises(PubMedClientError) as exc_info:
            await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        assert "Failed to retrieve one or more abstracts" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_api_error(biopython_pubmed_client):
    """Test error handling when the NCBI API returns an error for multiple IDs."""
    with patch("Bio.Entrez.efetch", side_effect=Exception("API error")):
        # Verify that an exception is raised
        with pytest.raises(PubMedClientError) as exc_info:
            await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        assert "Failed to retrieve" in str(exc_info.value)
