import urllib.error
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data_pipelines.clients.biopython_pubmed_client import BioPythonPubMedClient
from src.data_pipelines.clients.pubmed_client import (
    PubMedClientError,
    PubMedRateLimitError,
)


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
async def test_get_abstract_by_id_rate_limit_error(biopython_pubmed_client):
    """Test that HTTP 429 errors raise the specific PubMedRateLimitError."""
    http_error = urllib.error.HTTPError(
        url="https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        code=429,
        msg="Too Many Requests",
        hdrs={},
        fp=None,
    )

    with patch("Bio.Entrez.efetch", side_effect=http_error):
        # Verify that a PubMedRateLimitError is raised
        with pytest.raises(PubMedRateLimitError) as exc_info:
            await biopython_pubmed_client.get_abstract_by_id("12345")

        # Check error details
        assert "Rate limit exceeded" in str(exc_info.value)
        assert exc_info.value.status_code == 429


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_success(
    biopython_pubmed_client, mock_record, mock_handle
):
    """Test successful retrieval of multiple abstracts."""
    record1 = mock_record
    record2 = {**mock_record, "PMID": "67890", "TI": "Second Test Article"}

    with patch.object(
        biopython_pubmed_client, "get_abstract_by_id", new_callable=AsyncMock
    ) as mock_get_abstract:
        # Configure the mock to return different abstract for each ID
        mock_get_abstract.side_effect = [
            # Format each record with _format_record
            {
                "id": "12345",
                "title": "Test Article Title",
                "abstract": "This is a test abstract for the BioPython PubMed client.",
                "authors": ["Smith, John", "Doe, Jane"],
                "publication_date": "2024 Jan 15",
                "journal": "Journal of Testing",
                "doi": "10.1234/test.12345",
                "keywords": ["Bioinformatics", "Testing", "Python"],
                "mesh_terms": ["Bioinformatics", "Testing", "Python"],
            },
            {
                "id": "67890",
                "title": "Second Test Article",
                "abstract": "This is a test abstract for the BioPython PubMed client.",
                "authors": ["Smith, John", "Doe, Jane"],
                "publication_date": "2024 Jan 15",
                "journal": "Journal of Testing",
                "doi": "10.1234/test.12345",
                "keywords": ["Bioinformatics", "Testing", "Python"],
                "mesh_terms": ["Bioinformatics", "Testing", "Python"],
            },
        ]

        # Call the method
        results = await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        # Verify the results
        assert len(results) == 2
        assert results[0]["id"] == "12345"
        assert results[1]["id"] == "67890"
        assert results[1]["title"] == "Second Test Article"

        # Verify get_abstract_by_id was called for each ID
        assert mock_get_abstract.call_count == 2
        mock_get_abstract.assert_any_call("12345")
        mock_get_abstract.assert_any_call("67890")


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_missing_records(biopython_pubmed_client):
    """Test handling when some records are not found."""
    with patch.object(
        biopython_pubmed_client, "get_abstract_by_id", new_callable=AsyncMock
    ) as mock_get_abstract:
        # Configure mock to raise exception for some IDs
        async def side_effect(pubmed_id):
            if pubmed_id == "12345":
                return {
                    "id": "12345",
                    "title": "Test Article Title",
                    "abstract": "This is a test abstract.",
                    "authors": ["Smith, John"],
                    "publication_date": "2024 Jan 15",
                    "journal": "Journal of Testing",
                    "doi": "10.1234/test.12345",
                    "keywords": ["Testing"],
                    "mesh_terms": ["Testing"],
                }
            else:
                raise PubMedClientError(f"No abstract found for ID: {pubmed_id}")

        mock_get_abstract.side_effect = side_effect

        # Call the method - should not raise an exception but return partial results
        results = await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        # Should only have one result
        assert len(results) == 1
        assert results[0]["id"] == "12345"

        # Verify get_abstract_by_id was called for each ID
        assert mock_get_abstract.call_count == 2
        mock_get_abstract.assert_any_call("12345")
        mock_get_abstract.assert_any_call("67890")


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_api_error(biopython_pubmed_client):
    """Test handling when API errors occur for all IDs."""
    with patch.object(
        biopython_pubmed_client, "get_abstract_by_id", new_callable=AsyncMock
    ) as mock_get_abstract:
        # Configure mock to raise exception for all IDs
        mock_get_abstract.side_effect = PubMedClientError("API error")

        # Call the method - should not raise but return empty results
        results = await biopython_pubmed_client.get_abstracts_by_ids(["12345", "67890"])

        # Should have no results
        assert len(results) == 0

        # Verify get_abstract_by_id was called for each ID
        assert mock_get_abstract.call_count == 2
