"""Tests for the retry_failed.py script."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.retry_failed import retry_failed_urls


@pytest.fixture
def mock_failed_urls_file(tmp_path):
    """Create a mock failed_urls.json file."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    failed_urls = [
        "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
        "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
        "http://www.ncbi.nlm.nih.gov/pubmed/11223344",
    ]

    failed_urls_file = data_dir / "failed_urls.json"
    with open(failed_urls_file, "w") as f:
        json.dump(failed_urls, f)

    return {
        "data_dir": data_dir,
        "failed_urls_file": failed_urls_file,
        "failed_urls": failed_urls,
    }


@pytest.mark.asyncio
async def test_retry_failed_urls_success(
    mock_pubmed_client, mock_pubmed_abstract, mock_failed_urls_file
):
    """Test successful retry of failed URLs."""
    # Setup
    mock_pubmed_client.get_abstract_by_id.return_value = mock_pubmed_abstract

    # Set up so all URLs are successfully fetched
    with (
        patch(
            "src.retry_failed.BioPythonPubMedClient",
            return_value=mock_pubmed_client,
        ),
        patch("src.retry_failed.DataFetcher") as mock_data_fetcher_class,
    ):
        # Configure mock DataFetcher
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.fetch_all_abstracts = AsyncMock()
        mock_data_fetcher.fetch_all_abstracts.return_value = [
            mock_pubmed_abstract
        ] * len(mock_failed_urls_file["failed_urls"])
        mock_data_fetcher.failed_urls = set()  # No failures
        mock_data_fetcher.abstracts_dir = (
            mock_failed_urls_file["data_dir"] / "abstracts"
        )

        mock_data_fetcher_class.return_value = mock_data_fetcher

        # Run the function under test
        result = await retry_failed_urls(
            email="test@example.com", data_dir=str(mock_failed_urls_file["data_dir"])
        )

        # Assertions
        assert result == len(mock_failed_urls_file["failed_urls"])

        # Check if DataFetcher was initialized correctly
        mock_data_fetcher_class.assert_called_once()

        # Verify fetch_all_abstracts was called with the correct URLs
        mock_data_fetcher.fetch_all_abstracts.assert_called_once()
        called_urls = mock_data_fetcher.fetch_all_abstracts.call_args[0][0]
        assert isinstance(called_urls, set)
        assert called_urls == set(mock_failed_urls_file["failed_urls"])

        # Verify failed_urls.json file is removed when all fetches succeed
        assert not mock_failed_urls_file["failed_urls_file"].exists()


@pytest.mark.asyncio
async def test_retry_failed_urls_partial_success(
    mock_pubmed_client, mock_pubmed_abstract, mock_failed_urls_file
):
    """Test partial success when retrying failed URLs."""
    # Setup
    mock_pubmed_client.get_abstract_by_id.return_value = mock_pubmed_abstract

    # Set up for partial success (only 2 of 3 URLs succeed)
    with (
        patch(
            "src.retry_failed.BioPythonPubMedClient",
            return_value=mock_pubmed_client,
        ),
        patch("src.retry_failed.DataFetcher") as mock_data_fetcher_class,
    ):
        # Configure mock DataFetcher
        mock_data_fetcher = MagicMock()
        mock_data_fetcher.fetch_all_abstracts = AsyncMock()
        mock_data_fetcher.fetch_all_abstracts.return_value = [
            mock_pubmed_abstract
        ] * 2  # Only 2 succeed

        # One URL still fails
        remaining_failed_url = mock_failed_urls_file["failed_urls"][0]
        mock_data_fetcher.failed_urls = {remaining_failed_url}
        mock_data_fetcher.abstracts_dir = (
            mock_failed_urls_file["data_dir"] / "abstracts"
        )

        mock_data_fetcher_class.return_value = mock_data_fetcher

        # Run the function under test
        result = await retry_failed_urls(
            email="test@example.com", data_dir=str(mock_failed_urls_file["data_dir"])
        )

        # Assertions
        assert result == 2  # 2 out of 3 succeeded

        # Check if failed_urls.json was updated with the remaining failed URL
        assert mock_failed_urls_file["failed_urls_file"].exists()
        with open(mock_failed_urls_file["failed_urls_file"], "r") as f:
            updated_failed_urls = json.load(f)

        assert len(updated_failed_urls) == 1
        assert updated_failed_urls[0] == remaining_failed_url


@pytest.mark.asyncio
async def test_retry_failed_urls_no_file(mock_pubmed_client):
    """Test handling of missing failed_urls.json file."""
    # Setup a temp directory with no failed_urls.json
    tmp_dir = Path(os.path.join(os.path.dirname(__file__), "temp_test_dir"))
    tmp_dir.mkdir(exist_ok=True)

    try:
        with patch(
            "src.retry_failed.BioPythonPubMedClient",
            return_value=mock_pubmed_client,
        ):
            # Run the function under test
            result = await retry_failed_urls(
                email="test@example.com", data_dir=str(tmp_dir)
            )

            # Assertions
            assert result == 0  # No URLs processed
    finally:
        # Cleanup
        if tmp_dir.exists():
            import shutil

            shutil.rmtree(tmp_dir)


@pytest.mark.asyncio
async def test_retry_failed_urls_empty_file(mock_pubmed_client, mock_failed_urls_file):
    """Test handling of empty failed_urls.json file."""
    # Setup - empty the failed_urls.json file
    with open(mock_failed_urls_file["failed_urls_file"], "w") as f:
        json.dump([], f)

    with patch(
        "src.retry_failed.BioPythonPubMedClient",
        return_value=mock_pubmed_client,
    ):
        # Run the function under test
        result = await retry_failed_urls(
            email="test@example.com", data_dir=str(mock_failed_urls_file["data_dir"])
        )

        # Assertions
        assert result == 0  # No URLs processed
