"""Tests for the PubMedURLCollector class."""

import json
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import mock_open, patch

import pytest

from src.data_collection.pubmed_url_collector import PubMedURLCollector


@pytest.fixture
def mock_bioasq_data() -> Dict[str, List[Dict[str, Any]]]:
    """Return mock BioASQ data for testing."""
    return {
        "questions": [
            {
                "body": "Test question 1?",
                "documents": [
                    "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
                    "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
                    "http://www.ncbi.nlm.nih.gov/pubmed/11223344",
                ],
            },
            {
                "body": "Test question 2?",
                "documents": [
                    "http://www.ncbi.nlm.nih.gov/pubmed/12345678",  # Duplicate URL
                    "http://www.ncbi.nlm.nih.gov/pubmed/55667788",
                ],
            },
        ]
    }


class TestPubMedURLCollector:
    """Tests for the PubMedURLCollector class."""

    def test_init(self):
        """Test PubMedURLCollector initialization."""
        collector = PubMedURLCollector(data_dir="/test/path")
        assert collector.data_dir == Path("/test/path")
        assert collector.training_dir == Path("/test/path/BioASQ-12b/training")
        assert collector.goldset_dir == Path("/test/path/BioASQ-12b/goldset")
        assert collector.unique_urls == set()

    def test_extract_urls_from_file(self, mock_bioasq_data):
        """Test _extract_urls_from_file method."""
        collector = PubMedURLCollector()

        # Mock file opening and JSON loading
        mock_file = mock_open(read_data=json.dumps(mock_bioasq_data))

        with (
            patch("builtins.open", mock_file),
            patch("json.load", return_value=mock_bioasq_data),
        ):
            urls = collector._extract_urls_from_file(Path("mock_file.json"))

        assert urls == {
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
            "http://www.ncbi.nlm.nih.gov/pubmed/11223344",
            "http://www.ncbi.nlm.nih.gov/pubmed/55667788",
        }
        assert len(urls) == 4  # Checking deduplication

    def test_extract_urls_from_file_error_handling(self):
        """Test error handling in _extract_urls_from_file method."""
        collector = PubMedURLCollector()

        # Test with a file that doesn't exist
        with patch("builtins.open", side_effect=FileNotFoundError):
            urls = collector._extract_urls_from_file(Path("nonexistent_file.json"))

        assert urls == set()

    def test_collect_urls(self):
        """Test collect_urls method."""
        collector = PubMedURLCollector()

        # Mock the _extract_urls_from_file method
        training_urls = {
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
        }
        goldset_urls = {
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/99887766",
        }

        with (
            patch.object(collector, "_extract_urls_from_file") as mock_extract,
            patch("pathlib.Path.glob") as mock_glob,
        ):
            # Setup mock for training files
            mock_glob.side_effect = [
                [Path("training_file.json")],  # Training files
                [
                    Path("goldset_file1.json"),
                    Path("goldset_file2.json"),
                ],  # Goldset files
            ]

            # Setup mock for _extract_urls_from_file
            mock_extract.side_effect = [training_urls, goldset_urls, set()]

            # Call the method
            result = collector.collect_urls()

        # Verify results
        assert result == {
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
            "http://www.ncbi.nlm.nih.gov/pubmed/99887766",
        }
        assert len(result) == 3  # Check deduplication

    def test_save_urls_to_file(self, tmp_path):
        """Test save_urls_to_file method."""
        collector = PubMedURLCollector()
        collector.unique_urls = {
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
            "http://www.ncbi.nlm.nih.gov/pubmed/11223344",
        }

        # Create a temporary output file
        output_file = tmp_path / "test_urls.txt"

        # Call the method
        collector.save_urls_to_file(str(output_file))

        # Verify the output file
        assert output_file.exists()

        # Read the file and check its contents
        with open(output_file, "r") as f:
            lines = f.readlines()

        # Check that all URLs are in the file (in sorted order)
        expected_urls = sorted(
            [
                "http://www.ncbi.nlm.nih.gov/pubmed/11223344",
                "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
                "http://www.ncbi.nlm.nih.gov/pubmed/87654321",
            ]
        )

        for i, url in enumerate(expected_urls):
            assert lines[i].strip() == url
