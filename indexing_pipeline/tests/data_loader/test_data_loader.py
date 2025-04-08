from pathlib import Path

import pytest
from src.data_loader import DataLoader
from src.models.pubmed import Document


class TestDataLoader:
    """Test suite for the DataLoader class."""

    def test_init_with_corpus_path(self):
        """Test initialization with a corpus path."""
        corpus_path = Path("/path/to/corpus.jsonl")
        loader = DataLoader(corpus_path=corpus_path)
        assert loader.corpus_path == corpus_path

    def test_init_without_corpus_path(self):
        """Test initialization without a corpus path."""
        loader = DataLoader()
        assert loader.corpus_path is None

    def test_load_abstracts_from_file(self, temp_corpus_file):
        """Test loading abstracts from a file."""
        loader = DataLoader(corpus_path=temp_corpus_file)
        abstracts_gen = loader.load_abstracts_from_file()
        abstracts = next(abstracts_gen)

        assert len(abstracts) == 5
        assert all(isinstance(abstract, Document) for abstract in abstracts)

        # Check the first abstract
        first_abstract = abstracts[0]
        assert first_abstract.id == "123456"
        assert first_abstract.title == "Sample Medical Abstract 1"
        assert first_abstract.text == "This is sample abstract 1 about a medical topic."
        assert first_abstract.url == "https://www.ncbi.nlm.nih.gov/pubmed/123456"
        assert first_abstract.publication_date == "2023-01-15"
        assert first_abstract.journal == "Journal of Medical Testing"
        assert first_abstract.authors == ["Smith, J", "Johnson, A"]
        assert first_abstract.keywords == ["testing", "embeddings"]
        assert first_abstract.mesh_terms == ["Testing", "Embeddings"]
        assert first_abstract.doi == "10.1234/test.5678"

    def test_load_abstracts_with_batch_size(self, temp_corpus_file):
        """Test loading abstracts with a batch size."""
        loader = DataLoader(corpus_path=temp_corpus_file)
        abstracts_gen = loader.load_abstracts_from_file(batch_size=2)

        # First batch
        batch1 = next(abstracts_gen)
        assert len(batch1) == 2

        # Second batch
        batch2 = next(abstracts_gen)
        assert len(batch2) == 2

        # Third batch (should have only 1 item)
        batch3 = next(abstracts_gen)
        assert len(batch3) == 1

        # No more batches
        with pytest.raises(StopIteration):
            next(abstracts_gen)

    def test_load_abstracts_with_limit(self, temp_corpus_file):
        """Test loading abstracts with a limit."""
        loader = DataLoader(corpus_path=temp_corpus_file)
        abstracts_gen = loader.load_abstracts_from_file(limit=3)
        abstracts = next(abstracts_gen)

        assert len(abstracts) == 3

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised when corpus file doesn't exist."""
        non_existent_path = Path("/non/existent/path.jsonl")
        loader = DataLoader(corpus_path=non_existent_path)

        with pytest.raises(FileNotFoundError):
            next(loader.load_abstracts_from_file())
