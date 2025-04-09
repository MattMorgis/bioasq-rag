from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from src.embedding.sentence_tranformer_embedder import SentenceTransformerEmbedder
from src.models.document import (
    Document,
    DocumentChunk,
    EmbeddedDocumentChunk,
)


@pytest.fixture
def sample_pubmed_abstract():
    """Return a sample Document for testing."""
    return Document(
        id="123456",
        title="Sample Medical Abstract for Testing",
        text="This is a sample abstract about a medical topic.",
        url="https://www.ncbi.nlm.nih.gov/pubmed/123456",
        publication_date="2023-01-15",
        journal="Journal of Medical Testing",
        authors=["Smith, J", "Johnson, A"],
        keywords=["testing", "embeddings"],
        mesh_terms=["Testing", "Embeddings"],
        doi="10.1234/test.5678",
    )


@pytest.fixture
def sample_pubmed_chunks(sample_pubmed_abstract):
    """Return sample DocumentChunk objects for testing."""
    chunk1 = DocumentChunk(
        chunk_id=f"{sample_pubmed_abstract.id}-1",
        text="This is a sample abstract",
        document=sample_pubmed_abstract,
        metadata={"position": 1},
    )

    chunk2 = DocumentChunk(
        chunk_id=f"{sample_pubmed_abstract.id}-2",
        text="about a medical topic.",
        document=sample_pubmed_abstract,
        metadata={"position": 2},
    )

    return [chunk1, chunk2]


# Mock the entire SentenceTransformerEmbedder class to avoid loading actual models
@patch("src.embedding.sentence_tranformer_embedder.SentenceTransformer")
def test_embed_batch(mock_sentence_transformer, sample_pubmed_chunks):
    """Test that embed_batch correctly embeds chunks."""
    # Create mock embeddings
    mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    # Setup the mock
    mock_model = MagicMock()
    mock_model.encode.return_value = mock_embeddings
    mock_sentence_transformer.return_value = mock_model

    # Create embedder with the mock
    model_name = "all-MiniLM-L6-v2"  # Use a default model name
    embedder = SentenceTransformerEmbedder(model_name=model_name)

    # Run embed_batch
    embedded_chunks = embedder.embed_batch(sample_pubmed_chunks)

    # Assert model was called correctly
    mock_model.encode.assert_called_once_with(
        [chunk.text for chunk in sample_pubmed_chunks], batch_size=32
    )

    # Check the results
    assert len(embedded_chunks) == 2

    # Check first embedded chunk
    assert isinstance(embedded_chunks[0], EmbeddedDocumentChunk)
    assert embedded_chunks[0].chunk == sample_pubmed_chunks[0]
    assert np.array_equal(embedded_chunks[0].embedding, mock_embeddings[0])
    assert embedded_chunks[0].embedding_model == model_name

    # Check second embedded chunk
    assert isinstance(embedded_chunks[1], EmbeddedDocumentChunk)
    assert embedded_chunks[1].chunk == sample_pubmed_chunks[1]
    assert np.array_equal(embedded_chunks[1].embedding, mock_embeddings[1])
    assert embedded_chunks[1].embedding_model == model_name
