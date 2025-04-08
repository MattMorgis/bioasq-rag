import pytest
from src.chunker.word_chunker import WordChunker
from src.models.pubmed import Document, DocumentChunk


@pytest.fixture
def sample_pubmed_abstract():
    """Return a sample Document for testing."""
    return Document(
        id="123456",
        title="Sample Medical Abstract for Testing",
        text="This is a long sample abstract about a medical topic. It contains multiple sentences that should be chunked appropriately by the HaystackChunker. The chunker should split this text into manageable pieces while preserving the meaning and context of the content. This abstract discusses important biomedical information that would be relevant in a RAG system.",
        url="https://www.ncbi.nlm.nih.gov/pubmed/123456",
        publication_date="2023-01-15",
        journal="Journal of Medical Testing",
        authors=["Smith, J", "Johnson, A", "Williams, B"],
        keywords=["testing", "chunking", "abstracts"],
        mesh_terms=["Testing", "Document Processing", "Information Retrieval"],
        doi="10.1234/test.5678",
    )


def test_chunker_init():
    """Test the initialization of the HaystackChunker."""
    # Test with default parameters
    chunker = WordChunker()
    assert chunker.chunk_size == 200
    assert chunker.chunk_overlap == 20

    # Test with custom parameters
    chunker = WordChunker(chunk_size=300, chunk_overlap=50)
    assert chunker.chunk_size == 300
    assert chunker.chunk_overlap == 50


def test_chunking_behavior(sample_pubmed_abstract):
    """Test the actual chunking behavior with different parameters."""
    # Create a chunker with small chunk size to ensure multiple chunks
    chunker = WordChunker(chunk_size=10, chunk_overlap=0)

    # Process the abstract
    chunks = chunker.chunk_abstract(sample_pubmed_abstract)

    # We should get multiple chunks with this small size
    assert len(chunks) > 3

    # Test with larger chunk size - should result in fewer chunks
    chunker_large = WordChunker(chunk_size=100, chunk_overlap=0)
    chunks_large = chunker_large.chunk_abstract(sample_pubmed_abstract)

    # Larger chunks should result in fewer chunks
    assert len(chunks) > len(chunks_large)

    # Test with overlap
    chunker_overlap = WordChunker(chunk_size=20, chunk_overlap=10)
    chunks_overlap = chunker_overlap.chunk_abstract(sample_pubmed_abstract)

    # With overlap, we should have more chunks than without overlap at same size
    chunker_no_overlap = WordChunker(chunk_size=20, chunk_overlap=0)
    chunks_no_overlap = chunker_no_overlap.chunk_abstract(sample_pubmed_abstract)

    assert len(chunks_overlap) >= len(chunks_no_overlap)


def test_metadata_preservation(sample_pubmed_abstract):
    """Test that metadata is preserved in all chunks."""
    chunker = WordChunker(chunk_size=20, chunk_overlap=0)
    chunks = chunker.chunk_abstract(sample_pubmed_abstract)

    # Verify each chunk has the correct structure and preserved metadata
    for i, chunk in enumerate(chunks):
        # Check chunk is a DocumentChunk
        assert isinstance(chunk, DocumentChunk)

        # Check chunk ID format
        assert chunk.chunk_id == f"{sample_pubmed_abstract.id}-{i + 1}"

        # Check metadata preservation through the document reference
        assert chunk.document.title == sample_pubmed_abstract.title
        assert chunk.document.id == sample_pubmed_abstract.id
        assert chunk.document.url == sample_pubmed_abstract.url
        assert chunk.document.journal == sample_pubmed_abstract.journal
        assert (
            chunk.document.publication_date == sample_pubmed_abstract.publication_date
        )
        assert chunk.document.authors == sample_pubmed_abstract.authors
        assert chunk.document.doi == sample_pubmed_abstract.doi


def test_chunking_text_content(sample_pubmed_abstract):
    """Test that the chunked text content makes sense and preserves meaningful segments."""
    chunker = WordChunker(chunk_size=30, chunk_overlap=0)
    chunks = chunker.chunk_abstract(sample_pubmed_abstract)

    # The chunks should contain parts of the original text
    full_text = sample_pubmed_abstract.text

    # Check if all chunks are substrings of the original text
    for chunk in chunks:
        # The chunk text should be a continuous segment from the original text
        # (accounting for potential whitespace differences)
        assert chunk.text.strip() in full_text

    # Reconstruct the text from chunks (simplified check)
    # This is a rough check - chunking might not preserve exact whitespace or order
    combined_length = sum(len(chunk.text) for chunk in chunks)
    # Combined length of chunks should be at least the length of the original text
    # (might be longer due to overlap)
    assert combined_length >= len(full_text)


def test_respect_sentence_boundary():
    """Test that the chunker respects sentence boundaries."""
    # Test text with clear sentence boundaries
    text = "This is sentence one. This is sentence two. This is sentence three."
    abstract = Document(
        id="test123",
        title="Sentence Test",
        text=text,
        url="https://example.com",
        publication_date="2023-01-01",
        journal="Test Journal",
        authors=["Tester, T"],
        keywords=["test"],
        mesh_terms=["Testing"],
        doi="10.1234/test",
    )

    # Create a chunker with a chunk size that would split within sentences if not respecting boundaries
    chunker = WordChunker(chunk_size=10, chunk_overlap=0)
    chunks = chunker.chunk_abstract(abstract)

    # Check that each chunk doesn't end in the middle of a sentence
    for chunk in chunks:
        # Skip the last chunk which might be a complete sentence
        if chunk != chunks[-1]:
            # A chunk should either end with a period or be a complete sentence
            assert chunk.text.strip().endswith(".") or chunk.text.strip() in text.split(
                ". "
            )
