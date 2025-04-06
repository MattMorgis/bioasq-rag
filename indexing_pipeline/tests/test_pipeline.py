from unittest.mock import Mock

import pytest
from src.chunker.chunker import AbstractChunker
from src.embedding.embedder import Embedder
from src.models.pubmed import PubMedAbstract, PubMedChunk, PubMedEmbeddedChunk
from src.pipeline import Pipeline, PipelineStep


@pytest.fixture
def mock_chunker():
    """Mock chunker implementation."""
    chunker = Mock(spec=AbstractChunker)

    # Configure the mock to return predefined chunks
    def chunk_abstract(abstract):
        return [
            PubMedChunk(
                chunk_id=f"{abstract.id}-1",
                text="First chunk of test abstract",
                abstract=abstract,
            ),
            PubMedChunk(
                chunk_id=f"{abstract.id}-2",
                text="Second chunk of test abstract",
                abstract=abstract,
            ),
        ]

    chunker.chunk_abstract.side_effect = chunk_abstract
    return chunker


@pytest.fixture
def mock_embedder():
    """Mock embedder implementation."""
    embedder = Mock(spec=Embedder)

    # Configure the mock to return predefined embedded chunks
    def embed_batch(chunks):
        return [
            PubMedEmbeddedChunk(
                chunk=chunk,
                embedding=[0.1, 0.2, 0.3],  # Simplified embedding
                embedding_model="test-model",
            )
            for chunk in chunks
        ]

    embedder.embed_batch.side_effect = embed_batch
    return embedder


class TestPipeline:
    """Test suite for the Pipeline class."""

    def test_pipeline_initialization(self, mock_chunker, mock_embedder):
        """Test initializing the pipeline with different configurations."""
        # Test with all steps (default)
        pipeline = Pipeline(chunker=mock_chunker, embedder=mock_embedder)
        assert PipelineStep.CHUNK in pipeline.steps
        assert PipelineStep.EMBED in pipeline.steps

        # Test with only chunking step
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, steps={PipelineStep.CHUNK}
        )
        assert PipelineStep.CHUNK in pipeline.steps
        assert PipelineStep.EMBED not in pipeline.steps

        # Test with only embedding step
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, steps={PipelineStep.EMBED}
        )
        assert PipelineStep.EMBED in pipeline.steps
        assert PipelineStep.CHUNK not in pipeline.steps

    def test_pipeline_initialization_validation(self, mock_chunker, mock_embedder):
        """Test validation during pipeline initialization."""
        # Test invalid configuration: CHUNK step without chunker
        with pytest.raises(ValueError):
            Pipeline(chunker=None, embedder=mock_embedder, steps={PipelineStep.CHUNK})

        # Test invalid configuration: EMBED step without embedder
        with pytest.raises(ValueError):
            Pipeline(chunker=mock_chunker, embedder=None, steps={PipelineStep.EMBED})

    def test_process_documents_all_steps(
        self, mock_chunker, mock_embedder, sample_pubmed_abstract
    ):
        """Test processing documents with all pipeline steps."""
        # Create a pipeline with all steps
        pipeline = Pipeline(chunker=mock_chunker, embedder=mock_embedder)

        # Process a sample document
        result = pipeline.process_documents([sample_pubmed_abstract])

        # Verify chunker was called with the abstract
        mock_chunker.chunk_abstract.assert_called_once_with(sample_pubmed_abstract)

        # Verify embedder was called with the chunks returned from chunker
        assert mock_embedder.embed_batch.called
        assert "embedded_chunks" in result

        # Verify results contain both chunks and embedded chunks
        assert "chunks" in result
        assert "embedded_chunks" in result
        assert len(result["chunks"]) == 2
        assert len(result["embedded_chunks"]) == 2

        # Verify structure of returned objects
        assert isinstance(result["chunks"][0], PubMedChunk)
        assert isinstance(result["embedded_chunks"][0], PubMedEmbeddedChunk)

    def test_process_documents_chunk_only(
        self, mock_chunker, mock_embedder, sample_pubmed_abstract
    ):
        """Test processing documents with only the chunking step."""
        # Create a pipeline with only the chunking step
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, steps={PipelineStep.CHUNK}
        )

        # Process a sample document
        result = pipeline.process_documents([sample_pubmed_abstract])

        # Verify chunker was called
        mock_chunker.chunk_abstract.assert_called_once_with(sample_pubmed_abstract)

        # Verify embedder was not called
        mock_embedder.embed_batch.assert_not_called()

        # Verify results contain only chunks
        assert "chunks" in result
        assert "embedded_chunks" not in result
        assert len(result["chunks"]) == 2

    def test_process_documents_with_multiple_abstracts(
        self, mock_chunker, mock_embedder
    ):
        """Test processing multiple documents at once."""
        # Create sample abstracts
        abstracts = [
            PubMedAbstract(
                id=f"id{i}",
                title=f"Title {i}",
                text=f"Abstract text {i}",
                url=f"https://example.com/{i}",
                publication_date="2023-01-01",
                journal="Test Journal",
                authors=["Author A"],
                keywords=["keyword"],
                mesh_terms=["mesh term"],
            )
            for i in range(3)
        ]

        # Create pipeline
        pipeline = Pipeline(chunker=mock_chunker, embedder=mock_embedder)

        # Process multiple documents
        result = pipeline.process_documents(abstracts)

        # Verify chunker was called for each abstract
        assert mock_chunker.chunk_abstract.call_count == 3

        # Each abstract produces 2 chunks (based on our mock), so we expect 6 total
        assert len(result["chunks"]) == 6
        assert len(result["embedded_chunks"]) == 6

    def test_empty_document_list(self, mock_chunker, mock_embedder):
        """Test processing an empty document list."""
        pipeline = Pipeline(chunker=mock_chunker, embedder=mock_embedder)
        result = pipeline.process_documents([])

        # No chunking or embedding should happen
        mock_chunker.chunk_abstract.assert_not_called()
        mock_embedder.embed_batch.assert_not_called()

        # Result should be empty
        assert "chunks" in result
        assert len(result["chunks"]) == 0
        assert "embedded_chunks" not in result
