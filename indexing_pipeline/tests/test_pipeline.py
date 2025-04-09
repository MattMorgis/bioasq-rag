from unittest.mock import Mock

import pytest
from src.chunker.chunker import DocumentChunker
from src.embedding.embedder import Embedder
from src.indexing.indexer import Indexer
from src.models.document import (
    Document,
    DocumentChunk,
    EmbeddedDocumentChunk,
)
from src.pipeline import Pipeline, PipelineStep


@pytest.fixture
def mock_chunker():
    """Mock chunker implementation."""
    chunker = Mock(spec=DocumentChunker)

    # Configure the mock to return predefined chunks
    def chunk_abstract(abstract):
        return [
            DocumentChunk(
                chunk_id=f"{abstract.id}-1",
                text="First chunk of test abstract",
                document=abstract,
            ),
            DocumentChunk(
                chunk_id=f"{abstract.id}-2",
                text="Second chunk of test abstract",
                document=abstract,
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
            EmbeddedDocumentChunk(
                chunk=chunk,
                embedding=[0.1, 0.2, 0.3],  # Simplified embedding
                embedding_model="test-model",
            )
            for chunk in chunks
        ]

    embedder.embed_batch.side_effect = embed_batch
    return embedder


@pytest.fixture
def mock_indexer():
    """Mock indexer implementation."""
    indexer = Mock(spec=Indexer)

    # Configure default behavior
    indexer.is_ready.return_value = True
    indexer.add_chunks.return_value = None
    indexer.size = 0

    return indexer


class TestPipeline:
    """Test suite for the Pipeline class."""

    def test_pipeline_initialization(self, mock_chunker, mock_embedder, mock_indexer):
        """Test initializing the pipeline with different configurations."""
        # Test with all steps (default)
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, indexer=mock_indexer
        )
        assert PipelineStep.CHUNK in pipeline.steps
        assert PipelineStep.EMBED in pipeline.steps
        assert PipelineStep.INDEX in pipeline.steps

        # Test with only chunking step
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, steps={PipelineStep.CHUNK}
        )
        assert PipelineStep.CHUNK in pipeline.steps
        assert PipelineStep.EMBED not in pipeline.steps
        assert PipelineStep.INDEX not in pipeline.steps

        # Test with only embedding step
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, steps={PipelineStep.EMBED}
        )
        assert PipelineStep.EMBED in pipeline.steps
        assert PipelineStep.CHUNK not in pipeline.steps
        assert PipelineStep.INDEX not in pipeline.steps

        # Test with only indexing step (should fail as it requires embedded chunks)
        with pytest.raises(ValueError):
            Pipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                indexer=mock_indexer,
                steps={PipelineStep.INDEX},
            )

    def test_pipeline_initialization_validation(
        self, mock_chunker, mock_embedder, mock_indexer
    ):
        """Test validation during pipeline initialization."""
        # Test invalid configuration: CHUNK step without chunker
        with pytest.raises(ValueError):
            Pipeline(chunker=None, embedder=mock_embedder, steps={PipelineStep.CHUNK})

        # Test invalid configuration: EMBED step without embedder
        with pytest.raises(ValueError):
            Pipeline(chunker=mock_chunker, embedder=None, steps={PipelineStep.EMBED})

        # Test invalid configuration: INDEX step without indexer
        with pytest.raises(ValueError):
            Pipeline(
                chunker=mock_chunker,
                embedder=mock_embedder,
                indexer=None,
                steps={PipelineStep.INDEX},
            )

    def test_process_documents_all_steps(
        self, mock_chunker, mock_embedder, mock_indexer, sample_pubmed_abstract
    ):
        """Test processing documents with all pipeline steps."""
        # Create a pipeline with all steps
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, indexer=mock_indexer
        )

        # Process a sample document
        result = pipeline.process_documents([sample_pubmed_abstract])

        # Verify chunker was called with the abstract
        mock_chunker.chunk_abstract.assert_called_once_with(sample_pubmed_abstract)

        # Verify embedder was called with the chunks returned from chunker
        assert mock_embedder.embed_batch.called
        assert "embedded_chunks" in result

        # Verify indexer was called with the embedded chunks
        mock_indexer.is_ready.assert_called_once()
        mock_indexer.add_chunks.assert_called_once()
        chunks_indexed = mock_indexer.add_chunks.call_args[0][0]
        assert len(chunks_indexed) == 2
        assert all(isinstance(chunk, EmbeddedDocumentChunk) for chunk in chunks_indexed)

        # Verify results contain both chunks and embedded chunks
        assert "chunks" in result
        assert "embedded_chunks" in result
        assert len(result["chunks"]) == 2
        assert len(result["embedded_chunks"]) == 2

        # Verify structure of returned objects
        assert isinstance(result["chunks"][0], DocumentChunk)
        assert isinstance(result["embedded_chunks"][0], EmbeddedDocumentChunk)

    def test_process_documents_chunk_only(
        self, mock_chunker, mock_embedder, mock_indexer, sample_pubmed_abstract
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

        # Verify embedder and indexer were not called
        mock_embedder.embed_batch.assert_not_called()
        mock_indexer.add_chunks.assert_not_called()

        # Verify results contain only chunks
        assert "chunks" in result
        assert "embedded_chunks" not in result
        assert len(result["chunks"]) == 2

    def test_process_documents_with_multiple_abstracts(
        self, mock_chunker, mock_embedder, mock_indexer
    ):
        """Test processing multiple documents at once."""
        # Create sample abstracts
        abstracts = [
            Document(
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

        # Create pipeline with all steps
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, indexer=mock_indexer
        )

        # Process multiple documents
        result = pipeline.process_documents(abstracts)

        # Verify chunker was called for each abstract
        assert mock_chunker.chunk_abstract.call_count == 3

        # Each abstract produces 2 chunks (based on our mock), so we expect 6 total
        assert len(result["chunks"]) == 6
        assert len(result["embedded_chunks"]) == 6

        # Verify indexer was called once with all chunks
        mock_indexer.add_chunks.assert_called_once()
        chunks_indexed = mock_indexer.add_chunks.call_args[0][0]
        assert len(chunks_indexed) == 6

    def test_empty_document_list(self, mock_chunker, mock_embedder, mock_indexer):
        """Test processing an empty document list."""
        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, indexer=mock_indexer
        )
        result = pipeline.process_documents([])

        # No chunking, embedding, or indexing should happen
        mock_chunker.chunk_abstract.assert_not_called()
        mock_embedder.embed_batch.assert_not_called()
        mock_indexer.add_chunks.assert_not_called()

        # Result should be empty
        assert "chunks" in result
        assert len(result["chunks"]) == 0
        assert "embedded_chunks" not in result

    def test_indexer_not_ready_error(
        self, mock_chunker, mock_embedder, mock_indexer, sample_pubmed_abstract
    ):
        """Test error handling when indexer is not ready."""
        # Configure indexer to report not ready
        mock_indexer.is_ready.return_value = False

        pipeline = Pipeline(
            chunker=mock_chunker, embedder=mock_embedder, indexer=mock_indexer
        )

        # Process should raise RuntimeError
        with pytest.raises(RuntimeError, match="Indexer is not initialized"):
            pipeline.process_documents([sample_pubmed_abstract])

        # Verify that chunks and embeddings were still created
        mock_chunker.chunk_abstract.assert_called_once()
        mock_embedder.embed_batch.assert_called_once()
        # But indexing was not attempted
        mock_indexer.add_chunks.assert_not_called()
