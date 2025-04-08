import unittest
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
from src.indexing.qdrant_indexer import QdrantIndexer
from src.models.pubmed import PubMedAbstract, PubMedChunk, PubMedEmbeddedChunk


class TestQdrantIndexer(unittest.TestCase):
    """Tests for the QdrantIndexer implementation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock for the Qdrant client
        self.mock_client_patcher = patch("src.indexing.qdrant_indexer.QdrantClient")
        self.mock_client = self.mock_client_patcher.start()

        # Create instance with mocked client
        self.indexer = QdrantIndexer(host="test-host", port=1234)
        self.indexer.client = MagicMock()

        # Set up test data
        self.test_abstract = PubMedAbstract(
            id="123456",
            title="Test Medical Paper",
            text="This is a test abstract for a medical paper about testing.",
            url="https://pubmed.ncbi.nlm.nih.gov/123456/",
            publication_date="2023-01-01",
            journal="Journal of Testing",
            authors=["Smith, John", "Doe, Jane"],
            keywords=["testing", "medicine", "research"],
            mesh_terms=["Testing", "Medicine", "Research Methodology"],
            doi="10.1234/test.123456",
        )

        self.test_chunk = PubMedChunk(
            chunk_id="123456-1",
            text="This is a test abstract for a medical paper about testing.",
            abstract=self.test_abstract,
            metadata={"position": 0, "is_title": False},
        )

        self.test_embedding = np.random.rand(768).astype(
            np.float32
        )  # Typical embedding dimension

        self.test_embedded_chunk = PubMedEmbeddedChunk(
            chunk=self.test_chunk,
            embedding=self.test_embedding,
            embedding_model="test-model",
        )

    def tearDown(self):
        """Clean up after each test method."""
        self.mock_client_patcher.stop()

    def test_initialization(self):
        """Test initializing the QdrantIndexer."""
        self.assertEqual(self.indexer._index_name, None)
        self.assertEqual(self.indexer._dimension, None)

    def test_initialize_creates_new_collection(self):
        """Test that initialize creates a new collection when it doesn't exist."""
        # Mock the get_collections response to indicate no collections exist
        mock_collections = MagicMock()
        mock_collections.collections = []
        self.indexer.client.get_collections.return_value = mock_collections

        # Call initialize
        self.indexer.initialize(index_name="test-index", dimension=768)

        # Verify create_collection was called with correct parameters
        self.indexer.client.create_collection.assert_called_once()
        # Verify create_payload_index was called for each field
        self.assertEqual(self.indexer.client.create_payload_index.call_count, 3)

        # Verify stored properties
        self.assertEqual(self.indexer._index_name, "test-index")
        self.assertEqual(self.indexer._dimension, 768)

    def test_initialize_skips_creation_when_collection_exists(self):
        """Test that initialize skips creation when collection exists."""
        # Mock the get_collections response to indicate collection exists
        mock_collection = MagicMock()
        mock_collection.name = "test-index"
        mock_collections = MagicMock()
        mock_collections.collections = [mock_collection]
        self.indexer.client.get_collections.return_value = mock_collections

        # Call initialize
        self.indexer.initialize(index_name="test-index", dimension=768)

        # Verify create_collection was not called
        self.indexer.client.create_collection.assert_not_called()

        # Verify stored properties
        self.assertEqual(self.indexer._index_name, "test-index")
        self.assertEqual(self.indexer._dimension, 768)

    def test_add_chunks_converts_and_inserts_correctly(self):
        """Test that add_chunks correctly converts PubMedEmbeddedChunks to Qdrant points."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"
        self.indexer._dimension = 768

        # Call add_chunks with test data
        self.indexer.add_chunks([self.test_embedded_chunk])

        # Verify upsert was called once
        self.indexer.client.upsert.assert_called_once()

        # Get the first call arguments
        args, kwargs = self.indexer.client.upsert.call_args

        # Verify collection name
        self.assertEqual(kwargs["collection_name"], "test-index")

        # Verify points structure
        points = kwargs["points"]
        self.assertEqual(len(points), 1)

        # Verify the point structure
        point = points[0]

        # Verify the point ID is a valid UUID
        try:
            uuid.UUID(str(point.id))
        except ValueError:
            self.fail("Point ID is not a valid UUID")

        # Check vector values (should be same as the test embedding)
        if hasattr(point, "vector"):
            self.assertTrue(np.array_equal(np.array(point.vector), self.test_embedding))

        # Check payload contains essential fields and the original chunk_id
        if hasattr(point, "payload"):
            payload = point.payload
            self.assertEqual(
                payload["chunk_id"], "123456-1"
            )  # Original chunk_id preserved in payload
            self.assertEqual(payload["abstract_id"], "123456")
            self.assertEqual(payload["title"], "Test Medical Paper")
            self.assertEqual(payload["journal"], "Journal of Testing")
            self.assertEqual(payload["embedding_model"], "test-model")

    def test_add_chunks_raises_error_when_not_initialized(self):
        """Test that add_chunks raises error when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Verify error is raised
        with self.assertRaises(ValueError):
            self.indexer.add_chunks([self.test_embedded_chunk])

    def test_size_returns_correct_vector_count(self):
        """Test that size property returns correct vector count."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"

        # Mock collection info response
        mock_collection_info = MagicMock()
        mock_collection_info.vectors_count = 42
        self.indexer.client.get_collection.return_value = mock_collection_info

        # Check size
        self.assertEqual(self.indexer.size, 42)

        # Verify get_collection was called with correct name
        self.indexer.client.get_collection.assert_called_once_with("test-index")

    def test_size_returns_zero_when_not_initialized(self):
        """Test that size property returns zero when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Check size
        self.assertEqual(self.indexer.size, 0)

        # Verify get_collection was not called
        self.indexer.client.get_collection.assert_not_called()

    def test_search_converts_and_searches_correctly(self):
        """Test that search correctly converts and searches for vectors."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"

        # Mock search results
        mock_result = MagicMock()
        mock_result.id = "123456-1"
        mock_result.score = 0.95
        mock_result.payload = {"title": "Test Medical Paper"}
        self.indexer.client.search.return_value = [mock_result]

        # Call search
        results = self.indexer.search(self.test_embedding, limit=5)

        # Verify search was called with correct parameters
        self.indexer.client.search.assert_called_once()
        args, kwargs = self.indexer.client.search.call_args
        self.assertEqual(kwargs["collection_name"], "test-index")
        self.assertEqual(kwargs["limit"], 5)
        self.assertTrue(kwargs["with_payload"])

        # Verify results format
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["chunk_id"], "123456-1")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["payload"]["title"], "Test Medical Paper")

    def test_search_raises_error_when_not_initialized(self):
        """Test that search raises error when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Verify error is raised
        with self.assertRaises(ValueError):
            self.indexer.search(self.test_embedding)

    def test_is_ready_returns_correct_status(self):
        """Test that is_ready returns correct status based on collection state."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"

        # Import within test to avoid global import issues
        from qdrant_client.http import models as rest

        # Mock collection info response - GREEN status
        mock_collection_info = MagicMock()
        mock_collection_info.status = rest.CollectionStatus.GREEN
        self.indexer.client.get_collection.return_value = mock_collection_info

        # Check is_ready - should be True
        self.assertTrue(self.indexer.is_ready())

        # Mock collection info response - YELLOW status
        mock_collection_info.status = rest.CollectionStatus.YELLOW

        # Check is_ready - should still be True since not RED
        self.assertTrue(self.indexer.is_ready())

        # Mock exception in get_collection
        self.indexer.client.get_collection.side_effect = Exception(
            "Collection not found"
        )

        # Check is_ready - should be False due to exception
        self.assertFalse(self.indexer.is_ready())

    def test_is_ready_returns_false_when_not_initialized(self):
        """Test that is_ready returns False when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Check is_ready
        self.assertFalse(self.indexer.is_ready())

        # Verify get_collection was not called
        self.indexer.client.get_collection.assert_not_called()

    def test_delete_removes_points_correctly(self):
        """Test that delete removes points with specified IDs."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"

        # Call delete
        chunk_ids = ["123456-1", "123456-2"]
        self.indexer.delete(chunk_ids)

        # Verify delete was called with correct parameters
        self.indexer.client.delete.assert_called_once()
        args, kwargs = self.indexer.client.delete.call_args
        self.assertEqual(kwargs["collection_name"], "test-index")

        # Verify points selector contains correct IDs
        points_selector = kwargs["points_selector"]
        self.assertEqual(points_selector.points, chunk_ids)

    def test_delete_raises_error_when_not_initialized(self):
        """Test that delete raises error when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Verify error is raised
        with self.assertRaises(ValueError):
            self.indexer.delete(["123456-1"])

    def test_save_creates_snapshot(self):
        """Test that save method creates a snapshot."""
        # Initialize the indexer
        self.indexer._index_name = "test-index"

        # Call save
        self.indexer.save("/path/to/snapshot")

        # Verify create_snapshot was called with correct collection name
        self.indexer.client.create_snapshot.assert_called_once_with(
            collection_name="test-index"
        )

    def test_save_raises_error_when_not_initialized(self):
        """Test that save raises error when index is not initialized."""
        # Ensure index is not initialized
        self.indexer._index_name = None

        # Verify error is raised
        with self.assertRaises(ValueError):
            self.indexer.save("/path/to/snapshot")

    def test_load_raises_error_when_not_implemented(self):
        """Test that load method raises NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            self.indexer.load("/path/to/snapshot")


if __name__ == "__main__":
    unittest.main()
