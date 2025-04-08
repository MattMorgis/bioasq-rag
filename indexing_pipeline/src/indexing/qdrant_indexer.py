import uuid
from typing import Any, Dict, List, Union

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.models import Distance, VectorParams
from src.indexing.indexer import Indexer
from src.models.pubmed import PubMedEmbeddedChunk


class QdrantIndexer(Indexer):
    """Implementation of the Indexer interface using Qdrant vector database."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        **kwargs: Any,
    ):
        """
        Initialize the Qdrant client connection.

        Args:
            host: Qdrant server host
            port: Qdrant server HTTP port
            grpc_port: Qdrant server gRPC port
            **kwargs: Additional client configuration parameters
        """
        self.client = QdrantClient(host=host, port=port, grpc_port=grpc_port, **kwargs)
        self._index_name = None
        self._dimension = None

    def initialize(self, index_name: str, dimension: int, **kwargs: Any) -> None:
        """
        Initialize a collection in Qdrant with the specified name and dimension.

        Args:
            index_name: Name of the collection to create
            dimension: Vector dimension
            **kwargs: Additional collection parameters
        """
        # Store for later reference
        self._index_name = index_name
        self._dimension = dimension

        # Set defaults for optional parameters
        distance = kwargs.get("distance", Distance.COSINE)

        # Check if collection exists and create if it doesn't
        collections = self.client.get_collections().collections
        collection_names = [collection.name for collection in collections]

        if index_name not in collection_names:
            # Create the collection
            self.client.create_collection(
                collection_name=index_name,
                vectors_config=VectorParams(size=dimension, distance=distance),
                # Define payload schema for efficient filtering
                optimizers_config=rest.OptimizersConfigDiff(
                    indexing_threshold=0,  # Index immediately
                ),
            )

            # Create payload indexes for efficient filtering
            self.client.create_payload_index(
                collection_name=index_name,
                field_name="publication_date",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=index_name,
                field_name="journal",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=index_name,
                field_name="mesh_terms",
                field_schema=rest.PayloadSchemaType.KEYWORD,
            )

    def add_chunks(self, chunks: List[PubMedEmbeddedChunk]) -> None:
        """
        Add a batch of PubMedEmbeddedChunks to the index.

        Args:
            chunks: List of PubMedEmbeddedChunk objects to be indexed
        """
        if not self._index_name:
            raise ValueError("Index not initialized. Call initialize() first.")

        # Prepare points for batch insertion
        points = []
        for chunk in chunks:
            # Extract vector
            vector = chunk.embedding
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Generate a UUID for the point ID
            point_id = str(uuid.uuid4())

            # Prepare payload with metadata
            abstract = chunk.chunk.abstract
            payload = {
                # Chunk metadata
                "chunk_id": chunk.chunk.chunk_id,  # Store original chunk_id in payload
                "text": chunk.chunk.text,
                # Abstract metadata
                "abstract_id": abstract.id,
                "title": abstract.title,
                "url": abstract.url,
                "publication_date": abstract.publication_date,
                "journal": abstract.journal,
                "authors": abstract.authors,
                "keywords": abstract.keywords,
                "mesh_terms": abstract.mesh_terms,
                "doi": abstract.doi,
                # Embedding metadata
                "embedding_model": chunk.embedding_model,
                # Any additional chunk-specific metadata
                **chunk.chunk.metadata,
            }

            # Add point using UUID as the point ID
            points.append(rest.PointStruct(id=point_id, vector=vector, payload=payload))

        # Execute batch insert
        self.client.upsert(collection_name=self._index_name, points=points)

    @property
    def size(self) -> int:
        """
        Get the current number of vectors in the index.

        Returns:
            Number of indexed vectors
        """
        if not self._index_name:
            return 0

        collection_info = self.client.get_collection(self._index_name)
        return collection_info.vectors_count

    def search(
        self, query_vector: Union[List[float], np.ndarray], limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors in the index.

        Args:
            query_vector: The query vector to search for
            limit: Maximum number of results to return

        Returns:
            List of search results with similarity scores and metadata
        """
        if not self._index_name:
            raise ValueError("Index not initialized. Call initialize() first.")

        # Convert numpy array to list if necessary
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        # Execute search
        results = self.client.search(
            collection_name=self._index_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "chunk_id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                }
            )

        return formatted_results

    def save(self, path: str) -> None:
        """
        Save the collection snapshot to disk.

        Args:
            path: Path where to save the snapshot
        """
        if not self._index_name:
            raise ValueError("Index not initialized. Call initialize() first.")

        # Qdrant stores data on its own, but we can trigger a snapshot
        self.client.create_snapshot(collection_name=self._index_name)

    def load(self, path: str) -> None:
        """
        Load a collection from a snapshot.

        Args:
            path: Path to the snapshot
        """
        # Implementation would depend on how you manage Qdrant snapshots
        # This is a placeholder implementation
        raise NotImplementedError("Loading snapshots is not yet implemented")

    def delete(self, chunk_ids: List[str]) -> None:
        """
        Remove vectors with the specified IDs from the index.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        if not self._index_name:
            raise ValueError("Index not initialized. Call initialize() first.")

        self.client.delete(
            collection_name=self._index_name,
            points_selector=rest.PointIdsList(points=chunk_ids),
        )

    def is_ready(self) -> bool:
        """
        Check if the index is initialized and ready for use.

        Returns:
            True if the index is ready, False otherwise
        """
        if not self._index_name:
            return False

        try:
            collection_info = self.client.get_collection(self._index_name)
            return collection_info.status in [
                rest.CollectionStatus.GREEN,
                rest.CollectionStatus.YELLOW,
            ]
        except Exception:
            return False
