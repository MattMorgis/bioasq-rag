from abc import ABC, abstractmethod
from typing import Any, List

from src.models.document import EmbeddedDocumentChunk


class Indexer(ABC):
    """Abstract base class defining the interface for indexing embedded document chunks."""

    @abstractmethod
    def initialize(self, index_name: str, dimension: int, **kwargs: Any) -> None:
        """
        Initialize a new vector index with the specified name and dimension.

        Args:
            index_name: Name of the index to initialize
            dimension: The dimensionality of the vectors to be indexed
            **kwargs: Additional implementation-specific parameters
        """
        pass

    @abstractmethod
    def add_chunks(self, chunks: List[EmbeddedDocumentChunk]) -> None:
        """
        Add a batch of embedded document chunks to the index.

        Args:
            chunks: List of EmbeddedDocumentChunk objects to be indexed
        """
        pass

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Get the current number of vectors in the index.

        Returns:
            Number of indexed vectors
        """
        pass

    @abstractmethod
    def delete(self, chunk_ids: List[str]) -> None:
        """
        Remove vectors with the specified IDs from the index.

        Args:
            chunk_ids: List of chunk IDs to remove
        """
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Check if the index is initialized and ready for use.

        Returns:
            True if the index is ready, False otherwise
        """
        pass
