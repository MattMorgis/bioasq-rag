from abc import ABC, abstractmethod
from typing import Any, List

from src.models.pubmed import PubMedEmbeddedChunk


class Indexer(ABC):
    """Abstract base class defining the interface for indexing PubMedEmbeddedChunks."""

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
    def add_chunks(self, chunks: List[PubMedEmbeddedChunk]) -> None:
        """
        Add a batch of PubMedEmbeddedChunks to the index.

        Args:
            chunks: List of PubMedEmbeddedChunk objects to be indexed
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
