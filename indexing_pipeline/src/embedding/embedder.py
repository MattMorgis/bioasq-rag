from abc import ABC, abstractmethod
from typing import List

from src.models.pubmed import PubMedChunk, PubMedEmbeddedChunk


class Embedder(ABC):
    """Interface for embedding a chunk of text in the RAG pipeline."""

    @abstractmethod
    def embed_batch(self, chunks: List[PubMedChunk]) -> List[PubMedEmbeddedChunk]:
        """
        Process a list of PubMedChunks and embed them.

        Args:
            chunks: A list of PubMedChunk objects to embed

        Returns:
            A list of PubMedEmbeddedChunk objects with embedded text and metadata
        """
        pass
