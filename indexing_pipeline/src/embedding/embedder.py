from abc import ABC, abstractmethod
from typing import List

from src.models.pubmed import DocumentChunk, EmbeddedDocumentChunk


class Embedder(ABC):
    """Interface for embedding a chunk of text in the RAG pipeline."""

    @abstractmethod
    def embed_batch(self, chunks: List[DocumentChunk]) -> List[EmbeddedDocumentChunk]:
        """
        Process a list of DocumentChunks and embed them.

        Args:
            chunks: A list of DocumentChunk objects to embed

        Returns:
            A list of EmbeddedDocumentChunk objects with embedded text and metadata
        """
        pass
