from abc import ABC, abstractmethod
from typing import List

from src.models.document import Document, DocumentChunk


class DocumentChunker(ABC):
    """Interface for document chunking in the RAG pipeline."""

    @abstractmethod
    def chunk_abstract(self, abstract: Document) -> List[DocumentChunk]:
        """
        Process a Document and split it into chunks.

        Args:
            abstract: A Document object containing the document text and metadata

        Returns:
            A list of DocumentChunk objects with preserved metadata
        """
        pass
