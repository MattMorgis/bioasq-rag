from abc import ABC, abstractmethod
from typing import List

from src.models.pubmed import PubMedAbstract, PubMedChunk


class AbstractChunker(ABC):
    """Interface for document chunking in the RAG pipeline."""

    @abstractmethod
    def chunk_abstract(self, abstract: PubMedAbstract) -> List[PubMedChunk]:
        """
        Process a PubMedAbstract and split it into chunks.

        Args:
            abstract: A PubMedAbstract object containing the document text and metadata

        Returns:
            A list of PubMedChunk objects with preserved metadata
        """
        pass
