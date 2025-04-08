from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union

from src.chunker.chunker import AbstractChunker
from src.embedding.embedder import Embedder
from src.indexing.indexer import Indexer
from src.models.pubmed import Document, DocumentChunk, EmbeddedDocumentChunk


class PipelineStep(Enum):
    """Enum representing different steps in the pipeline."""

    CHUNK = auto()
    EMBED = auto()
    INDEX = auto()


class Pipeline:
    """
    Flexible pipeline for processing documents.
    Supports chunking, embedding, and indexing with configurable steps.
    """

    def __init__(
        self,
        chunker: AbstractChunker,
        embedder: Embedder,
        indexer: Optional[Indexer] = None,
        steps: Optional[Set[PipelineStep]] = None,
    ):
        """
        Initialize the pipeline with the given components.

        Args:
            chunker: The chunker implementation to use
            embedder: The embedder implementation to use
            indexer: Optional indexer implementation for storing embedded chunks
            steps: Set of steps to execute in the pipeline. If None, all steps will be executed.

        Raises:
            ValueError: If required components are missing for enabled steps or if steps are in invalid combination
        """
        self.chunker = chunker
        self.embedder = embedder
        self.indexer = indexer
        self.steps = steps or {
            PipelineStep.CHUNK,
            PipelineStep.EMBED,
            PipelineStep.INDEX,
        }

        # Add INDEX step if indexer is provided and steps not explicitly set
        if indexer and not steps:
            self.steps.add(PipelineStep.INDEX)

        # Validate steps
        for step in self.steps:
            if step == PipelineStep.CHUNK and not chunker:
                raise ValueError("Chunker is required when CHUNK step is enabled")
            if step == PipelineStep.EMBED and not embedder:
                raise ValueError("Embedder is required when EMBED step is enabled")
            if step == PipelineStep.INDEX and not indexer:
                raise ValueError("Indexer is required when INDEX step is enabled")

        # Validate step combinations
        if PipelineStep.INDEX in self.steps:
            if PipelineStep.EMBED not in self.steps:
                raise ValueError("INDEX step requires EMBED step to be enabled")
            if PipelineStep.CHUNK not in self.steps:
                raise ValueError("INDEX step requires CHUNK step to be enabled")

    def process_documents(
        self, documents: List[Document]
    ) -> Dict[str, Union[List[DocumentChunk], List[EmbeddedDocumentChunk]]]:
        """
        Process documents through the pipeline.

        Args:
            documents: List of Document objects to process

        Returns:
            Dictionary with 'chunks' and/or 'embedded_chunks' keys depending on
            which steps were executed. Note that indexed chunks are not returned
            but are stored in the indexer.

        Raises:
            RuntimeError: If indexing is enabled but the indexer is not initialized
        """
        result: Dict[str, Union[List[DocumentChunk], List[EmbeddedDocumentChunk]]] = {}

        # Chunking step
        chunks: List[DocumentChunk] = []
        if PipelineStep.CHUNK in self.steps:
            print(f"Chunking {len(documents)} documents...")
            for doc in documents:
                doc_chunks = self.chunker.chunk_abstract(doc)
                chunks.extend(doc_chunks)
            result["chunks"] = chunks
            print(f"Created {len(chunks)} chunks")

        # Embedding step
        embedded_chunks: List[EmbeddedDocumentChunk] = []
        if PipelineStep.EMBED in self.steps and chunks:
            print(f"Embedding {len(chunks)} chunks...")
            embedded_chunks = self.embedder.embed_batch(chunks)
            result["embedded_chunks"] = embedded_chunks
            print(f"Created {len(embedded_chunks)} embedded chunks")

        # Indexing step
        if PipelineStep.INDEX in self.steps and embedded_chunks:
            # Since we validated in __init__ that indexer exists if INDEX step is enabled,
            # we can safely assert it's not None here
            assert self.indexer is not None, (
                "Indexer cannot be None when INDEX step is enabled"
            )

            if not self.indexer.is_ready():
                raise RuntimeError(
                    "Indexer is not initialized. Call initialize() first."
                )
            print(f"Indexing {len(embedded_chunks)} embedded chunks...")
            self.indexer.add_chunks(embedded_chunks)
            print(f"Successfully indexed {len(embedded_chunks)} chunks")

        return result
