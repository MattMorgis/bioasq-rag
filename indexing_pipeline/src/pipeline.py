from enum import Enum, auto
from typing import Dict, List, Optional, Set, Union

from src.chunker.chunker import AbstractChunker
from src.embedding.embedder import Embedder
from src.models.pubmed import PubMedAbstract, PubMedChunk, PubMedEmbeddedChunk


class PipelineStep(Enum):
    """Enum representing different steps in the pipeline."""

    CHUNK = auto()
    EMBED = auto()


class Pipeline:
    """
    Flexible pipeline for processing PubMed abstracts.
    Supports chunking and embedding with configurable steps.
    """

    def __init__(
        self,
        chunker: AbstractChunker,
        embedder: Embedder,
        steps: Optional[Set[PipelineStep]] = None,
    ):
        """
        Initialize the pipeline with the given components.

        Args:
            chunker: The chunker implementation to use
            embedder: The embedder implementation to use
            steps: Set of steps to execute in the pipeline. If None, all steps will be executed.
        """
        self.chunker = chunker
        self.embedder = embedder
        self.steps = steps or {PipelineStep.CHUNK, PipelineStep.EMBED}

        # Validate steps
        for step in self.steps:
            if step == PipelineStep.CHUNK and not chunker:
                raise ValueError("Chunker is required when CHUNK step is enabled")
            if step == PipelineStep.EMBED and not embedder:
                raise ValueError("Embedder is required when EMBED step is enabled")

    def process_documents(
        self, documents: List[PubMedAbstract]
    ) -> Dict[str, Union[List[PubMedChunk], List[PubMedEmbeddedChunk]]]:
        """
        Process documents through the pipeline.

        Args:
            documents: List of PubMedAbstract documents to process

        Returns:
            Dictionary with 'chunks' and/or 'embedded_chunks' keys depending on
            which steps were executed
        """
        result = {}

        # Chunking step
        chunks = []
        if PipelineStep.CHUNK in self.steps:
            print(f"Chunking {len(documents)} documents...")
            for doc in documents:
                doc_chunks = self.chunker.chunk_abstract(doc)
                chunks.extend(doc_chunks)
            result["chunks"] = chunks
            print(f"Created {len(chunks)} chunks")

        # Embedding step
        if PipelineStep.EMBED in self.steps and chunks:
            print(f"Embedding {len(chunks)} chunks...")
            embedded_chunks = self.embedder.embed_batch(chunks)
            result["embedded_chunks"] = embedded_chunks
            print(f"Created {len(embedded_chunks)} embedded chunks")

        return result
