from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class PubMedAbstract:
    """Represents a PubMed abstract from the corpus."""

    id: str
    title: str
    text: str
    url: str
    publication_date: str
    journal: str
    authors: List[str]
    keywords: List[str]
    mesh_terms: List[str]
    doi: Optional[str] = None


@dataclass
class PubMedChunk:
    """Represents a chunk of text from a PubMed abstract with full metadata preservation."""

    # Chunk-specific content
    chunk_id: str
    text: str

    # Reference to original abstract
    abstract: PubMedAbstract

    # Additional metadata specific to this chunk
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PubMedEmbeddedChunk:
    """
    Represents a PubMed chunk with an embedding vector representation.
    Contains the original chunk object and its embedding information.
    """

    # The original chunk
    chunk: PubMedChunk

    # Embedding information
    embedding: Union[List[float], np.ndarray]
    embedding_model: str
