from typing import List

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from src.chunker.chunker import AbstractChunker
from src.models.pubmed import PubMedAbstract, PubMedChunk


class WordChunker(AbstractChunker):
    def __init__(self, chunk_size: int = 200, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = DocumentSplitter(
            split_by="word",
            split_length=chunk_size,
            split_overlap=chunk_overlap,
            respect_sentence_boundary=True,
        )
        self.splitter.warm_up()

    def chunk_abstract(self, abstract: PubMedAbstract) -> List[PubMedChunk]:
        doc = Document(content=abstract.text)
        result = self.splitter.run(documents=[doc])
        haystack_chunks = result["documents"]

        pubmed_chunks = []
        for i, chunk in enumerate(haystack_chunks):
            chunk_id = f"{abstract.id}-{i + 1}"
            pubmed_chunk = PubMedChunk(
                chunk_id=chunk_id,
                text=chunk.content,
                abstract=abstract,
                metadata=chunk.meta,
            )
            pubmed_chunks.append(pubmed_chunk)

        return pubmed_chunks
