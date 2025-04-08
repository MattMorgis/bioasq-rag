from typing import List

from haystack import Document as HaystackDocument
from haystack.components.preprocessors import DocumentSplitter
from src.chunker.chunker import AbstractChunker
from src.models.pubmed import Document, DocumentChunk


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

    def chunk_abstract(self, abstract: Document) -> List[DocumentChunk]:
        doc = HaystackDocument(content=abstract.text)
        result = self.splitter.run(documents=[doc])
        haystack_chunks = result["documents"]

        document_chunks = []
        for i, chunk in enumerate(haystack_chunks):
            chunk_id = f"{abstract.id}-{i + 1}"
            document_chunk = DocumentChunk(
                chunk_id=chunk_id,
                text=chunk.content,
                document=abstract,
                metadata=chunk.meta,
            )
            document_chunks.append(document_chunk)

        return document_chunks
