from typing import List

from sentence_transformers import SentenceTransformer
from src.embedding.embedder import Embedder
from src.models.document import DocumentChunk, EmbeddedDocumentChunk


class SentenceTransformerEmbedder(Embedder):
    """Embedder using sentence-transformers models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        batch_size: int = 32,
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.model_name = model_name

    def embed_batch(self, chunks: List[DocumentChunk]) -> List[EmbeddedDocumentChunk]:
        """Get embeddings for a batch of chunks."""
        # Extract text from each chunk
        texts = [chunk.text for chunk in chunks]

        # Generate embeddings in batches
        embeddings = self.model.encode(texts, batch_size=self.batch_size)

        # Create EmbeddedDocumentChunk objects
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            embedded_chunk = EmbeddedDocumentChunk(
                chunk=chunk, embedding=embeddings[i], embedding_model=self.model_name
            )
            embedded_chunks.append(embedded_chunk)

        return embedded_chunks
