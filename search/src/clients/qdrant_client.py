import os
from typing import List, Optional

from qdrant_client import QdrantClient as OriginalQdrantClient
from sentence_transformers import SentenceTransformer
from src.clients.search_client import SearchClient
from src.models.models import SearchResult


class QdrantSearchClient(SearchClient):
    """Implementation of SearchClient using Qdrant vector database."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize the Qdrant search client.

        Args:
            model_name: Name of the sentence transformer model to use
            host: Qdrant server host
            port: Qdrant server port
            collection_name: Name of the collection to search in
        """
        self.model = SentenceTransformer(model_name)
        self.host = host or os.getenv("QDRANT_HOST", "localhost")
        self.port = port or int(os.getenv("QDRANT_PORT", "6333"))
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "pubmed-all-MiniLM-L12-v2-200w-20o"
        )
        self.client = OriginalQdrantClient(host=self.host, port=self.port)

    def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for documents similar to the query.

        Args:
            query: The search query text
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        # Convert query to embedding vector
        query_vector = self.model.encode(query).tolist()

        # Search in Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        # Process and format results
        formatted_results = []
        for result in search_results:
            payload = result.payload
            if payload is not None:
                formatted_results.append(
                    SearchResult(
                        abstract_id=payload.get("abstract_id", ""),
                        title=payload.get("title", ""),
                        text=payload.get("text", ""),
                        score=float(result.score),
                        url=payload.get("url"),
                        publication_date=payload.get("publication_date"),
                        journal=payload.get("journal"),
                        authors=payload.get("authors", []),
                    )
                )

        return formatted_results
