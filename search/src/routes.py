import os

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from src.models.models import SearchResponse, SearchResult

# Load environment variables
load_dotenv()

# Initialize Qdrant client and embedding model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")
qdrant_client = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)
collection_name = os.getenv("QDRANT_COLLECTION", "pubmed-all-MiniLM-L12-v2-200w-20o")

router = APIRouter()


@router.get("/")
def read_root():
    return {"Hello": "World"}


@router.get("/search/vector", response_model=SearchResponse)
async def search(
    query: str = Query(..., description="The search query"),
    limit: int = Query(10, description="Number of results to return"),
):
    """
    Search the PubMed abstracts using semantic search.

    Converts the query to an embedding and searches for similar chunks in the
    Qdrant vector database.
    """
    try:
        # Convert query to embedding vector
        query_vector = model.encode(query).tolist()

        # Search in Qdrant
        search_results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,
        )

        # Process and format results
        formatted_results = []
        for result in search_results:
            payload = result.payload
            if payload is not None:  # Add null check to fix linter errors
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

        return SearchResponse(
            results=formatted_results, query=query, total_results=len(formatted_results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
