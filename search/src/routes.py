from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from src.clients.qdrant_client import QdrantSearchClient
from src.models.models import SearchResponse

# Load environment variables
load_dotenv()

# Initialize search client
search_client = QdrantSearchClient()

router = APIRouter()


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
        # Search using the client
        results = search_client.search(query=query, limit=limit)

        return SearchResponse(results=results, query=query, total_results=len(results))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")
