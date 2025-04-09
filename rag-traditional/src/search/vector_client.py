import os
from typing import List, Optional
import httpx
from pydantic import ValidationError
import logging
from .models import SearchResponse, SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorSearchClient:
    """
    Client for the vector search service.
    
    This client allows querying the vector search service to find relevant documents
    based on semantic similarity to the input query.
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the vector search client.
        
        Args:
            base_url: Optional base URL for the search service. 
                     If not provided, will use SEARCH_API_URL from environment variables
                     or default to http://localhost:8000
        """
        self.base_url = base_url or os.getenv("SEARCH_API_URL", "http://localhost:8000")
        logger.info(f"Initialized vector search client with base URL: {self.base_url}")
        
    async def search(self, query: str, limit: int = 10) -> SearchResponse:
        """
        Search for documents semantically similar to the query.
        
        Args:
            query: The search query text
            limit: Maximum number of results to return (default: 10)
            
        Returns:
            SearchResponse object containing the search results
            
        Raises:
            httpx.HTTPStatusError: If the search service returns an error status code
            ConnectionError: If connection to the search service fails
            ValidationError: If the response cannot be parsed into the expected format
        """
        try:
            # Construct the search endpoint URL with query parameters
            url = f"{self.base_url}/search/vector"
            params = {"query": query, "limit": limit}
            
            logger.debug(f"Sending search request to {url} with params: {params}")
            
            # Make the request to the search service
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, timeout=30.0)
                response.raise_for_status()  # Raise exception for error status codes
                
                # Parse the response JSON into our SearchResponse model
                response_data = response.json()
                search_response = SearchResponse(**response_data)
                
                logger.info(f"Search request successful. Found {search_response.total_results} results.")
                return search_response
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Search service returned error status: {e.response.status_code}"
            if e.response.content:
                error_msg += f", Details: {e.response.text}"
            logger.error(error_msg)
            raise
            
        except httpx.ConnectError:
            logger.error(f"Failed to connect to search service at {self.base_url}")
            raise ConnectionError(f"Unable to connect to search service at {self.base_url}")
            
        except ValidationError as e:
            logger.error(f"Failed to parse search response: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error during search: {str(e)}")
            raise


# Function for easier importing when used as a module
async def search_documents(query: str, limit: int = 10) -> List[SearchResult]:
    """
    Convenience function to search for documents matching a query.
    
    Args:
        query: The search query
        limit: Maximum number of results to return
        
    Returns:
        List of SearchResult objects
    """
    client = VectorSearchClient()
    response = await client.search(query=query, limit=limit)
    return response.results
