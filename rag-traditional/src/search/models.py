from typing import List, Optional
from pydantic import BaseModel


class SearchResult(BaseModel):
    """Represents a single search result from the vector search service."""
    abstract_id: str
    title: str
    text: str
    score: float
    url: Optional[str] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    authors: Optional[List[str]] = None


class SearchResponse(BaseModel):
    """Represents the complete response from the vector search service."""
    results: List[SearchResult]
    query: str
    total_results: int
