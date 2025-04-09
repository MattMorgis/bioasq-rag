from typing import List, Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    """Request model for the RAG API endpoint."""

    query: str
    max_results: Optional[int] = 5
    temperature: Optional[float] = 0.7


class RAGResponse(BaseModel):
    """Response model for the RAG API endpoint."""

    query: str
    answer: str
    sources: List[dict]
