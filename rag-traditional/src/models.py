from typing import List, Optional

from pydantic import BaseModel


class RAGRequest(BaseModel):
    """Request model for the RAG API endpoint."""

    query: str
    max_results: Optional[int] = 10
    temperature: Optional[float] = 0.2
    log_prompt: Optional[bool] = False


class RAGResponse(BaseModel):
    """Response model for the RAG API endpoint."""

    query: str
    answer: str
    sources: List[dict]
