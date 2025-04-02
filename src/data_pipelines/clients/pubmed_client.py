from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class PubMedClient(ABC):
    """Interface for retrieving PubMed abstracts."""

    @abstractmethod
    async def get_abstract_by_id(self, pubmed_id: str) -> Dict[str, Any]:
        """
        Retrieve a PubMed abstract by its ID.

        Args:
            pubmed_id: The PubMed ID of the article

        Returns:
            A dictionary containing the abstract data with at least the following keys:
            - id: The PubMed ID
            - title: The article title
            - abstract: The article abstract
            - authors: List of authors
            - publication_date: Publication date

        Raises:
            PubMedRateLimitError: If the request is rate limited
            PubMedClientError: If there's another error retrieving the abstract
        """
        pass

    @abstractmethod
    async def get_abstracts_by_ids(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple PubMed abstracts by their IDs.

        Args:
            pubmed_ids: List of PubMed IDs

        Returns:
            List of dictionaries containing the abstract data

        Raises:
            PubMedRateLimitError: If the request is rate limited
            PubMedClientError: If there's another error retrieving the abstracts
        """
        pass


class PubMedClientError(Exception):
    """Base exception for PubMed client errors."""

    def __init__(self, message: str, status_code: Optional[int] = None):
        """
        Initialize the exception.

        Args:
            message: Error message
            status_code: Optional HTTP status code
        """
        self.status_code = status_code
        super().__init__(message)


class PubMedRateLimitError(PubMedClientError):
    """Exception for rate limit errors when accessing PubMed API."""

    def __init__(self, message: str, status_code: int = 429):
        """
        Initialize the rate limit exception.

        Args:
            message: Error message
            status_code: HTTP status code (defaults to 429 Too Many Requests)
        """
        super().__init__(message, status_code)
