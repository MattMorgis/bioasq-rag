from abc import ABC, abstractmethod
from typing import Any, Dict, List


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
            PubMedClientError: If there's an error retrieving the abstract
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
            PubMedClientError: If there's an error retrieving the abstracts
        """
        pass


class PubMedClientError(Exception):
    """Base exception for PubMed client errors."""

    pass
