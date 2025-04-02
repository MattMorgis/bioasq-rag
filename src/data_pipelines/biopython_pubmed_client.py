import asyncio
import logging
from typing import Any, Dict, List, Optional

from Bio import Entrez, Medline

from src.data_pipelines.pubmed_client import PubMedClient, PubMedClientError


class BioPythonPubMedClient(PubMedClient):
    """Implementation of PubMedClient using BioPython."""

    def __init__(
        self, email: str, api_key: Optional[str] = None, tool: str = "bioasq-rag"
    ):
        """
        Initialize the BioPython PubMed client.

        Args:
            email: Email address to identify yourself to NCBI
            api_key: Optional NCBI API key for higher request limits
            tool: Name of the application/tool making the request
        """
        self.logger = logging.getLogger(__name__)
        Entrez.email = email  # type: ignore
        Entrez.tool = tool  # type: ignore
        if api_key:
            Entrez.api_key = api_key  # type: ignore

    async def get_abstract_by_id(self, pubmed_id: str) -> Dict[str, Any]:
        """
        Retrieve a PubMed abstract by its ID using BioPython.

        Args:
            pubmed_id: The PubMed ID of the article

        Returns:
            A dictionary containing the abstract data

        Raises:
            PubMedClientError: If there's an error retrieving the abstract
        """
        try:
            return await asyncio.to_thread(self._fetch_abstract, pubmed_id)
        except Exception as e:
            self.logger.error(f"Error fetching PubMed abstract {pubmed_id}: {str(e)}")
            raise PubMedClientError(
                f"Failed to retrieve abstract for ID: {pubmed_id}"
            ) from e

    async def get_abstracts_by_ids(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Retrieve multiple PubMed abstracts by their IDs using BioPython.

        Args:
            pubmed_ids: List of PubMed IDs

        Returns:
            List of dictionaries containing the abstract data

        Raises:
            PubMedClientError: If there's an error retrieving the abstracts
        """
        try:
            return await asyncio.to_thread(self._fetch_multiple_abstracts, pubmed_ids)
        except Exception as e:
            self.logger.error(f"Error fetching multiple PubMed abstracts: {str(e)}")
            raise PubMedClientError("Failed to retrieve one or more abstracts") from e

    def _fetch_abstract(self, pubmed_id: str) -> Dict[str, Any]:
        """
        Fetch a single abstract using BioPython's synchronous API.

        Args:
            pubmed_id: The PubMed ID

        Returns:
            Formatted abstract data
        """
        handle = Entrez.efetch(
            db="pubmed", id=pubmed_id, rettype="medline", retmode="text"
        )
        records = Medline.parse(handle)
        record = next(records, None)
        handle.close()

        if not record:
            raise PubMedClientError(f"No abstract found for ID: {pubmed_id}")

        return self._format_record(record, pubmed_id)

    def _fetch_multiple_abstracts(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch multiple abstracts using BioPython's batch retrieval.

        Args:
            pubmed_ids: List of PubMed IDs

        Returns:
            List of formatted abstract data
        """
        handle = Entrez.efetch(
            db="pubmed", id=",".join(pubmed_ids), rettype="medline", retmode="text"
        )
        records = list(Medline.parse(handle))
        handle.close()

        if not records or len(records) != len(pubmed_ids):
            raise PubMedClientError("Could not retrieve all requested abstracts")

        return [
            self._format_record(record, pubmed_id)
            for record, pubmed_id in zip(records, pubmed_ids)
        ]

    def _format_record(self, record: Dict[str, Any], pubmed_id: str) -> Dict[str, Any]:
        """
        Format a Medline record into the expected abstract format.

        Args:
            record: Medline record from BioPython
            pubmed_id: The PubMed ID (used as fallback if not in record)

        Returns:
            Formatted abstract data
        """
        # Extract authors from record
        authors = []
        if "AU" in record:
            authors = record["AU"]

        # Format publication date
        pub_date = record.get("DP", "Unknown")

        return {
            "id": record.get("PMID", pubmed_id),
            "title": record.get("TI", "No title available"),
            "abstract": record.get("AB", "No abstract available"),
            "authors": authors,
            "publication_date": pub_date,
            "journal": record.get("JT", "Unknown journal"),
            "doi": record.get("LID", "").replace(" [doi]", "")
            if "LID" in record
            else None,
            "keywords": record.get("MH", []),
            "mesh_terms": record.get("MH", []),
        }
