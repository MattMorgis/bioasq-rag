import logging
from typing import Any, Dict, Optional

from src.data_pipelines.clients.pubmed_client import PubMedClient


class DataFetcher:
    """
    Class for fetching data from PubMed based on URLs.
    Uses a PubMedClient for the actual retrieval.
    """

    def __init__(self, pubmed_client: PubMedClient):
        """
        Initialize the DataFetcher with a PubMedClient implementation.

        Args:
            pubmed_client: An implementation of PubMedClient
        """
        self.pubmed_client = pubmed_client
        self.logger = logging.getLogger(__name__)

        # Hardcoded PubMed URL
        self.pubmed_url = "http://www.ncbi.nlm.nih.gov/pubmed/15858239"

    def _extract_pubmed_id(self, url: str) -> str:
        """
        Extract the PubMed ID from a PubMed URL.

        Args:
            url: PubMed URL

        Returns:
            The PubMed ID extracted from the URL
        """
        # Simple extraction based on URL structure
        return url.split("/")[-1]

    async def fetch_single_abstract(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a single abstract from a PubMed URL.

        Args:
            url: PubMed URL

        Returns:
            Dictionary containing abstract data or None if retrieval failed
        """
        pubmed_id = self._extract_pubmed_id(url)
        self.logger.info(f"Fetching abstract for URL: {url}")
        self.logger.info(f"Extracted PubMed ID: {pubmed_id}")

        try:
            abstract = await self.pubmed_client.get_abstract_by_id(pubmed_id)
            self.logger.info(f"Successfully fetched abstract for {url}")
            self.logger.info(f"Abstract: {abstract['abstract']}")
            return abstract
        except Exception as e:
            self.logger.error(f"Error fetching abstract for {url}: {str(e)}")
            return None

    async def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the DataFetcher to fetch the abstract from the hardcoded URL.

        Returns:
            Dictionary containing abstract data or None if retrieval failed
        """
        self.logger.info(f"Running DataFetcher with hardcoded URL: {self.pubmed_url}")
        abstract = await self.fetch_single_abstract(self.pubmed_url)

        if abstract:
            print("\nSuccessfully fetched abstract:")
            print(f"Title: {abstract['title']}")
            print(f"Authors: {', '.join(abstract['authors'])}")
            print(f"Journal: {abstract['journal']}")
            print(f"Publication Date: {abstract['publication_date']}")
            print(f"ID: {abstract['id']}")
            print("\nAbstract:")
            print(abstract["abstract"])
        else:
            print(f"\nFailed to fetch abstract from {self.pubmed_url}")

        return abstract
