import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from src.data_pipelines.clients.pubmed_client import PubMedClient, PubMedClientError
from src.data_pipelines.pubmed_url_collector import PubMedURLCollector


class DataFetcher:
    """
    Class for fetching data from PubMed based on URLs.
    Uses a PubMedClient for the actual retrieval and handles batch processing and rate limiting.
    """

    def __init__(
        self,
        pubmed_client: PubMedClient,
        data_dir: str = "data",
        batch_size: int = 100,
        rate_limit_per_min: int = 10,
        max_retries: int = 3,
        retry_delay: int = 5,
    ):
        """
        Initialize the DataFetcher with a PubMedClient implementation.

        Args:
            pubmed_client: An implementation of PubMedClient
            data_dir: Directory to save abstracts to
            batch_size: Number of abstracts to fetch in a single batch
            rate_limit_per_min: Maximum number of API requests per minute
            max_retries: Maximum number of retries for failed requests
            retry_delay: Delay in seconds between retries
        """
        self.pubmed_client = pubmed_client
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.abstracts_dir = self.data_dir / "abstracts"
        self.abstracts_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.rate_limit_per_min = rate_limit_per_min
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Calculate delay between requests to respect rate limit
        self.request_delay = 60.0 / rate_limit_per_min

        # For tracking rate limits
        self.last_request_time = 0.0

        # URL collector for getting PubMed URLs
        self.url_collector = PubMedURLCollector(data_dir=data_dir)

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
        Fetch a single abstract from a PubMed URL with retry logic.

        Args:
            url: PubMed URL

        Returns:
            Dictionary containing abstract data or None if retrieval failed
        """
        pubmed_id = self._extract_pubmed_id(url)
        self.logger.info(f"Fetching abstract for URL: {url}")

        # Check if already saved
        abstract_path = self.abstracts_dir / f"{pubmed_id}.json"
        if abstract_path.exists():
            self.logger.info(f"Abstract for {pubmed_id} already exists. Skipping.")
            with open(abstract_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Implement rate limiting
        now = time.time()
        elapsed = now - self.last_request_time
        if elapsed < self.request_delay:
            await asyncio.sleep(self.request_delay - elapsed)

        # Record request time
        self.last_request_time = time.time()

        # Retry logic
        for attempt in range(self.max_retries):
            try:
                abstract = await self.pubmed_client.get_abstract_by_id(pubmed_id)
                self.logger.info(f"Successfully fetched abstract for {url}")

                # Save the abstract
                with open(abstract_path, "w", encoding="utf-8") as f:
                    json.dump(abstract, f, indent=2)

                return abstract
            except PubMedClientError as e:
                if "rate limit" in str(e).lower() and attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (attempt + 1)
                    self.logger.warning(
                        f"Rate limit hit for {url}. Retrying in {wait_time} seconds..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Error fetching abstract for {url}: {str(e)}")
                    return None
            except Exception as e:
                self.logger.error(
                    f"Unexpected error fetching abstract for {url}: {str(e)}"
                )
                return None

        return None

    async def fetch_batch(self, urls: List[str]) -> List[Dict[str, Any]]:
        """
        Fetch a batch of abstracts in parallel.

        Args:
            urls: List of PubMed URLs to fetch

        Returns:
            List of successfully fetched abstracts
        """
        tasks = [self.fetch_single_abstract(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [r for r in results if r is not None]

    async def fetch_all_abstracts(self, urls: Set[str]) -> List[Dict[str, Any]]:
        """
        Fetch all abstracts from a set of URLs using batching.

        Args:
            urls: Set of PubMed URLs to fetch

        Returns:
            List of successfully fetched abstracts
        """
        all_abstracts = []
        url_list = list(urls)
        total_urls = len(url_list)

        self.logger.info(
            f"Starting to fetch {total_urls} abstracts in batches of {self.batch_size}"
        )

        for i in range(0, total_urls, self.batch_size):
            batch_urls = url_list[i : i + self.batch_size]
            self.logger.info(
                f"Fetching batch {i // self.batch_size + 1}/{(total_urls + self.batch_size - 1) // self.batch_size}: {len(batch_urls)} URLs"
            )

            batch_results = await self.fetch_batch(batch_urls)
            all_abstracts.extend(batch_results)

            self.logger.info(
                f"Fetched {len(all_abstracts)}/{total_urls} abstracts so far"
            )

            # Brief pause between batches to prevent overwhelming the API
            await asyncio.sleep(1)

        return all_abstracts

    async def run(self) -> Optional[Dict[str, Any]]:
        """
        Run the DataFetcher to fetch all abstracts from the URL collector.

        Returns:
            Dictionary with summary of the fetch operation
        """
        self.logger.info("Running DataFetcher with PubMedURLCollector")

        # Get all URLs from the collector
        urls = self.url_collector.collect_urls()
        total_urls = len(urls)
        self.logger.info(f"Collected {total_urls} URLs from PubMedURLCollector")

        if not urls:
            self.logger.warning("No URLs found. Nothing to fetch.")
            return None

        # Fetch all abstracts
        abstracts = await self.fetch_all_abstracts(urls)
        successful_fetches = len(abstracts)

        # Summary
        summary = {
            "total_urls": total_urls,
            "successful_fetches": successful_fetches,
            "failed_fetches": total_urls - successful_fetches,
            "abstracts_dir": str(self.abstracts_dir),
        }

        # Save summary to file
        summary_path = self.data_dir / "fetch_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\nAbstract fetching complete:")
        print(f"Total URLs: {total_urls}")
        print(f"Successfully fetched: {successful_fetches}")
        print(f"Failed: {total_urls - successful_fetches}")
        print(f"Abstracts saved to: {self.abstracts_dir}")

        return summary
