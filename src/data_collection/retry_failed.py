#!/usr/bin/env python
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Set

from dotenv import load_dotenv

from src.data_collection.clients.biopython_pubmed_client import BioPythonPubMedClient
from src.data_collection.data_fetcher import DataFetcher
from src.data_collection.utils.logging_utils import setup_logging


async def retry_failed_urls(
    email: str,
    api_key: Optional[str] = None,
    data_dir: str = "data",
    batch_size: int = 10,
    rate_limit: int = 3,
    max_retries: int = 5,
    retry_delay: int = 10,
):
    """
    Retry fetching abstracts for URLs that previously failed.

    Args:
        email: Email address for NCBI API
        api_key: NCBI API key for higher rate limits
        data_dir: Directory containing data files
        batch_size: Number of abstracts to fetch in a single batch
        rate_limit: Maximum requests per second
        max_retries: Maximum number of retries for failed requests
        retry_delay: Delay in seconds between retries

    Returns:
        Number of successfully fetched abstracts
    """
    logger = logging.getLogger(__name__)
    data_dir_path = Path(data_dir)
    failed_urls_path = data_dir_path / "failed_urls.json"

    if not failed_urls_path.exists():
        logger.error(f"Failed URLs file not found at {failed_urls_path}")
        return 0

    # Load failed URLs
    with open(failed_urls_path, "r", encoding="utf-8") as f:
        failed_urls: List[str] = json.load(f)

    if not failed_urls:
        logger.info("No failed URLs to retry")
        return 0

    logger.info(f"Found {len(failed_urls)} failed URLs to retry")

    # Create the client - using lower rate limits and more retries
    pubmed_client = BioPythonPubMedClient(
        email=email, api_key=api_key, tool="bioasq-rag-retry"
    )

    # Create a data fetcher with more conservative settings
    data_fetcher = DataFetcher(
        pubmed_client=pubmed_client,
        data_dir=data_dir,
        batch_size=batch_size,
        rate_limit_per_sec=rate_limit,
        max_retries=max_retries,
        retry_delay=retry_delay,
        concurrent_requests=min(rate_limit, 5),  # Limit concurrent requests
    )

    # Convert list to set
    urls_to_retry: Set[str] = set(failed_urls)

    # Fetch the abstracts
    abstracts = await data_fetcher.fetch_all_abstracts(urls_to_retry)
    successful_fetches = len(abstracts)

    # Update the failed URLs file with remaining failures
    if data_fetcher.failed_urls:
        with open(failed_urls_path, "w", encoding="utf-8") as f:
            json.dump(list(data_fetcher.failed_urls), f, indent=2)
        logger.info(
            f"Updated failed URLs file with {len(data_fetcher.failed_urls)} remaining failed URLs"
        )
    else:
        # All URLs were fetched successfully, so remove the file
        failed_urls_path.unlink(missing_ok=True)
        logger.info("All URLs fetched successfully. Removed failed URLs file.")

    # Print summary
    print("\nRetry complete:")
    print(f"Total URLs attempted: {len(urls_to_retry)}")
    print(f"Successfully fetched: {successful_fetches}")
    print(f"Failed: {len(data_fetcher.failed_urls)}")
    print(f"Abstracts saved to: {data_fetcher.abstracts_dir}")

    return successful_fetches


async def main():
    """Run the retry script to fetch previously failed PubMed abstracts."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Retry fetching failed PubMed abstracts"
    )
    parser.add_argument(
        "--email", required=True, help="Email address for NCBI API (required)"
    )
    parser.add_argument("--api-key", help="NCBI API key for higher rate limits")
    parser.add_argument(
        "--data-dir", default="data", help="Directory containing data files"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Number of abstracts per batch"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=3,
        help="Maximum requests per second (3 without API key, 10 with API key)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=5, help="Maximum retries for failed requests"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=10,
        help="Delay in seconds between retries",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-file",
        default="pubmed_retry.log",
        help="File to save logs to (use 'none' to disable file logging)",
    )

    args = parser.parse_args()

    # Set up logging
    log_file = None if args.log_file.lower() == "none" else args.log_file
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)

    # Use the API key from .env file if not provided via command line
    api_key = args.api_key or os.environ.get("NCBI_API_KEY")

    logger.info("Initializing retry script for failed PubMed abstracts")
    logger.info(f"Using email: {args.email}")
    logger.info(f"API key provided: {bool(api_key)}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info("Using more conservative settings for retry:")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Rate limit: {args.rate_limit} requests per second")
    logger.info(f"  Max retries: {args.max_retries}")
    logger.info(f"  Retry delay: {args.retry_delay} seconds")

    try:
        # Retry fetching the failed URLs
        successful_fetches = await retry_failed_urls(
            email=args.email,
            api_key=api_key,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            rate_limit=args.rate_limit,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        logger.info(
            f"Retry complete. Successfully fetched {successful_fetches} abstracts."
        )

    except Exception as e:
        logger.exception(f"Error running retry script: {e}")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(asyncio.run(main()))
