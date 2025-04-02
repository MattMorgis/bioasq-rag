#!/usr/bin/env python
import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

from src.data_pipelines.clients.biopython_pubmed_client import BioPythonPubMedClient
from src.data_pipelines.data_fetcher import DataFetcher


def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("pubmed_fetcher.log"),
        ],
    )


async def main():
    """Run the DataFetcher to collect PubMed abstracts."""
    # Load environment variables from .env file
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fetch PubMed abstracts")
    parser.add_argument(
        "--email", required=True, help="Email address for NCBI API (required)"
    )
    parser.add_argument("--api-key", help="NCBI API key for higher rate limits")
    parser.add_argument(
        "--data-dir", default="data", help="Directory to save abstracts to"
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Number of abstracts per batch"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=10,
        help="Maximum requests per second (3 without API key, 10 with API key)",
    )
    parser.add_argument(
        "--max-retries", type=int, default=3, help="Maximum retries for failed requests"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay in seconds between retries",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # Use the API key from .env file if not provided via command line
    api_key = args.api_key or os.environ.get("NCBI_API_KEY")

    logger.info("Initializing PubMed abstract fetcher")
    logger.info(f"Using email: {args.email}")
    logger.info(f"API key provided: {bool(api_key)}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Rate limit: {args.rate_limit} requests per second")

    try:
        # Create the client
        pubmed_client = BioPythonPubMedClient(
            email=args.email, api_key=api_key, tool="bioasq-rag"
        )

        # Create and run the fetcher
        data_fetcher = DataFetcher(
            pubmed_client=pubmed_client,
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            rate_limit_per_sec=args.rate_limit,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            concurrent_requests=args.rate_limit,  # Set concurrent requests to match rate limit
        )

        result = await data_fetcher.run()

        if result:
            logger.info(
                f"Fetching complete. {result['successful_fetches']}/{result['total_urls']} "
                f"abstracts successfully fetched. Failed: {result['failed_fetches']}"
            )
            logger.info(f"Results saved to {result['abstracts_dir']}")
        else:
            logger.error("Fetching failed - no URLs found or other error occurred")
            return 1

    except Exception as e:
        logger.exception(f"Error running fetcher: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
