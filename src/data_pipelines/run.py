#!/usr/bin/env python
import asyncio
import logging
import sys

from src.data_pipelines.biopython_pubmed_client import BioPythonPubMedClient
from src.data_pipelines.data_fetcher import DataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)


async def main():
    # Initialize the PubMed client
    pubmed_client = BioPythonPubMedClient(
        email="your.email@example.com",  # Replace with your email
        tool="bioasq-rag",
    )

    # Initialize the DataFetcher with the PubMed client
    data_fetcher = DataFetcher(pubmed_client)

    # Run the DataFetcher
    await data_fetcher.run()


if __name__ == "__main__":
    asyncio.run(main())
