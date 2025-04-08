"""Script to clear the Qdrant index for the BioASQ RAG system."""

import logging

from src.indexing.qdrant_indexer import QdrantIndexer


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Initialize indexer with same parameters as main.py
    logger.info("Connecting to Qdrant...")
    indexer = QdrantIndexer(host="localhost", port=6333, grpc_port=6334)

    try:
        # Initialize with same collection name and dimension as main.py
        logger.info("Attempting to delete existing collection...")
        indexer.client.delete_collection(collection_name="bioasq-12b-rag-dataset")
        logger.info("Successfully deleted collection 'bioasq-12b-rag-dataset'")
    except Exception as e:
        logger.info(f"Collection might not exist or other error occurred: {e}")

    # Reinitialize the collection
    logger.info("Reinitializing collection...")
    indexer.initialize("bioasq-12b-rag-dataset", 384)
    logger.info("Successfully reinitialized empty collection")


if __name__ == "__main__":
    main()
