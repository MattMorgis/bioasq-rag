import logging
from pathlib import Path

from src.chunker.word_chunker import WordChunker
from src.data_loader import DataLoader
from src.embedding.sentence_tranformer_embedder import SentenceTransformerEmbedder
from src.indexing.qdrant_indexer import QdrantIndexer
from src.pipeline import Pipeline, PipelineStep
from tqdm import tqdm


def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Configure paths and batch size
    data_path = Path("data/bioasq-12b-rag-dataset/data/corpus.jsonl")
    BATCH_SIZE = 1000  # Process 100 documents at a time

    data_loader = DataLoader(corpus_path=data_path)

    # Create pipeline components
    chunker = WordChunker(chunk_size=200, chunk_overlap=50)
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", batch_size=32
    )

    logger.info("Initializing Qdrant vector database...")
    indexer = QdrantIndexer(host="localhost", port=6333, grpc_port=6334)
    indexer.initialize("bioasq-12b-rag-all-MiniLM-L12-v2-200w-20o", 384)
    logger.info("Vector database initialized successfully")

    # Create pipeline with all steps
    pipeline = Pipeline(
        chunker=chunker,
        embedder=embedder,
        indexer=indexer,
        steps={PipelineStep.CHUNK, PipelineStep.EMBED, PipelineStep.INDEX},
    )

    logger.info("Starting full dataset processing pipeline...")

    try:
        # Initialize counters
        total_docs = 0
        total_chunks = 0

        # Process in batches
        for batch in tqdm(
            data_loader.load_abstracts_from_file(batch_size=BATCH_SIZE),
            desc="Processing documents",
        ):
            try:
                results = pipeline.process_documents(batch)

                # Update counters
                total_docs += len(batch)
                total_chunks += len(results.get("chunks", []))

                # Log progress every 1000 documents
                if total_docs % 1000 == 0:
                    logger.info(
                        f"Progress: Processed {total_docs} documents, {total_chunks} chunks"
                    )

            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                logger.error("Continuing with next batch...")
                continue

        logger.info("Pipeline completed successfully!")
        logger.info(f"Total documents processed: {total_docs}")
        logger.info(f"Total chunks created and indexed: {total_chunks}")

    except Exception as e:
        logger.error(f"Fatal error in pipeline: {str(e)}")
        raise


if __name__ == "__main__":
    main()
