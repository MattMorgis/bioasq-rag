from pathlib import Path

from src.chunker.word_chunker import WordChunker
from src.data_loader import DataLoader
from src.embedding.sentence_tranformer_embedder import SentenceTransformerEmbedder
from src.pipeline import Pipeline


def main():
    data_path = Path("data/bioasq-12b-rag-dataset/data/corpus.jsonl")
    data_loader = DataLoader(corpus_path=data_path)

    # Create pipeline components
    chunker = WordChunker(chunk_size=200, chunk_overlap=50)
    embedder = SentenceTransformerEmbedder(
        model_name="sentence-transformers/all-MiniLM-L12-v2", batch_size=32
    )

    # Create pipeline with all steps
    pipeline = Pipeline(chunker=chunker, embedder=embedder)

    print("Processing a small batch of documents...")
    # Get a small batch of documents
    # change limit to batch_size to process the entire dataset
    for batch in data_loader.load_abstracts_from_file(limit=5):
        results = pipeline.process_documents(batch)
        print(f"Processed {len(batch)} documents")
        print(f"Generated {len(results.get('chunks', []))} chunks")
        print(f"Generated {len(results.get('embedded_chunks', []))} embedded chunks")
        # Write results to output files
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Pretty print the first embedded chunk for inspection
        if results.get("embedded_chunks"):
            first_embedded_chunk = results["embedded_chunks"][0]
            print("\nSample Embedded Chunk:")
            print(f"  Document ID: {first_embedded_chunk.chunk.abstract.url}")
            print(f"  Chunk ID: {first_embedded_chunk.chunk.chunk_id}")
            print(f"  Journal: {first_embedded_chunk.chunk.abstract.journal}")
            print(f"  Title: {first_embedded_chunk.chunk.abstract.title}")
            print(f"  Text: {first_embedded_chunk.chunk.text}")
            print(f"  Embedding shape: {len(first_embedded_chunk.embedding)}")
            print(f"  Embedding model: {first_embedded_chunk.embedding_model}")

    # Example: Configure for larger scale processing
    # Uncommenting this would process the entire dataset
    """
    print("\nProcessing full dataset in batches...")

    # Process in batches of 100 documents
    count = 0
    for batch in data_loader.load_abstracts_from_file(batch_size=100):
        results = pipeline.process_documents(batch)
        count += len(batch)
        print(f"Processed {count} documents so far")
    """


if __name__ == "__main__":
    main()
