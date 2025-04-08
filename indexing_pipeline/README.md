# Indexing Pipeline

A tool for processing biomedical abstracts from the BioASQ dataset.

## What it does

This pipeline:

- Loads biomedical abstracts from the BioASQ dataset
- Splits them into smaller chunks
- Creates vector embeddings for these chunks
- Indexes the embeddings in Qdrant for efficient retrieval

## Components

- **Data Loader**: Reads abstracts from files
- **Chunker**: Splits text into smaller pieces
- **Embedder**: Converts text chunks into vector embeddings
- **Indexer**: Stores embeddings in Qdrant vector database
- **Pipeline**: Connects all the components together

## Setup

### Installation

Install dependencies:

```bash
uv sync
```

### Testing

Run tests:

```bash
uv run pytest indexing_pipeline
```

## Usage

### Running the Pipeline

Before running the pipeline, make sure Qdrant is running (see `db/README.md`).

To run the indexing pipeline:

```bash
uv run indexing_pipeline/main.py
```

### Managing the Index

The pipeline will automatically create and update the Qdrant index. However, if you need to clear and reinitialize the index (e.g., to remove duplicates or start fresh), you can use:

```bash
uv run indexing_pipeline/clear_index.py
```

This will:

1. Delete the existing collection if it exists
2. Create a fresh collection with the correct schema
3. Set up necessary payload indexes

After clearing the index, you can run the main pipeline again to reindex your documents.
