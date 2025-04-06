# Indexing Pipeline

A tool for processing biomedical abstracts from the BioASQ dataset.

## What it does

This pipeline:

- Loads biomedical abstracts from the BioASQ dataset
- Splits them into smaller chunks
- Creates vector embeddings for these chunks
- Prepares the data for efficient retrieval

## Components

- **Data Loader**: Reads abstracts from files
- **Chunker**: Splits text into smaller pieces
- **Embedder**: Converts text chunks into vector embeddings
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
