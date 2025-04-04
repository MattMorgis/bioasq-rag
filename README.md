# BioASQ RAG

A Retrieval-Augmented Generation (RAG) system built on the BioASQ dataset for biomedical question answering.

## Project Overview

This project implements a RAG-based approach to biomedical question answering using the BioASQ dataset. The system retrieves relevant PubMed abstracts for a given biomedical question and generates accurate, evidence-based answers.

## Project Structure

The project is organized into modular components, each handling a specific part of the RAG pipeline:

### [Data Acquisition](data_acquisition/README.md)

The data acquisition module handles downloading and processing PubMed abstracts referenced in the BioASQ dataset:

- Extracts PubMed URLs from BioASQ questions
- Downloads abstracts using the NCBI E-utilities API
- Processes and stores the abstracts for later use in the RAG pipeline

For details on how to use this module, see the [Data Acquisition README](data_acquisition/README.md).

### [Data Processing](data_processing/README.md)

The data processing module prepares BioASQ data into a structured format for RAG applications:

- Creates a corpus of PubMed abstracts in JSONL format
- Processes BioASQ questions into development and test datasets
- Prepares the dataset structure for publication on Hugging Face
- Includes validation and metadata creation for the final dataset

For details on how to use this module, see the [Data Processing README](data_processing/README.md).
Final dataset can be found here: [huggingface.co/datasets/mattmorgis/bioasq-12b-rag](https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag)

## Development Setup

This project uses `uv` for Python package management.

```bash
# Install dependencies
uv sync

# Run commands in the project environment
uv run <script.py>

# Add dependencies
uv add <package>

# Add development dependencies
uv add <package> --dev
```

## Running Tests

```bash
# Run all tests
uv run pytest

# Run tests for a specific module
uv run pytest <module_directory>
```
