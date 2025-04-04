# BioASQ RAG Data Processing

This module processes BioASQ data into a structured format for RAG applications and prepares it for publication on Hugging Face.

## Published Dataset

The processed dataset is available on Hugging Face:
[mattmorgis/bioasq-12b-rag](https://huggingface.co/datasets/mattmorgis/bioasq-12b-rag)

## Dataset Structure

The processed dataset will be created with the following structure:

```
data/bioasq-12b-rag-dataset/
├── .gitattributes
├── README.md
├── dataset-info.json
└── data/
    ├── corpus.jsonl     # All PubMed abstracts
    ├── dev.jsonl        # Development questions
    └── test.jsonl       # Test questions
```

## Usage

Run the script with default parameters:

```bash
uv run data_processing/main.py
```

Or specify custom paths:

```bash
uv run data_processing/main.py --abstracts_dir /path/to/abstracts --training_file /path/to/training.json --goldset_dir /path/to/goldset --output_dir /path/to/output
```

### Parameters

- `--abstracts_dir`: Directory containing PubMed abstract JSON files (default: "data/abstracts")
- `--training_file`: Path to BioASQ training file (default: "data/BioASQ-12b/training/training12b_new.json")
- `--goldset_dir`: Directory containing BioASQ goldset files (default: "data/BioASQ-12b/goldset")
- `--output_dir`: Output directory for the processed dataset (default: "data/bioasq-12b-rag-dataset")

## Output Format

### Corpus (corpus.jsonl)

Each line is a JSON object with:

- `id`: PubMed ID
- `title`: Title of the paper
- `text`: Abstract text
- `url`: PubMed URL
- `publication_date`: Publication date
- `journal`: Journal name
- `authors`: List of authors
- `doi`: Digital Object Identifier (if available)
- `keywords`: Keywords
- `mesh_terms`: MeSH terms

### Questions (dev.jsonl, test.jsonl)

Each line is a JSON object with:

- `question_id`: Unique identifier for the question
- `question`: The question text
- `answer`: Ideal answer
- `relevant_passage_ids`: List of PubMed IDs for relevant abstracts
- `type`: Question type (e.g., factoid, list, yes/no, summary)
- `snippets`: Relevant snippets from abstracts

## Example Dataset Usage

There is an example script that demonstrates how to load the dataset from Hugging Face:

```bash
uv run data_processing/example/bioasq_demo.py
```

The script includes all required dependencies in script headers and shows:

- How to load the dataset from Hugging Face
- Basic dataset statistics (corpus size, number of questions)
- Sample document and question data structure
