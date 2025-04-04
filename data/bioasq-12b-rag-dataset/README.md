---
language:
  - en
license: cc-by-nc-sa-4.0
datasets:
  - bioasq
task_categories:
  - question-answering
  - sentence-similarity
tags:
  - biomedical
  - rag
  - pubmed
  - bioasq
  - biomedical-qa
library_name: huggingface
pretty_name: BioASQ 12B RAG Dataset
---

# BioASQ 12B RAG Dataset

A processed version of the BioASQ 12B dataset optimized for Retrieval-Augmented Generation (RAG) applications in biomedical question answering.

This dataset provides a structured collection of biomedical questions paired with relevant PubMed abstracts and gold standard answers. It is specifically formatted for RAG pipelines, making it ideal for training and evaluating systems that need to retrieve relevant biomedical information from a corpus and generate accurate, evidence-based answers to complex biomedical questions.

## Dataset Structure

The dataset contains three main components:

1. **Corpus** (`data/corpus.jsonl`): A collection of PubMed abstracts including metadata.

   - Each line is a JSON object containing:
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

2. **Dev Questions** (`data/dev.jsonl`): Development set of biomedical questions.

   - Each line is a JSON object containing:
     - `question_id`: Unique identifier for the question
     - `question`: The question text
     - `answer`: Ideal answer
     - `relevant_passage_ids`: List of PubMed IDs for relevant abstracts
     - `type`: Question type (e.g., factoid, list, yes/no, summary)
     - `snippets`: Relevant snippets from abstracts

3. **Test Questions** (`data/test.jsonl`): Test set of biomedical questions.
   - Same structure as dev questions

## Usage

This dataset is designed for training and evaluating RAG systems for biomedical question answering.

### Loading the Dataset

You can load the dataset using the Hugging Face `datasets` library:

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("mattmorgis/bioasq-12b-rag-dataset")

# Access the corpus
corpus = dataset["corpus"]

# Access the development questions
dev_questions = dataset["dev"]

# Access the test questions
test_questions = dataset["test"]
```

### Example RAG Application

This dataset can be used to build a biomedical RAG system:

1. Index the corpus using a vector database (e.g., FAISS, Chroma)
2. Embed questions using a biomedical or general purpose text embedding model
3. Retrieve relevant documents from the corpus based on question embeddings
4. Generate answers using a large language model (LLM) with the retrieved context

### Evaluation

The dataset provides gold standard answers and relevant passage IDs that can be used to evaluate:

- Retrieval accuracy
- Answer quality
- Domain-specific knowledge incorporation

## Source

This dataset is derived from the [BioASQ Challenge](http://bioasq.org/) task 12b dataset.

Anastasia Krithara, Anastasios Nentidis, Konstantinos Bougiatiotis, Georgios Paliouras. BioASQ-QA: A manually curated corpus for Biomedical Question Answering. bioRxiv 2022.12.14.520213; doi: https://doi.org/10.1101/2022.12.14.520213
