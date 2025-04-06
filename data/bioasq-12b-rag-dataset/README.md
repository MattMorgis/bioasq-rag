---
configs:
  - config_name: text-corpus
    data_files: "data/corpus.jsonl"
  - config_name: question-answer-passages
    data_files:
      - split: dev
        path: "data/dev.jsonl"
      - split: eval
        path: "data/eval.jsonl"
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

This dataset contains two distinct subsets specifically designed for RAG applications:

1. **A text corpus of PubMed abstracts** ready for indexing and retrieval, containing detailed metadata and full abstract text.

2. **An evaluation dataset** consisting of biomedical questions, each paired with an ideal answer and a list of passage IDs that are relevant to answering the question.

This structure makes it ideal for building and evaluating RAG systems that retrieve relevant biomedical information from a corpus and generate accurate, evidence-based answers to complex biomedical questions.

The code to generate this dataset is here: https://github.com/MattMorgis/bioasq-rag

## Dataset Structure

The dataset contains three main components:

1. **Corpus** (`data/corpus.jsonl`): A collection of PubMed abstracts including metadata.

   - The corpus is accessible through the "train" split of the "text-corpus" config
   - Each document contains:
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

   - The dev questions are accessible through the "dev" split of the "question-answer-passages" config
   - Each question contains:
     - `question_id`: Unique identifier for the question
     - `question`: The question text
     - `answer`: Ideal answer
     - `relevant_passage_ids`: List of PubMed IDs for relevant abstracts
     - `type`: Question type (e.g., factoid, list, yes/no, summary)
     - `snippets`: Relevant snippets from abstracts

3. **Eval Questions** (`data/eval.jsonl`): Eval set of biomedical questions.
   - Same structure as dev questions, accessible through the "eval" split

## Usage

This dataset is designed for training and evaluating RAG systems for biomedical question answering.

### Loading the Dataset

You can load the dataset using the Hugging Face `datasets` library. **Note that you must specify a config name**:

```python
from datasets import load_dataset

# Load the corpus of PubMed abstracts
corpus_dataset = load_dataset("mattmorgis/bioasq-12b-rag", "text-corpus")

# Load the question-answer dataset
questions_dataset = load_dataset("mattmorgis/bioasq-12b-rag", "question-answer-passages")

# Access the corpus data (note: the corpus is stored in the "train" split)
corpus_docs = corpus_dataset["train"]

# Access the development questions
dev_questions = questions_dataset["dev"]

# Access the eval questions
eval_questions = questions_dataset["eval"]
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
