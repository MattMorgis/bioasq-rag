# Evaluations

This module runs the RAG pipeline for the BioASQ 12b goldset, capturing both retrieved chunks and final generated answers for analysis.

## Overview

The evaluations module:

1. Processes questions from the BioASQ 12b goldset
2. Runs each question through the complete RAG pipeline
3. Captures and stores:
   - Retrieved document chunks
   - Generated answers
   - Reference (gold standard) answers

## Metrics

The module uses the [RAGAS metrics](https://ar5iv.labs.arxiv.org/html/2309.15217) to evaluate each question/answer pair using four different metrics:

- [Answer Relevancy](https://docs.confident-ai.com/docs/metrics-answer-relevancy)
- [Faithfulness](https://docs.confident-ai.com/docs/metrics-faithfulness)
- [Contextual Precision](https://docs.confident-ai.com/docs/metrics-contextual-precision)
- [Contextual Recall](https://docs.confident-ai.com/docs/metrics-contextual-recall)

We also use one more additional metric

- [Contextual Relevancy](https://docs.confident-ai.com/docs/metrics-contextual-relevancy)

## Usage

First, install dependencies:

```bash
uv sync
```

### Running the RAG Pipeline

To process questions through the RAG pipeline and save results:

```bash
uv run evaluations/src/run_rag.py <eval_file_path> <output_file_path> [max_workers]
```

Example:

```bash
uv run evaluations/src/run_rag.py data/bioasq-12b-rag-dataset/data/eval.jsonl results/rag_evaluation.jsonl 2
```

### Running Evaluations

To evaluate the generated answers against ground truth:

```bash
uv run evaluations/src/rag_deep_eval.py
```

This will process the `rag_evaluation.jsonl` file and generate a detailed evaluation report in the `results/eval-1` directory.
