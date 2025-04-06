# /// script
# dependencies = [
#   "pandas",
#   "datasets",
# ]
# ///
"""
BioASQ RAG Dataset Demo

This script demonstrates how to load the BioASQ 12B RAG dataset
from Hugging Face and display basic information about it.
"""

import pandas as pd
from datasets import load_dataset

# Load the dataset with specified configs
print("Loading BioASQ 12B RAG dataset from Hugging Face...")
corpus_dataset = load_dataset("mattmorgis/bioasq-12b-rag", "text-corpus")
questions_dataset = load_dataset(
    "mattmorgis/bioasq-12b-rag", "question-answer-passages"
)

# Extract the actual corpus data from the nested structure
corpus = corpus_dataset["train"]

# Access the questions
dev_questions = questions_dataset["dev"]
eval_questions = questions_dataset["eval"]

print(f"Loaded {len(corpus)} documents in corpus")
print(f"Loaded {len(dev_questions)} development questions")
print(f"Loaded {len(eval_questions)} evaluation questions")

# Convert to DataFrames for easier manipulation
corpus_df = pd.DataFrame(corpus)
dev_df = pd.DataFrame(dev_questions)

# Debug: Print the column names available in each dataset
print("\nCorpus columns:")
print(corpus_df.columns.tolist() if not corpus_df.empty else "No columns found")
print("\nDev questions columns:")
print(dev_df.columns.tolist() if not dev_df.empty else "No columns found")

# Display sample document from corpus - the first document
print("\nSample document from corpus (first row):")
if not corpus_df.empty:
    sample_doc = corpus_df.iloc[0]
    print("Available keys:")
    for key in sample_doc.index:
        value = sample_doc[key]
        if isinstance(value, str):
            preview = value[:50]
        else:
            preview = str(value)[:50]
        print(f"  - {key}: {preview}...")
else:
    print("No documents found in corpus.")

# Display sample question
print("\nSample question:")
if not dev_df.empty:
    sample_question = dev_df.iloc[0]
    print(f"Question: {sample_question.get('question', 'Not available')}")

    answer = sample_question.get("answer", "Not available")
    if isinstance(answer, str):
        answer_preview = answer[:200] + "..." if len(answer) > 200 else answer
    else:
        answer_preview = str(answer)
    print(f"Answer: {answer_preview}")

    print(f"Type: {sample_question.get('type', 'Not available')}")

    relevant_passages = sample_question.get("relevant_passage_ids", [])
    relevant_preview = relevant_passages[:3] if relevant_passages else "None available"
    print(f"Relevant passages: {relevant_preview} ...")
else:
    print("No questions found in development set.")
