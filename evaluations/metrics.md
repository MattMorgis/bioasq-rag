# RAG Evaluation Metrics for BioASQ

This document explains the key metrics used to evaluate our RAG system using PubMed abstracts and the BioASQ dataset.

## Retrieval Metrics

### Contextual Precision

Contextual precision measures whether relevant documents in your retrieval results are ranked higher than irrelevant ones. It focuses on the ordering and prioritization of retrieved information.

**Example:**
For a query about "CRISPR applications in cancer therapy":

| Rank | Document                                                    | Relevance |
| ---- | ----------------------------------------------------------- | --------- |
| 1    | Recent study on CRISPR-Cas9 for targeting lung cancer cells | High      |
| 2    | CRISPR gene editing mechanisms and technical limitations    | Medium    |
| 3    | Advances in immunotherapy for cancer treatment              | Low       |
| 4    | CRISPR applications in treating genetic disorders           | Low       |

**High Precision System:** Ranks the highly relevant document about CRISPR for lung cancer first, followed by the somewhat relevant document about CRISPR mechanisms, with less relevant documents ranked lower.

**Low Precision System:** Might rank the immunotherapy document (which doesn't focus on CRISPR) higher than the directly relevant CRISPR cancer therapy documents.

### Contextual Recall

Contextual recall measures how completely the retrieved documents cover the information needed for a comprehensive answer. It focuses on whether all necessary information has been retrieved, regardless of ranking.

**Example:**
For the same query about "CRISPR applications in cancer therapy", a complete answer should include:

- CRISPR applications in solid tumors
- CRISPR for hematological malignancies
- Delivery methods for CRISPR cancer therapies
- Clinical trial status and results
- Safety considerations

**High Recall System:** Retrieves documents covering all or most of these aspects, providing comprehensive information for answering the query.

**Low Recall System:** Might only retrieve documents about CRISPR in solid tumors, missing important information about other applications, delivery methods, or clinical status.

### Contextual Relevancy

Contextual relevancy measures the overall topical relevance of retrieved documents to the query, regardless of ranking or completeness.

**Example:**
For the query about "CRISPR applications in cancer therapy":

**High Relevancy System:** Retrieves documents where most or all are related to CRISPR use in cancer treatment, even if they cover different aspects (mechanisms, trials, reviews, etc.).

**Medium Relevancy System:** Retrieves some documents about CRISPR in cancer but also includes documents about CRISPR generally or cancer therapies generally.

**Low Relevancy System:** Retrieves mostly unrelated documents (e.g., about non-CRISPR genetic techniques or non-cancer CRISPR applications).

### Key Differences Between Retrieval Metrics:

- **Contextual Precision** focuses on the ranking quality (are the most relevant documents at the top?)
- **Contextual Recall** focuses on information completeness (is all necessary information retrieved?)
- **Contextual Relevancy** focuses on topical alignment (are the retrieved documents on-topic?)

## Generation Metrics

### Answer Relevancy

Answer relevancy evaluates how well the generated answer addresses the specific query.

**Example:**
For a query about "What are the mechanisms of antibiotic resistance in Pseudomonas aeruginosa?"

**High Relevancy Answer:** Discusses efflux pumps, beta-lactamases, reduced membrane permeability, and biofilm formation in P. aeruginosa specifically.

**Low Relevancy Answer:** Mostly discusses general antibiotic classes, treatment guidelines, or resistance in other bacteria without focusing on the specific mechanisms in P. aeruginosa.

### Answer Faithfulness

Answer faithfulness evaluates whether the generated answer is supported by the retrieved context, without adding unsupported claims.

**Example:**
Retrieved context only discusses three mechanisms of resistance: efflux pumps, beta-lactamases, and reduced permeability.

**Faithful Answer:** "P. aeruginosa exhibits antibiotic resistance through three primary mechanisms: efflux pumps that expel antibiotics, beta-lactamases that degrade certain antibiotics, and reduced membrane permeability limiting antibiotic entry into the cell."

**Unfaithful Answer:** "P. aeruginosa exhibits antibiotic resistance through four primary mechanisms: efflux pumps, beta-lactamases, reduced permeability, and plasmid transfer of resistance genes." (The context doesn't mention plasmid transfer)

### Important Note on Faithfulness vs. Correctness

A response can be faithful to the retrieved context yet factually incorrect if the retrieved information itself is flawed.

**Example:**
Retrieved context (containing outdated information): "A 2010 study found that P. aeruginosa cannot develop resistance to polymyxin antibiotics."

**Faithful but Incorrect Answer:** "According to research, P. aeruginosa cannot develop resistance to polymyxin antibiotics."

This answer is faithful to the context but factually wrong, as newer research has documented polymyxin resistance in P. aeruginosa. This highlights why high-quality retrieval is crucial - the generation metrics only measure how well the system works with whatever information it retrieves.

## Comprehensive Evaluation

A high-quality RAG system needs to excel in both retrieval and generation metrics:

1. **Good retrieval** ensures the system finds relevant, comprehensive, and properly ranked information
2. **Good generation** ensures the system creates answers that address the query while staying faithful to the retrieved information

Weaknesses in either component will affect overall system performance. For example, perfect generation cannot compensate for retrieving irrelevant or incomplete information, and excellent retrieval is wasted if the generation process introduces hallucinations or fails to address the user's query.
