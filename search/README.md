# BioASQ Search API

A FastAPI-based search API for querying the BioASQ dataset using RAG capabilities.

## Installation

1. Install the dependencies:

```bash
uv sync
```

## Running the API

Start the FastAPI development server:

```bash
uv run uvicorn main:app --reload
```

The server will start at `http://localhost:8000`.

## API Endpoints

### GET `/search`

Performs a semantic search over the PubMed abstracts.

**Parameters:**

- `query` (required): The search query
- `limit` (optional, default: 10): Number of results to return

**Example Request:**

```
GET /search?query=What is the role of IL-17 in asthma?&limit=5
```

**Example Response:**

```json
{
  "results": [
    {
      "abstract_id": "12345678",
      "title": "The role of IL-17 in asthma pathogenesis",
      "text": "IL-17 has been shown to play a critical role in asthma by...",
      "score": 0.85,
      "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/",
      "publication_date": "2020-01-15",
      "journal": "Journal of Immunology",
      "authors": ["Smith, J.", "Jones, T."]
    },
    ...
  ],
  "query": "What is the role of IL-17 in asthma?",
  "total_results": 5
}
```

## Interactive Documentation

FastAPI generates interactive API documentation automatically:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
