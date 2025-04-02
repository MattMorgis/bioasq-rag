# BioASQ RAG

## Data Pipelines

### Fetching PubMed Abstracts

The system includes a data fetcher that can download thousands of PubMed abstracts in parallel while respecting rate limits. The fetcher uses the PubMedURLCollector to gather URLs from the BioASQ dataset and then downloads abstracts for each URL.

To run the fetcher:

```bash
# Basic usage
uv run src/data_pipelines/run.py --email your.email@example.com

# Advanced usage with all parameters
uv run src/data_pipelines/run.py \
  --email your.email@example.com \
  --api-key YOUR_NCBI_API_KEY \
  --data-dir data \
  --batch-size 100 \
  --rate-limit 10 \
  --max-retries 3 \
  --retry-delay 5 \
  --log-level INFO
```

**Parameters:**

- `--email` (required): Your email address for the NCBI API
- `--api-key`: NCBI API key for higher rate limits (optional but recommended)
- `--data-dir`: Directory to save abstracts to (default: "data")
- `--batch-size`: Number of abstracts to fetch in parallel per batch (default: 100)
- `--rate-limit`: Maximum requests per minute (default: 10, use 3 without API key)
- `--max-retries`: Maximum retries for failed requests (default: 3)
- `--retry-delay`: Delay in seconds between retries (default: 5)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

The fetcher will:

1. Collect all unique PubMed URLs from the BioASQ dataset, both training and gold sets.
2. Download abstracts in parallel batches while respecting rate limits
3. Handle retries for rate limit errors
4. Skip abstracts that have already been downloaded
5. Save abstracts as JSON files in the `data/abstracts` directory

Fetching all ~50,000 abstracts should take 1.5~ hours with the rate limit of 10 per second.
