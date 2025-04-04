# Data Acquisition

The `data_acquisition` module contains the necessary components to collect and download PubMed abstracts for the BioASQ dataset.

### Components

#### 1. PubMedURLCollector

The `PubMedURLCollector` is responsible for extracting and collecting all unique PubMed URLs from the BioASQ dataset:

- Processes both BioASQ 12b training and golden sets
- Extracts all URLs from the `documents` field of each question
- Deduplicates URLs to ensure each abstract is only downloaded once
- Can save the collected URLs to a file for reference

#### 2. BioPythonPubMedClient

The `BioPythonPubMedClient` handles the actual retrieval of abstracts from PubMed:

- Uses Biopython's Entrez API to fetch PubMed abstracts
- Supports API key authentication for higher rate limits
- Handles rate limiting and retries gracefully
- Extracts and formats abstract data including title, authors, publication date, etc.

#### 3. DataFetcher

The `DataFetcher` orchestrates the entire process:

- Uses the `PubMedURLCollector` to gather all required PubMed URLs
- Leverages `BioPythonPubMedClient` to download abstracts
- Implements parallel fetching with configurable batch size
- Respects NCBI rate limits (3/second without API key, 10/second with API key)
- Handles retries and error logging
- Saves abstracts as JSON files in the data directory

### Fetching PubMed Abstracts

To run the fetcher:

```bash
# Basic usage
uv run data_acquisition/main.py --email your.email@example.com

# Advanced usage with all parameters
uv run data_acquisition/main.py \
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
- `--rate-limit`: Maximum requests per second (default: 10, use 3 without API key)
- `--max-retries`: Maximum retries for failed requests (default: 3)
- `--retry-delay`: Delay in seconds between retries (default: 5)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Retrying Failed Downloads

During the initial data fetching process, some abstracts might fail to download due to various reasons (network issues, rate limiting, etc.). The `retry_failed.py` script allows you to retry these failed downloads with more conservative settings:

```bash
# Basic usage
uv run data_acquisition/src/retry_failed.py --email your.email@example.com

# Advanced usage with all parameters
uv run data_acquisition/src/retry_failed.py \
  --email your.email@example.com \
  --api-key YOUR_NCBI_API_KEY \
  --data-dir data \
  --batch-size 10 \
  --rate-limit 3 \
  --max-retries 5 \
  --retry-delay 10 \
  --log-level INFO
```

The retry script:

- Uses more conservative default settings than the main fetcher
- Reads the failed URLs from the `failed_urls.json` file created during the initial run
- Attempts to fetch only those abstracts that previously failed
- Updates the `failed_urls.json` file after completion, or removes it if all abstracts were successfully fetched

**Default Retry Parameters:**

- `--batch-size`: 10 (smaller than main fetcher for more careful processing)
- `--rate-limit`: 3 (lower than main fetcher to avoid rate limiting)
- `--max-retries`: 5 (more retries per URL)
- `--retry-delay`: 10 (longer delay between retries)

### Rate Limits and Performance

- **Without API key**: Limited to 3 requests per second
- **With API key**: Up to 10 requests per second
- Fetching all ~50,000 abstracts typically takes 2-4 hours depending on network speed

### Getting an NCBI API Key

To obtain an API key for higher rate limits:

1. Create an NCBI account at https://www.ncbi.nlm.nih.gov/ if you don't already have one
2. Log into your NCBI account
3. Click on your username at the top right corner to access the "Settings" page
4. Look for the "API Key Management" section
5. Click "Create an API Key" and copy the generated key
6. Use the key in your requests via the `--api-key` parameter

For more information, visit [the official NCBI documentation](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/).

### Running Tests

To run the unit tests for the data acquisition module:

```bash
# From within the data_acquisition directory
uv run pytest
```

or from the project root:

```bash
uv run pytest data_acquisition
```

The tests verify the functionality of the URL collector, PubMed client, and data fetcher components.
