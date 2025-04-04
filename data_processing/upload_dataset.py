import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

# Load environment variables from .env file
load_dotenv()

# Initialize the Hugging Face API client
api = HfApi()

# Set your variables
TOKEN = os.getenv("HUGGINGFACE_TOKEN")  # Get from .env file
REPO_ID = "mattmorgis/bioasq-12b-rag"
DATASET_DIR = "data/bioasq-12b-rag-dataset"

# Upload the dataset files
files_to_upload = [
    os.path.join(DATASET_DIR, "README.md"),
    os.path.join(DATASET_DIR, "dataset-info.json"),
    os.path.join(DATASET_DIR, ".gitattributes"),
    os.path.join(DATASET_DIR, "data/corpus.jsonl"),
    os.path.join(DATASET_DIR, "data/dev.jsonl"),
    os.path.join(DATASET_DIR, "data/eval.jsonl"),
]

# Create a list of (path_in_repo, local_path) tuples
upload_pairs = []
for file_path in files_to_upload:
    if os.path.exists(file_path):
        path_in_repo = os.path.relpath(file_path, DATASET_DIR)
        upload_pairs.append((path_in_repo, file_path))
        print(f"Will upload {file_path} to {path_in_repo}")
    else:
        print(f"Warning: File {file_path} does not exist, skipping")

# Upload each file to Hugging Face
for path_in_repo, local_path in upload_pairs:
    print(f"Uploading {local_path} to {path_in_repo}...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path_in_repo,
        repo_id=REPO_ID,
        repo_type="dataset",
        token=TOKEN,
    )
    print(f"Successfully uploaded {path_in_repo}")

print("Upload complete!")
