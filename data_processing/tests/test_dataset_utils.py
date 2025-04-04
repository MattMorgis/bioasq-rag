import json
import os

from src.dataset_utils import (
    validate_dataset,
)


def test_validate_dataset_success(temp_output_dir):
    """Test dataset validation with valid dataset."""
    # Create all required files
    os.makedirs(os.path.join(temp_output_dir, "data"), exist_ok=True)

    # Create corpus.jsonl
    with open(
        os.path.join(temp_output_dir, "data/corpus.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"id": "12345678", "text": "Sample abstract"}) + "\n")

    # Create dev.jsonl
    with open(
        os.path.join(temp_output_dir, "data/dev.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"question_id": "q1", "question": "Sample question"}) + "\n")

    # Create test.jsonl
    with open(
        os.path.join(temp_output_dir, "data/test.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"question_id": "q2", "question": "Sample question"}) + "\n")

    # Create dataset-info.json
    with open(
        os.path.join(temp_output_dir, "dataset-info.json"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"name": "test-dataset"}) + "\n")

    # Create README.md
    with open(os.path.join(temp_output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Test Dataset\n")

    # Validate dataset
    result = validate_dataset(temp_output_dir)

    # Check that validation succeeds
    assert result is True


def test_validate_dataset_failure_missing_file(temp_output_dir):
    """Test dataset validation with missing file."""
    # Create some but not all required files
    os.makedirs(os.path.join(temp_output_dir, "data"), exist_ok=True)

    # Create corpus.jsonl but skip other files
    with open(
        os.path.join(temp_output_dir, "data/corpus.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"id": "12345678", "text": "Sample abstract"}) + "\n")

    # Validate dataset
    result = validate_dataset(temp_output_dir)

    # Check that validation fails
    assert result is False


def test_validate_dataset_failure_invalid_json(temp_output_dir):
    """Test dataset validation with invalid JSON."""
    # Create all required files
    os.makedirs(os.path.join(temp_output_dir, "data"), exist_ok=True)

    # Create corpus.jsonl with valid JSON
    with open(
        os.path.join(temp_output_dir, "data/corpus.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"id": "12345678", "text": "Sample abstract"}) + "\n")

    # Create dev.jsonl with invalid JSON
    with open(
        os.path.join(temp_output_dir, "data/dev.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write("This is not valid JSON\n")

    # Create test.jsonl with valid JSON
    with open(
        os.path.join(temp_output_dir, "data/test.jsonl"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"question_id": "q2", "question": "Sample question"}) + "\n")

    # Create dataset-info.json
    with open(
        os.path.join(temp_output_dir, "dataset-info.json"), "w", encoding="utf-8"
    ) as f:
        f.write(json.dumps({"name": "test-dataset"}) + "\n")

    # Create README.md
    with open(os.path.join(temp_output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("# Test Dataset\n")

    # Validate dataset
    result = validate_dataset(temp_output_dir)

    # Check that validation fails
    assert result is False
