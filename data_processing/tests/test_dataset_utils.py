import json
import os

from src.dataset_utils import (
    create_dataset_metadata,
    create_dataset_readme,
    prepare_huggingface_structure,
    validate_dataset,
)


def test_create_dataset_metadata(temp_output_dir):
    """Test creating dataset metadata."""
    # Create dataset metadata
    create_dataset_metadata(temp_output_dir, 100, 50, 25)

    # Check that the metadata file exists
    metadata_path = os.path.join(temp_output_dir, "dataset-info.json")
    assert os.path.exists(metadata_path)

    # Check the content of the metadata file
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

        # Check the metadata structure
        assert metadata["name"] == "bioasq-12b-rag-dataset"
        assert metadata["version"] == "1.0.0"
        assert "description" in metadata

        # Check the split information
        assert metadata["splits"]["corpus"]["num_examples"] == 100
        assert metadata["splits"]["dev"]["num_examples"] == 50
        assert metadata["splits"]["test"]["num_examples"] == 25

        # Check the feature information
        assert "corpus" in metadata["features"]
        assert "questions" in metadata["features"]
        assert len(metadata["features"]["corpus"]) > 0
        assert len(metadata["features"]["questions"]) > 0


def test_create_dataset_readme(temp_output_dir):
    """Test creating dataset README."""
    # Create README
    create_dataset_readme(temp_output_dir)

    # Check that the README file exists
    readme_path = os.path.join(temp_output_dir, "README.md")
    assert os.path.exists(readme_path)

    # Check the content of the README file
    with open(readme_path, "r", encoding="utf-8") as f:
        readme_content = f.read()

        # Check that the README contains important sections
        assert "BioASQ 12B RAG Dataset" in readme_content
        assert "Dataset Structure" in readme_content
        assert "Corpus" in readme_content
        assert "Dev Questions" in readme_content
        assert "Test Questions" in readme_content
        assert "Usage" in readme_content
        assert "License" in readme_content


def test_prepare_huggingface_structure(temp_output_dir):
    """Test preparing Hugging Face structure."""
    # Prepare Hugging Face structure
    prepare_huggingface_structure(temp_output_dir)

    # Check that the .gitattributes file exists
    gitattributes_path = os.path.join(temp_output_dir, ".gitattributes")
    assert os.path.exists(gitattributes_path)

    # Check the content of the .gitattributes file
    with open(gitattributes_path, "r", encoding="utf-8") as f:
        content = f.read()

        # Check that the file contains the expected content
        assert "*.jsonl filter=lfs diff=lfs merge=lfs -text" in content


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
