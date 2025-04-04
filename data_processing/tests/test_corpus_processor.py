import json
import os

from src.corpus_processor import create_corpus, process_abstract


def test_process_abstract(sample_abstract_file, sample_abstract_data):
    """Test that an abstract file is processed correctly."""
    # Process the sample abstract
    result = process_abstract(sample_abstract_file)

    # Check that the result matches expectations
    assert result is not None
    assert result["id"] == "12345678"
    assert result["title"] == sample_abstract_data["title"]
    assert result["text"] == sample_abstract_data["abstract"]
    assert result["url"] == "http://www.ncbi.nlm.nih.gov/pubmed/12345678"
    assert result["authors"] == sample_abstract_data["authors"]
    assert result["journal"] == sample_abstract_data["journal"]
    assert result["doi"] == sample_abstract_data["doi"]
    assert result["keywords"] == sample_abstract_data["keywords"]
    assert result["mesh_terms"] == sample_abstract_data["mesh_terms"]


def test_process_abstract_with_invalid_file(tmp_path):
    """Test that processing an invalid abstract file returns None."""
    # Create an invalid file
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("This is not valid JSON")

    # Process the invalid file
    result = process_abstract(invalid_file)

    # Check that the result is None
    assert result is None


def test_create_corpus(sample_abstracts_dir, temp_output_dir):
    """Test that a corpus is created correctly from a directory of abstracts."""
    # Define the output path
    output_path = os.path.join(temp_output_dir, "data/corpus.jsonl")

    # Create the corpus
    count = create_corpus(str(sample_abstracts_dir), output_path)

    # Check that the correct number of abstracts was processed
    assert count == 3

    # Check that the output file exists
    assert os.path.exists(output_path)

    # Check the content of the output file
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 3

        # Check each line is valid JSON
        for line in lines:
            entry = json.loads(line)
            assert "id" in entry
            assert "title" in entry
            assert "text" in entry
            assert "url" in entry


def test_create_corpus_with_empty_dir(tmp_path, temp_output_dir):
    """Test that creating a corpus from an empty directory works."""
    # Create an empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    # Define the output path
    output_path = os.path.join(temp_output_dir, "data/empty_corpus.jsonl")

    # Create the corpus
    count = create_corpus(str(empty_dir), output_path)

    # Check that no abstracts were processed
    assert count == 0

    # Check that the output file exists
    assert os.path.exists(output_path)

    # Check that the output file is empty
    with open(output_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 0
