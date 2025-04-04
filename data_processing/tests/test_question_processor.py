import json
import os

from src.question_processor import (
    create_question_datasets,
    extract_pubmed_id,
    process_question,
    process_question_file,
)


def test_extract_pubmed_id():
    """Test extraction of PubMed ID from URL."""
    # Test with valid URL
    url = "http://www.ncbi.nlm.nih.gov/pubmed/12345678"
    pubmed_id = extract_pubmed_id(url)
    assert pubmed_id == "12345678"

    # Test with invalid URL
    url = "http://example.com/not-pubmed"
    pubmed_id = extract_pubmed_id(url)
    assert pubmed_id is None


def test_process_question(sample_question_data):
    """Test processing a single question."""
    # Process the sample question
    result = process_question(sample_question_data)

    # Check that the result matches expectations
    assert result is not None
    assert result["question_id"] == sample_question_data["id"]
    assert result["question"] == sample_question_data["body"]
    assert result["answer"] == sample_question_data["ideal_answer"][0]
    assert result["type"] == sample_question_data["type"]
    assert len(result["relevant_passage_ids"]) == 2
    assert "12345678" in result["relevant_passage_ids"]
    assert "23456789" in result["relevant_passage_ids"]
    assert result["snippets"] == sample_question_data["snippets"]


def test_process_question_with_invalid_data():
    """Test that processing an invalid question returns None."""
    # Create invalid question data
    invalid_question = {"invalid": "data"}

    # Process the invalid question
    result = process_question(invalid_question)

    # Check that the result is None
    assert result is None


def test_process_question_file(sample_questions_file):
    """Test processing a question file."""
    # Process the sample questions file
    questions = process_question_file(str(sample_questions_file))

    # Check that the correct number of questions was processed
    assert len(questions) == 2

    # Check that the questions have the correct IDs
    question_ids = [q["question_id"] for q in questions]
    assert "test_question_id" in question_ids
    assert "test_question_id_2" in question_ids


def test_process_question_file_with_invalid_file(tmp_path):
    """Test that processing an invalid question file returns an empty list."""
    # Create an invalid file
    invalid_file = tmp_path / "invalid.json"
    with open(invalid_file, "w") as f:
        f.write("This is not valid JSON")

    # Process the invalid file
    questions = process_question_file(str(invalid_file))

    # Check that the result is an empty list
    assert questions == []


def test_create_question_datasets(
    sample_questions_file, sample_goldset_dir, temp_output_dir
):
    """Test creating question datasets."""
    # Define output paths
    dev_path = os.path.join(temp_output_dir, "data/dev.jsonl")
    test_path = os.path.join(temp_output_dir, "data/test.jsonl")

    # Create the datasets
    dev_count, test_count = create_question_datasets(
        str(sample_questions_file), str(sample_goldset_dir), dev_path, test_path
    )

    # Check that the correct number of questions was processed
    assert dev_count == 2  # 2 questions in the sample file
    assert test_count == 2  # 2 goldset files with 1 question each

    # Check that the output files exist
    assert os.path.exists(dev_path)
    assert os.path.exists(test_path)

    # Check the content of the dev file
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_questions = [json.loads(line) for line in f]
        assert len(dev_questions) == 2

        # Check each question has the required fields
        for question in dev_questions:
            assert "question_id" in question
            assert "question" in question
            assert "answer" in question
            assert "relevant_passage_ids" in question

    # Check the content of the test file
    with open(test_path, "r", encoding="utf-8") as f:
        test_questions = [json.loads(line) for line in f]
        assert len(test_questions) == 2

        # Check each question has the required fields and expected IDs
        for question in test_questions:
            assert "question_id" in question
            assert "question" in question
            assert "answer" in question
            assert "relevant_passage_ids" in question
            assert question["question_id"].startswith("goldset_question_")
