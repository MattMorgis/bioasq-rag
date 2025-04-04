import json
import os
import tempfile

import pytest


@pytest.fixture
def sample_abstract_data():
    """Fixture providing sample abstract data for testing."""
    return {
        "id": "12345678",
        "title": "Test Abstract Title",
        "abstract": "This is a test abstract for unit testing the corpus processor.",
        "authors": ["Test Author 1", "Test Author 2"],
        "publication_date": "2023 Jan 01",
        "journal": "Test Journal of Medicine",
        "doi": "10.1234/test.12345",
        "keywords": ["test", "abstract", "biomedical"],
        "mesh_terms": ["Test Term 1", "Test Term 2"],
    }


@pytest.fixture
def sample_abstract_file(tmp_path, sample_abstract_data):
    """Fixture creating a temporary abstract JSON file."""
    abstract_dir = tmp_path / "abstracts"
    abstract_dir.mkdir()

    # Create a sample abstract file
    file_path = abstract_dir / "12345678.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_abstract_data, f)

    return file_path


@pytest.fixture
def sample_abstracts_dir(tmp_path, sample_abstract_data):
    """Fixture creating a temporary directory with multiple abstract files."""
    abstract_dir = tmp_path / "abstracts"
    abstract_dir.mkdir()

    # Create multiple abstract files
    pmids = ["12345678", "23456789", "34567890"]

    for i, pmid in enumerate(pmids):
        abstract = sample_abstract_data.copy()
        abstract["id"] = pmid
        abstract["title"] = f"Test Abstract {i + 1}"

        file_path = abstract_dir / f"{pmid}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(abstract, f)

    return abstract_dir


@pytest.fixture
def sample_question_data():
    """Fixture providing sample question data for testing."""
    return {
        "id": "test_question_id",
        "body": "What is the mechanism of action of aspirin?",
        "type": "summary",
        "documents": [
            "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
            "http://www.ncbi.nlm.nih.gov/pubmed/23456789",
        ],
        "ideal_answer": ["Aspirin works by inhibiting cyclooxygenase enzymes."],
        "snippets": [
            {
                "document": "http://www.ncbi.nlm.nih.gov/pubmed/12345678",
                "text": "Aspirin inhibits the enzyme cyclooxygenase.",
                "beginSection": "abstract",
                "endSection": "abstract",
                "offsetInBeginSection": 10,
                "offsetInEndSection": 50,
            }
        ],
    }


@pytest.fixture
def sample_questions_file(tmp_path, sample_question_data):
    """Fixture creating a temporary questions JSON file."""
    questions_dir = tmp_path / "questions"
    questions_dir.mkdir()

    # Create a questions file with multiple questions
    file_path = questions_dir / "training.json"

    # Create a second sample question
    question2 = sample_question_data.copy()
    question2["id"] = "test_question_id_2"
    question2["body"] = "What are the side effects of ibuprofen?"

    data = {"questions": [sample_question_data, question2]}

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    return file_path


@pytest.fixture
def sample_goldset_dir(tmp_path, sample_question_data):
    """Fixture creating a temporary goldset directory with question files."""
    goldset_dir = tmp_path / "goldset"
    goldset_dir.mkdir()

    # Create multiple goldset files
    for batch in range(1, 3):
        # Create a slightly different question for each batch
        question = sample_question_data.copy()
        question["id"] = f"goldset_question_{batch}"
        question["body"] = f"Test goldset question {batch}?"

        file_path = goldset_dir / f"batch{batch}_golden.json"

        data = {"questions": [question]}

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    return goldset_dir


@pytest.fixture
def temp_output_dir():
    """Fixture providing a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        yield temp_dir
