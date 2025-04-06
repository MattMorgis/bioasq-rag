from typing import Any, Dict, List

import pytest

from src.clients.pubmed_client import PubMedClient, PubMedClientError


class MockPubMedClient(PubMedClient):
    def __init__(self):
        self.error_ids = {"ERROR_ID", "999999999"}  # IDs that should trigger errors

    async def get_abstract_by_id(self, pubmed_id: str) -> Dict[str, Any]:
        if pubmed_id in self.error_ids:
            raise PubMedClientError(f"Failed to retrieve abstract for ID: {pubmed_id}")

        return {
            "id": pubmed_id,
            "title": "Test Title",
            "abstract": "Test Abstract",
            "authors": ["Author 1", "Author 2"],
            "publication_date": "2024-01-01",
        }

    async def get_abstracts_by_ids(self, pubmed_ids: List[str]) -> List[Dict[str, Any]]:
        if any(pid in self.error_ids for pid in pubmed_ids):
            raise PubMedClientError("Failed to retrieve one or more abstracts")
        return [await self.get_abstract_by_id(pid) for pid in pubmed_ids]


@pytest.fixture
def pubmed_client() -> PubMedClient:
    return MockPubMedClient()


@pytest.mark.asyncio
async def test_get_abstract_by_id_success(pubmed_client: PubMedClient):
    """Test successful retrieval of a single abstract."""
    result = await pubmed_client.get_abstract_by_id("12345")
    assert result["id"] == "12345"
    assert "title" in result
    assert "abstract" in result
    assert "authors" in result
    assert "publication_date" in result


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_success(pubmed_client: PubMedClient):
    """Test successful retrieval of multiple abstracts."""
    pubmed_ids = ["12345", "67890"]
    results = await pubmed_client.get_abstracts_by_ids(pubmed_ids)
    assert len(results) == 2
    assert results[0]["id"] == "12345"
    assert results[1]["id"] == "67890"


@pytest.mark.asyncio
async def test_get_abstract_by_id_error(pubmed_client: PubMedClient):
    """Test error handling when retrieving a single abstract."""
    with pytest.raises(PubMedClientError) as exc_info:
        await pubmed_client.get_abstract_by_id("ERROR_ID")
    assert "Failed to retrieve abstract" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_abstracts_by_ids_error(pubmed_client: PubMedClient):
    """Test error handling when retrieving multiple abstracts."""
    with pytest.raises(PubMedClientError) as exc_info:
        await pubmed_client.get_abstracts_by_ids(["12345", "ERROR_ID", "67890"])
    assert "Failed to retrieve" in str(exc_info.value)
