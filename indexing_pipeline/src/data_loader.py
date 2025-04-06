import json
from pathlib import Path
from typing import Generator, List, Optional, Union

from src.models.pubmed import PubMedAbstract


class DataLoader:
    """Loads PubMed abstracts from corpus files or Hugging Face datasets."""

    def __init__(self, corpus_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.

        Args:
            corpus_path: Optional path to the local corpus.jsonl file.
                         If None, will use Hugging Face datasets.
        """
        self.corpus_path = Path(corpus_path) if corpus_path else None

    def load_abstracts_from_file(
        self, batch_size: Optional[int] = None, limit: Optional[int] = None
    ) -> Generator[List[PubMedAbstract], None, None]:
        """
        Load PubMed abstracts from a local JSONL file.

        Args:
            batch_size: Number of abstracts to yield in each batch.
                        If None, returns all abstracts in one batch.
            limit: Maximum number of abstracts to load. If None, loads all abstracts.

        Yields:
            Batches of PubMedAbstract objects.
        """
        if not self.corpus_path or not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus file not found at {self.corpus_path}")

        batch = []
        count = 0

        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if limit is not None and count >= limit:
                    break

                data = json.loads(line)
                abstract = PubMedAbstract(
                    id=data["id"],
                    title=data["title"],
                    text=data["text"],
                    url=data["url"],
                    publication_date=data["publication_date"],
                    journal=data["journal"],
                    authors=data["authors"],
                    keywords=data.get("keywords", []),
                    mesh_terms=data.get("mesh_terms", []),
                    doi=data.get("doi"),
                )

                batch.append(abstract)
                count += 1

                if batch_size and len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:  # Yield remaining items
                yield batch
