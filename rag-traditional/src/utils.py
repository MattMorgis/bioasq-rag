from typing import List

from src.search.models import SearchResult


def clean_title(title: str) -> str:
    """
    Clean a title by removing square brackets if present.

    Args:
        title: The title string to clean

    Returns:
        The cleaned title string
    """
    if title is None:
        return None

    if not title:
        return title

    # Trim whitespace from the entire string first
    title = title.strip()

    # Remove trailing period if it exists after a closing bracket
    if title.endswith("]."):
        title = title[:-1]

    # Check if the title is completely enclosed in square brackets
    if title.startswith("[") and title.endswith("]"):
        # Extract content between brackets and trim whitespace
        return title[1:-1].strip()

    return title


def format_context_from_results(results: List[SearchResult]) -> str:
    """
    Format search results into a context string for the LLM.

    Args:
        results: List of SearchResult objects from vector search

    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant information found."

    context_parts = []

    for i, result in enumerate(results, 1):
        source_info = f"Title: {clean_title(result.title)}"
        if result.authors:
            source_info += f"\nAuthors: {', '.join(result.authors)}"
        if result.journal:
            source_info += f"\nJournal: {result.journal}"
        if result.publication_date:
            source_info += f"\nDate: {result.publication_date}"

        context_parts.append(
            f"SOURCE {i}:\n{source_info}\n\nABSTRACT:\n{result.text}\n"
        )

    return "\n\n".join(context_parts)
