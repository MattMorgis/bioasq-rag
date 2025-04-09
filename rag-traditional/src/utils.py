from typing import List

from src.search.models import SearchResult


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
        source_info = f"Title: {result.title}"
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
