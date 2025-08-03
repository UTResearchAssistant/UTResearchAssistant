"""Search service.

This module provides simple search capabilities.  In a production
system, this would query a vector database or an external API to
retrieve documents relevant to the user's query.  Here we return a
hard‑coded list for demonstration purposes.
"""

from typing import List


def search_documents(query: str) -> List[str]:
    """Search the knowledge base for documents matching a query.

    Parameters
    ----------
    query : str
        The search term.

    Returns
    -------
    list[str]
        A list of document identifiers or titles.
    """
    # TODO: integrate with vector search service or external APIs
    return [f"Document {i}: result for '{query}'" for i in range(1, 4)]
