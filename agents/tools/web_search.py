"""Simple web search tool.

This module defines a minimal search function to demonstrate how
external capabilities can be encapsulated as tools.  In practice, this
could call a real search API (e.g. Bing or Google) and parse the
results.  Here it returns placeholder titles.
"""

from typing import List


def simple_web_search(query: str) -> List[str]:
    """Perform a trivial search and return dummy results.

    Parameters
    ----------
    query : str
        The search term.

    Returns
    -------
    list[str]
        A list of strings representing search result titles.
    """
    # TODO: integrate with a real search API when available
    return [f"Result {i} for '{query}'" for i in range(1, 4)]
