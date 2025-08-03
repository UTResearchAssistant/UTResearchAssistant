"""Research service for deep research queries.

This module coordinates multiple agents to perform complex queries.
Given a research question, it first calls a browsing agent to obtain
high‑level results and then uses a summariser agent to produce a
concise answer.  The implementation uses stub agents defined in
``agents/`` and can be expanded later to include citations and
context retrieval from the vector database.
"""

from typing import Dict, List

from agents.browsing_agent import BrowsingAgent
from agents.summarizer_agent import SummarizerAgent


_browsing_agent = BrowsingAgent()
_summariser_agent = SummarizerAgent()


def deep_research(query: str) -> Dict[str, List[str] | str]:
    """Perform a deep research query using the available agents.

    Parameters
    ----------
    query : str
        The research question.

    Returns
    -------
    dict
        A dictionary with an answer and sources (references).
    """
    # Use browsing agent to get initial search results
    results = _browsing_agent.run(query)
    # Combine the results into a single text to summarise
    combined = "\n".join(results)
    summary_data = _summariser_agent.run(combined)
    return {"answer": summary_data["summary"], "sources": summary_data["references"]}
