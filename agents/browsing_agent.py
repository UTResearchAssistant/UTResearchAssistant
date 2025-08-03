"""Agent for web browsing and exploration.

This agent demonstrates how one could implement a multi‑step search
that queries external search engines, scrapes pages and extracts
useful information.  For now it merely echoes the query back.
"""

from typing import List

from .base_agent import BaseAgent
from .tools.web_search import simple_web_search


class BrowsingAgent(BaseAgent):
    """An agent that searches the web for a query and returns results."""

    def run(self, query: str, **kwargs: object) -> List[str]:  # type: ignore[override]
        # Delegate to the simple web search tool; in a real system this
        # would involve multiple steps and perhaps summarisation of each
        # result.  This stub simply returns search result titles.
        return simple_web_search(query)
