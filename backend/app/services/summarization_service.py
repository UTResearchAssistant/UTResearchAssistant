"""Summarisation service.

This module provides functions for summarising text.  In a real
implementation, this would invoke an LLM via the ``agents`` layer to
generate a concise summary and collect citations.  For now we return
static dummy data to illustrate the interface.
"""

from typing import Dict, List


def summarise_text(text: str) -> Dict[str, List[str] | str]:
    """Generate a summary and references for a given text.

    Parameters
    ----------
    text : str
        The input text to summarise.

    Returns
    -------
    dict
        A dictionary containing a summary and a list of reference IDs.
    """
    # TODO: integrate with summariser agents and citation extraction
    summary = f"Summary of input: {text[:50]}..."
    references: List[str] = []
    return {"summary": summary, "references": references}
