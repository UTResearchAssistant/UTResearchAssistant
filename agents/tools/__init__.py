"""Tools usable by agents.

Tools are standalone functions that implement specific capabilities (e.g.
performing a web search, fetching a URL or accessing a database).  They
are designed to be composed by agents to accomplish complex tasks.
"""

from .web_search import simple_web_search  # noqa: F401

__all__ = ["simple_web_search"]
