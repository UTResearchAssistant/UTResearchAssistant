"""Version 1 API package.

This package groups API routers corresponding to version 1 of the public
interface.  Additional versions can be introduced later by adding new
packages (e.g. ``v2``) without breaking existing clients.
"""

from . import summarization, search, health, dataset, training, research  # noqa: F401

__all__ = [
    "summarization",
    "search",
    "health",
    "dataset",
    "training",
    "research",
]
