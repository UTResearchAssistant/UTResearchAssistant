"""Embedding model package.

This package provides functions to convert text into vector
representations.  Embeddings are used for semantic search and
retrieving relevant context from the document store.  Actual model
loading and inference lives in ``embedder.py``.
"""

from .embedder import embed_text  # noqa: F401

__all__ = ["embed_text"]
