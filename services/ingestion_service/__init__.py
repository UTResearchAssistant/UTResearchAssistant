"""Ingestion service package.

This package contains scripts responsible for crawling external
repositories of research papers, parsing and processing documents and
storing their metadata and embeddings.  Running these scripts
regularly keeps the database and vector store up to date without
blocking the main API server.
"""

__all__: list[str] = []
