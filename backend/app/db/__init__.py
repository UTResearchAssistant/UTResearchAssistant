"""Database layer for the research assistant.

This package contains SQLAlchemy models and session management code.
Abstracting the database logic behind a common interface makes it
possible to swap out the underlying database technology with minimal
changes elsewhere in the codebase.
"""

from .session import SessionLocal
from .models import Base, Paper, Author  # noqa: F401

__all__ = ["SessionLocal", "Base", "Paper", "Author"]
