"""Database session management.

This module creates an SQLAlchemy engine and session factory based on
configuration defined in ``Settings``.  Using a session factory
abstracts away the database connection so that services can obtain
sessions via dependency injection.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import get_settings


def _create_engine():
    settings = get_settings()
    # For SQLite, we need to set ``check_same_thread`` to False to allow
    # multiple threads.  For other databases, the keyword arguments can be
    # adjusted as needed.
    if settings.database_url.startswith("sqlite"):
        return create_engine(
            settings.database_url, connect_args={"check_same_thread": False}
        )
    return create_engine(settings.database_url)


engine = _create_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
