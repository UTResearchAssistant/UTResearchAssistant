"""SQLAlchemy ORM models for persistent storage.

These models define tables for papers and authors.  They are minimal
and intended for demonstration purposes.  In a production system you
would extend them with additional fields and relationships, or use
migration tools such as Alembic to manage schema changes.
"""

from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, relationship


Base = declarative_base()


class Paper(Base):
    """Represents a research paper in the database."""

    __tablename__ = "papers"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(256), nullable=False)
    abstract = Column(Text, nullable=True)
    author_id = Column(Integer, ForeignKey("authors.id"))

    author = relationship("Author", back_populates="papers")


class Author(Base):
    """Represents an author of research papers."""

    __tablename__ = "authors"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(128), nullable=False)

    papers = relationship("Paper", back_populates="author")
