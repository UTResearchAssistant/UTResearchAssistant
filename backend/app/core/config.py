"""Application settings.

Configuration parameters are loaded from environment variables or
defaults using Pydantic's ``BaseSettings``.  Storing configuration in
a dedicated class makes it easy to manage secrets, API keys and other
environment‑specific parameters in a single place.
"""

from functools import lru_cache
from typing import Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Configuration for the application.

    Attributes
    ----------
    database_url : str
        URL for connecting to the relational database.
    openai_api_key : Optional[str]
        API key for OpenAI models, if available.
    vector_db_url : Optional[str]
        Connection URL for the vector database (e.g. Qdrant or Pinecone).
    """

    database_url: str = Field("sqlite:///./db.sqlite3", env="DATABASE_URL")
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    vector_db_url: Optional[str] = Field(default=None, env="VECTOR_DB_URL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Retrieve cached settings.

    Using a cache prevents re‑reading environment variables on every
    import.  FastAPI can use this in dependency injection via
    ``Depends(get_settings)``.
    """
    return Settings()
