"""Enhanced configuration for the Research Assistant with advanced features."""

from pydantic_settings import BaseSettings
from typing import Optional, List
import os


class EnhancedSettings(BaseSettings):
    """Enhanced application settings with additional API keys and configurations."""

    # Database
    database_url: str = "sqlite:///./research_agent.db"

    # Core API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # Academic Database APIs
    semantic_scholar_api_key: Optional[str] = None
    pubmed_api_key: Optional[str] = None
    ieee_api_key: Optional[str] = None
    springer_api_key: Optional[str] = None

    # Audio/Video Processing
    elevenlabs_api_key: Optional[str] = None
    whisper_api_key: Optional[str] = None

    # Translation Services
    google_translate_api_key: Optional[str] = None
    deepl_api_key: Optional[str] = None

    # Email/Notification Services
    sendgrid_api_key: Optional[str] = None
    smtp_server: Optional[str] = None
    smtp_port: int = 587
    smtp_username: Optional[str] = None
    smtp_password: Optional[str] = None

    # Vector Database
    vector_db_url: str = "http://localhost:6333"
    qdrant_api_key: Optional[str] = None

    # Redis for caching
    redis_url: str = "redis://localhost:6379"

    # File Storage
    storage_path: str = "./storage"
    max_file_size: int = 100 * 1024 * 1024  # 100MB

    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # Features Flags
    enable_podcasts: bool = True
    enable_video_analysis: bool = True
    enable_multilingual: bool = True
    enable_citation_networks: bool = True
    enable_ai_detection: bool = True
    enable_plagiarism_check: bool = True

    # Supported Languages
    supported_languages: List[str] = [
        "en",
        "es",
        "fr",
        "de",
        "it",
        "pt",
        "nl",
        "ru",
        "zh",
        "ja",
        "ko",
        "ar",
    ]

    # Audio Settings
    default_voice: str = "alloy"  # OpenAI TTS voice
    audio_quality: str = "high"
    max_audio_length: int = 3600  # 1 hour in seconds

    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
enhanced_settings = EnhancedSettings()
