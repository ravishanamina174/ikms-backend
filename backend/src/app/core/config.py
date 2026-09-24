"""Configuration management for the multi-agent RAG system.

This module loads environment variables for the Gemini LLM, Pinecone settings,
and other runtime options.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Gemini Configuration
    gemini_api_key: str
    gemini_model_name: str = "gemini-3.6-flash"
    gemini_embedding_model_name: str = "gemini-embedding-001"
    gemini_embedding_dimension: int = 3072

    # Pinecone Configuration
    pinecone_api_key: str
    pinecone_index_name: str

    # Retrieval Configuration
    retrieval_k: int = 5

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


# Create a singleton settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get the application settings instance (singleton pattern).

    Returns:
        Settings instance with all configuration values loaded.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
