# config.py
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central application configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    OLLAMA_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.2:latest"
    CHROMA_PATH: Path = Path("chroma_db")
    MAX_RESULTS: int = Field(default=3, ge=1)
    CONFIDENCE_THRESHOLD: float = Field(
        default=1.0, gt=0.0, le=2.0
    )
    DEBUG: bool = False

    @field_validator("OLLAMA_URL")
    @classmethod
    def validate_ollama_url(cls, value: str) -> str:
        value = value.strip().rstrip("/")
        if not value.startswith(("http://", "https://")):
            raise ValueError("OLLAMA_URL must use HTTP or HTTPS")
        return value

    @field_validator("MODEL_NAME")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("MODEL_NAME cannot be empty")
        return value


@lru_cache
def get_settings() -> Settings:
    """Return one validated settings instance per process."""
    return Settings()