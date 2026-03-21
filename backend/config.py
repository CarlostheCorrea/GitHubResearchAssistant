from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI GitHub Research Assistant"
    openai_api_key: str | None = None
    openai_chat_model: str = "gpt-4.1-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    github_token: str | None = None

    request_timeout_seconds: int = 30
    max_file_bytes: int = 250_000
    max_total_repo_bytes: int = 6_000_000
    max_files_per_repo: int = 250
    embedding_batch_size: int = 32
    vector_query_k: int = 16
    answer_context_k: int = 6

    data_dir: Path = Field(default_factory=lambda: Path("data"))
    cache_dir: Path = Field(default_factory=lambda: Path("data") / "cache")
    chroma_dir: Path = Field(default_factory=lambda: Path("data") / "cache" / "chroma")
    manifest_dir: Path = Field(default_factory=lambda: Path("data") / "cache" / "manifests")

    def ensure_directories(self) -> None:
        for directory in (self.data_dir, self.cache_dir, self.chroma_dir, self.manifest_dir):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
