"""Application settings loaded from environment variables."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: str | None = None
    llm_model: str = "llama-3.3-70b-versatile"
    llm_provider: str = "groq"
    llm_timeout: float = 30.0

    top_k: int = 5
    rerank: bool = True
    rerank_model: str = "ms-marco-TinyBERT-L-2-v2"

    data_dir: Path = ROOT / "data"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    chroma_dir: Path = ROOT / "data" / "chroma"
    web_dir: Path = ROOT / "web"


@lru_cache
def get_settings() -> Settings:
    return Settings()
