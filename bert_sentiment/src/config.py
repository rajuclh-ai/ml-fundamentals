"""Central configuration — reads from environment / .env file."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model
    model_dir: Path = BASE_DIR / "bert_sentiment_model"
    max_length: int = 512
    device: str = "cpu"

    # Inference
    batch_size: int = 16


settings = Settings()
