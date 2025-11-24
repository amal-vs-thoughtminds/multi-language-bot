from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str
    response_model: str = "gpt-4o-mini"
    tts_model: str = "gpt-4o-mini-tts"
    tts_voice: str = "alloy"
    embedding_model: str = "text-embedding-3-small"
    fish_data_path: Path = Path(__file__).resolve().parent.parent / "fish_rate.json"
    supported_languages_path: Path = Path(__file__).resolve().parent.parent / "supported_languages.json"
    max_context_items: int = 4
    database_url: str = (
        "postgresql+psycopg://voicebot:voicebot@postgres:5432/voicebot"
    )
    conversation_recent_turns: int = 6
    conversation_semantic_turns: int = 3
    summary_trigger_turns: int = 8
    summary_max_turns: int = 20
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_region: str = "ap-south-1"
    s3_bucket_name: str = "blog-616"
    assemblyai_api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

