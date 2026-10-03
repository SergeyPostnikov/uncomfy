from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "sqlite+aiosqlite:///./uncomfy.db"
    redis_url: str | None = None
    comfy_path: Path = Path(r"C:\ComfyUI")

    tg_bot_token: str = ""
    tg_chat_id: str = ""


settings = Settings()
