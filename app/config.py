from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    telegram_bot_token: str = Field(default="")
    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/one_c_bot"
    google_spreadsheet_id: str = ""
    google_sheet_name: str = "Заявки"
    google_credentials_file: Path = Path("secrets/voronka-510810-a2733be4f3ce.json")
    image_storage_path: Path = Path("data/images")
    roi_config_path: Path = Path("config/roi.yaml")
    ocr_language: str = "ru"
    low_confidence_threshold: float = Field(default=0.8, ge=0, le=1)
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
