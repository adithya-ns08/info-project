"""Application configuration utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


@dataclass
class AppConfig:
    """Runtime configuration loaded from environment variables."""

    alpha_vantage_api_key: Optional[str]
    smtp_host: Optional[str]
    smtp_port: int
    smtp_user: Optional[str]
    smtp_password: Optional[str]
    email_from: Optional[str]
    email_to: Optional[str]
    slack_webhook_url: Optional[str]
    log_level: str


def load_config() -> AppConfig:
    """Load application settings from `.env` and environment variables."""

    load_dotenv()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    return AppConfig(
        alpha_vantage_api_key=os.getenv("ALPHA_VANTAGE_API_KEY"),
        smtp_host=os.getenv("SMTP_HOST"),
        smtp_port=int(os.getenv("SMTP_PORT", "587")),
        smtp_user=os.getenv("SMTP_USER"),
        smtp_password=os.getenv("SMTP_PASSWORD"),
        email_from=os.getenv("EMAIL_FROM"),
        email_to=os.getenv("EMAIL_TO"),
        slack_webhook_url=os.getenv("SLACK_WEBHOOK_URL"),
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )

