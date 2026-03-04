"""Email and Slack alert integrations."""

from __future__ import annotations

import json
import logging
import smtplib
from dataclasses import dataclass
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional

import requests

LOGGER = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    """Configuration for outbound alert channels."""

    smtp_host: Optional[str]
    smtp_port: int
    smtp_user: Optional[str]
    smtp_password: Optional[str]
    email_from: Optional[str]
    email_to: Optional[str]
    slack_webhook_url: Optional[str]


def signal_label(signal: int) -> str:
    """Convert integer signal to text label."""

    if signal == 1:
        return "BUY"
    if signal == -1:
        return "SELL"
    return "HOLD"


def build_alert_message(ticker: str, signal: int, price: float) -> str:
    """Build a formatted signal alert message."""

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return (
        f"Ticker: {ticker}\n"
        f"Signal: {signal_label(signal)} ({signal})\n"
        f"Current Price: ${price:.2f}\n"
        f"Timestamp: {timestamp}"
    )


def send_email_alert(config: AlertConfig, subject: str, body: str) -> bool:
    """Send alert email via SMTP using TLS."""

    required = [
        config.smtp_host,
        config.smtp_user,
        config.smtp_password,
        config.email_from,
        config.email_to,
    ]
    if not all(required):
        LOGGER.warning("Email alert skipped: SMTP/email credentials are incomplete.")
        return False

    message = MIMEMultipart()
    message["From"] = config.email_from
    message["To"] = config.email_to
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(config.smtp_user, config.smtp_password)
            server.sendmail(config.email_from, config.email_to, message.as_string())
        LOGGER.info("Email alert sent to %s", config.email_to)
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Email alert failed: %s", exc)
        return False


def send_slack_alert(config: AlertConfig, message: str) -> bool:
    """Send alert message to Slack incoming webhook."""

    if not config.slack_webhook_url:
        LOGGER.warning("Slack alert skipped: SLACK_WEBHOOK_URL is not configured.")
        return False

    payload = {"text": message}
    try:
        response = requests.post(
            config.slack_webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        LOGGER.info("Slack alert sent.")
        return True
    except Exception as exc:  # noqa: BLE001
        LOGGER.exception("Slack alert failed: %s", exc)
        return False

