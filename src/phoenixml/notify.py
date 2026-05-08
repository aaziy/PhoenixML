"""Slack notification helpers (Phase 3 — stub for now)."""

from __future__ import annotations

import json
import logging

import requests

from phoenixml.config import Settings, get_settings

logger = logging.getLogger(__name__)


def send_slack_message(text: str, settings: Settings | None = None) -> bool:
    """Send a plain-text Slack message. Returns True on success."""
    cfg = settings or get_settings()
    if not cfg.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not set — skipping notification.")
        return False

    payload = {"text": text}
    resp = requests.post(
        cfg.slack_webhook_url,
        data=json.dumps(payload),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    if resp.status_code != 200:
        logger.error("Slack webhook returned %d: %s", resp.status_code, resp.text)
        return False

    logger.info("Slack notification sent.")
    return True
