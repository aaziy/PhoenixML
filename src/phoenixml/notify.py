"""Slack notification helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import requests

from phoenixml.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_alert_payload(
    *,
    batch_id: int,
    model_version: str,
    prauc: float,
    drift_detected: bool,
    drift_share: float,
    run_url: str,
    prauc_threshold: float,
    trigger_retrain: bool,
    promote_url: str = "",
) -> dict:
    """Build a rich Slack Block Kit payload for a monitoring alert."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    status_icon = "🔴" if trigger_retrain else "🟡"
    drift_icon = "⚠️" if drift_detected else "✅"
    prauc_icon = "⚠️" if prauc < prauc_threshold else "✅"

    header = f"{status_icon} PhoenixML Monitor — Batch {batch_id}"
    if trigger_retrain:
        retrain_line = "*Action:* 🔄 Retrain triggered → challenger will be registered to Staging"
        if promote_url:
            retrain_line += f"\nOnce retrain completes, <{promote_url}|approve Promotion here>."
    else:
        retrain_line = "*Action:* No retrain needed"

    return {
        "blocks": [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": header, "emoji": True},
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Model version:*\n`fraud-detector v{model_version}`",
                    },
                    {"type": "mrkdwn", "text": f"*Timestamp:*\n{ts}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*PR-AUC:* {prauc_icon}\n`{prauc:.4f}` (threshold: {prauc_threshold})",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Drift:* {drift_icon}\n`{drift_share:.1%}` of features drifted",
                    },
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": retrain_line},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "View MLflow Run"},
                        "url": run_url,
                        "style": "primary",
                    },
                    *(
                        [
                            {
                                "type": "button",
                                "text": {"type": "plain_text", "text": "Approve Promotion"},
                                "url": promote_url,
                                "style": "danger",
                            }
                        ]
                        if promote_url
                        else []
                    ),
                ],
            },
        ]
    }


def send_alert(
    *,
    batch_id: int,
    model_version: str,
    prauc: float,
    drift_detected: bool,
    drift_share: float,
    run_url: str,
    trigger_retrain: bool,
    promote_url: str = "",
    settings: Settings | None = None,
) -> bool:
    """Send a monitoring alert to Slack. Returns True on success."""
    cfg = settings or get_settings()

    if not cfg.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured — skipping Slack notification.")
        return False

    payload = _build_alert_payload(
        batch_id=batch_id,
        model_version=model_version,
        prauc=prauc,
        drift_detected=drift_detected,
        drift_share=drift_share,
        run_url=run_url,
        prauc_threshold=cfg.prauc_alert_threshold,
        trigger_retrain=trigger_retrain,
        promote_url=promote_url,
    )

    try:
        resp = requests.post(
            cfg.slack_webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        if resp.status_code != 200:
            logger.error("Slack returned %d: %s", resp.status_code, resp.text)
            return False
        logger.info("Slack alert sent for batch %d.", batch_id)
        return True
    except requests.RequestException as exc:
        logger.error("Slack request failed: %s", exc)
        return False


def send_slack_message(text: str, settings: Settings | None = None) -> bool:
    """Send a plain-text Slack message (convenience wrapper)."""
    cfg = settings or get_settings()
    if not cfg.slack_webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured — skipping.")
        return False

    payload = {"text": text}
    try:
        resp = requests.post(
            cfg.slack_webhook_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return resp.status_code == 200
    except requests.RequestException as exc:
        logger.error("Slack request failed: %s", exc)
        return False
