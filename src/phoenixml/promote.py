"""Staging → Production gated promotion with offline eval gate."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.metrics import average_precision_score

from phoenixml.config import Settings, get_settings
from phoenixml.data import get_feature_columns, prepare_data

logger = logging.getLogger(__name__)


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _setup_mlflow(cfg: Settings) -> None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow_tracking_password
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)


def _get_model_by_alias(
    client: mlflow.tracking.MlflowClient,
    model_name: str,
    alias: str,
):
    """Return (sklearn_model, version_str) for a given alias, or (None, None)."""
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
        model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")
        logger.info("Loaded '%s' alias='%s' version=%s", model_name, alias, mv.version)
        return model, mv.version
    except Exception as exc:
        logger.warning("Alias '%s' not found for '%s': %s", alias, model_name, exc)
        return None, None


def _eval_prauc(model, X, y) -> float:
    """Score model on (X, y) and return PR-AUC. Returns 0.0 if no positives."""
    if y.sum() == 0:
        logger.warning("No positive labels in eval set — PR-AUC set to 0.")
        return 0.0
    proba = model.predict_proba(X)[:, 1]
    return float(average_precision_score(y, proba))


# ── Core promote logic ────────────────────────────────────────────────────────


def run_promote(settings: Settings | None = None) -> dict[str, Any]:
    """Compare Staging vs Production on the held-out eval split.

    Promotes Staging → Production alias only if:
        challenger_prauc > champion_prauc + cfg.prauc_promote_delta

    If no Production model exists yet, promotes Staging automatically.

    Returns a result dict with:
        promoted        bool
        reason          str
        staging_version str
        staging_prauc   float
        production_version  str | None
        production_prauc    float | None
        delta           float
    """
    cfg = settings or get_settings()
    _setup_mlflow(cfg)

    client = mlflow.tracking.MlflowClient()
    model_name = cfg.registered_model_name

    # ── Load candidates ───────────────────────────────────────────────────────
    challenger, staging_ver = _get_model_by_alias(client, model_name, "Staging")
    if challenger is None:
        raise RuntimeError("No model with alias 'Staging' found. Run `make train` first.")

    champion, production_ver = _get_model_by_alias(client, model_name, "Production")

    # ── Load eval data ────────────────────────────────────────────────────────
    logger.info("Loading eval split …")
    train_df, eval_df, _ = prepare_data(cfg)
    feature_cols = get_feature_columns(train_df, cfg)
    X_eval = eval_df[feature_cols].values
    y_eval = eval_df[cfg.target_col].values

    # ── Score challenger ──────────────────────────────────────────────────────
    staging_prauc = _eval_prauc(challenger, X_eval, y_eval)
    logger.info("Staging  v%s  PR-AUC: %.4f", staging_ver, staging_prauc)

    # ── Score champion (if exists) ────────────────────────────────────────────
    if champion is not None:
        production_prauc: float | None = _eval_prauc(champion, X_eval, y_eval)
        logger.info("Production v%s  PR-AUC: %.4f", production_ver, production_prauc)
        delta = staging_prauc - production_prauc
        promote = delta > cfg.prauc_promote_delta
        if promote:
            reason = (
                f"Staging PR-AUC {staging_prauc:.4f} beats Production "
                f"{production_prauc:.4f} by Δ{delta:.4f} > ε{cfg.prauc_promote_delta}"
            )
        else:
            reason = (
                f"Staging PR-AUC {staging_prauc:.4f} does NOT beat Production "
                f"{production_prauc:.4f} by required Δ{cfg.prauc_promote_delta} "
                f"(actual Δ{delta:.4f})"
            )
    else:
        # First ever production deploy — promote automatically
        production_prauc = None
        delta = staging_prauc
        promote = True
        reason = f"No Production model exists yet — auto-promoting Staging v{staging_ver}"
        logger.info(reason)

    # ── Execute promotion ─────────────────────────────────────────────────────
    if promote:
        client.set_registered_model_alias(
            name=model_name,
            alias="Production",
            version=staging_ver,
        )
        logger.info(
            "✅  Version %s of '%s' promoted → alias 'Production'",
            staging_ver,
            model_name,
        )
        _send_promotion_slack(
            cfg=cfg,
            staging_ver=staging_ver,
            staging_prauc=staging_prauc,
            production_ver=production_ver,
            production_prauc=production_prauc,
            delta=delta,
        )
    else:
        logger.info("❌  Promotion rejected: %s", reason)

    result: dict[str, Any] = {
        "promoted": promote,
        "reason": reason,
        "staging_version": staging_ver,
        "staging_prauc": staging_prauc,
        "production_version": production_ver,
        "production_prauc": production_prauc,
        "delta": delta,
    }
    return result


# ── Slack notification for promotion ─────────────────────────────────────────


def _send_promotion_slack(
    *,
    cfg: Settings,
    staging_ver: str,
    staging_prauc: float,
    production_ver: str | None,
    production_prauc: float | None,
    delta: float,
) -> None:
    from phoenixml.notify import send_slack_message

    champion_line = (
        f"Previous Production v{production_ver} PR-AUC: `{production_prauc:.4f}`\n"
        if production_ver and production_prauc is not None
        else "First Production deploy — no previous champion.\n"
    )
    text = (
        f"✅ *PhoenixML — Model Promoted to Production*\n"
        f"New Production: `fraud-detector v{staging_ver}`  PR-AUC: `{staging_prauc:.4f}`\n"
        f"{champion_line}"
        f"Δ PR-AUC: `{delta:+.4f}`"
    )
    send_slack_message(text, settings=cfg)


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    result = run_promote()
    print("\n── Promote result ──────────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")
    if not result["promoted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
