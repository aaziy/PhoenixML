"""Monitoring loop: PR-AUC + drift detection, MLflow logging, alerting."""

from __future__ import annotations

import logging
import os
from typing import Any

import mlflow
import mlflow.sklearn
from sklearn.metrics import average_precision_score

from phoenixml.config import Settings, get_settings
from phoenixml.data import get_feature_columns, prepare_data, prod_batch_iterator
from phoenixml.drift import compute_drift
from phoenixml.notify import send_alert

logger = logging.getLogger(__name__)


# ── MLflow helpers ────────────────────────────────────────────────────────────


def _setup_mlflow(cfg: Settings) -> None:
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow_tracking_password
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.experiment_monitor)


def _load_production_model(cfg: Settings):
    """Load the model pinned to the 'Production' alias.

    Falls back to 'Staging' alias if Production is not yet set.
    Returns (model, version_str).
    """
    client = mlflow.tracking.MlflowClient()
    model_name = cfg.registered_model_name

    for alias in ("Production", "Staging"):
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            model_uri = f"models:/{model_name}@{alias}"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded model '%s' alias='%s' version=%s", model_name, alias, mv.version)
            return model, mv.version
        except Exception:
            logger.warning("Alias '%s' not found for model '%s', trying next.", alias, model_name)

    raise RuntimeError(
        f"No 'Production' or 'Staging' alias found for model '{model_name}'. "
        "Run `make train` first."
    )


# ── GitHub dispatch helper ────────────────────────────────────────────────────


def _fire_retrain_dispatch(cfg: Settings) -> bool:
    """Send a repository_dispatch event to trigger the Train workflow."""
    import json

    import requests

    pat = cfg.dispatch_pat
    repo = cfg.github_repo

    if not pat:
        logger.warning("DISPATCH_PAT not set — skipping repository_dispatch.")
        return False

    url = f"https://api.github.com/repos/{repo}/dispatches"
    headers = {
        "Authorization": f"Bearer {pat}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"event_type": "retrain", "client_payload": {"source": "monitor"}}

    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
    if resp.status_code == 204:
        logger.info("repository_dispatch 'retrain' fired successfully.")
        return True
    else:
        logger.error("repository_dispatch failed: %d %s", resp.status_code, resp.text)
        return False


# ── Core monitor loop ─────────────────────────────────────────────────────────


def run_monitor(
    batch_id: int = 0,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Score one production batch, detect drift, log to MLflow, alert if needed.

    Args:
        batch_id: Which production batch (0-indexed) to evaluate.

    Returns a decision dict with keys:
        batch_id, model_version, prauc, drift_detected, drift_share,
        trigger_retrain, mlflow_run_id, mlflow_run_url
    """
    cfg = settings or get_settings()
    _setup_mlflow(cfg)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data splits …")
    train_df, _, prod_pool_df = prepare_data(cfg)
    feature_cols = get_feature_columns(train_df, cfg)

    # Pick the requested batch
    batches = list(prod_batch_iterator(prod_pool_df, cfg))
    if batch_id >= len(batches):
        raise ValueError(f"batch_id={batch_id} out of range (max {len(batches) - 1})")
    _, batch_df = batches[batch_id]
    logger.info("Batch %d: %d rows", batch_id, len(batch_df))

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_version = _load_production_model(cfg)

    # ── Score batch ───────────────────────────────────────────────────────────
    X_batch = batch_df[feature_cols].values
    y_batch = batch_df[cfg.target_col].values

    if y_batch.sum() == 0:
        logger.warning(
            "Batch %d has no positive labels — PR-AUC undefined, setting to 0.", batch_id
        )
        prauc = 0.0
    else:
        y_proba = model.predict_proba(X_batch)[:, 1]
        prauc = float(average_precision_score(y_batch, y_proba))

    logger.info(
        "Batch %d PR-AUC: %.4f (threshold: %.2f)", batch_id, prauc, cfg.prauc_alert_threshold
    )

    # ── Drift detection ───────────────────────────────────────────────────────
    drift_result = compute_drift(train_df, batch_df, feature_cols, cfg)
    drift_detected: bool = drift_result["drift_detected"]
    drift_share: float = drift_result["drift_share"]

    # ── Decision ─────────────────────────────────────────────────────────────
    prauc_alert = prauc < cfg.prauc_alert_threshold
    trigger_retrain = prauc_alert or drift_detected

    logger.info(
        "Decision — prauc_alert=%s drift=%s trigger_retrain=%s",
        prauc_alert,
        drift_detected,
        trigger_retrain,
    )

    # ── Log to MLflow monitoring experiment ───────────────────────────────────
    with mlflow.start_run(run_name=f"monitor-batch-{batch_id}") as run:
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "batch_id": batch_id,
                "model_version": model_version,
                "model_name": cfg.registered_model_name,
                "batch_rows": len(batch_df),
            }
        )
        mlflow.log_metrics(
            {
                "prauc": prauc,
                "drift_share": drift_share,
                "n_drifted_features": drift_result["n_drifted"],
                "prauc_alert": int(prauc_alert),
                "drift_detected": int(drift_detected),
                "trigger_retrain": int(trigger_retrain),
            }
        )

    run_url = (
        f"{cfg.mlflow_tracking_uri}/#/experiments/"
        f"{mlflow.get_experiment_by_name(cfg.experiment_monitor).experiment_id}"
        f"/runs/{run_id}"
    )

    # ── Slack alert ───────────────────────────────────────────────────────────
    promote_url = f"https://github.com/{cfg.github_repo}/actions/workflows/promote.yml"
    if trigger_retrain:
        send_alert(
            batch_id=batch_id,
            model_version=str(model_version),
            prauc=prauc,
            drift_detected=drift_detected,
            drift_share=drift_share,
            run_url=run_url,
            trigger_retrain=trigger_retrain,
            promote_url=promote_url,
            settings=cfg,
        )
        _fire_retrain_dispatch(cfg)
    else:
        logger.info("No alert threshold breached — all good for batch %d.", batch_id)

    result = {
        "batch_id": batch_id,
        "model_version": model_version,
        "prauc": prauc,
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "trigger_retrain": trigger_retrain,
        "mlflow_run_id": run_id,
        "mlflow_run_url": run_url,
    }
    return result


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run PhoenixML monitoring for one batch.")
    parser.add_argument("--batch-id", type=int, default=0, help="Production batch index (0-based)")
    args = parser.parse_args()

    result = run_monitor(batch_id=args.batch_id)
    print("\n── Monitor result ──────────────────────────────────")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
