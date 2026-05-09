"""Evidently-based data drift detection (Evidently 0.7+ API)."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from phoenixml.config import Settings, get_settings

logger = logging.getLogger(__name__)


def compute_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Run Evidently DataDriftPreset on current batch vs. training reference.

    Returns a dict with:
        drift_detected  bool   – True if share of drifted columns > threshold
        drift_share     float  – fraction of feature columns that drifted
        n_drifted       int    – absolute count of drifted columns
        n_features      int    – total features checked
        per_column      dict   – {col: {"drifted": bool, "p_value": float}}
    """
    cfg = settings or get_settings()

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as e:
        raise ImportError(
            "evidently is required for drift detection. Run: pip install evidently"
        ) from e

    ref = reference_df[feature_cols].reset_index(drop=True)
    cur = current_df[feature_cols].reset_index(drop=True)

    report = Report(metrics=[DataDriftPreset(drift_share=cfg.drift_share_threshold)])
    snapshot = report.run(reference_data=ref, current_data=cur)
    result = snapshot.dict()

    # Parse the structured metrics from Evidently 0.7
    # Metric 0: DriftedColumnsCount → {count, share}
    # Metric 1+: ValueDrift per column → p_value float
    n_drifted = 0
    drift_share = 0.0
    n_features = len(feature_cols)
    per_column: dict[str, dict] = {}

    for metric in result.get("metrics", []):
        name: str = metric["metric_name"]
        value = metric["value"]

        if name.startswith("DriftedColumnsCount"):
            if isinstance(value, dict):
                n_drifted = int(value.get("count", 0))
                drift_share = float(value.get("share", 0.0))

        elif name.startswith("ValueDrift"):
            # Extract column name from e.g. "ValueDrift(column=V1,method=K-S p_value,...)"
            col = _parse_column_from_metric_name(name)
            if col:
                p_value = float(value) if isinstance(value, (int, float)) else 0.0
                # Evidently flags drift when p_value < threshold (default 0.05)
                per_column[col] = {
                    "drifted": p_value < 0.05,
                    "p_value": p_value,
                }

    drift_detected = drift_share > cfg.drift_share_threshold

    logger.info(
        "Drift report — drifted: %s  share: %.3f  (%d/%d features)",
        drift_detected,
        drift_share,
        n_drifted,
        n_features,
    )

    return {
        "drift_detected": drift_detected,
        "drift_share": drift_share,
        "n_drifted": n_drifted,
        "n_features": n_features,
        "per_column": per_column,
    }


def _parse_column_from_metric_name(name: str) -> str | None:
    """Extract column name from 'ValueDrift(column=V1,method=...)'."""
    try:
        inner = name[name.index("(") + 1 : name.index(")")]
        for part in inner.split(","):
            if part.strip().startswith("column="):
                return part.strip()[len("column=") :]
    except (ValueError, IndexError):
        pass
    return None


def perturb_batch(
    batch_df: pd.DataFrame,
    feature_cols: list[str],
    noise_scale: float = 2.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Add Gaussian noise to V* features to simulate distribution shift.

    Used by simulate_drift.py to create batches that trigger the drift flag.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    df = batch_df.copy()
    v_cols = [c for c in feature_cols if c.startswith("V")]
    noise = rng.normal(0, noise_scale, size=(len(df), len(v_cols)))
    df[v_cols] = df[v_cols].values + noise
    return df
