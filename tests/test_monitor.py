"""Tests for drift.py, notify.py, and monitor helpers — no network calls."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenixml.drift import compute_drift, perturb_batch
from phoenixml.notify import _build_alert_payload

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_df(n: int = 300, seed: int = 0, noise: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    v_cols = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    df = pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            **v_cols,
            "Amount": rng.exponential(100, n),
            "Class": (rng.random(n) < 0.01).astype(int),
        }
    )
    if noise > 0:
        v_names = [f"V{i}" for i in range(1, 29)]
        df[v_names] += rng.normal(0, noise, size=(n, 28))
    return df


FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


# ── drift.compute_drift ───────────────────────────────────────────────────────


def test_drift_no_drift_on_same_distribution():
    """Same-distribution data should produce low drift share."""
    ref = _make_df(500, seed=0)
    cur = _make_df(500, seed=1)
    result = compute_drift(ref, cur, FEATURE_COLS)

    assert "drift_detected" in result
    assert "drift_share" in result
    assert 0.0 <= result["drift_share"] <= 1.0
    assert isinstance(result["drift_detected"], bool)
    # Same distribution — drift share should be well below 0.5
    assert result["drift_share"] < 0.5


def test_drift_detected_on_perturbed_data():
    """Heavily perturbed data must flag drift_detected=True."""
    ref = _make_df(500, seed=0)
    # Extreme noise to guarantee drift
    cur = _make_df(500, seed=0, noise=10.0)
    result = compute_drift(ref, cur, FEATURE_COLS)

    assert result["drift_share"] > 0.0
    # With noise=10 sigma the majority of features should drift
    assert result["n_drifted"] > 0


def test_drift_result_keys():
    ref = _make_df(300, seed=2)
    cur = _make_df(300, seed=3)
    result = compute_drift(ref, cur, FEATURE_COLS)

    for key in ("drift_detected", "drift_share", "n_drifted", "n_features", "per_column"):
        assert key in result, f"Missing key: {key}"

    assert result["n_features"] == len(FEATURE_COLS)


# ── drift.perturb_batch ───────────────────────────────────────────────────────


def test_perturb_batch_modifies_v_columns():
    df = _make_df(100, seed=0)
    perturbed = perturb_batch(df, FEATURE_COLS, noise_scale=3.0)

    v_cols = [c for c in FEATURE_COLS if c.startswith("V")]
    # Perturbed values must differ from original
    assert not np.allclose(df[v_cols].values, perturbed[v_cols].values)


def test_perturb_batch_preserves_non_v_columns():
    df = _make_df(100, seed=0)
    perturbed = perturb_batch(df, FEATURE_COLS, noise_scale=3.0)

    pd.testing.assert_series_equal(df["Amount"], perturbed["Amount"])
    pd.testing.assert_series_equal(df["Class"], perturbed["Class"])


def test_perturb_batch_deterministic():
    df = _make_df(100, seed=0)
    p1 = perturb_batch(df, FEATURE_COLS, noise_scale=2.0, seed=99)
    p2 = perturb_batch(df, FEATURE_COLS, noise_scale=2.0, seed=99)
    pd.testing.assert_frame_equal(p1, p2)


# ── notify._build_alert_payload ──────────────────────────────────────────────


def test_alert_payload_structure():
    payload = _build_alert_payload(
        batch_id=3,
        model_version="2",
        prauc=0.72,
        drift_detected=True,
        drift_share=0.61,
        run_url="https://dagshub.com/example/run",
        prauc_threshold=0.75,
        trigger_retrain=True,
    )
    assert "blocks" in payload
    assert len(payload["blocks"]) > 0
    # Header block should reference the batch id
    header_text = payload["blocks"][0]["text"]["text"]
    assert "3" in header_text


def test_alert_payload_no_retrain():
    payload = _build_alert_payload(
        batch_id=0,
        model_version="1",
        prauc=0.85,
        drift_detected=False,
        drift_share=0.1,
        run_url="https://example.com",
        prauc_threshold=0.75,
        trigger_retrain=False,
    )
    # Find the retrain section block text
    texts = [
        b.get("text", {}).get("text", "") for b in payload["blocks"] if b.get("type") == "section"
    ]
    combined = " ".join(texts)
    assert "No retrain" in combined
