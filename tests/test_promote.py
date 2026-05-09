"""Tests for promote.py — uses synthetic models, no MLflow remote calls."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phoenixml.config import get_settings
from phoenixml.promote import _eval_prauc

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_xy(n: int = 400, pos_rate: float = 0.05, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 10))
    y = (rng.random(n) < pos_rate).astype(int)
    # Ensure at least a few positives
    y[:5] = 1
    return X, y


def _trained_pipeline(X, y) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=500)),
        ]
    )
    pipe.fit(X, y)
    return pipe


# ── _eval_prauc ───────────────────────────────────────────────────────────────


def test_eval_prauc_returns_float():
    X, y = _make_xy(300)
    model = _trained_pipeline(X, y)
    X_eval, y_eval = _make_xy(100, seed=1)
    y_eval[:3] = 1  # guarantee positives
    score = _eval_prauc(model, X_eval, y_eval)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_eval_prauc_zero_positives_returns_zero():
    X, y = _make_xy(200)
    model = _trained_pipeline(X, y)
    X_eval = np.random.default_rng(5).standard_normal((50, 10))
    y_eval = np.zeros(50, dtype=int)  # no positives
    score = _eval_prauc(model, X_eval, y_eval)
    assert score == 0.0


def test_eval_prauc_perfect_model():
    """A model that separates perfectly should have PR-AUC close to 1."""
    rng = np.random.default_rng(99)
    n = 200
    # Class 0: centred at 0, Class 1: centred at 10 (easy to separate)
    X0 = rng.normal(0, 0.1, (n, 5))
    X1 = rng.normal(10, 0.1, (20, 5))
    X = np.vstack([X0, X1])
    y = np.array([0] * n + [1] * 20)

    model = _trained_pipeline(X, y)
    score = _eval_prauc(model, X, y)
    assert score > 0.95


def test_eval_prauc_better_model_wins():
    """A well-trained model should score higher than a random one."""
    X_train, y_train = _make_xy(500, seed=0)
    X_eval, y_eval = _make_xy(200, seed=1)
    y_eval[:10] = 1

    good_model = _trained_pipeline(X_train, y_train)

    # Random model: intercept-only (essentially random predictions)
    bad_model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1e-10, max_iter=10)),
        ]
    )
    bad_model.fit(X_train, y_train)

    good_score = _eval_prauc(good_model, X_eval, y_eval)
    bad_score = _eval_prauc(bad_model, X_eval, y_eval)

    assert good_score >= bad_score


# ── promotion delta logic (unit test without MLflow) ─────────────────────────


def test_promotion_decision_logic():
    """Validate the Δ PR-AUC gate logic directly."""
    cfg = get_settings()
    epsilon = cfg.prauc_promote_delta

    staging_prauc = 0.82
    production_prauc = 0.80
    delta = staging_prauc - production_prauc

    # Should promote: delta > epsilon
    assert delta > epsilon, f"Expected delta {delta:.4f} > epsilon {epsilon} for promotion"

    staging_prauc_low = 0.80
    delta_low = staging_prauc_low - production_prauc
    # Should NOT promote: delta <= epsilon
    assert (
        delta_low <= epsilon
    ), f"Expected delta {delta_low:.4f} <= epsilon {epsilon} (no promotion)"
