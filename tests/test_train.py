"""Tests for train.py — uses synthetic data, no MLflow remote calls."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score

from phoenixml.config import get_settings
from phoenixml.train import build_pipeline


def _make_xy(n: int = 500, seed: int = 0) -> tuple:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 29))
    y = (rng.random(n) < 0.01).astype(int)
    return X, y


# ── build_pipeline ─────────────────────────────────────────────────────────────


def test_pipeline_has_scaler_and_clf():
    cfg = get_settings()
    pipe = build_pipeline(cfg)
    step_names = [name for name, _ in pipe.steps]
    assert "scaler" in step_names
    assert "clf" in step_names


def test_pipeline_fit_predict():
    cfg = get_settings()
    pipe = build_pipeline(cfg)
    X, y = _make_xy(300)
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)
    assert proba.shape == (300, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_pipeline_prauc_positive():
    """PR-AUC must be strictly > 0 on non-trivial synthetic data."""
    cfg = get_settings()
    pipe = build_pipeline(cfg)
    X_train, y_train = _make_xy(500, seed=1)
    X_eval, y_eval = _make_xy(200, seed=2)
    # ensure at least some positives in eval
    y_eval[:5] = 1

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_eval)[:, 1]
    prauc = average_precision_score(y_eval, proba)
    assert prauc > 0.0


def test_pipeline_is_sklearn_compatible():
    """Pipeline must expose fit/predict_proba and support get_params."""
    cfg = get_settings()
    pipe = build_pipeline(cfg)
    params = pipe.get_params()
    assert "clf__class_weight" in params
    assert params["clf__class_weight"] == "balanced"
