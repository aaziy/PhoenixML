"""Smoke tests for api.py — stub model, no MLflow remote calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

# ── Build a fake trained model ────────────────────────────────────────────────

N_FEATURES = 29  # V1-V28 + Amount


def _fake_model():
    """Return a mock sklearn pipeline that behaves like a real one."""
    model = MagicMock()
    model.predict_proba.side_effect = lambda X: np.column_stack(
        [1 - np.full(len(X), 0.1), np.full(len(X), 0.1)]
    )
    return model


# ── Patch _STATE before importing the app ────────────────────────────────────


@pytest.fixture()
def client():
    """TestClient with a pre-loaded fake model (lifespan patched — no MLflow call)."""
    from phoenixml import api as api_module

    fake = _fake_model()

    def _noop_load(cfg):
        api_module._STATE.update(
            {
                "model": fake,
                "version": "1",
                "run_id": "abc123",
                "registered_at": "2026-05-09T00:00:00+00:00",
                "alias": "Production",
            }
        )

    with patch.object(api_module, "_load_model", side_effect=_noop_load):
        with TestClient(api_module.app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture()
def client_no_model():
    """TestClient with no model loaded (lifespan patched to simulate startup failure)."""
    from phoenixml import api as api_module

    _empty = {"model": None, "version": None, "run_id": None, "registered_at": None, "alias": None}

    def _fail_load(cfg):
        # Explicitly clear state to simulate a failed startup
        api_module._STATE.update(_empty)

    with patch.object(api_module, "_load_model", side_effect=_fail_load):
        with TestClient(api_module.app, raise_server_exceptions=False) as c:
            yield c


# ── /health ───────────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_model_loaded_true(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1"


def test_health_model_not_loaded(client_no_model):
    resp = client_no_model.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["model_loaded"] is False
    assert data["model_version"] is None


# ── /predict ──────────────────────────────────────────────────────────────────


def _valid_row():
    return [float(i) for i in range(N_FEATURES)]


def test_predict_single_row(client):
    resp = client.post("/predict", json={"data": [_valid_row()]})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["probabilities"]) == 1
    assert len(data["labels"]) == 1
    assert data["n_rows"] == 1
    assert data["model_version"] == "1"


def test_predict_batch(client):
    rows = [_valid_row() for _ in range(5)]
    resp = client.post("/predict", json={"data": rows})
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_rows"] == 5
    assert len(data["probabilities"]) == 5
    assert len(data["labels"]) == 5


def test_predict_probabilities_in_range(client):
    resp = client.post("/predict", json={"data": [_valid_row(), _valid_row()]})
    for p in resp.json()["probabilities"]:
        assert 0.0 <= p <= 1.0


def test_predict_labels_are_binary(client):
    resp = client.post("/predict", json={"data": [_valid_row()]})
    for label in resp.json()["labels"]:
        assert label in (0, 1)


def test_predict_wrong_feature_count(client):
    resp = client.post("/predict", json={"data": [[1.0, 2.0, 3.0]]})  # only 3 features
    assert resp.status_code == 422


def test_predict_empty_data_rejected(client):
    resp = client.post("/predict", json={"data": []})
    assert resp.status_code == 422


def test_predict_no_model_returns_503(client_no_model):
    resp = client_no_model.post("/predict", json={"data": [_valid_row()]})
    assert resp.status_code == 503


# ── /model-info ───────────────────────────────────────────────────────────────


def test_model_info_returns_200(client):
    resp = client.get("/model-info")
    assert resp.status_code == 200


def test_model_info_fields(client):
    data = client.get("/model-info").json()
    assert data["model_name"] == "fraud-detector"
    assert data["version"] == "1"
    assert data["alias"] == "Production"
    assert data["run_id"] == "abc123"
    assert "tracking_uri" in data


def test_model_info_no_model_returns_503(client_no_model):
    resp = client_no_model.get("/model-info")
    assert resp.status_code == 503
