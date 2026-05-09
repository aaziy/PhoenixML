"""FastAPI inference service — /health, /predict, /model-info."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from phoenixml.config import Settings, get_settings

logger = logging.getLogger(__name__)

# ── Global model state (populated at startup) ─────────────────────────────────

_STATE: dict[str, Any] = {
    "model": None,
    "version": None,
    "run_id": None,
    "registered_at": None,
    "alias": None,
}


# ── Pydantic schemas ──────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Batch prediction input.

    Each row must contain the 29 features: V1–V28 + Amount (in that order).
    """

    data: list[list[float]] = Field(
        ...,
        description="List of feature vectors. Each vector: [V1, V2, …, V28, Amount]",
        min_length=1,
    )


class PredictResponse(BaseModel):
    probabilities: list[float] = Field(..., description="Fraud probability per row")
    labels: list[int] = Field(..., description="Binary label (1=fraud) using predict_cutoff")
    model_version: str
    threshold: float
    n_rows: int


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    alias: str
    run_id: str | None
    registered_at: str | None
    tracking_uri: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str | None


# ── Model loading ─────────────────────────────────────────────────────────────


def _load_model(cfg: Settings) -> None:
    """Load the Production (or Staging) model into _STATE at startup."""
    os.environ["MLFLOW_TRACKING_USERNAME"] = cfg.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = cfg.mlflow_tracking_password
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    model_name = cfg.registered_model_name

    for alias in ("Production", "Staging"):
        try:
            mv = client.get_model_version_by_alias(model_name, alias)
            model = mlflow.sklearn.load_model(f"models:/{model_name}@{alias}")

            _STATE["model"] = model
            _STATE["version"] = mv.version
            _STATE["run_id"] = mv.run_id
            _STATE["alias"] = alias
            _STATE["registered_at"] = (
                datetime.fromtimestamp(mv.creation_timestamp / 1000, tz=timezone.utc).isoformat()
                if mv.creation_timestamp
                else None
            )
            logger.info(
                "Loaded model '%s' alias='%s' version=%s run_id=%s",
                model_name,
                alias,
                mv.version,
                mv.run_id,
            )
            return
        except Exception as exc:
            logger.warning("Alias '%s' unavailable: %s", alias, exc)

    logger.error(
        "No Production or Staging model found for '%s' — /predict will return 503.",
        model_name,
    )


# ── FastAPI lifespan ──────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_settings()
    logger.info("Loading model from MLflow …")
    try:
        _load_model(cfg)
    except Exception as exc:
        logger.error("Model load failed at startup: %s", exc)
    yield
    # teardown (nothing needed)
    _STATE["model"] = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="PhoenixML Fraud Detector",
    description="Serves the Production fraud-detection model from MLflow.",
    version="0.1.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness check — always returns 200."""
    return HealthResponse(
        status="ok",
        model_loaded=_STATE["model"] is not None,
        model_version=_STATE["version"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Score a batch of transactions.

    Returns fraud probability and binary label for each row.
    Feature order: V1, V2, …, V28, Amount (29 features total).
    """
    if _STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health and MLflow connectivity.",
        )

    cfg = get_settings()
    expected_features = 29  # V1-V28 + Amount

    X = np.array(request.data, dtype=float)
    if X.ndim != 2 or X.shape[1] != expected_features:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each row must have exactly {expected_features} features "
                f"(V1–V28 + Amount). Got shape {X.shape}."
            ),
        )

    probabilities: list[float] = _STATE["model"].predict_proba(X)[:, 1].tolist()
    labels: list[int] = [1 if p >= cfg.predict_cutoff else 0 for p in probabilities]

    return PredictResponse(
        probabilities=probabilities,
        labels=labels,
        model_version=str(_STATE["version"]),
        threshold=cfg.predict_cutoff,
        n_rows=len(X),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["ops"])
def model_info() -> ModelInfoResponse:
    """Return metadata about the currently loaded model."""
    if _STATE["version"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check /health and MLflow connectivity.",
        )

    cfg = get_settings()
    return ModelInfoResponse(
        model_name=cfg.registered_model_name,
        version=str(_STATE["version"]),
        alias=str(_STATE["alias"]),
        run_id=_STATE["run_id"],
        registered_at=_STATE["registered_at"],
        tracking_uri=cfg.mlflow_tracking_uri,
    )
