"""FastAPI inference service (Phase 5 — stub)."""

from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="PhoenixML Fraud Detector", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


# Full /predict and /model-info endpoints implemented in Phase 5.
