# PhoenixML

> **Fully automated MLOps pipeline for credit-card fraud detection.**
>
> When PR-AUC on a live batch drops below threshold *or* Evidently reports input drift, the system fires a Slack alert, triggers an automated retrain via `repository_dispatch`, registers the challenger to MLflow `Staging`, and waits for a human approval gate before promoting to `Production`. A FastAPI container continuously serves the current Production model.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions (free tier)                       │
│                                                                         │
│  ┌──────────┐   push/dispatch   ┌────────────────────────────────────┐  │
│  │ train.yml│ ◄──────────────── │ monitor.yml  (cron every 6h)       │  │
│  │          │                   │                                    │  │
│  │ 1. Download Kaggle data      │ 1. Load Production model           │  │
│  │ 2. Train LogReg pipeline     │ 2. Score next production batch     │  │
│  │ 3. Log metrics + artifacts   │ 3. Compute PR-AUC + Evidently drift│  │
│  │ 4. Register → Staging        │ 4. Log to MLflow monitoring exp    │  │
│  └──────────┘                   │ 5. If breach → Slack + dispatch ──►│  │
│       │ MLflow alias            └────────────────────────────────────┘  │
│       │                                                                 │
│  ┌────▼──────────────────────────────────────────────────────────────┐  │
│  │ promote.yml  (workflow_dispatch + GitHub Environment gate)        │  │
│  │                                                                   │  │
│  │  Load Staging + Production → eval on held-out split               │  │
│  │  If Δ PR-AUC > ε  →  set alias "Production" on Staging version   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ serve.yml  (push to main → GHCR)                                  │  │
│  │  docker buildx → linux/amd64 + linux/arm64 → ghcr.io/aaziy/      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
         │                                  │
         ▼                                  ▼
 ┌───────────────┐                 ┌─────────────────┐
 │ DagsHub MLflow│                 │  Slack channel  │
 │               │                 │                 │
 │ Experiments   │                 │  Alert payloads │
 │ Model Registry│                 │  PR-AUC, drift  │
 │ Artifacts     │                 │  deep link →    │
 └───────────────┘                 │  Promote run    │
                                   └─────────────────┘
         │
         ▼
 ┌───────────────────────────────────────────────────┐
 │  FastAPI inference service  (Docker / GHCR)       │
 │                                                   │
 │  GET  /health       liveness check                │
 │  GET  /model-info   version · alias · run_id      │
 │  POST /predict      batch → probabilities + labels│
 └───────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Tool |
|---|---|
| Data | Kaggle Credit Card Fraud (`mlg-ulb/creditcardfraud`) |
| Modeling | scikit-learn `LogisticRegression(class_weight='balanced')` |
| Tracking + Registry | MLflow 3 on DagsHub |
| Drift detection | Evidently `DataDriftPreset` + PR-AUC threshold |
| Serving | FastAPI + Uvicorn → Docker → GHCR |
| Orchestration | GitHub Actions — cron, `repository_dispatch`, manual gate |
| Notifications | Slack Incoming Webhook (Block Kit payload) |
| Tests | pytest + ruff + black |
| Config | Pydantic Settings + Hydra-style `conf/config.yaml` |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for local serve testing)
- A [DagsHub](https://dagshub.com) account with an MLflow-linked repo
- A [Kaggle](https://www.kaggle.com) account with an API token
- A Slack Incoming Webhook URL

### 1 — Clone and install

```bash
git clone https://github.com/aaziy/PhoenixML.git
cd PhoenixML

cp .env.example .env   # fill in your secrets (see table below)

make install-dev       # pip install -r requirements-dev.txt + editable package
```

### 2 — Smoke-test the connection

```bash
make hello             # connects to DagsHub MLflow and lists experiments
```

### 3 — Train baseline

```bash
make train             # downloads Kaggle data, trains, logs to MLflow
```

The new model version will appear in your DagsHub MLflow registry under the alias **Staging**.

### 4 — Run monitoring

```bash
make monitor           # scores next production batch, checks drift, may Slack
```

### 5 — Simulate drift end-to-end

```bash
make simulate          # walks all production batches with increasing noise
# Options:
make simulate ARGS="--start-noise 0.0 --end-noise 2.0 --batches 10"
```

### 6 — Serve locally with Docker

```bash
docker pull ghcr.io/aaziy/phoenixml-serve:latest

docker run -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=https://dagshub.com/<user>/PhoenixML.mlflow \
  -e MLFLOW_TRACKING_USERNAME=<user> \
  -e MLFLOW_TRACKING_PASSWORD=<token> \
  ghcr.io/aaziy/phoenixml-serve:latest
```

Then test:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"data": [[-1.36, 0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
                  0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
                  0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
                  0.13, -0.19, 0.13, -0.02, 149.62]]}'
```

---

## API Reference

### `GET /health`

Liveness probe.

```json
{ "status": "ok", "model_loaded": true, "model_version": "4" }
```

### `GET /model-info`

Returns metadata about the currently loaded model.

```json
{
  "model_name": "fraud-detector",
  "version": "4",
  "alias": "Staging",
  "run_id": "f3783fceb481...",
  "registered_at": "2026-05-09T08:30:27+00:00",
  "tracking_uri": "https://dagshub.com/.../PhoenixML.mlflow"
}
```

### `POST /predict`

Input: a list of feature vectors (29 floats each: `V1`–`V28` + `Amount`).

```json
{ "data": [[v1, v2, ..., v28, amount]] }
```

Response:

```json
{
  "probabilities": [0.0012, 0.9843],
  "labels": [0, 1],
  "model_version": "4",
  "threshold": 0.5,
  "n_rows": 2
}
```

---

## Secrets

Add these to **GitHub → Settings → Secrets and variables → Actions**:

| Secret | Where to get it |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/PhoenixML.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token (Settings → Tokens) |
| `SLACK_WEBHOOK_URL` | Slack App → Incoming Webhooks |
| `KAGGLE_USERNAME` | Kaggle → Account → Settings → API |
| `KAGGLE_KEY` | Kaggle → Account → Settings → API |
| `DISPATCH_PAT` | GitHub → Settings → Personal Access Tokens (`repo` write scope) |

---

## GitHub Actions Workflows

| Workflow | Trigger | What it does |
|---|---|---|
| `ci.yml` | push / PR to `main` | ruff lint, black format check, pytest |
| `train.yml` | push to `src/` or `requirements.txt`; `repository_dispatch: retrain`; `workflow_dispatch` | Download data, train, log to MLflow, register to Staging |
| `monitor.yml` | cron `0 */6 * * *`; `workflow_dispatch` | Score latest production batch, compute drift, Slack alert if breach, dispatch retrain |
| `promote.yml` | `workflow_dispatch` + GitHub Environment approval | Compare Staging vs Production PR-AUC, promote if Δ > ε |
| `serve.yml` | push to `src/phoenixml/api.py`, `docker/serve.Dockerfile`, `requirements.txt`; `workflow_dispatch` | Build multi-arch image (`amd64` + `arm64`), push to GHCR |

### Gated promotion setup

1. Go to **GitHub → Settings → Environments → New environment** → name it `production`.
2. Add yourself (or your team) as a **Required reviewer**.
3. The `promote.yml` workflow will pause and send a review request before the promotion step runs.

---

## Repo Layout

```
.github/workflows/
  ci.yml            # ruff + black + pytest on every push / PR
  train.yml         # training pipeline, triggered by code push or dispatch
  monitor.yml       # monitoring loop, cron every 6h
  promote.yml       # gated Staging → Production promotion
  serve.yml         # builds + pushes multi-arch Docker image to GHCR
src/phoenixml/
  __init__.py
  config.py         # Pydantic settings, thresholds, MLflow names
  data.py           # Kaggle download, time-sorted 70/15/15 split, batch iterator
  drift.py          # Evidently DataDriftPreset wrappers
  train.py          # scikit-learn pipeline, MLflow logging, Staging alias
  monitor.py        # score batch, detect drift, Slack + dispatch
  promote.py        # eval gate: Δ PR-AUC > ε before promoting
  api.py            # FastAPI: /health /predict /model-info
  notify.py         # Slack Block Kit payload builder
  simulate_drift.py # drift simulation harness (end-to-end demo)
docker/
  train.Dockerfile  # multi-stage builder for training job
  serve.Dockerfile  # lean uvicorn runtime image
tests/
  test_data.py      # split + batch logic
  test_train.py     # pipeline build + scoring
  test_monitor.py   # Evidently wrappers + Slack payload
  test_promote.py   # eval gate logic
  test_api.py       # FastAPI endpoint smoke tests
conf/
  config.yaml       # thresholds, MLflow experiment names, data paths
data/               # gitignored — populated by make train
notebooks/          # exploration (not in CI)
.env.example        # secret template — copy to .env and fill in
pyproject.toml      # project metadata, ruff + black + isort config
requirements.txt    # runtime dependencies
requirements-dev.txt# dev/test dependencies
Makefile            # make install | train | monitor | api | test | simulate
```

---

## Phases Completed

| Phase | Description | Status |
|---|---|---|
| 0 — Setup | Repo skeleton, DagsHub, Slack, pre-commit | ✅ |
| 1 — Baseline | Data pipeline, LogReg training, MLflow registry | ✅ |
| 2 — Dockerize | Multi-stage Dockerfiles, train GitHub Actions workflow | ✅ |
| 3 — Monitoring | Evidently drift, PR-AUC scoring, Slack alerts, monitor workflow | ✅ |
| 4 — Auto-retrain | `repository_dispatch` retrain, gated promotion, promote workflow | ✅ |
| 5 — FastAPI | `/predict /health /model-info`, GHCR image, smoke tests | ✅ |
| 6 — Simulation | Drift harness, end-to-end demo with progressive perturbation | ✅ |
| 7 — Polish | README, CI hardening, `v0.1` tag | ✅ |

---

## What I'd Do Next (v0.2)

1. **XGBoost upgrade** — swap scikit-learn LogReg for XGBoost; the promotion gate already handles the A/B comparison automatically.
2. **Optimal decision threshold** — replace the fixed `0.5` cutoff in `/predict` with the F1-optimal threshold derived from the eval-split PR curve, stored as a logged parameter.
3. **SHAP explanations** — add a `GET /explain` endpoint returning per-feature SHAP values for a given prediction.
4. **Feature store** — replace the raw Kaggle CSV with a versioned Feast feature store so the training and serving features stay in sync.
5. **Kubernetes deployment** — replace the single Docker container with a Helm chart + HPA for autoscaling.
6. **Live A/B testing** — run a champion-challenger split at the nginx/Envoy layer and log per-variant metrics.
7. **DagsHub DVC** — version the raw dataset and preprocessed splits with DVC so every training run is fully reproducible.
8. **Alerting SLA** — add PagerDuty escalation if the Slack alert isn't acknowledged within 30 minutes.

---

## License

MIT
