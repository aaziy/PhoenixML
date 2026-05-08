# PhoenixML 🔥

**Automated MLOps pipeline for credit-card fraud detection.**

When PR-AUC on a new production batch drops below threshold *or* Evidently reports input drift, the system Slacks an alert, fires a `repository_dispatch` to retrain a challenger model in GitHub Actions, registers it to MLflow `Staging`, and waits for human approval before promoting to `Production`. A FastAPI container serves the current Production model.

## Stack

| Layer | Tool |
|---|---|
| Data | Kaggle Credit Card Fraud (`mlg-ulb/creditcardfraud`) |
| Modeling | scikit-learn `LogisticRegression(class_weight='balanced')` |
| Tracking + Registry | MLflow on DagsHub |
| Drift | Evidently (input) + PR-AUC (performance) |
| Serving | FastAPI + Docker → GHCR |
| Orchestration | GitHub Actions (cron + dispatch + manual gate) |
| Notifications | Slack Incoming Webhook |

## Quick Start

```bash
# 1. clone
git clone https://github.com/aaziy/PhoenixML && cd PhoenixML

# 2. copy env
cp .env.example .env   # fill in your secrets

# 3. install (dev)
make install-dev

# 4. smoke test — connect to DagsHub MLflow
make hello

# 5. train baseline
make train
```

## Secrets (GitHub → Settings → Secrets)

| Secret | Source |
|---|---|
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/muhammadaaziq179/PhoenixML.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook |
| `KAGGLE_USERNAME` | Kaggle account → Settings → API |
| `KAGGLE_KEY` | Kaggle account → Settings → API |
| `DISPATCH_PAT` | Fine-scoped GitHub PAT with `repo` write |

## Repo Layout

```
.github/workflows/
  ci.yml          # pytest + ruff on PRs
  train.yml       # triggered by code push or repository_dispatch
  monitor.yml     # cron every 6h + workflow_dispatch   [Phase 3]
  promote.yml     # manual gate, environment: production [Phase 4]
  serve.yml       # build + push FastAPI to GHCR        [Phase 5]
src/phoenixml/
  config.py       # pydantic settings
  data.py         # Kaggle download, split, batch iterator
  drift.py        # Evidently wrappers                  [Phase 3]
  train.py        # train + MLflow + register
  monitor.py      # score batch, detect drift           [Phase 3]
  promote.py      # Staging → Production                [Phase 4]
  api.py          # FastAPI /predict /health /model-info[Phase 5]
  notify.py       # Slack webhook
  simulate_drift.py                                     [Phase 6]
docker/
  train.Dockerfile
  serve.Dockerfile
tests/
  test_data.py  test_train.py  test_monitor.py  test_api.py
conf/config.yaml
```

## Phases

| Phase | Status | Description |
|---|---|---|
| 0 — Setup | ✅ | Repo skeleton, DagsHub, Slack |
| 1 — Baseline | ✅ | Data + training + MLflow |
| 2 — Dockerize | 🔜 | train.Dockerfile + train.yml |
| 3 — Monitoring | 🔜 | Evidently + PR-AUC + Slack |
| 4 — Auto-retrain | 🔜 | Dispatch + gated promotion |
| 5 — FastAPI | 🔜 | /predict /health /model-info |
| 6 — Simulation | 🔜 | Drift harness + demo |
| 7 — Polish | 🔜 | README, CI, tag v0.1 |
