# PhoenixML — Implementation Plan

## North Star

A fully automated MLOps pipeline for credit-card fraud detection. When PR-AUC on a new production batch drops below threshold **or** Evidently reports input drift, the system Slacks an alert, fires a `repository_dispatch` to retrain a challenger model in GitHub Actions, registers it to MLflow `Staging`, and waits for a human approval (GitHub Environment) before promoting to `Production`. A FastAPI container serves the current Production model and exposes its version via `/model-info`.

## Stack

- **Data**: Kaggle Credit Card Fraud (`mlg-ulb/creditcardfraud`)
- **Modeling**: scikit-learn LogReg (`class_weight='balanced'`); XGBoost as a later upgrade
- **Tracking + Registry**: MLflow on DagsHub
- **Drift**: Evidently (input drift) + PR-AUC (performance drift)
- **Serving**: FastAPI in Docker (image pushed to GHCR)
- **Orchestration**: GitHub Actions — cron + `repository_dispatch` + `workflow_dispatch`
- **Compute**: GH-hosted runners (`ubuntu-latest`) for everything; local laptop for exploration only
- **Notifications**: Slack Incoming Webhook
- **Tests**: pytest + ruff

## Critical architectural decisions

1. **All automation runs in GitHub Actions.** Train, monitor, promote, serve — every workflow on free GH-hosted runners. The autonomous retrain story works end-to-end without a laptop being online.
2. **PR-AUC, not accuracy.** Fraud is ~0.17% positive class — accuracy is meaningless. Default trigger: `PR-AUC < 0.75`. Pick the real threshold after seeing baseline numbers.
3. **Two-signal drift.** Evidently dataset-drift flag **or** PR-AUC below threshold triggers retrain. Both signals logged to MLflow under a `monitoring` experiment for a metric-over-time view.
4. **Gated promotion.** Retrain → MLflow `Staging` automatic. `Staging` → `Production` is manual, behind a GitHub Environment with a required reviewer.
5. **Deterministic drift simulation.** Time-sort the dataset, take the last 15% as N "production" batches, progressively perturb the `V*` features in later batches. Reproducible alerts.

## Repo layout

```text
.github/workflows/
  train.yml       # ubuntu-latest; on path push or repository_dispatch
  monitor.yml     # ubuntu-latest; cron 0 */6 * * * + workflow_dispatch
  promote.yml     # ubuntu-latest; workflow_dispatch only, environment: production
  serve.yml       # ubuntu-latest; build + push FastAPI image to GHCR
  ci.yml          # ubuntu-latest; pytest + ruff on PRs
src/phoenixml/
  __init__.py
  config.py       # pydantic settings, thresholds, MLflow names
  data.py         # Kaggle download, time-sort split, batch generator
  drift.py        # Evidently wrappers
  train.py        # train + log to MLflow, register, transition → Staging
  monitor.py      # load Production, score batch, PR-AUC + drift, alert + dispatch
  promote.py      # Staging → Production with offline eval gate
  api.py          # FastAPI: /predict /health /model-info
  notify.py       # Slack webhook payload builder
  simulate_drift.py
docker/
  train.Dockerfile
  serve.Dockerfile
tests/
  test_data.py test_train.py test_monitor.py test_api.py
conf/config.yaml
data/             # gitignored
notebooks/        # exploration; not in CI
.env.example
pyproject.toml requirements.txt requirements-dev.txt
Makefile          # make train | monitor | api | test | simulate
README.md PLAN.md
```

## Secrets (GitHub repo → Settings → Secrets and variables → Actions)

| Secret | Source |
| --- | --- |
| `MLFLOW_TRACKING_URI` | `https://dagshub.com/<user>/PhoenixML.mlflow` |
| `MLFLOW_TRACKING_USERNAME` | DagsHub username |
| `MLFLOW_TRACKING_PASSWORD` | DagsHub access token |
| `SLACK_WEBHOOK_URL` | Slack Incoming Webhook |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle account → Settings → API |
| `DISPATCH_PAT` | Fine-scoped PAT so `monitor` can fire `repository_dispatch` |

## Phase plan (2 weeks)

### Phase 0 — Setup (Day 1, half day)

- Create GitHub repo, push skeleton, init pre-commit (ruff, black, isort)
- Create DagsHub repo linked to GitHub; copy MLflow URI + token
- Create Slack workspace + Incoming Webhook; smoke-test with `curl`
- Add all secrets to GitHub
- **Acceptance**: `make hello` connects to DagsHub MLflow and lists experiments

### Phase 1 — Baseline training + MLflow (Days 1-2)

- `data.py`: Kaggle API download → cache → time-sorted 70/15/15 split → batch iterator over the last 15%
- `train.py`: `Pipeline([StandardScaler, LogisticRegression(class_weight='balanced')])`; log params, PR-AUC on eval split, model with signature + input example; register as `fraud-detector`, transition new version → `Staging`
- **Acceptance**: run + registered version visible in DagsHub MLflow UI

### Phase 2 — Dockerize + train workflow (Days 3-4)

- `train.Dockerfile`: multi-stage, slim final image
- `train.yml`: `runs-on: ubuntu-latest`; triggers = push to `src/phoenixml/train.py` / `requirements.txt`, plus `repository_dispatch: retrain`; cache pip + Kaggle dataset
- **Acceptance**: pushing a change re-runs training and produces a new MLflow version on DagsHub

### Phase 3 — Monitoring + drift detection (Days 5-6)

- `drift.py`: Evidently `DataDriftPreset` on live batch vs. training reference; return overall flag + per-column scores
- `monitor.py`: pull `Production` model → score next batch → PR-AUC + drift report → log to MLflow `monitoring` experiment → return decision
- `notify.py`: Slack payload with model version, PR-AUC, drift score, batch id, timestamp, MLflow run URL
- `monitor.yml`: `cron: 0 */6 * * *` + `workflow_dispatch`
- **Acceptance**: `workflow_dispatch` posts a Slack message; perturbed batch flips the drift flag

### Phase 4 — Auto-retrain + gated promotion (Days 7-8)

- On a fired trigger, `monitor.py` calls GitHub API → `repository_dispatch: retrain` (uses `DISPATCH_PAT`)
- `promote.py`: load `Staging` + `Production` candidates, score on held-out eval split, promote only if `Δ PR-AUC > ε`
- `promote.yml`: `workflow_dispatch` only, `environment: production` (configure required reviewer in GH UI)
- Slack message includes deep link to the pending Promote run
- **Acceptance**: forced-drift batch → Slack → retrain → new Staging version → manual Promote → Production version updated

### Phase 5 — FastAPI inference (Days 9-10)

- `api.py`:
  - `GET /health` — liveness
  - `POST /predict` — batch input, returns probabilities + threshold-based labels
  - `GET /model-info` — current `Production` version, run id, registered timestamp pulled from MLflow
- `serve.Dockerfile`: lean uvicorn image
- `serve.yml`: build + push to GHCR on push to `main`
- pytest smoke tests for each endpoint with a stubbed model
- **Acceptance**: `docker run -p 8000:8000 ghcr.io/<user>/phoenixml-serve` works; `/model-info` reflects DagsHub state

### Phase 6 — Drift simulation harness (Days 11-12)

- `simulate_drift.py`: walks N production batches with increasing perturbation, calls the monitor flow against each
- Capture screenshots and a demo recording for README
- **Acceptance**: end-to-end run — clean batch (no alert) → noisy batch (alert + retrain + Staging) → manual approve → Production updated

### Phase 7 — Polish (Days 13-14)

- README: architecture diagram, quickstart, screenshots, demo GIF, "what I'd do next"
- `ci.yml`: pytest + ruff on PRs
- Tag `v0.1`

## Open risks / revisit

- **DagsHub free-tier artifact limits** — check before logging large artifacts; prune old runs if needed.
- **Kaggle API rate limits in CI** — cache dataset in GH Actions cache keyed on dataset slug.
- **LogReg PR-AUC** on this dataset is typically ~0.70-0.75; set the alert threshold after the baseline run, not blind.
- **Predict threshold** — `predict_proba` cutoff for `/predict` should come from the eval-split PR curve, not 0.5.
- **GH Actions runtime** — free tier is 2000 min/month; cron-every-6h burns ~60 min/month, plenty of headroom.

## Explicitly out of scope (v0.1)

- XGBoost upgrade (only after baseline is end-to-end)
- Live A/B / champion-challenger at inference time
- Feature store, streaming inference
- Multi-region, autoscaling serving

## Day-1 checklist

- [ ] GitHub repo created, skeleton pushed
- [ ] DagsHub linked, MLflow URI confirmed reachable
- [ ] Slack webhook tested with `curl`
- [ ] All 6 secrets in GitHub (MLflow x3, Slack, Kaggle x2, DISPATCH_PAT)
- [ ] `make hello` lists DagsHub experiments
