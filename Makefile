.PHONY: hello install install-dev train monitor api test simulate lint format clean

# ── env ────────────────────────────────────────────────────────────────────
-include .env
export

# ── setup ──────────────────────────────────────────────────────────────────
install:
	pip install -e .
	pip install -r requirements.txt

install-dev:
	pip install -e .
	pip install -r requirements-dev.txt
	pre-commit install

# ── smoke test: list DagsHub MLflow experiments ────────────────────────────
hello:
	python - <<'EOF'
import mlflow, os
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
exps = mlflow.search_experiments()
print(f"Connected to {os.environ['MLFLOW_TRACKING_URI']}")
print(f"Experiments found: {len(exps)}")
for e in exps:
    print(f"  [{e.experiment_id}] {e.name}")
EOF

# ── ML pipeline ────────────────────────────────────────────────────────────
train:
	python -m phoenixml.train

monitor:
	python -m phoenixml.monitor

promote:
	python -m phoenixml.promote

# ── FastAPI ────────────────────────────────────────────────────────────────
api:
	uvicorn phoenixml.api:app --host 0.0.0.0 --port 8000 --reload

# ── drift simulation ───────────────────────────────────────────────────────
simulate:
	python -m phoenixml.simulate_drift

# ── tests + lint ───────────────────────────────────────────────────────────
test:
	pytest tests/ -v --cov=phoenixml --cov-report=term-missing

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	ruff check --fix src/ tests/
	black src/ tests/
	isort src/ tests/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache/ .ruff_cache/ htmlcov/ .coverage
