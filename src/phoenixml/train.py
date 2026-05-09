"""Train a fraud-detection model, log to MLflow, register to Staging."""

from __future__ import annotations

import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from phoenixml.config import Settings, get_settings
from phoenixml.data import get_feature_columns, prepare_data

logger = logging.getLogger(__name__)


# ── MLflow setup ──────────────────────────────────────────────────────────────


def setup_mlflow(settings: Settings) -> None:
    """Configure MLflow tracking URI and authenticate with DagsHub."""
    os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.experiment_train)
    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)


# ── Build pipeline ────────────────────────────────────────────────────────────


def build_pipeline(settings: Settings) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=settings.logreg_max_iter,
                    solver="lbfgs",
                    random_state=settings.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )


# ── Train + evaluate ──────────────────────────────────────────────────────────


def train_and_log(settings: Settings | None = None) -> str:
    """Train the model, log to MLflow, register as Staging. Returns run_id."""
    cfg = settings or get_settings()
    setup_mlflow(cfg)

    logger.info("Preparing data …")
    train_df, eval_df, _ = prepare_data(cfg)

    feature_cols = get_feature_columns(train_df, cfg)
    X_train = train_df[feature_cols].values
    y_train = train_df[cfg.target_col].values
    X_eval = eval_df[feature_cols].values
    y_eval = eval_df[cfg.target_col].values

    pipeline = build_pipeline(cfg)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info("MLflow run_id: %s", run_id)

        # ── log params ────────────────────────────────────────────────────
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "max_iter": cfg.logreg_max_iter,
                "solver": "lbfgs",
                "train_rows": len(train_df),
                "eval_rows": len(eval_df),
                "n_features": len(feature_cols),
                "random_state": cfg.random_state,
            }
        )

        # ── train ─────────────────────────────────────────────────────────
        logger.info("Fitting pipeline on %d samples …", len(X_train))
        pipeline.fit(X_train, y_train)

        # ── evaluate ──────────────────────────────────────────────────────
        y_proba = pipeline.predict_proba(X_eval)[:, 1]
        y_pred = (y_proba >= cfg.predict_cutoff).astype(int)
        prauc = float(average_precision_score(y_eval, y_proba))
        pos_rate = float(y_eval.mean())

        mlflow.log_metrics(
            {
                "eval_prauc": prauc,
                "eval_pos_rate": pos_rate,
                "eval_rows": len(y_eval),
            }
        )
        logger.info("Eval PR-AUC: %.4f  (positive rate %.4f%%)", prauc, 100 * pos_rate)

        # classification report as a text artifact
        report = classification_report(y_eval, y_pred, target_names=["legit", "fraud"])
        mlflow.log_text(report, "eval_classification_report.txt")

        # ── log model with signature + input example ───────────────────────
        from mlflow.models.signature import infer_signature

        input_example = pd.DataFrame(X_eval[:5], columns=feature_cols)
        signature = infer_signature(input_example, y_proba[:5])

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=cfg.registered_model_name,
        )
        logger.info("Model logged and registered as '%s'", cfg.registered_model_name)

    # ── transition latest version → Staging ───────────────────────────────
    _transition_to_staging(cfg, run_id)

    return run_id


def _transition_to_staging(cfg: Settings, run_id: str) -> None:
    """Find the model version created from run_id and set the 'Staging' alias."""
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{cfg.registered_model_name}'")

    target = None
    for v in versions:
        if v.run_id == run_id:
            target = v
            break

    if target is None:
        logger.warning("Could not find registered version for run_id %s", run_id)
        return

    # MLflow 3.x: use aliases instead of deprecated stages
    client.set_registered_model_alias(
        name=cfg.registered_model_name,
        alias="Staging",
        version=target.version,
    )
    logger.info(
        "Version %s of '%s' → alias 'Staging' set",
        target.version,
        cfg.registered_model_name,
    )


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )
    run_id = train_and_log()
    print(f"\nDone. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
