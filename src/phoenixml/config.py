"""Pydantic settings — all config and thresholds loaded from env + conf/config.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_FILE = _REPO_ROOT / "conf" / "config.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_FILE) as f:
        return yaml.safe_load(f)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # ── MLflow / DagsHub ──────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="https://dagshub.com/muhammadaaziq179/PhoenixML.mlflow"
    )
    mlflow_tracking_username: str = Field(default="muhammadaaziq179")
    mlflow_tracking_password: str = Field(default="")

    # ── Slack ─────────────────────────────────────────────────────────────
    slack_webhook_url: str = Field(default="")

    # ── Kaggle ────────────────────────────────────────────────────────────
    kaggle_username: str = Field(default="muhammadaziqrauf")
    kaggle_key: str = Field(default="")

    # ── GitHub dispatch ───────────────────────────────────────────────────
    dispatch_pat: str = Field(default="")
    github_repo: str = Field(default="aaziy/PhoenixML")

    # ── Derived from yaml (loaded lazily) ─────────────────────────────────
    @property
    def yaml(self) -> dict:
        return _load_yaml()

    # shortcuts ─────────────────────────────────────────────────
    @property
    def experiment_train(self) -> str:
        return self.yaml["mlflow"]["experiment_train"]

    @property
    def experiment_monitor(self) -> str:
        return self.yaml["mlflow"]["experiment_monitor"]

    @property
    def registered_model_name(self) -> str:
        return self.yaml["mlflow"]["registered_model_name"]

    @property
    def prauc_alert_threshold(self) -> float:
        return self.yaml["thresholds"]["prauc_alert"]

    @property
    def prauc_promote_delta(self) -> float:
        return self.yaml["thresholds"]["prauc_promote_delta"]

    @property
    def predict_cutoff(self) -> float:
        return self.yaml["thresholds"]["predict_cutoff"]

    @property
    def drift_share_threshold(self) -> float:
        return self.yaml["drift"]["drift_share_threshold"]

    @property
    def dataset_slug(self) -> str:
        return self.yaml["data"]["dataset_slug"]

    @property
    def raw_dir(self) -> Path:
        return _REPO_ROOT / self.yaml["data"]["raw_dir"]

    @property
    def processed_dir(self) -> Path:
        return _REPO_ROOT / self.yaml["data"]["processed_dir"]

    @property
    def train_frac(self) -> float:
        return self.yaml["data"]["train_frac"]

    @property
    def eval_frac(self) -> float:
        return self.yaml["data"]["eval_frac"]

    @property
    def n_prod_batches(self) -> int:
        return self.yaml["data"]["n_prod_batches"]

    @property
    def batch_size(self) -> int:
        return self.yaml["data"]["batch_size"]

    @property
    def target_col(self) -> str:
        return self.yaml["data"]["target_col"]

    @property
    def random_state(self) -> int:
        return self.yaml["model"]["random_state"]

    @property
    def logreg_max_iter(self) -> int:
        return self.yaml["model"]["logreg_max_iter"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
