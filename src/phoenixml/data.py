"""Data layer: Kaggle download, time-sorted splits, and production batch iterator."""

from __future__ import annotations

import logging
import os
import zipfile
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from phoenixml.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ── Download ──────────────────────────────────────────────────────────────────


def download_dataset(settings: Settings | None = None) -> Path:
    """Download the creditcard dataset from Kaggle if not already cached.

    Returns the path to creditcard.csv.
    """
    cfg = settings or get_settings()
    raw_dir: Path = cfg.raw_dir
    csv_path = raw_dir / "creditcard.csv"

    if csv_path.exists():
        logger.info("Dataset already cached at %s", csv_path)
        return csv_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    # Kaggle API credentials must be in env or ~/.kaggle/kaggle.json
    os.environ.setdefault("KAGGLE_USERNAME", cfg.kaggle_username)
    if cfg.kaggle_key:
        os.environ.setdefault("KAGGLE_KEY", cfg.kaggle_key)

    try:
        from kaggle import api as kaggle_api  # type: ignore

        kaggle_api.authenticate()
        logger.info("Downloading dataset %s …", cfg.dataset_slug)
        kaggle_api.dataset_download_files(cfg.dataset_slug, path=str(raw_dir), unzip=False)
    except Exception as e:
        raise RuntimeError(
            f"Kaggle download failed: {e}\n" "Make sure KAGGLE_USERNAME and KAGGLE_KEY are set."
        ) from e

    zip_candidates = list(raw_dir.glob("*.zip"))
    if zip_candidates:
        zip_path = zip_candidates[0]
        logger.info("Extracting %s …", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)
        zip_path.unlink()

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Expected {csv_path} after download but it was not found. "
            "Check the dataset slug in conf/config.yaml."
        )

    logger.info("Dataset ready: %s  (%d bytes)", csv_path, csv_path.stat().st_size)
    return csv_path


# ── Load + split ──────────────────────────────────────────────────────────────


def load_raw(settings: Settings | None = None) -> pd.DataFrame:
    """Load the raw CSV, sort by Time (proxy for temporal order)."""
    cfg = settings or get_settings()
    csv_path = download_dataset(cfg)
    df = pd.read_csv(csv_path)
    # Sort by the 'Time' column — seconds elapsed since first transaction
    df = df.sort_values("Time").reset_index(drop=True)
    logger.info(
        "Loaded %d rows, %d positives (%.4f%%)",
        len(df),
        df["Class"].sum(),
        100 * df["Class"].mean(),
    )
    return df


def split_dataset(
    df: pd.DataFrame,
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Time-sorted 70/15/15 split.

    Returns (train_df, eval_df, prod_pool_df).
    The prod_pool is the last 15% used as the production batch simulation.
    """
    cfg = settings or get_settings()
    n = len(df)
    train_end = int(n * cfg.train_frac)
    eval_end = int(n * (cfg.train_frac + cfg.eval_frac))

    train_df = df.iloc[:train_end].copy()
    eval_df = df.iloc[train_end:eval_end].copy()
    prod_pool_df = df.iloc[eval_end:].copy()

    logger.info(
        "Split sizes — train: %d  eval: %d  prod_pool: %d",
        len(train_df),
        len(eval_df),
        len(prod_pool_df),
    )
    return train_df, eval_df, prod_pool_df


def save_splits(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    prod_pool_df: pd.DataFrame,
    settings: Settings | None = None,
) -> None:
    """Persist processed splits to disk."""
    cfg = settings or get_settings()
    out_dir: Path = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    eval_df.to_parquet(out_dir / "eval.parquet", index=False)
    prod_pool_df.to_parquet(out_dir / "prod_pool.parquet", index=False)
    logger.info("Saved splits to %s", out_dir)


def load_splits(
    settings: Settings | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-processed splits from disk (fast path after first run)."""
    cfg = settings or get_settings()
    out_dir: Path = cfg.processed_dir
    return (
        pd.read_parquet(out_dir / "train.parquet"),
        pd.read_parquet(out_dir / "eval.parquet"),
        pd.read_parquet(out_dir / "prod_pool.parquet"),
    )


def splits_exist(settings: Settings | None = None) -> bool:
    cfg = settings or get_settings()
    d = cfg.processed_dir
    return all((d / f).exists() for f in ("train.parquet", "eval.parquet", "prod_pool.parquet"))


def get_feature_columns(df: pd.DataFrame, settings: Settings | None = None) -> list[str]:
    """Return feature column names (all except Time and Class)."""
    cfg = settings or get_settings()
    return [c for c in df.columns if c not in ("Time", cfg.target_col)]


# ── Production batch iterator ─────────────────────────────────────────────────


def prod_batch_iterator(
    prod_pool_df: pd.DataFrame,
    settings: Settings | None = None,
) -> Iterator[tuple[int, pd.DataFrame]]:
    """Yield (batch_id, batch_df) slices from the production pool.

    batch_id is 0-indexed. The pool is divided into cfg.n_prod_batches windows.
    """
    cfg = settings or get_settings()
    n = len(prod_pool_df)
    batch_size = n // cfg.n_prod_batches

    for i in range(cfg.n_prod_batches):
        start = i * batch_size
        end = start + batch_size if i < cfg.n_prod_batches - 1 else n
        yield i, prod_pool_df.iloc[start:end].copy()


# ── Convenience: prepare everything in one call ───────────────────────────────


def prepare_data(
    settings: Settings | None = None,
    force_reprocess: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download (if needed), split, cache, and return (train, eval, prod_pool)."""
    cfg = settings or get_settings()
    if not force_reprocess and splits_exist(cfg):
        logger.info("Loading cached splits from %s", cfg.processed_dir)
        return load_splits(cfg)

    df = load_raw(cfg)
    train_df, eval_df, prod_pool_df = split_dataset(df, cfg)
    save_splits(train_df, eval_df, prod_pool_df, cfg)
    return train_df, eval_df, prod_pool_df
