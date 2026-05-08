"""Tests for data.py — no network calls, uses synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from phoenixml.config import get_settings
from phoenixml.data import (
    get_feature_columns,
    prod_batch_iterator,
    split_dataset,
)


def _make_synthetic_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Create a minimal synthetic dataframe that mimics the real CSV schema."""
    rng = np.random.default_rng(seed)
    v_cols = {f"V{i}": rng.standard_normal(n) for i in range(1, 29)}
    df = pd.DataFrame(
        {
            "Time": np.arange(n, dtype=float),
            **v_cols,
            "Amount": rng.exponential(100, n),
            "Class": (rng.random(n) < 0.002).astype(int),  # ~0.2% fraud
        }
    )
    return df


# ── split_dataset ─────────────────────────────────────────────────────────────


def test_split_sizes():
    df = _make_synthetic_df(1000)
    cfg = get_settings()
    train, eval_, prod = split_dataset(df, cfg)

    total = len(train) + len(eval_) + len(prod)
    assert total == len(df), "splits must cover entire dataset"

    # allow ±1 for integer rounding
    assert abs(len(train) - int(1000 * cfg.train_frac)) <= 1
    assert abs(len(eval_) - int(1000 * cfg.eval_frac)) <= 1


def test_split_temporal_order():
    """Ensure train < eval < prod in temporal order."""
    df = _make_synthetic_df(900)
    cfg = get_settings()
    train, eval_, prod = split_dataset(df, cfg)

    assert train["Time"].max() <= eval_["Time"].min()
    assert eval_["Time"].max() <= prod["Time"].min()


def test_split_no_overlap():
    df = _make_synthetic_df(900)
    cfg = get_settings()
    train, eval_, prod = split_dataset(df, cfg)

    train_idx = set(train.index)
    eval_idx = set(eval_.index)
    prod_idx = set(prod.index)

    assert train_idx.isdisjoint(eval_idx)
    assert train_idx.isdisjoint(prod_idx)
    assert eval_idx.isdisjoint(prod_idx)


# ── get_feature_columns ───────────────────────────────────────────────────────


def test_feature_columns_excludes_time_and_class():
    df = _make_synthetic_df(100)
    features = get_feature_columns(df)
    assert "Time" not in features
    assert "Class" not in features
    assert len(features) == 29  # V1–V28 + Amount


# ── prod_batch_iterator ───────────────────────────────────────────────────────


def test_batch_iterator_yields_all_rows():
    df = _make_synthetic_df(500)
    cfg = get_settings()
    _, _, prod = split_dataset(df, cfg)

    batches = list(prod_batch_iterator(prod, cfg))
    assert len(batches) == cfg.n_prod_batches

    total_rows = sum(len(b) for _, b in batches)
    assert total_rows == len(prod)


def test_batch_ids_are_sequential():
    df = _make_synthetic_df(500)
    cfg = get_settings()
    _, _, prod = split_dataset(df, cfg)

    ids = [bid for bid, _ in prod_batch_iterator(prod, cfg)]
    assert ids == list(range(cfg.n_prod_batches))
