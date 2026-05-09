"""Drift simulation harness.

Walks N production batches with progressively increasing Gaussian noise on the
V* features. For each batch it runs the full monitor pipeline (score + drift
detection + MLflow logging). Produces a colour-coded summary table showing when
alerts trigger.

Usage (local):
    make simulate                          # default: 10 batches, max noise 3.0
    python -m phoenixml.simulate_drift --n-batches 10 --max-noise 4.0
    python -m phoenixml.simulate_drift --dry-run    # skip Slack / dispatch

Acceptance criteria (Phase 6):
    clean batch (batch 0)  → no alert
    noisy batch (last few) → alert + Slack + repository_dispatch fires
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import mlflow

from phoenixml.config import Settings, get_settings
from phoenixml.data import get_feature_columns, prepare_data, prod_batch_iterator
from phoenixml.drift import compute_drift, perturb_batch
from phoenixml.monitor import (
    _fire_retrain_dispatch,
    _load_production_model,
    _setup_mlflow,
    score_batch,
)
from phoenixml.notify import send_alert

logger = logging.getLogger(__name__)

# ANSI colour helpers (fallback gracefully if terminal doesn't support them)
_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_BOLD = "\033[1m"
_RESET = "\033[0m"


def _c(text: str, colour: str) -> str:
    if os.getenv("NO_COLOR") or not os.isatty(1):
        return text
    return f"{colour}{text}{_RESET}"


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class BatchResult:
    batch_id: int
    noise_scale: float
    n_rows: int
    prauc: float
    drift_share: float
    n_drifted: int
    trigger_retrain: bool
    mlflow_run_id: str = ""
    action: str = ""


# ── Noise schedule ────────────────────────────────────────────────────────────


def _noise_schedule(batch_id: int, n_batches: int, max_noise: float) -> float:
    """Linear ramp: batch 0 → 0.0, last batch → max_noise."""
    if n_batches <= 1:
        return 0.0
    return max_noise * batch_id / (n_batches - 1)


# ── Core simulation ───────────────────────────────────────────────────────────


def run_simulation(
    n_batches: int | None = None,
    max_noise: float = 3.0,
    dry_run: bool = False,
    stop_on_alert: bool = False,
    settings: Settings | None = None,
) -> list[BatchResult]:
    """Run the drift simulation.

    Args:
        n_batches:      How many production batches to walk (default: cfg.n_prod_batches).
        max_noise:      Maximum noise sigma applied to V* features on the last batch.
        dry_run:        If True, skip Slack alerts and repository_dispatch.
        stop_on_alert:  If True, stop after the first retrain trigger.

    Returns a list of BatchResult, one per batch.
    """
    cfg = settings or get_settings()
    _setup_mlflow(cfg)
    mlflow.set_experiment(cfg.experiment_monitor)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data splits …")
    train_df, _, prod_pool_df = prepare_data(cfg)
    feature_cols = get_feature_columns(train_df, cfg)
    all_batches = list(prod_batch_iterator(prod_pool_df, cfg))
    n_batches = n_batches or len(all_batches)
    n_batches = min(n_batches, len(all_batches))

    # ── Load model ────────────────────────────────────────────────────────────
    model, model_version = _load_production_model(cfg)
    promote_url = f"https://github.com/{cfg.github_repo}/actions/workflows/promote.yml"

    results: list[BatchResult] = []
    retrain_fired = False

    print(f"\n{_c('PhoenixML Drift Simulation', _BOLD)}")
    print(
        f"  Model: fraud-detector v{model_version}  |  Batches: {n_batches}  |  Max noise: {max_noise}σ"
    )
    print(f"  dry_run={dry_run}  stop_on_alert={stop_on_alert}")
    print(f"  PR-AUC alert threshold: {cfg.prauc_alert_threshold}")
    print(f"  Drift share threshold:  {cfg.drift_share_threshold}")
    _print_table_header()

    for batch_idx in range(n_batches):
        _, batch_df = all_batches[batch_idx]
        noise = _noise_schedule(batch_idx, n_batches, max_noise)

        # Apply progressive perturbation
        if noise > 0:
            batch_df = perturb_batch(batch_df, feature_cols, noise_scale=noise, seed=batch_idx)

        # Score
        prauc = score_batch(model, batch_df, feature_cols, cfg.target_col)

        # Drift
        drift_result = compute_drift(train_df, batch_df, feature_cols, cfg)
        drift_share: float = drift_result["drift_share"]
        n_drifted: int = drift_result["n_drifted"]

        prauc_alert = prauc < cfg.prauc_alert_threshold
        drift_alert = drift_result["drift_detected"]
        trigger_retrain = prauc_alert or drift_alert

        # Log to MLflow
        run_id = ""
        with mlflow.start_run(run_name=f"sim-batch-{batch_idx}-noise-{noise:.1f}") as run:
            run_id = run.info.run_id
            mlflow.log_params(
                {
                    "batch_id": batch_idx,
                    "noise_scale": noise,
                    "model_version": model_version,
                    "simulated": True,
                    "n_rows": len(batch_df),
                }
            )
            mlflow.log_metrics(
                {
                    "prauc": prauc,
                    "drift_share": drift_share,
                    "n_drifted_features": n_drifted,
                    "trigger_retrain": int(trigger_retrain),
                }
            )

        action = "OK"
        if trigger_retrain and not retrain_fired:
            action = "🔴 RETRAIN"
            retrain_fired = True
            if not dry_run:
                exp = mlflow.get_experiment_by_name(cfg.experiment_monitor)
                run_url = (
                    f"{cfg.mlflow_tracking_uri}/#/experiments/" f"{exp.experiment_id}/runs/{run_id}"
                )
                send_alert(
                    batch_id=batch_idx,
                    model_version=str(model_version),
                    prauc=prauc,
                    drift_detected=drift_alert,
                    drift_share=drift_share,
                    run_url=run_url,
                    trigger_retrain=True,
                    promote_url=promote_url,
                    settings=cfg,
                )
                _fire_retrain_dispatch(cfg)
            else:
                logger.info("[dry-run] Skipping Slack alert and repository_dispatch.")
        elif trigger_retrain:
            action = "⚠️  (already dispatched)"

        result = BatchResult(
            batch_id=batch_idx,
            noise_scale=noise,
            n_rows=len(batch_df),
            prauc=prauc,
            drift_share=drift_share,
            n_drifted=n_drifted,
            trigger_retrain=trigger_retrain,
            mlflow_run_id=run_id,
            action=action,
        )
        results.append(result)
        _print_table_row(result, cfg)

        if stop_on_alert and trigger_retrain:
            logger.info("stop_on_alert=True — stopping after first trigger.")
            break

    _print_summary(results, cfg)
    _save_csv(results, cfg)
    return results


# ── Table printing ────────────────────────────────────────────────────────────

_COL = "  {:<6} {:<8} {:<7} {:<10} {:<10} {:<6} {}"
_HEADER = _COL.format("Batch", "Noise σ", "PR-AUC", "Drift%", "Drifted", "Alert", "Action")
_SEP = "  " + "-" * 68


def _print_table_header() -> None:
    print(f"\n{_c(_HEADER, _BOLD)}")
    print(_SEP)


def _print_table_row(r: BatchResult, cfg: Settings) -> None:
    prauc_str = f"{r.prauc:.4f}"
    drift_pct = f"{r.drift_share:.1%}"
    alert = "YES" if r.trigger_retrain else "no"

    colour = (
        _RED
        if r.trigger_retrain
        else (_YELLOW if r.prauc < cfg.prauc_alert_threshold + 0.05 else _GREEN)
    )
    row = _COL.format(
        r.batch_id,
        f"{r.noise_scale:.2f}",
        prauc_str,
        drift_pct,
        f"{r.n_drifted} cols",
        alert,
        r.action,
    )
    print(_c(row, colour))


def _print_summary(results: list[BatchResult], cfg: Settings) -> None:
    print(_SEP)
    triggered = [r for r in results if r.trigger_retrain]
    print(f"\n  Batches run:    {len(results)}")
    print(f"  Alerts fired:   {len(triggered)}")
    if triggered:
        first = triggered[0]
        print(
            f"  First alert:    batch {first.batch_id} (noise={first.noise_scale:.2f}σ,  PR-AUC={first.prauc:.4f},  drift={first.drift_share:.1%})"
        )
    else:
        print(f"  No alerts fired (all PR-AUC ≥ {cfg.prauc_alert_threshold})")
    print()


def _save_csv(results: list[BatchResult], cfg: Settings) -> None:
    out_path = Path("data") / "simulation_results.csv"
    out_path.parent.mkdir(exist_ok=True)
    fieldnames = [
        "batch_id",
        "noise_scale",
        "n_rows",
        "prauc",
        "drift_share",
        "n_drifted",
        "trigger_retrain",
        "mlflow_run_id",
        "action",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "batch_id": r.batch_id,
                    "noise_scale": round(r.noise_scale, 3),
                    "n_rows": r.n_rows,
                    "prauc": round(r.prauc, 4),
                    "drift_share": round(r.drift_share, 4),
                    "n_drifted": r.n_drifted,
                    "trigger_retrain": r.trigger_retrain,
                    "mlflow_run_id": r.mlflow_run_id,
                    "action": r.action,
                }
            )
    logger.info("Simulation results saved to %s", out_path)


# ── n_features_label patch ────────────────────────────────────────────────────
# Add a computed field to BatchResult for display


def _add_features_label(r: BatchResult) -> BatchResult:
    object.__setattr__(r, "n_features_label", str(r.n_drifted))
    return r


# Monkey-patch a display helper onto BatchResult
BatchResult.n_features_label = property(lambda self: str(self.n_drifted))  # type: ignore[attr-defined]


# ── CLI entry point ────────────────────────────────────────────────────────────


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )

    parser = argparse.ArgumentParser(description="PhoenixML drift simulation harness.")
    parser.add_argument(
        "--n-batches",
        type=int,
        default=None,
        help="Number of production batches to simulate (default: all from config)",
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=3.0,
        help="Maximum Gaussian noise sigma applied to V* features on the last batch (default: 3.0)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip Slack alerts and repository_dispatch (safe for local testing)",
    )
    parser.add_argument(
        "--stop-on-alert",
        action="store_true",
        help="Stop after the first retrain trigger",
    )
    args = parser.parse_args()

    run_simulation(
        n_batches=args.n_batches,
        max_noise=args.max_noise,
        dry_run=args.dry_run,
        stop_on_alert=args.stop_on_alert,
    )


if __name__ == "__main__":
    main()
