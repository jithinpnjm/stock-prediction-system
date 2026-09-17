"""
promotion_gate.py
-------------------
The direct fix for the past failure mode (trusting 1-2 training runs'
results). A candidate config is NOT promoted to the MLflow Model Registry
unless it passes every check below — never on a single good run or a
single good fold.

Usage:
    python3 promotion_gate.py --candidate-run-id <parent_run_id> --baseline-run-id <parent_run_id>
    python3 promotion_gate.py --candidate-run-id <run1> --seed-run-ids <run2> <run3>   # seed-consistency check
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlflow_config import set_tracking  # noqa: E402

import mlflow  # noqa: E402
from mlflow.tracking import MlflowClient  # noqa: E402

PRIMARY_METRIC = "pr_auc"
MIN_FOLD_WIN_RATIO = 0.0  # overridden by --min-fold-win-ratio; see main()
MAX_CROSS_FOLD_STD = 0.15  # overridden by --max-std
MAX_SEED_SPREAD = 0.05     # overridden by --max-seed-spread


@dataclass
class GateResult:
    passed: bool
    reasons: list = field(default_factory=list)

    def add(self, ok: bool, message: str):
        self.reasons.append(("PASS" if ok else "FAIL") + ": " + message)
        if not ok:
            self.passed = False


def get_child_fold_metrics(client: MlflowClient, parent_run_id: str) -> dict:
    """Returns {fold_id: {metric_name: value}} for a parent run's fold children."""
    children = client.search_runs(
        experiment_ids=[client.get_run(parent_run_id).info.experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
    )
    out = {}
    for run in children:
        fold_id = int(run.data.params.get("fold_id"))
        out[fold_id] = run.data.metrics
    return out


def check_all_folds_evaluated(parent_run, fold_metrics: dict) -> tuple[bool, str]:
    n_total = int(parent_run.data.params.get("n_folds_total", -1))
    n_run = len(fold_metrics)
    ok = (n_run == n_total) and n_total > 0
    return ok, f"{n_run}/{n_total} folds evaluated"


def check_beats_baseline(candidate_folds: dict, baseline_folds: dict, min_win_ratio: float) -> tuple[bool, str]:
    shared_folds = sorted(set(candidate_folds) & set(baseline_folds))
    if not shared_folds:
        return False, "no shared fold_ids with baseline run — can't compare"
    wins = sum(
        1 for f in shared_folds
        if candidate_folds[f].get(PRIMARY_METRIC, -1) > baseline_folds[f].get(PRIMARY_METRIC, -1)
    )
    ratio = wins / len(shared_folds)
    ok = ratio >= min_win_ratio
    return ok, f"beat baseline on {wins}/{len(shared_folds)} folds ({ratio:.0%}, need >={min_win_ratio:.0%})"


def check_cross_fold_std(candidate_folds: dict, max_std: float) -> tuple[bool, str]:
    values = [m.get(PRIMARY_METRIC) for m in candidate_folds.values() if m.get(PRIMARY_METRIC) is not None]
    std = float(np.std(values)) if values else float("nan")
    ok = std <= max_std
    return ok, f"cross-fold {PRIMARY_METRIC} std={std:.4f} (max allowed {max_std})"


def check_seed_consistency(client: MlflowClient, run_ids: list, max_spread: float) -> tuple[bool, str]:
    """Same hyperparameter config, different seeds — their AGGREGATE metric
    must agree, or the earlier config is just noise, not a real result."""
    means = []
    for rid in run_ids:
        run = client.get_run(rid)
        mean_metric = run.data.metrics.get(f"{PRIMARY_METRIC}_mean")
        if mean_metric is not None:
            means.append(mean_metric)
    if len(means) < 2:
        return False, f"need >=2 seed runs with logged {PRIMARY_METRIC}_mean, got {len(means)}"
    spread = float(np.max(means) - np.min(means))
    ok = spread <= max_spread
    return ok, f"seed-run {PRIMARY_METRIC}_mean spread={spread:.4f} across {len(means)} seeds (max allowed {max_spread})"


def evaluate_gate(
    candidate_run_id: str,
    baseline_run_id: str | None = None,
    seed_run_ids: list | None = None,
    min_fold_win_ratio: float = 0.75,
    max_cross_fold_std: float = MAX_CROSS_FOLD_STD,
    max_seed_spread: float = MAX_SEED_SPREAD,
) -> GateResult:
    set_tracking()
    client = MlflowClient()
    result = GateResult(passed=True)

    candidate_run = client.get_run(candidate_run_id)
    candidate_folds = get_child_fold_metrics(client, candidate_run_id)

    ok, msg = check_all_folds_evaluated(candidate_run, candidate_folds)
    result.add(ok, msg)

    ok, msg = check_cross_fold_std(candidate_folds, max_cross_fold_std)
    result.add(ok, msg)

    if baseline_run_id:
        baseline_folds = get_child_fold_metrics(client, baseline_run_id)
        ok, msg = check_beats_baseline(candidate_folds, baseline_folds, min_fold_win_ratio)
        result.add(ok, msg)
    else:
        result.add(False, "no --baseline-run-id given — cannot verify candidate beats a reference model")

    if seed_run_ids:
        all_seed_runs = [candidate_run_id] + list(seed_run_ids)
        ok, msg = check_seed_consistency(client, all_seed_runs, max_seed_spread)
        result.add(ok, msg)
    else:
        result.add(False, "no --seed-run-ids given — cannot verify the result reproduces across seeds "
                           "(this is the exact gap that caused the past failure)")

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-run-id", required=True)
    parser.add_argument("--baseline-run-id", default=None)
    parser.add_argument("--seed-run-ids", nargs="*", default=None)
    parser.add_argument("--min-fold-win-ratio", type=float, default=0.75)
    parser.add_argument("--max-std", type=float, default=MAX_CROSS_FOLD_STD)
    parser.add_argument("--max-seed-spread", type=float, default=MAX_SEED_SPREAD)
    args = parser.parse_args()

    result = evaluate_gate(
        candidate_run_id=args.candidate_run_id,
        baseline_run_id=args.baseline_run_id,
        seed_run_ids=args.seed_run_ids,
        min_fold_win_ratio=args.min_fold_win_ratio,
        max_cross_fold_std=args.max_std,
        max_seed_spread=args.max_seed_spread,
    )

    print("=== Promotion gate ===")
    for reason in result.reasons:
        print(f"  {reason}")
    print(f"\nOverall: {'PASS — eligible for registry promotion' if result.passed else 'FAIL — not promoted'}")

    if not result.passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
