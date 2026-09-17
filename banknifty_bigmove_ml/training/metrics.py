"""
metrics.py
-----------
What every fold logs (Phase 2 spec): ROC-AUC, PR-AUC (primary, given
imbalance), calibration (Brier score), precision/recall at a few thresholds,
confusion matrix — never just accuracy.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, confusion_matrix,
)

THRESHOLDS = (0.3, 0.5, 0.7)


@dataclass
class FoldMetrics:
    n_samples: int
    positive_rate: float
    roc_auc: float
    pr_auc: float
    brier: float
    precision_at: dict
    recall_at: dict
    confusion_at_0_5: list  # [[tn, fp], [fn, tp]]

    def as_flat_dict(self) -> dict:
        d = {
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier": self.brier,
        }
        for t in THRESHOLDS:
            d[f"precision_at_{t}"] = self.precision_at[t]
            d[f"recall_at_{t}"] = self.recall_at[t]
        return d


def compute_fold_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> FoldMetrics:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(np.unique(y_true)) < 2:
        # degenerate fold (all one class) — AUC undefined; still report what we can
        roc_auc = float("nan")
        pr_auc = float("nan")
    else:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)

    brier = brier_score_loss(y_true, y_prob)

    precision_at, recall_at = {}, {}
    for t in THRESHOLDS:
        preds = (y_prob >= t).astype(int)
        precision_at[t] = precision_score(y_true, preds, zero_division=0)
        recall_at[t] = recall_score(y_true, preds, zero_division=0)

    preds_50 = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, preds_50, labels=[0, 1]).tolist()

    return FoldMetrics(
        n_samples=len(y_true),
        positive_rate=float(y_true.mean()),
        roc_auc=float(roc_auc),
        pr_auc=float(pr_auc),
        brier=float(brier),
        precision_at=precision_at,
        recall_at=recall_at,
        confusion_at_0_5=cm,
    )


@dataclass
class DirectionalStrategyMetrics:
    """The REAL business metric (locked 2026-09-13): given both heads'
    probabilities, take whichever direction is favored above `threshold`,
    else skip — exactly the strategy rule, not a proxy for it. Win rate here
    is directly comparable to the 33.3% breakeven point at 100pt target /
    50pt stop (see triple_barrier_label.py's docstring for that derivation)."""
    n_candidates: int          # total candles evaluated
    n_trades: int              # candles where a trade was actually taken
    n_long: int
    n_short: int
    win_rate: float            # over trades TAKEN only
    long_win_rate: float
    short_win_rate: float
    expected_points_per_trade: float  # win_rate*target - (1-win_rate)*stop


def compute_directional_strategy_metrics(
    p_long: np.ndarray, p_short: np.ndarray,
    long_wins: np.ndarray, short_wins: np.ndarray,
    threshold: float = 0.5, target_pts: float = 100.0, stop_pts: float = 50.0,
) -> DirectionalStrategyMetrics:
    p_long, p_short = np.asarray(p_long), np.asarray(p_short)
    long_wins, short_wins = np.asarray(long_wins), np.asarray(short_wins)

    take_long = (p_long >= threshold) & (p_long > p_short)
    take_short = (p_short >= threshold) & (p_short > p_long)
    # ties (p_long == p_short, both above threshold) are ambiguous -> skip, never guess

    n_long, n_short = int(take_long.sum()), int(take_short.sum())
    n_trades = n_long + n_short

    long_outcomes = long_wins[take_long]
    short_outcomes = short_wins[take_short]
    all_outcomes = np.concatenate([long_outcomes, short_outcomes]) if n_trades else np.array([])

    win_rate = float(all_outcomes.mean()) if n_trades else float("nan")
    long_win_rate = float(long_outcomes.mean()) if n_long else float("nan")
    short_win_rate = float(short_outcomes.mean()) if n_short else float("nan")
    expected_points = win_rate * target_pts - (1 - win_rate) * stop_pts if n_trades else float("nan")

    return DirectionalStrategyMetrics(
        n_candidates=len(p_long), n_trades=n_trades, n_long=n_long, n_short=n_short,
        win_rate=win_rate, long_win_rate=long_win_rate, short_win_rate=short_win_rate,
        expected_points_per_trade=expected_points,
    )
