"""Purged k-fold walk-forward with embargo (López de Prado, AFML ch. 7).

Composes with existing statistical_controls.compute_deflated_sharpe and
leakage_detector to give an honest evaluation pipeline for the alpha catalog.

The current strategy lab uses simple chronological folds with no purging,
which leaks information across boundaries when label windows (e.g. max_hold_days)
overlap the test region. This module produces leak-free fold boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    """A single train/test split with purged train indices and embargoed buffer.

    train_mask and test_mask are boolean arrays indexed against the input
    timeline. observed labels in the training set whose label-window overlaps
    the test window are excluded (purged). An additional N-day embargo after
    the test window is also excluded from training to defeat serial correlation.
    """

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    embargo_end: date
    train_mask: np.ndarray
    test_mask: np.ndarray

    def n_train(self) -> int:
        return int(self.train_mask.sum())

    def n_test(self) -> int:
        return int(self.test_mask.sum())


@dataclass
class PurgedKFoldConfig:
    n_splits: int = 5
    label_horizon_days: int = 10
    embargo_days: int = 5
    min_train_days: int = 252

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if self.label_horizon_days < 0:
            raise ValueError("label_horizon_days must be >= 0")
        if self.embargo_days < 0:
            raise ValueError("embargo_days must be >= 0")


def _as_dates(timeline: Sequence) -> np.ndarray:
    arr = pd.to_datetime(pd.Series(list(timeline))).dt.date.to_numpy()
    if not np.all(arr[:-1] <= arr[1:]):
        raise ValueError("timeline must be sorted non-decreasing")
    return arr


def generate_purged_folds(
    timeline: Sequence,
    config: Optional[PurgedKFoldConfig] = None,
) -> List[PurgedFold]:
    """Produce n_splits walk-forward folds with purge + embargo.

    Test windows are contiguous and non-overlapping, advancing through the
    timeline in order. For each test window, training data is every prior bar
    EXCEPT those whose label window (entry_date + label_horizon_days) lands
    within [test_start - label_horizon, embargo_end].

    Args:
        timeline: ordered sequence of dates (one per bar).
        config: purged k-fold configuration.

    Returns:
        list of PurgedFold. Folds with insufficient training data
        (< min_train_days) are dropped.
    """

    cfg = config or PurgedKFoldConfig()
    dates = _as_dates(timeline)
    n = len(dates)
    if n < cfg.n_splits * 2:
        return []

    # Equal-size test slices from the back, leaving room for an initial train block.
    test_size = n // (cfg.n_splits + 1)
    if test_size < 1:
        return []

    folds: List[PurgedFold] = []
    for k in range(cfg.n_splits):
        test_lo = n - (cfg.n_splits - k) * test_size
        test_hi = test_lo + test_size
        if test_lo < cfg.min_train_days or test_hi > n:
            continue

        test_start = dates[test_lo]
        test_end = dates[test_hi - 1]
        embargo_end = test_end + timedelta(days=cfg.embargo_days)

        # Purge boundary: any train bar whose label window touches the test
        # window must be dropped. Label window for a bar at date d covers
        # (d, d + label_horizon_days]. So purge train bars in
        # [test_start - label_horizon, embargo_end].
        purge_lo = test_start - timedelta(days=cfg.label_horizon_days)
        purge_hi = embargo_end

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_lo:test_hi] = True

        train_mask = np.zeros(n, dtype=bool)
        train_mask[:test_lo] = True
        # purge by date, not by index — bar dates aren't necessarily contiguous
        train_dates = dates
        purge_zone = (train_dates >= purge_lo) & (train_dates <= purge_hi)
        train_mask &= ~purge_zone

        if int(train_mask.sum()) < cfg.min_train_days:
            continue

        train_indices = np.flatnonzero(train_mask)
        train_start = dates[train_indices[0]]
        train_end = dates[train_indices[-1]]

        folds.append(
            PurgedFold(
                fold_index=k,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                embargo_end=embargo_end,
                train_mask=train_mask,
                test_mask=test_mask,
            )
        )

    return folds


def fold_boundaries_for_leakage_check(folds: Sequence[PurgedFold]) -> List[dict]:
    """Adapt PurgedFold list to the dict format expected by LeakageDetector."""
    return [
        {
            "train_start": f.train_start,
            "train_end": f.train_end,
            "test_start": f.test_start,
            "test_end": f.test_end,
        }
        for f in folds
    ]


@dataclass
class FoldEvaluation:
    """Per-fold evaluation result."""

    fold_index: int
    n_train_trades: int
    n_test_trades: int
    train_sharpe: float
    test_sharpe: float
    train_pf: float
    test_pf: float
    test_returns: List[float] = field(default_factory=list)


def evaluate_returns_on_folds(
    fold_results: Sequence[FoldEvaluation],
) -> dict:
    """Aggregate per-fold test metrics + report train-vs-test degradation.

    Degradation is the proportional drop from train to test Sharpe. High
    degradation indicates overfitting.
    """
    if not fold_results:
        return {
            "n_folds": 0,
            "mean_test_sharpe": 0.0,
            "median_test_sharpe": 0.0,
            "mean_test_pf": 0.0,
            "degradation_pct": float("nan"),
            "all_test_returns": [],
        }

    train_sharpes = np.array([f.train_sharpe for f in fold_results])
    test_sharpes = np.array([f.test_sharpe for f in fold_results])
    test_pfs = np.array([f.test_pf for f in fold_results])
    all_test_returns: List[float] = []
    for f in fold_results:
        all_test_returns.extend(f.test_returns)

    mean_train = float(np.nanmean(train_sharpes)) if train_sharpes.size else 0.0
    mean_test = float(np.nanmean(test_sharpes)) if test_sharpes.size else 0.0
    if mean_train > 0:
        degradation = max(0.0, (mean_train - mean_test) / mean_train)
    else:
        degradation = float("nan")

    return {
        "n_folds": len(fold_results),
        "mean_train_sharpe": mean_train,
        "mean_test_sharpe": mean_test,
        "median_test_sharpe": float(np.nanmedian(test_sharpes)),
        "mean_test_pf": float(np.nanmean(test_pfs)) if test_pfs.size else 0.0,
        "degradation_pct": degradation,
        "all_test_returns": all_test_returns,
    }
