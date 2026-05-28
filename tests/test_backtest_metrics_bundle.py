from datetime import date

import pandas as pd

from autotrade.backtesting.contracts import FoldResult, MetricBundle
from autotrade.backtesting.metrics import compute_extended_metrics


def test_metric_bundle_from_fold_results_populates_phase3_fields() -> None:
    folds = [
        FoldResult(
            fold_index=0,
            train_start=date(2025, 1, 1),
            train_end=date(2025, 1, 20),
            test_start=date(2025, 1, 21),
            test_end=date(2025, 1, 30),
            is_oos=True,
            trades_count=2,
            pnl_dollars=100.0,
            pnl_pct=1.0,
            sharpe_ratio=1.2,
            sortino_ratio=1.4,
            max_drawdown_pct=4.0,
            calmar_ratio=0.8,
            turnover=0.15,
            trades=[
                {"pnl_dollars": 80.0, "pnl_pct": 1.6},
                {"pnl_dollars": 20.0, "pnl_pct": 0.4},
            ],
            metrics={"train": {"sharpe_ratio": 1.0}},
        ),
        FoldResult(
            fold_index=1,
            train_start=date(2025, 1, 11),
            train_end=date(2025, 1, 30),
            test_start=date(2025, 1, 31),
            test_end=date(2025, 2, 9),
            is_oos=True,
            trades_count=2,
            pnl_dollars=-40.0,
            pnl_pct=-0.4,
            sharpe_ratio=0.6,
            sortino_ratio=0.9,
            max_drawdown_pct=6.0,
            calmar_ratio=0.5,
            turnover=0.25,
            trades=[
                {"pnl_dollars": -50.0, "pnl_pct": -1.0},
                {"pnl_dollars": 10.0, "pnl_pct": 0.2},
            ],
            metrics={"train": {"sharpe_ratio": 0.7}},
        ),
    ]

    bundle = MetricBundle.from_fold_results(folds)

    assert bundle.total_trades == 4
    assert bundle.winning_trades == 3
    assert bundle.losing_trades == 1
    assert bundle.win_rate == 0.75
    assert bundle.hit_rate == bundle.win_rate
    assert bundle.sharpe_ratio > 0
    assert bundle.sortino_ratio > 0
    assert bundle.max_drawdown_pct == 6.0
    assert bundle.turnover > 0
    assert bundle.expectancy != 0
    assert bundle.raw_metrics["fold_count"] == 2
    assert len(bundle.fold_metrics) == 2


def test_compute_extended_metrics_includes_precision_recall_and_capacity() -> None:
    trades = [
        {"pnl_dollars": 100.0, "pnl_pct": 1.0},
        {"pnl_dollars": -20.0, "pnl_pct": -0.2},
        {"pnl_dollars": 30.0, "pnl_pct": 0.3},
    ]
    equity_curve = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-06"]),
            "equity": [100000.0, 100100.0, 100080.0, 100110.0],
        }
    )
    predictions = [True, False, True]

    metrics = compute_extended_metrics(
        trades=trades,
        equity_curve=equity_curve,
        initial_cash=100_000.0,
        predictions=predictions,
    )

    assert "precision" in metrics and "recall" in metrics
    assert "turnover" in metrics and "capacity_proxy" in metrics
    assert "annual_return_pct" in metrics and "annual_volatility_pct" in metrics
    assert metrics["total_trades"] == 3
