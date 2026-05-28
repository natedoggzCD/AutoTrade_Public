from datetime import date

from autotrade.backtesting.contracts import BacktestResultArtifact, FoldResult
from autotrade.backtesting.evaluation import evaluate_artifact_with_controls
from autotrade.backtesting.statistical_controls import apply_statistical_controls


def _sample_fold(idx: int = 0) -> FoldResult:
    return FoldResult(
        fold_index=idx,
        train_start=date(2025, 1, 1),
        train_end=date(2025, 1, 10),
        test_start=date(2025, 1, 11),
        test_end=date(2025, 1, 20),
        is_oos=True,
        trades_count=10,
        pnl_dollars=250.0,
        pnl_pct=1.2,
        sharpe_ratio=1.1,
        win_rate=0.6,
        trades=[
            {"pnl_pct": 1.0, "pnl_dollars": 100.0},
            {"pnl_pct": -0.5, "pnl_dollars": -50.0},
            {"pnl_pct": 0.7, "pnl_dollars": 70.0},
        ],
    )


def test_evaluate_artifact_with_controls_attaches_report_and_metrics() -> None:
    artifact = BacktestResultArtifact(
        strategy_id="phase4_demo",
        start_date="2025-01-01",
        end_date="2025-01-31",
        metrics={"total_trades": 50, "sharpe_ratio": 1.3, "max_drawdown_pct": 8.0},
        folds=[_sample_fold()],
    )

    out = evaluate_artifact_with_controls(
        artifact,
        config={"bootstrap_samples": 20, "min_out_of_sample_trades": 10},
    )

    assert isinstance(out.statistical_report, dict)
    assert "raw_metrics" in out.statistical_report
    assert "adjusted_metrics" in out.statistical_report
    assert "statistical_confidence_decision" in (out.metrics or {})
    assert "statistical_reason_codes" in (out.metrics or {})
    assert out.promotion_reason_code is not None


def test_evaluate_artifact_without_preexisting_report_does_not_crash() -> None:
    artifact = BacktestResultArtifact(
        strategy_id="phase4_no_report",
        start_date="2025-02-01",
        end_date="2025-02-28",
        metrics={"total_trades": 40, "sharpe_ratio": 1.6, "max_drawdown_pct": 7.0},
        folds=[_sample_fold()],
    )

    out = evaluate_artifact_with_controls(
        artifact,
        config={"bootstrap_samples": 20, "min_out_of_sample_trades": 10},
    )

    assert isinstance(out.promotion_eligible, bool)
    assert out.promotion_reason_code is not None


def test_apply_statistical_controls_hac_path() -> None:
    candidate_metrics = [
        {
            "pvalue": 0.03,
            "in_sample_sharpe": 1.1,
            "out_of_sample_sharpe": 0.9,
            "sharpe_ratio": 0.9,
            "returns": [0.01, -0.005, 0.008, 0.002, -0.003],
        },
        {
            "pvalue": 0.04,
            "in_sample_sharpe": 1.0,
            "out_of_sample_sharpe": 0.8,
            "sharpe_ratio": 0.8,
            "returns": [0.009, -0.004, 0.007, 0.001, -0.002],
        },
    ]

    out = apply_statistical_controls(
        candidate_metrics=candidate_metrics,
        method="dsr+spa",
        min_candidates=2,
        bootstrap_samples=20,
        alpha=0.05,
        enable_pbo=True,
        pbo_min_folds=2,
        robust_sharpe_test="hac",
    )

    assert out["robust_sharpe"] is not None
    assert out["robust_sharpe"]["test_type"] == "hac"
