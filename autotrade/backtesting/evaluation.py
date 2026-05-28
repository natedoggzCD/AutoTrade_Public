"""
Evaluation suite for backtesting with statistical controls integration.

Implements:
- Selection bias and data-snooping controls
- PBO estimation for candidate ranking workflows
- Deflated Sharpe Ratio computation for promoted strategies
- Reality Check / SPA test for multi-strategy comparisons
- Promotion eligibility assessment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)
from autotrade.backtesting.statistical_controls import (
    DeflatedSharpeResult,
    PBOResult,
    SPAResult,
    WhiteRealityCheckResult,
    apply_statistical_controls,
    benjamini_hochberg_correction,
    bonferroni_correction,
    bootstrap_sharpe_inference,
    compute_deflated_sharpe,
    compute_pbo,
    hac_adjusted_sharpe,
    hansens_spa_test,
    whites_reality_check,
)


@dataclass
class StatisticalControlReport:
    """Report from statistical controls applied to backtest results."""

    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    adjusted_metrics: Dict[str, Any] = field(default_factory=dict)

    pbo_estimate: Optional[float] = None
    pbo_passed: bool = False

    deflated_sharpe: Optional[float] = None
    dsr_passed: bool = False

    spa_pvalue: Optional[float] = None
    spa_significant: bool = False

    robust_sharpe_ci_lower: Optional[float] = None
    robust_sharpe_ci_upper: Optional[float] = None
    robust_sharpe_pvalue: Optional[float] = None
    robust_sharpe_significant: bool = False

    wrc_pvalue: Optional[float] = None
    wrc_significant: bool = False

    multiple_testing_method: Optional[str] = None
    n_significant_after_correction: int = 0

    reason_codes: List[str] = field(default_factory=list)
    confidence_decision: str = "pending"

    promotion_eligible: bool = False
    promotion_reason_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_metrics": self.raw_metrics,
            "adjusted_metrics": self.adjusted_metrics,
            "pbo_estimate": self.pbo_estimate,
            "pbo_passed": self.pbo_passed,
            "deflated_sharpe": self.deflated_sharpe,
            "dsr_passed": self.dsr_passed,
            "spa_pvalue": self.spa_pvalue,
            "spa_significant": self.spa_significant,
            "robust_sharpe_ci_lower": self.robust_sharpe_ci_lower,
            "robust_sharpe_ci_upper": self.robust_sharpe_ci_upper,
            "robust_sharpe_pvalue": self.robust_sharpe_pvalue,
            "robust_sharpe_significant": self.robust_sharpe_significant,
            "wrc_pvalue": self.wrc_pvalue,
            "wrc_significant": self.wrc_significant,
            "multiple_testing_method": self.multiple_testing_method,
            "n_significant_after_correction": self.n_significant_after_correction,
            "reason_codes": self.reason_codes,
            "confidence_decision": self.confidence_decision,
            "promotion_eligible": self.promotion_eligible,
            "promotion_reason_code": self.promotion_reason_code,
        }


class EvaluationSuite:
    """
    Evaluation suite with statistical controls integration.

    Implements the EvaluationSuite protocol from interfaces.py.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        min_candidates: int = 20,
        bootstrap_samples: int = 1000,
        enable_pbo: bool = True,
        pbo_min_folds: int = 8,
        robust_sharpe_test: str = "bootstrap",
        multiple_testing_method: str = "dsr+spa",
        min_out_of_sample_trades: int = 30,
    ):
        self.alpha = alpha
        self.min_candidates = min_candidates
        self.bootstrap_samples = bootstrap_samples
        self.enable_pbo = enable_pbo
        self.pbo_min_folds = pbo_min_folds
        self.robust_sharpe_test = robust_sharpe_test
        self.multiple_testing_method = multiple_testing_method
        self.min_out_of_sample_trades = min_out_of_sample_trades

    def compute_metrics(
        self,
        trades: List[Dict[str, Any]],
        equity_curve: Any = None,
    ) -> MetricBundle:
        """Compute evaluation metrics from trades and equity curve."""
        bundle = MetricBundle()

        if not trades:
            return bundle

        bundle.total_trades = len(trades)
        winning = [t for t in trades if t.get("pnl_dollars", 0) > 0]
        losing = [t for t in trades if t.get("pnl_dollars", 0) < 0]

        bundle.winning_trades = len(winning)
        bundle.losing_trades = len(losing)

        pnls = [t.get("pnl_dollars", 0) for t in trades]
        bundle.total_pnl_dollars = sum(pnls)
        bundle.avg_trade_dollars = (
            bundle.total_pnl_dollars / bundle.total_trades
            if bundle.total_trades
            else 0.0
        )

        bundle.win_rate = (
            len(winning) / bundle.total_trades if bundle.total_trades else 0.0
        )

        total_wins = sum(t.get("pnl_dollars", 0) for t in winning) if winning else 0
        total_losses = (
            abs(sum(t.get("pnl_dollars", 0) for t in losing)) if losing else 1
        )
        bundle.profit_factor = total_wins / total_losses if total_losses > 0 else 0.0

        if equity_curve is not None:
            returns = equity_curve.get("returns", [])
            if len(returns) > 1:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns, ddof=1)
                if std_ret > 0:
                    bundle.sharpe_ratio = float(mean_ret / std_ret * np.sqrt(252))

                neg_returns = returns[returns < 0]
                if len(neg_returns) > 1:
                    downside_std = np.std(neg_returns, ddof=1)
                    if downside_std > 0:
                        bundle.sortino_ratio = float(
                            mean_ret / downside_std * np.sqrt(252)
                        )

                cum_ret = equity_curve.get("cumulative_returns", [])
                if len(cum_ret) > 1:
                    peak = cum_ret[0]
                    max_dd = 0.0
                    for ret in cum_ret:
                        if ret > peak:
                            peak = ret
                        dd = (peak - ret) / (1 + peak) if peak > 0 else 0
                        max_dd = max(max_dd, dd)
                    bundle.max_drawdown_pct = float(max_dd * 100)

        bundle.hit_rate = bundle.win_rate
        if winning and losing:
            avg_win = sum(t.get("pnl_dollars", 0) for t in winning) / len(winning)
            avg_loss = abs(sum(t.get("pnl_dollars", 0) for t in losing) / len(losing))
            bundle.expectancy = (
                bundle.win_rate * avg_win - (1 - bundle.win_rate) * avg_loss
            )

        bundle.raw_metrics = bundle.to_dict()

        return bundle

    def apply_statistical_controls_to_folds(
        self,
        folds: List[FoldResult],
    ) -> StatisticalControlReport:
        """Apply statistical controls to fold results."""
        report = StatisticalControlReport()

        if not folds:
            return report

        oos_folds = [f for f in folds if f.is_oos]
        if not oos_folds:
            return report

        report.raw_metrics = {
            "avg_sharpe": np.mean([f.sharpe_ratio for f in oos_folds]).item(),
            "avg_win_rate": np.mean([f.win_rate for f in oos_folds]).item(),
            "total_trades": sum(f.trades_count for f in oos_folds),
            "avg_pnl_pct": np.mean([f.pnl_pct for f in oos_folds]).item(),
        }

        candidate_metrics = []
        for fold in oos_folds:
            returns = []
            for trade in fold.trades:
                ret = trade.get("pnl_pct", 0.0) / 100.0
                returns.append(ret)

            candidate_metrics.append(
                {
                    "sharpe_ratio": fold.sharpe_ratio,
                    "win_rate": fold.win_rate,
                    "returns": returns,
                    "pnl_pct": fold.pnl_pct,
                    "fold_index": fold.fold_index,
                }
            )

        control_results = apply_statistical_controls(
            candidate_metrics=candidate_metrics,
            method=self.multiple_testing_method,
            min_candidates=self.min_candidates,
            bootstrap_samples=self.bootstrap_samples,
            alpha=self.alpha,
            enable_pbo=self.enable_pbo,
            pbo_min_folds=self.pbo_min_folds,
            robust_sharpe_test=self.robust_sharpe_test,
        )

        if control_results.get("pbo"):
            report.pbo_estimate = control_results["pbo"].get("pbo_estimate")
            report.pbo_passed = (
                report.pbo_estimate < 0.6 if report.pbo_estimate else False
            )
            if report.pbo_passed:
                report.reason_codes.append("pbo_acceptable")

        if control_results.get("deflated_sharpe"):
            report.deflated_sharpe = control_results["deflated_sharpe"].get("dsr")
            report.dsr_passed = (
                report.deflated_sharpe > 0.5 if report.deflated_sharpe else False
            )
            if report.dsr_passed:
                report.reason_codes.append("dsr_acceptable")

        if control_results.get("spa"):
            report.spa_pvalue = control_results["spa"].get("pvalue")
            report.spa_significant = (
                report.spa_pvalue < self.alpha if report.spa_pvalue else False
            )
            if report.spa_significant:
                report.reason_codes.append("spa_significant")

        if control_results.get("robust_sharpe"):
            rs = control_results["robust_sharpe"]
            report.robust_sharpe_ci_lower = rs.get("ci_lower")
            report.robust_sharpe_ci_upper = rs.get("ci_upper")
            report.robust_sharpe_pvalue = rs.get("pvalue")
            report.robust_sharpe_significant = (
                rs.get("sharpe", 0) > 0 and rs.get("pvalue", 1) < self.alpha
            )
            if report.robust_sharpe_significant:
                report.reason_codes.append("robust_sharpe_significant")

        if control_results.get("multiple_testing"):
            mt = control_results["multiple_testing"]
            report.multiple_testing_method = mt.get("method")
            report.n_significant_after_correction = mt.get("n_significant", 0)
            if report.n_significant_after_correction > 0:
                report.reason_codes.append("multiple_testing_passed")

        if control_results.get("reason_codes"):
            report.reason_codes = control_results["reason_codes"]

        report.adjusted_metrics = {
            "pbo_estimate": report.pbo_estimate,
            "deflated_sharpe": report.deflated_sharpe,
            "spa_pvalue": report.spa_pvalue,
            "robust_sharpe_ci": [
                report.robust_sharpe_ci_lower,
                report.robust_sharpe_ci_upper,
            ],
            "robust_sharpe_pvalue": report.robust_sharpe_pvalue,
        }

        if report.reason_codes:
            report.confidence_decision = "pass"
        else:
            report.confidence_decision = "fail"

        return report

    def assess_promotion_eligibility(
        self,
        artifact: BacktestResultArtifact,
    ) -> tuple[bool, Optional[str]]:
        """Assess if a strategy is eligible for promotion."""
        if not artifact.metrics:
            return False, "no_metrics"

        metrics = artifact.metrics
        total_trades = metrics.get("total_trades", 0)

        if total_trades < self.min_out_of_sample_trades:
            return False, f"insufficient_trades_{total_trades}"

        sharpe = metrics.get("sharpe_ratio", 0)
        if sharpe <= 0:
            return False, "negative_sharpe"

        max_dd = abs(metrics.get("max_drawdown_pct", 0))
        if max_dd > 20:
            return False, "excessive_drawdown"

        oos_sharpe = metrics.get("out_of_sample_sharpe")
        if oos_sharpe is not None and oos_sharpe < 0.5:
            return False, "poor_oos_sharpe"

        report: Optional[StatisticalControlReport] = None
        if artifact.statistical_report:
            raw_report = artifact.statistical_report
            if isinstance(raw_report, StatisticalControlReport):
                report = raw_report
            elif isinstance(raw_report, dict):
                report = StatisticalControlReport(
                    raw_metrics=raw_report.get("raw_metrics", {}),
                    adjusted_metrics=raw_report.get("adjusted_metrics", {}),
                    pbo_estimate=raw_report.get("pbo_estimate"),
                    pbo_passed=bool(raw_report.get("pbo_passed", False)),
                    deflated_sharpe=raw_report.get("deflated_sharpe"),
                    dsr_passed=bool(raw_report.get("dsr_passed", False)),
                    spa_pvalue=raw_report.get("spa_pvalue"),
                    spa_significant=bool(raw_report.get("spa_significant", False)),
                    robust_sharpe_ci_lower=raw_report.get("robust_sharpe_ci_lower"),
                    robust_sharpe_ci_upper=raw_report.get("robust_sharpe_ci_upper"),
                    robust_sharpe_pvalue=raw_report.get("robust_sharpe_pvalue"),
                    robust_sharpe_significant=bool(
                        raw_report.get("robust_sharpe_significant", False)
                    ),
                    reason_codes=list(raw_report.get("reason_codes", [])),
                    confidence_decision=raw_report.get("confidence_decision", "pending"),
                    promotion_eligible=bool(raw_report.get("promotion_eligible", False)),
                    promotion_reason_code=raw_report.get("promotion_reason_code"),
                )

        if report is not None:

            if not report.pbo_passed and report.pbo_estimate is not None:
                return False, "pbo_too_high"

            if not report.dsr_passed and report.deflated_sharpe is not None:
                return False, "dsr_too_low"

            if (
                not report.robust_sharpe_significant
                and report.robust_sharpe_pvalue is not None
            ):
                return False, "robust_sharpe_not_significant"

            if report.promotion_eligible:
                return True, report.promotion_reason_code

        report_reason_codes = report.reason_codes if report is not None else []
        positive_reasons = [
            r
            for r in ["pbo_acceptable", "dsr_acceptable", "spa_significant"]
            if r in report_reason_codes
        ]

        if len(positive_reasons) >= 2 and sharpe > 1.0:
            return True, "statistical_controls_passed"

        if sharpe > 1.5 and max_dd < 10:
            return True, "strong_performance"

        return False, "did_not_meet_criteria"

    def compare_strategies(
        self,
        results: List[BacktestResultArtifact],
    ) -> Dict[str, Any]:
        """Compare multiple strategy results."""
        if not results:
            return {"rankings": [], "best_strategy": None}

        scored = []
        for artifact in results:
            metrics = artifact.metrics or {}
            sharpe = metrics.get("sharpe_ratio", 0)
            max_dd = abs(metrics.get("max_drawdown_pct", 0))
            total_trades = metrics.get("total_trades", 0)

            score = sharpe * 2 - max_dd / 100 + min(total_trades / 100, 1)
            scored.append((score, artifact))

        scored.sort(key=lambda x: x[0], reverse=True)

        rankings = []
        for rank, (score, artifact) in enumerate(scored, 1):
            rankings.append(
                {
                    "rank": rank,
                    "strategy_id": artifact.strategy_id,
                    "score": score,
                    "sharpe_ratio": artifact.metrics.get("sharpe_ratio", 0),
                    "max_drawdown_pct": artifact.metrics.get("max_drawdown_pct", 0),
                    "total_trades": artifact.metrics.get("total_trades", 0),
                    "promotion_eligible": artifact.promotion_eligible,
                }
            )

        best = scored[0][1] if scored else None

        return {
            "rankings": rankings,
            "best_strategy": best.strategy_id if best else None,
            "best_score": scored[0][0] if scored else None,
        }


def evaluate_artifact_with_controls(
    artifact: BacktestResultArtifact,
    config: Optional[Dict[str, Any]] = None,
) -> BacktestResultArtifact:
    """
    Evaluate a backtest artifact with statistical controls.

    Args:
        artifact: Backtest result artifact to evaluate
        config: Optional configuration for evaluation suite

    Returns:
        Artifact with statistical report and promotion eligibility
    """
    config = config or {}

    suite = EvaluationSuite(
        alpha=config.get("alpha", 0.05),
        min_candidates=config.get("min_candidates", 20),
        bootstrap_samples=config.get("bootstrap_samples", 1000),
        enable_pbo=config.get("enable_pbo", True),
        pbo_min_folds=config.get("pbo_min_folds", 8),
        robust_sharpe_test=config.get("robust_sharpe_test", "bootstrap"),
        multiple_testing_method=config.get("multiple_testing_method", "dsr+spa"),
        min_out_of_sample_trades=config.get("min_out_of_sample_trades", 30),
    )

    report = suite.apply_statistical_controls_to_folds(artifact.folds or [])
    artifact.statistical_report = report

    promotion_eligible, reason_code = suite.assess_promotion_eligibility(artifact)

    report.promotion_eligible = promotion_eligible
    report.promotion_reason_code = reason_code

    artifact.statistical_report = report.to_dict()
    artifact.promotion_eligible = promotion_eligible
    artifact.promotion_reason_code = reason_code

    if artifact.metrics:
        artifact.metrics["raw_metrics"] = report.raw_metrics
        artifact.metrics["adjusted_metrics"] = report.adjusted_metrics
        artifact.metrics["pbo_estimate"] = report.pbo_estimate
        artifact.metrics["deflated_sharpe"] = report.deflated_sharpe
        artifact.metrics["spa_pvalue"] = report.spa_pvalue
        artifact.metrics["statistical_confidence_decision"] = report.confidence_decision
        artifact.metrics["statistical_reason_codes"] = list(report.reason_codes)

    artifact.validation_notes.append(
        f"Statistical controls: {report.confidence_decision}"
    )
    if report.reason_codes:
        artifact.validation_notes.append(
            f"Reason codes: {', '.join(report.reason_codes)}"
        )

    artifact.leakage_check_passed = (
        artifact.leakage_check_passed and report.confidence_decision == "pass"
    )

    return artifact
