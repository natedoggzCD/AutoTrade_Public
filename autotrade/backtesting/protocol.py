from __future__ import annotations

import logging
from math import exp
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)
from autotrade.backtesting.evaluation import (
    EvaluationSuite,
    evaluate_artifact_with_controls,
)
from autotrade.backtesting.interfaces import generate_run_id
from autotrade.backtesting.engine import BacktestEngine
from autotrade.backtesting.leakage_detector import (
    LeakageDetector,
    create_leakage_detector,
)
from autotrade.backtesting.artifact_persistence import (
    ArtifactPersister,
    create_persister,
)

logger = logging.getLogger(__name__)


class BacktestProtocolError(Exception):
    """Exception raised for protocol violations."""

    pass


@dataclass
class NestedValidationResult:
    """Result from nested cross-validation."""

    fold_index: int
    inner_train_start: date
    inner_train_end: date
    inner_test_start: date
    inner_test_end: date
    best_params: Dict[str, Any]
    inner_sharpe: float
    metrics: Dict[str, Any] = field(default_factory=dict)


class BacktestProtocol:
    """
    Canonical backtest protocol enforcing:
    - Walk-forward structure
    - Optional nested validation
    - Out-of-sample fold accounting
    - Execution/cost model wiring through Stage 5 adapters
    """

    def __init__(
        self,
        request: BacktestRequest,
        engine: Optional[BacktestEngine] = None,
        config_override: Optional[Dict[str, Any]] = None,
    ):
        self.request = request
        self.config_override = config_override or {}

        self._engine = engine
        self._folds: List[FoldResult] = []
        self._nested_results: List[NestedValidationResult] = []

        self._validate_request()

    def _validate_request(self) -> None:
        """Validate the backtest request configuration."""
        errors = []

        if self.request.train_days <= 0:
            errors.append("train_days must be positive")
        if self.request.test_days <= 0:
            errors.append("test_days must be positive")
        if self.request.start_date >= self.request.end_date:
            errors.append("start_date must be before end_date")
        if self.request.initial_cash <= 0:
            errors.append("initial_cash must be positive")
        if self.request.nested_validation_enabled and self.request.inner_folds < 2:
            errors.append("inner_folds must be >= 2 when nested validation is enabled")

        if errors:
            raise BacktestProtocolError(
                f"Request validation failed: {', '.join(errors)}"
            )

    @property
    def engine(self) -> BacktestEngine:
        """Get or create the backtest engine."""
        if self._engine is None:
            exec_config = self._build_execution_config()
            override = {
                "commission_pct": exec_config.get(
                    "commission_pct", self.request.commission_pct
                ),
                "slippage_pct": exec_config.get(
                    "slippage_pct", self.request.slippage_pct
                ),
                "spread_pct": exec_config.get("spread_pct", self.request.spread_pct),
                "seed": self.request.seed,
                "initial_cash": self.request.initial_cash,
                "max_positions": self.request.max_positions,
                "max_position_value": self.request.max_position_value,
                "max_hold_days": self.request.max_hold_days,
                "walk_forward_train_days": self.request.train_days,
                "walk_forward_test_days": self.request.test_days,
            }
            override.update(self.config_override)
            self._engine = BacktestEngine(config_override=override)
        return self._engine

    def _build_execution_config(self) -> Dict[str, Any]:
        """Build execution configuration from request."""
        return {
            "commission_pct": self.request.commission_pct,
            "slippage_pct": self.request.slippage_pct,
            "spread_pct": self.request.spread_pct,
            "seed": self.request.seed,
        }

    def _build_candidate_metrics(
        self, folds: Sequence[FoldResult]
    ) -> List[Dict[str, Any]]:
        """Build candidate metrics payload for statistical controls."""
        candidates: List[Dict[str, Any]] = []
        for fold in folds:
            trades = fold.trades or []
            returns: List[float] = []
            for trade in trades:
                if "pnl_pct" in trade and trade.get("pnl_pct") is not None:
                    returns.append(float(trade.get("pnl_pct", 0.0)) / 100.0)
                elif self.request.initial_cash > 0:
                    returns.append(
                        float(trade.get("pnl_dollars", 0.0) or 0.0)
                        / float(self.request.initial_cash)
                    )

            train_sharpe = float(
                (fold.metrics or {}).get("train", {}).get("sharpe_ratio", 0.0) or 0.0
            )
            oos_sharpe = float(fold.sharpe_ratio or 0.0)
            candidates.append(
                {
                    "pvalue": float(exp(-abs(oos_sharpe))),
                    "in_sample_sharpe": train_sharpe,
                    "out_of_sample_sharpe": oos_sharpe,
                    "sharpe_ratio": oos_sharpe,
                    "returns": returns,
                }
            )
        return candidates

    def _apply_phase3_controls(
        self, metrics: MetricBundle, folds: Sequence[FoldResult]
    ) -> Dict[str, Any]:
        """
        Apply statistical controls and attach adjusted results to MetricBundle.

        Returns control result payload (or empty dict if unavailable).
        """
        try:
            from autotrade.backtesting.statistical_controls import (
                apply_statistical_controls,
            )
            from config.config_loader import get_backtest_protocol_config
        except Exception:
            return {}

        protocol_cfg = get_backtest_protocol_config()
        mt_cfg = protocol_cfg.multiple_testing_control
        sb_cfg = protocol_cfg.selection_bias_control
        inf_cfg = protocol_cfg.inference

        candidates = self._build_candidate_metrics(folds)
        if not candidates:
            return {}

        controls = apply_statistical_controls(
            candidate_metrics=candidates,
            method=mt_cfg.method,
            min_candidates=mt_cfg.min_candidate_count,
            bootstrap_samples=mt_cfg.bootstrap_samples,
            alpha=mt_cfg.alpha,
            enable_pbo=sb_cfg.enable_pbo,
            pbo_min_folds=sb_cfg.pbo_min_folds,
            robust_sharpe_test=inf_cfg.robust_sharpe_test,
        )

        metrics.raw_metrics = dict(metrics.raw_metrics or {})
        metrics.adjusted_metrics = dict(metrics.adjusted_metrics or {})
        metrics.raw_metrics["phase3_controls_config"] = {
            "method": mt_cfg.method,
            "min_candidate_count": mt_cfg.min_candidate_count,
            "bootstrap_samples": mt_cfg.bootstrap_samples,
            "alpha": mt_cfg.alpha,
            "enable_pbo": sb_cfg.enable_pbo,
            "pbo_min_folds": sb_cfg.pbo_min_folds,
            "robust_sharpe_test": inf_cfg.robust_sharpe_test,
            "robust_sharpe_min_pvalue": inf_cfg.robust_sharpe_min_pvalue,
        }
        metrics.adjusted_metrics["statistical_controls"] = controls

        pbo_result = controls.get("pbo") or {}
        dsr_result = controls.get("deflated_sharpe") or {}
        spa_result = controls.get("spa") or {}
        if "pbo_estimate" in pbo_result:
            metrics.pbo_estimate = float(pbo_result["pbo_estimate"])
        if "dsr" in dsr_result:
            metrics.deflated_sharpe = float(dsr_result["dsr"])
        if "pvalue" in spa_result:
            metrics.spa_pvalue = float(spa_result["pvalue"])

        return controls

    def _generate_fold_dates(
        self,
        start_date: date,
        end_date: date,
        train_days: Optional[int] = None,
        test_days: Optional[int] = None,
        max_folds: int = 100,
    ) -> List[Dict[str, date]]:
        """Generate walk-forward fold date windows."""
        train_days = train_days if train_days is not None else self.request.train_days
        test_days = test_days if test_days is not None else self.request.test_days

        current = start_date
        folds = []
        fold_idx = 0

        while True:
            train_start = current
            train_end = current + timedelta(days=train_days - 1)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_days - 1)

            if test_end > end_date:
                break

            folds.append(
                {
                    "fold_index": fold_idx,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )

            current = test_start
            fold_idx += 1

            if fold_idx >= max_folds:
                logger.warning(
                    "Fold limit reached (%d), stopping walk-forward", max_folds
                )
                break

        return folds

    def _run_single_fold(
        self,
        fold_config: Dict[str, date],
        symbols: Sequence[str],
        is_oos: bool = True,
    ) -> FoldResult:
        """Execute a single walk-forward fold."""
        train_start = fold_config["train_start"]
        train_end = fold_config["train_end"]
        test_start = fold_config["test_start"]
        test_end = fold_config["test_end"]

        train_result = self.engine._run_single_period(
            symbols=symbols,
            start_date=train_start,
            end_date=train_end,
            save_path=None,
            log_summary=False,
        )

        test_result = self.engine._run_single_period(
            symbols=symbols,
            start_date=test_start,
            end_date=test_end,
            save_path=None,
            log_summary=False,
        )

        fold_result = FoldResult(
            fold_index=fold_config["fold_index"],
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            is_oos=is_oos,
            trades_count=test_result.get("total_trades", 0),
            pnl_dollars=test_result.get("total_pnl", 0.0),
            pnl_pct=test_result.get("total_return_pct", 0.0),
            sharpe_ratio=test_result.get("sharpe_ratio", 0.0),
            sortino_ratio=test_result.get("sortino_ratio", 0.0),
            max_drawdown_pct=test_result.get("max_drawdown", 0.0) * 100.0,
            calmar_ratio=test_result.get("calmar_ratio", 0.0),
            win_rate=test_result.get("win_rate", 0.0),
            profit_factor=test_result.get("profit_factor", 0.0),
            avg_trade_dollars=test_result.get("avg_trade_pnl", 0.0),
            turnover=test_result.get("turnover", 0.0),
            trades=test_result.get("trades", []),
            metrics={
                "train": {
                    "total_trades": train_result.get("total_trades", 0),
                    "win_rate": train_result.get("win_rate", 0.0),
                    "profit_factor": train_result.get("profit_factor", 0.0),
                    "sharpe_ratio": train_result.get("sharpe_ratio", 0.0),
                },
                "test": {
                    "total_trades": test_result.get("total_trades", 0),
                    "win_rate": test_result.get("win_rate", 0.0),
                    "profit_factor": test_result.get("profit_factor", 0.0),
                    "sharpe_ratio": test_result.get("sharpe_ratio", 0.0),
                },
            },
        )

        return fold_result

    def _run_nested_validation(
        self,
        fold_config: Dict[str, date],
        symbols: Sequence[str],
    ) -> NestedValidationResult:
        """Run nested cross-validation within a fold."""
        train_start = fold_config["train_start"]
        train_end = fold_config["train_end"]

        available_days = max(1, (train_end - train_start).days + 1)
        inner_train_days = max(5, available_days // (self.request.inner_folds + 1))
        inner_test_days = max(2, inner_train_days // 3)
        inner_folds = self._generate_fold_dates(
            train_start,
            train_end,
            train_days=inner_train_days,
            test_days=inner_test_days,
            max_folds=self.request.inner_folds,
        )

        best_params = {}
        best_sharpe = float("-inf")

        for inner_fold in inner_folds:
            inner_result = self.engine._run_single_period(
                symbols=symbols,
                start_date=inner_fold["test_start"],
                end_date=inner_fold["test_end"],
                save_path=None,
                log_summary=False,
            )

            sharpe = inner_result.get("sharpe_ratio", 0.0)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = inner_result.get("config", {}).get("backtest", {})

        if best_sharpe == float("-inf"):
            best_sharpe = 0.0

        return NestedValidationResult(
            fold_index=fold_config["fold_index"],
            inner_train_start=train_start,
            inner_train_end=train_end,
            inner_test_start=inner_folds[0]["test_start"] if inner_folds else train_end,
            inner_test_end=inner_folds[-1]["test_end"] if inner_folds else train_end,
            best_params=best_params,
            inner_sharpe=best_sharpe,
            metrics={
                "requested_inner_folds": self.request.inner_folds,
                "evaluated_inner_folds": len(inner_folds),
                "inner_train_days": inner_train_days,
                "inner_test_days": inner_test_days,
            },
        )

    def run_walk_forward(
        self,
        symbols: Sequence[str],
    ) -> BacktestResultArtifact:
        """
        Execute walk-forward backtesting with protocol enforcement.

        Args:
            symbols: List of symbols to backtest

        Returns:
            Complete backtest result artifact with fold-level diagnostics
        """
        start_date = self.request.start_date
        end_date = self.request.end_date

        fold_configs = self._generate_fold_dates(start_date, end_date)

        if not fold_configs:
            raise BacktestProtocolError(
                f"Not enough data for walk-forward: "
                f"train_days={self.request.train_days}, test_days={self.request.test_days}"
            )

        self._folds = []
        self._nested_results = []

        for fold_config in fold_configs:
            logger.info(
                "Running fold %d: train=%s to %s, test=%s to %s",
                fold_config["fold_index"],
                fold_config["train_start"],
                fold_config["train_end"],
                fold_config["test_start"],
                fold_config["test_end"],
            )

            if self.request.nested_validation_enabled:
                nested_result = self._run_nested_validation(fold_config, symbols)
                self._nested_results.append(nested_result)

            fold_result = self._run_single_fold(fold_config, symbols, is_oos=True)
            self._folds.append(fold_result)

        metrics = MetricBundle.from_fold_results(self._folds)
        controls = self._apply_phase3_controls(metrics, self._folds)

        config_hash = self.request.to_config_hash()
        timestamp = pd.Timestamp.now().isoformat()
        run_id = generate_run_id(self.request.strategy_id, config_hash, timestamp)

        leakage_detector = create_leakage_detector()
        fold_dicts = [
            {
                "train_start": f.train_start,
                "train_end": f.train_end,
                "test_start": f.test_start,
                "test_end": f.test_end,
            }
            for f in self._folds
        ]
        leakage_report = leakage_detector.run_all_checks(folds=fold_dicts)

        min_oos_trades = 30
        try:
            from config.config_loader import get_backtest_protocol_config

            min_oos_trades = int(
                get_backtest_protocol_config().paper_gate.min_out_of_sample_trades
            )
        except Exception:
            pass
        passed_controls = bool((controls or {}).get("passed_controls"))
        leakage_passed = leakage_report.passed and leakage_report.error_count == 0
        promotion_eligible = (
            passed_controls
            and leakage_passed
            and int(metrics.total_trades) >= min_oos_trades
        )
        if promotion_eligible:
            promotion_reason_code = "phase3_controls_passed"
        elif int(metrics.total_trades) < min_oos_trades:
            promotion_reason_code = "insufficient_oos_trades"
        elif not leakage_passed:
            promotion_reason_code = "leakage_detected"
        else:
            promotion_reason_code = "phase3_controls_failed"
        validation_notes = list((controls or {}).get("reason_codes") or [])
        if leakage_report.warnings:
            validation_notes.extend([f"leakage: {w}" for w in leakage_report.warnings])

        persister = create_persister(strategy_id=self.request.strategy_id)
        try:
            artifact_paths = persister.persist_artifact(
                artifact=None,
                run_id=run_id,
                timestamp=timestamp,
                config_hash=config_hash,
                save_plots=True,
            )
        except Exception as e:
            logger.warning(f"Failed to persist artifacts: {e}")
            artifact_paths = None

        artifact = BacktestResultArtifact(
            schema_version="1.0",
            config_hash=config_hash,
            run_id=run_id,
            run_timestamp=timestamp,
            strategy_id=self.request.strategy_id,
            start_date=str(start_date),
            end_date=str(end_date),
            request=self.request.to_dict(),
            metrics=metrics.to_dict(),
            folds=self._folds,
            execution_config=self._build_execution_config(),
            cost_sensitivity={
                "commission_pct": self.request.commission_pct,
                "slippage_pct": self.request.slippage_pct,
                "spread_pct": self.request.spread_pct,
            },
            leakage_check_passed=leakage_passed,
            leakage_warnings=leakage_report.warnings,
            promotion_eligible=promotion_eligible,
            promotion_reason_code=promotion_reason_code,
            validation_notes=validation_notes,
            equity_curve_path=str(artifact_paths.equity_plot)
            if artifact_paths and artifact_paths.equity_plot.exists()
            else None,
            drawdown_plot_path=str(artifact_paths.drawdown_plot)
            if artifact_paths and artifact_paths.drawdown_plot.exists()
            else None,
        )

        if artifact_paths:
            try:
                persister.persist_artifact(
                    artifact=artifact,
                    run_id=run_id,
                    timestamp=timestamp,
                    config_hash=config_hash,
                    save_plots=True,
                )
            except Exception as e:
                logger.warning(f"Failed to persist artifact: {e}")

        try:
            from autotrade.backtesting.results import RunRegistry

            RunRegistry().register_run(artifact)
        except Exception as e:
            logger.warning(f"Failed to register run in registry: {e}")

        return artifact

    def run_single_period(
        self,
        symbols: Sequence[str],
    ) -> BacktestResultArtifact:
        """
        Execute single-period backtest (non walk-forward).

        Args:
            symbols: List of symbols to backtest

        Returns:
            Backtest result artifact
        """
        result = self.engine._run_single_period(
            symbols=symbols,
            start_date=self.request.start_date,
            end_date=self.request.end_date,
            save_path=None,
            log_summary=False,
        )

        fold_result = FoldResult(
            fold_index=0,
            train_start=self.request.start_date,
            train_end=self.request.start_date,
            test_start=self.request.start_date,
            test_end=self.request.end_date,
            is_oos=True,
            trades_count=result.get("total_trades", 0),
            pnl_dollars=result.get("total_pnl", 0.0),
            pnl_pct=result.get("total_return_pct", 0.0),
            sharpe_ratio=result.get("sharpe_ratio", 0.0),
            sortino_ratio=result.get("sortino_ratio", 0.0),
            max_drawdown_pct=result.get("max_drawdown", 0.0) * 100.0,
            calmar_ratio=result.get("calmar_ratio", 0.0),
            win_rate=result.get("win_rate", 0.0),
            profit_factor=result.get("profit_factor", 0.0),
            avg_trade_dollars=result.get("avg_trade_pnl", 0.0),
            turnover=result.get("turnover", 0.0),
            trades=result.get("trades", []),
            metrics={"single_period": result},
        )

        metrics = MetricBundle.from_fold_results([fold_result])
        controls = self._apply_phase3_controls(metrics, [fold_result])

        config_hash = self.request.to_config_hash()
        timestamp = pd.Timestamp.now().isoformat()
        run_id = generate_run_id(self.request.strategy_id, config_hash, timestamp)

        leakage_detector = create_leakage_detector()
        fold_dicts = [
            {
                "train_start": fold_result.train_start,
                "train_end": fold_result.train_end,
                "test_start": fold_result.test_start,
                "test_end": fold_result.test_end,
            }
        ]
        leakage_report = leakage_detector.run_all_checks(folds=fold_dicts)
        leakage_passed = leakage_report.passed and leakage_report.error_count == 0
        validation_notes = list((controls or {}).get("reason_codes") or [])
        if leakage_report.warnings:
            validation_notes.extend([f"leakage: {w}" for w in leakage_report.warnings])

        persister = create_persister(strategy_id=self.request.strategy_id)
        artifact = BacktestResultArtifact(
            schema_version="1.0",
            config_hash=config_hash,
            run_id=run_id,
            run_timestamp=timestamp,
            strategy_id=self.request.strategy_id,
            start_date=str(self.request.start_date),
            end_date=str(self.request.end_date),
            request=self.request.to_dict(),
            metrics=metrics.to_dict(),
            folds=[fold_result],
            execution_config=self._build_execution_config(),
            leakage_check_passed=leakage_passed,
            leakage_warnings=leakage_report.warnings,
            promotion_eligible=bool((controls or {}).get("passed_controls"))
            and leakage_passed,
            promotion_reason_code=(
                "phase3_controls_passed"
                if bool((controls or {}).get("passed_controls")) and leakage_passed
                else "phase3_controls_failed"
                if not leakage_passed
                else "leakage_detected"
            ),
            validation_notes=validation_notes,
        )

        try:
            from autotrade.backtesting.results import RunRegistry

            RunRegistry().register_run(artifact)
        except Exception as e:
            logger.warning(f"Failed to register run in registry: {e}")

        return artifact


class ProtocolAdapter:
    """
    Adapter to route legacy engine entry points through protocol.

    Preserves backward compatibility with existing code while enabling
    protocol-enforced backtesting.
    """

    @staticmethod
    def run_backtest_legacy(
        engine: BacktestEngine,
        symbols: Sequence[str],
        lookback_days: Optional[int] = 90,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
        walk_forward: bool = False,
    ) -> Dict[str, Any]:
        """
        Legacy backtest entry point routed through protocol.

        Uses protocol execution when an explicit date range is provided.
        Falls back to engine.run for legacy lookback-based calls.
        """
        if start_date is None or end_date is None:
            return engine.run(
                symbols=symbols,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                walk_forward=walk_forward,
            )

        request = BacktestRequest(
            strategy_id="legacy_engine",
            start_date=pd.to_datetime(start_date).date(),
            end_date=pd.to_datetime(end_date).date(),
            initial_cash=float(getattr(engine.config, "initial_cash", 100_000.0)),
            train_days=int(getattr(engine.config, "walk_forward_train_days", 60)),
            test_days=int(getattr(engine.config, "walk_forward_test_days", 20)),
            commission_pct=float(getattr(engine.config, "commission_pct", 0.001)),
            slippage_pct=float(getattr(engine.config, "slippage_pct", 0.0005)),
            spread_pct=float(getattr(engine.config, "spread_pct", 0.001)),
            seed=int(getattr(engine.config, "seed", 42)),
            max_positions=int(getattr(engine.config, "max_positions", 10)),
            max_position_value=float(
                getattr(engine.config, "max_position_value", 10_000.0)
            ),
            max_hold_days=int(getattr(engine.config, "max_hold_days", 5)),
        )
        protocol = BacktestProtocol(request=request, engine=engine)
        artifact = (
            protocol.run_walk_forward(symbols)
            if walk_forward
            else protocol.run_single_period(symbols)
        )
        return artifact.to_dict()

    @staticmethod
    def run_walk_forward_legacy(
        engine: BacktestEngine,
        symbols: Sequence[str],
        lookback_days: Optional[int] = 360,
        start_date: Optional[Any] = None,
        end_date: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Legacy walk-forward entry point.

        Uses protocol execution for explicit date ranges and falls back
        to engine.walk_forward when a lookback-only call is used.
        """
        return ProtocolAdapter.run_backtest_legacy(
            engine=engine,
            symbols=symbols,
            lookback_days=lookback_days,
            start_date=start_date,
            end_date=end_date,
            walk_forward=True,
        )


def create_protocol_from_config(
    config: Dict[str, Any],
    **kwargs,
) -> BacktestProtocol:
    """
    Create a backtest protocol from configuration dictionary.

    Args:
        config: Configuration dictionary (from trading_config.yaml or similar)
        **kwargs: Additional overrides

    Returns:
        Configured BacktestProtocol instance
    """
    from datetime import date as dt_date

    backtest_cfg = config.get("backtest", config)
    protocol_cfg = config.get("backtest_protocol", {})
    walk_forward_cfg = protocol_cfg.get("walk_forward", {})
    nested_cfg = protocol_cfg.get("nested_validation", {})

    request_config = {
        "strategy_id": config.get("strategy_id", "default"),
        "start_date": kwargs.get("start_date", dt_date.today() - timedelta(days=365)),
        "end_date": kwargs.get("end_date", dt_date.today()),
        "initial_cash": backtest_cfg.get("initial_cash", 100_000.0),
        "train_days": walk_forward_cfg.get(
            "train_days",
            backtest_cfg.get(
                "walk_forward_train_days", backtest_cfg.get("train_days", 90)
            ),
        ),
        "test_days": walk_forward_cfg.get(
            "test_days",
            backtest_cfg.get(
                "walk_forward_test_days", backtest_cfg.get("test_days", 30)
            ),
        ),
        "walk_forward_mode": walk_forward_cfg.get("mode", "rolling"),
        "commission_pct": backtest_cfg.get("commission_pct", 0.001),
        "slippage_pct": backtest_cfg.get("slippage_pct", 0.0005),
        "spread_pct": backtest_cfg.get("spread_pct", 0.001),
        "seed": backtest_cfg.get("seed", 42),
        "max_positions": backtest_cfg.get("max_positions", 10),
        "max_position_value": backtest_cfg.get("max_position_value", 10_000.0),
        "max_hold_days": backtest_cfg.get("max_hold_days", 5),
        "nested_validation_enabled": nested_cfg.get(
            "enabled",
            config.get(
                "nested_validation_enabled",
                backtest_cfg.get("nested_validation_enabled", False),
            ),
        ),
        "inner_folds": nested_cfg.get(
            "inner_folds",
            config.get("inner_folds", backtest_cfg.get("inner_folds", 3)),
        ),
    }
    request_config.update(kwargs)

    request = BacktestRequest.from_dict(request_config)
    return BacktestProtocol(request)


def run_backtest_with_protocol(
    request: BacktestRequest,
    symbols: Sequence[str],
    use_protocol: bool = True,
) -> BacktestResultArtifact:
    """
    Convenience function to run backtest with or without protocol.

    Args:
        request: Backtest request configuration
        symbols: Symbols to backtest
        use_protocol: If True, use protocol enforcement

    Returns:
        Backtest result artifact
    """
    if use_protocol:
        protocol = BacktestProtocol(request)

        train_days = request.train_days
        test_days = request.test_days
        total_days = train_days + test_days

        date_range = (request.end_date - request.start_date).days
        use_walk_forward = date_range > total_days * 2

        if use_walk_forward:
            return protocol.run_walk_forward(symbols)
        else:
            return protocol.run_single_period(symbols)
    else:
        engine = BacktestEngine()
        result = engine.run(
            symbols=symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            walk_forward=False,
        )

        config_hash = request.to_config_hash()
        timestamp = pd.Timestamp.now().isoformat()

        return BacktestResultArtifact(
            schema_version="1.0",
            config_hash=config_hash,
            run_id=generate_run_id(request.strategy_id, config_hash, timestamp),
            run_timestamp=timestamp,
            strategy_id=request.strategy_id,
            start_date=str(request.start_date),
            end_date=str(request.end_date),
            request=request.to_dict(),
            folds=[],
        )
