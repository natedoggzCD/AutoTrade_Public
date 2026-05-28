from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def _coerce_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    return date.fromisoformat(str(value))


@dataclass
class BacktestRequest:
    """Canonical request object for backtest runs."""

    strategy_id: str
    start_date: date
    end_date: date
    initial_cash: float = 100_000.0
    train_days: int = 90
    test_days: int = 30
    lookback_days: Optional[int] = None
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    spread_pct: float = 0.001
    seed: int = 42
    benchmark_symbol: str = "SPY"
    max_positions: int = 10
    max_position_value: float = 10_000.0
    max_hold_days: int = 7
    walk_forward_mode: str = "rolling"
    nested_validation_enabled: bool = False
    inner_folds: int = 3
    execution_model: Optional[str] = None
    signal_source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_config_hash(self) -> str:
        """Generate deterministic hash for this configuration."""
        config_dict = {
            "strategy_id": self.strategy_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_cash": self.initial_cash,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "lookback_days": self.lookback_days,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "spread_pct": self.spread_pct,
            "seed": self.seed,
            "benchmark_symbol": self.benchmark_symbol,
            "max_positions": self.max_positions,
            "max_position_value": self.max_position_value,
            "max_hold_days": self.max_hold_days,
            "walk_forward_mode": self.walk_forward_mode,
            "nested_validation_enabled": self.nested_validation_enabled,
            "inner_folds": self.inner_folds,
            "execution_model": self.execution_model,
            "signal_source": self.signal_source,
        }
        json_str = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy_id": self.strategy_id,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_cash": self.initial_cash,
            "train_days": self.train_days,
            "test_days": self.test_days,
            "lookback_days": self.lookback_days,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "spread_pct": self.spread_pct,
            "seed": self.seed,
            "benchmark_symbol": self.benchmark_symbol,
            "max_positions": self.max_positions,
            "max_position_value": self.max_position_value,
            "max_hold_days": self.max_hold_days,
            "walk_forward_mode": self.walk_forward_mode,
            "nested_validation_enabled": self.nested_validation_enabled,
            "inner_folds": self.inner_folds,
            "execution_model": self.execution_model,
            "signal_source": self.signal_source,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BacktestRequest":
        """Build request from serialized dictionary."""
        payload = dict(data)
        payload["start_date"] = _coerce_date(payload["start_date"])
        payload["end_date"] = _coerce_date(payload["end_date"])
        return cls(**payload)


@dataclass
class FoldResult:
    """Result from a single walk-forward fold."""

    fold_index: int
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    is_oos: bool = True
    trades_count: int = 0
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_dollars: float = 0.0
    turnover: float = 0.0
    equity_curve: Optional[pd.DataFrame] = None
    trades: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization (excludes DataFrame)."""
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start.isoformat(),
            "train_end": self.train_end.isoformat(),
            "test_start": self.test_start.isoformat(),
            "test_end": self.test_end.isoformat(),
            "is_oos": self.is_oos,
            "trades_count": self.trades_count,
            "pnl_dollars": self.pnl_dollars,
            "pnl_pct": self.pnl_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "calmar_ratio": self.calmar_ratio,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_dollars": self.avg_trade_dollars,
            "turnover": self.turnover,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FoldResult":
        """Build fold result from serialized dictionary."""
        payload = dict(data)
        payload["train_start"] = _coerce_date(payload["train_start"])
        payload["train_end"] = _coerce_date(payload["train_end"])
        payload["test_start"] = _coerce_date(payload["test_start"])
        payload["test_end"] = _coerce_date(payload["test_end"])
        return cls(**payload)


@dataclass
class MetricBundle:
    """Standardized metrics bundle for backtest evaluation."""

    schema_version: str = "1.0"

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0

    total_pnl_dollars: float = 0.0
    total_pnl_pct: float = 0.0
    avg_trade_dollars: float = 0.0
    avg_trade_pct: float = 0.0

    win_rate: float = 0.0
    profit_factor: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0

    annual_return_pct: float = 0.0
    annual_volatility_pct: float = 0.0

    hit_rate: float = 0.0
    expectancy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0

    turnover: float = 0.0
    capacity_proxy: float = 0.0

    risk_adjusted_return: float = 0.0

    in_sample_sharpe: Optional[float] = None
    out_of_sample_sharpe: Optional[float] = None

    fold_metrics: List[Dict[str, Any]] = field(default_factory=list)

    pbo_estimate: Optional[float] = None
    deflated_sharpe: Optional[float] = None
    spa_pvalue: Optional[float] = None

    raw_metrics: Dict[str, Any] = field(default_factory=dict)
    adjusted_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl_dollars": self.total_pnl_dollars,
            "total_pnl_pct": self.total_pnl_pct,
            "avg_trade_dollars": self.avg_trade_dollars,
            "avg_trade_pct": self.avg_trade_pct,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "calmar_ratio": self.calmar_ratio,
            "max_drawdown_pct": self.max_drawdown_pct,
            "max_drawdown_dollars": self.max_drawdown_dollars,
            "annual_return_pct": self.annual_return_pct,
            "annual_volatility_pct": self.annual_volatility_pct,
            "hit_rate": self.hit_rate,
            "expectancy": self.expectancy,
            "precision": self.precision,
            "recall": self.recall,
            "turnover": self.turnover,
            "capacity_proxy": self.capacity_proxy,
            "risk_adjusted_return": self.risk_adjusted_return,
            "in_sample_sharpe": self.in_sample_sharpe,
            "out_of_sample_sharpe": self.out_of_sample_sharpe,
            "fold_metrics": self.fold_metrics,
            "pbo_estimate": self.pbo_estimate,
            "deflated_sharpe": self.deflated_sharpe,
            "spa_pvalue": self.spa_pvalue,
            "raw_metrics": self.raw_metrics,
            "adjusted_metrics": self.adjusted_metrics,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MetricBundle":
        """Build metrics bundle from serialized dictionary."""
        return cls(**dict(data))

    @classmethod
    def from_fold_results(cls, folds: List[FoldResult]) -> MetricBundle:
        """Aggregate metrics from multiple fold results."""
        if not folds:
            return cls()

        is_oos_folds = [f for f in folds if f.is_oos]
        is_folds = [f for f in folds if not f.is_oos]
        all_trades = [t for f in folds for t in (f.trades or [])]

        total_pnl = sum(f.pnl_dollars for f in folds)
        total_trades = sum(f.trades_count for f in folds)
        total_pnl_pct = float(sum(f.pnl_pct for f in folds))
        winning = sum(
            1 for f in folds for t in (f.trades or []) if t.get("pnl_dollars", 0) > 0
        )
        losing = sum(
            1 for f in folds for t in (f.trades or []) if t.get("pnl_dollars", 0) < 0
        )
        pnl_pct_values = [
            float(t.get("pnl_pct", 0.0) or 0.0) for t in all_trades if t is not None
        ]
        wins_sum = sum(
            float(t.get("pnl_dollars", 0.0) or 0.0)
            for t in all_trades
            if float(t.get("pnl_dollars", 0.0) or 0.0) > 0
        )
        losses_sum = sum(
            float(t.get("pnl_dollars", 0.0) or 0.0)
            for t in all_trades
            if float(t.get("pnl_dollars", 0.0) or 0.0) < 0
        )

        bundle = cls()
        bundle.total_trades = total_trades
        bundle.winning_trades = winning
        bundle.losing_trades = losing
        bundle.total_pnl_dollars = total_pnl
        bundle.total_pnl_pct = total_pnl_pct
        bundle.avg_trade_dollars = total_pnl / total_trades if total_trades else 0.0
        bundle.avg_trade_pct = (
            float(np.mean(pnl_pct_values)) if pnl_pct_values else 0.0
        )
        bundle.win_rate = winning / total_trades if total_trades else 0.0
        bundle.hit_rate = bundle.win_rate
        bundle.expectancy = bundle.avg_trade_pct
        bundle.profit_factor = (
            float(wins_sum / abs(losses_sum))
            if losses_sum < 0
            else (999.0 if wins_sum > 0 else 0.0)
        )

        oos_sharpes = [f.sharpe_ratio for f in is_oos_folds if f.sharpe_ratio != 0]
        if oos_sharpes:
            bundle.sharpe_ratio = float(np.mean(oos_sharpes))
            bundle.out_of_sample_sharpe = bundle.sharpe_ratio

        is_sharpes = [f.sharpe_ratio for f in is_folds if f.sharpe_ratio != 0]
        if is_sharpes:
            bundle.in_sample_sharpe = float(np.mean(is_sharpes))

        oos_sortinos = [f.sortino_ratio for f in is_oos_folds if f.sortino_ratio != 0]
        if oos_sortinos:
            bundle.sortino_ratio = float(np.mean(oos_sortinos))

        oos_calmars = [f.calmar_ratio for f in is_oos_folds if f.calmar_ratio != 0]
        if oos_calmars:
            bundle.calmar_ratio = float(np.mean(oos_calmars))

        max_drawdowns = [float(f.max_drawdown_pct or 0.0) for f in is_oos_folds or folds]
        if max_drawdowns:
            bundle.max_drawdown_pct = float(max(max_drawdowns))

        turnovers = [float(f.turnover or 0.0) for f in folds]
        if turnovers:
            bundle.turnover = float(np.mean(turnovers))

        if folds:
            min_test_start = min(f.test_start for f in folds)
            max_test_end = max(f.test_end for f in folds)
            span_days = max(1, (max_test_end - min_test_start).days + 1)
            years = span_days / 252.0
            if years > 0:
                bundle.annual_return_pct = (
                    float((1.0 + (bundle.total_pnl_pct / 100.0)) ** (1.0 / years) - 1.0)
                    * 100.0
                )
            oos_fold_returns = [float(f.pnl_pct or 0.0) / 100.0 for f in is_oos_folds]
            if len(oos_fold_returns) >= 2:
                avg_test_days = np.mean(
                    [max(1, (f.test_end - f.test_start).days + 1) for f in is_oos_folds]
                )
                ann_scale = np.sqrt(252.0 / float(max(1.0, avg_test_days)))
                bundle.annual_volatility_pct = (
                    float(np.std(oos_fold_returns, ddof=0)) * ann_scale * 100.0
                )

        if bundle.max_drawdown_pct > 0:
            bundle.risk_adjusted_return = (
                float(bundle.annual_return_pct) / float(bundle.max_drawdown_pct)
            )

        bundle.raw_metrics = {
            "fold_count": len(folds),
            "oos_fold_count": len(is_oos_folds),
            "is_fold_count": len(is_folds),
        }
        bundle.fold_metrics = [f.to_dict() for f in folds]

        return bundle


@dataclass
class BacktestResultArtifact:
    """Canonical artifact format for backtest results."""

    schema_version: str = "1.0"
    config_hash: str = ""
    run_id: str = ""
    run_timestamp: str = ""

    strategy_id: str = ""
    start_date: str = ""
    end_date: str = ""

    request: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None

    folds: List[FoldResult] = field(default_factory=list)

    equity_curve_path: Optional[str] = None
    drawdown_plot_path: Optional[str] = None

    execution_config: Dict[str, Any] = field(default_factory=dict)
    cost_sensitivity: Dict[str, Any] = field(default_factory=dict)

    leakage_check_passed: bool = True
    leakage_warnings: List[str] = field(default_factory=list)

    promotion_eligible: bool = False
    promotion_reason_code: Optional[str] = None

    statistical_report: Optional[Dict[str, Any]] = None

    validation_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "schema_version": self.schema_version,
            "config_hash": self.config_hash,
            "run_id": self.run_id,
            "run_timestamp": self.run_timestamp,
            "strategy_id": self.strategy_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "request": self.request,
            "metrics": self.metrics,
            "folds": [f.to_dict() for f in self.folds],
            "equity_curve_path": self.equity_curve_path,
            "drawdown_plot_path": self.drawdown_plot_path,
            "execution_config": self.execution_config,
            "cost_sensitivity": self.cost_sensitivity,
            "leakage_check_passed": self.leakage_check_passed,
            "leakage_warnings": self.leakage_warnings,
            "promotion_eligible": self.promotion_eligible,
            "promotion_reason_code": self.promotion_reason_code,
            "statistical_report": self.statistical_report,
            "validation_notes": self.validation_notes,
        }

    def save(self, path: Path) -> None:
        """Save artifact to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BacktestResultArtifact":
        """Load artifact from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        payload = dict(data)
        folds = payload.get("folds") or []
        payload["folds"] = [FoldResult.from_dict(f) for f in folds]
        return cls(**payload)
