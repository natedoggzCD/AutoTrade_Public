"""
Backtesting results management, hyperparameter search, and promotion rules.

Phase 6 components:
- HyperparameterSearchRunner with deterministic seed policy
- Model/strategy promotion rules from backtest to paper-trading
- Run registry for Strategy Lab comparison
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)
from autotrade.backtesting.evaluation import (
    EvaluationSuite,
    StatisticalControlReport,
    evaluate_artifact_with_controls,
)
from autotrade.backtesting.protocol import BacktestProtocol

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result from a single hyperparameter search trial."""

    trial_id: str
    params: Dict[str, Any]
    metrics: Dict[str, Any]
    artifact: Optional[BacktestResultArtifact] = None
    seed: int = 42
    runtime_seconds: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "seed": self.seed,
            "runtime_seconds": self.runtime_seconds,
            "error": self.error,
        }


@dataclass
class SearchRun:
    """Complete hyperparameter search run with multiple trials."""

    run_id: str
    strategy_id: str
    search_space: Dict[str, List[Any]]
    search_type: str
    metric_optimize: str
    n_trials: int
    best_trial: Optional[SearchResult] = None
    trials: List[SearchResult] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config_hash: str = ""
    total_runtime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "strategy_id": self.strategy_id,
            "search_space": self.search_space,
            "search_type": self.search_type,
            "metric_optimize": self.metric_optimize,
            "n_trials": self.n_trials,
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "trials": [t.to_dict() for t in self.trials],
            "timestamp": self.timestamp,
            "config_hash": self.config_hash,
            "total_runtime_seconds": self.total_runtime_seconds,
        }


@dataclass
class PromotionRule:
    """Single promotion rule definition."""

    name: str
    metric: str
    operator: str
    threshold: float
    enabled: bool = True

    def evaluate(self, metrics: Dict[str, Any]) -> bool:
        """Evaluate if rule passes for given metrics."""
        if not self.enabled:
            return True

        value = metrics.get(self.metric)
        if value is None:
            return False

        if self.operator == ">":
            return value > self.threshold
        elif self.operator == ">=":
            return value >= self.threshold
        elif self.operator == "<":
            return value < self.threshold
        elif self.operator == "<=":
            return value <= self.threshold
        elif self.operator == "==":
            return value == self.threshold
        else:
            return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "enabled": self.enabled,
        }


@dataclass
class PromotionGate:
    """Complete promotion gate with multiple rules."""

    name: str
    rules: List[PromotionRule]
    min_oos_trades: int = 30
    require_all: bool = True

    def evaluate(
        self,
        metrics: Dict[str, Any],
        statistical_report: Optional[Dict[str, Any]] = None,
    ) -> tuple[bool, List[str]]:
        """
        Evaluate promotion gate.

        Returns:
            Tuple of (passed, list of reason codes)
        """
        total_trades = metrics.get("total_trades", 0)
        if total_trades < self.min_oos_trades:
            return False, [f"insufficient_oos_trades_{total_trades}"]

        passed_rules: List[str] = []
        failed_rules: List[str] = []
        stat_passed: List[str] = []
        stat_failed: List[str] = []

        for rule in self.rules:
            if rule.evaluate(metrics):
                passed_rules.append(rule.name)
            else:
                failed_rules.append(rule.name)

        if statistical_report:
            sr = statistical_report or {}

            if "pbo_passed" in sr:
                if bool(sr.get("pbo_passed")):
                    stat_passed.append("pbo_passed")
                else:
                    stat_failed.append("pbo_failed")

            if "dsr_passed" in sr:
                if bool(sr.get("dsr_passed")):
                    stat_passed.append("dsr_passed")
                else:
                    stat_failed.append("dsr_failed")

            if "spa_significant" in sr:
                if bool(sr.get("spa_significant")):
                    stat_passed.append("spa_significant")
                else:
                    stat_failed.append("spa_not_significant")

            if "robust_sharpe_passed" in sr:
                if bool(sr.get("robust_sharpe_passed")):
                    stat_passed.append("robust_sharpe_passed")
                else:
                    stat_failed.append("robust_sharpe_failed")

            if "confidence_passed" in sr:
                if bool(sr.get("confidence_passed")):
                    stat_passed.append("confidence_passed")
                else:
                    stat_failed.append("confidence_failed")

        if self.require_all:
            passed = len(failed_rules) == 0 and len(stat_failed) == 0
            reason_codes = (
                [*passed_rules, *stat_passed]
                if passed
                else [*failed_rules, *stat_failed]
            )
        else:
            passed = (len(passed_rules) + len(stat_passed)) > 0 and len(stat_failed) == 0
            reason_codes = (
                [*passed_rules, *stat_passed]
                if passed
                else ([*failed_rules, *stat_failed] or ["no_rules_passed"])
            )

        return passed, reason_codes

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "rules": [r.to_dict() for r in self.rules],
            "min_oos_trades": self.min_oos_trades,
            "require_all": self.require_all,
        }


class RunRegistry:
    """
    Registry for backtest runs enabling comparison across Strategy Lab and hypothesis flows.
    """

    def __init__(self, registry_path: Optional[Path] = None):
        self.registry_path = registry_path or Path("data/backtest_runs.json")
        self.runs: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    self.runs = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load run registry: {e}")
                self.runs = {}

    def _save(self) -> None:
        """Save registry to disk."""
        try:
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.registry_path, "w", encoding="utf-8") as f:
                json.dump(self.runs, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save run registry: {e}")

    def register_run(self, artifact: BacktestResultArtifact) -> None:
        """Register a backtest run artifact."""
        run_id = artifact.run_id or artifact.config_hash
        self.runs[run_id] = {
            "run_id": run_id,
            "config_hash": artifact.config_hash,
            "strategy_id": artifact.strategy_id,
            "timestamp": artifact.run_timestamp,
            "start_date": artifact.start_date,
            "end_date": artifact.end_date,
            "metrics": artifact.metrics,
            "promotion_eligible": artifact.promotion_eligible,
            "promotion_reason_code": artifact.promotion_reason_code,
            "leakage_check_passed": artifact.leakage_check_passed,
            "validation_notes": artifact.validation_notes,
            "statistical_report": artifact.statistical_report,
        }
        self._save()

    def register_search_run(self, search_run: SearchRun) -> None:
        """Register a hyperparameter search run."""
        self.runs[search_run.run_id] = search_run.to_dict()
        self._save()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific run by ID."""
        return self.runs.get(run_id)

    def get_runs_by_strategy(self, strategy_id: str) -> List[Dict[str, Any]]:
        """Get all runs for a specific strategy."""
        return [r for r in self.runs.values() if r.get("strategy_id") == strategy_id]

    def get_best_run(
        self, strategy_id: str, metric: str = "sharpe_ratio"
    ) -> Optional[Dict[str, Any]]:
        """Get the best run for a strategy by metric."""
        runs = self.get_runs_by_strategy(strategy_id)
        if not runs:
            return None

        best = None
        best_value = float("-inf")

        for run in runs:
            metrics = run.get("metrics") or {}
            value = metrics.get(metric, float("-inf"))
            if value > best_value:
                best_value = value
                best = run

        return best

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs by their metrics."""
        runs = [self.runs.get(rid) for rid in run_ids if rid in self.runs]
        if not runs:
            return {"error": "No valid runs found"}

        metrics_to_compare = [
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "win_rate",
            "total_trades",
            "annual_return_pct",
        ]

        comparison = {"runs": []}

        for run in runs:
            if run is None:
                continue
            run_id = run.get("run_id") if isinstance(run, dict) else None
            strategy_id = run.get("strategy_id") if isinstance(run, dict) else None
            timestamp = run.get("timestamp") if isinstance(run, dict) else None
            run_metrics = {
                "run_id": run_id,
                "strategy_id": strategy_id,
                "timestamp": timestamp,
            }
            metrics = run.get("metrics") or {} if isinstance(run, dict) else {}
            for m in metrics_to_compare:
                run_metrics[m] = metrics.get(m)
            comparison["runs"].append(run_metrics)

        return comparison


class HyperparameterSearchRunner:
    """
    Reproducible hyperparameter search runner with deterministic seed policy.
    """

    def __init__(
        self,
        strategy_id: str = "default",
        search_type: str = "grid",
        metric_optimize: str = "sharpe_ratio",
        seed: int = 42,
        max_trials: int = 100,
    ):
        self.strategy_id = strategy_id
        self.search_type = search_type
        self.metric_optimize = metric_optimize
        self.seed = seed
        self.max_trials = max_trials

        self._rng = random.Random(seed)

    def _generate_grid_trials(
        self, search_space: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate trials from grid search space."""
        import itertools

        keys = list(search_space.keys())
        values = list(search_space.values())

        trials = []
        for combo in itertools.product(*values):
            trial = dict(zip(keys, combo))
            trials.append(trial)

        self._rng.shuffle(trials)
        return trials[: self.max_trials]

    def _generate_random_trials(
        self, search_space: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate trials from random search space."""
        trials = []
        for _ in range(self.max_trials):
            trial = {
                key: self._rng.choice(values) for key, values in search_space.items()
            }
            trials.append(trial)
        return trials

    def _evaluate_trial(
        self,
        params: Dict[str, Any],
        request: BacktestRequest,
    ) -> SearchResult:
        """Evaluate a single trial."""
        import time

        start_time = time.time()

        trial_seed = self._rng.randint(0, 2**31 - 1)
        trial_params = dict(params)
        trial_index = trial_params.pop("_trial_index", None)

        request_copy = BacktestRequest(
            **{
                **request.to_dict(),
                **trial_params,
                "seed": trial_seed,
                "strategy_id": f"{self.strategy_id}_trial",
            }
        )

        result_params = dict(trial_params)
        if trial_index is not None:
            result_params["_trial_index"] = trial_index

        try:
            protocol = BacktestProtocol(request_copy)
            artifact = protocol.run_single_period(
                symbols=request_copy.metadata.get("symbols", ["SPY"])
            )

            metrics = artifact.metrics or {}

            return SearchResult(
                trial_id=f"trial_{result_params.get('_trial_index', 0)}",
                params=result_params,
                metrics=metrics,
                artifact=artifact,
                seed=trial_seed,
                runtime_seconds=time.time() - start_time,
            )

        except Exception as e:
            return SearchResult(
                trial_id=f"trial_{result_params.get('_trial_index', 0)}",
                params=result_params,
                metrics={},
                error=str(e),
                seed=trial_seed,
                runtime_seconds=time.time() - start_time,
            )

    def run_search(
        self,
        search_space: Dict[str, List[Any]],
        request: BacktestRequest,
    ) -> SearchRun:
        """
        Run hyperparameter search.

        Args:
            search_space: Dictionary of parameter names to lists of values
            request: Base BacktestRequest to use for each trial

        Returns:
            SearchRun with all trial results
        """
        import time

        total_start = time.time()

        if self.search_type not in {"grid", "random"}:
            raise ValueError(f"Unsupported search_type: {self.search_type}")

        if self.search_type == "grid":
            trials = self._generate_grid_trials(search_space)
        else:
            trials = self._generate_random_trials(search_space)

        config_dict = {
            "strategy_id": self.strategy_id,
            "search_space": search_space,
            "search_type": self.search_type,
            "metric_optimize": self.metric_optimize,
            "seed": self.seed,
        }
        config_hash = hashlib.sha256(
            json.dumps(config_dict, sort_keys=True, default=str).encode()
        ).hexdigest()[:16]

        run_id = f"{self.strategy_id}_search_{config_hash}"

        search_results = []
        for i, params in enumerate(trials):
            params["_trial_index"] = i
            result = self._evaluate_trial(params, request)
            search_results.append(result)

        best_trial = None
        best_value = float("-inf")

        for result in search_results:
            if result.error is None:
                value = result.metrics.get(self.metric_optimize, float("-inf"))
                if value > best_value:
                    best_value = value
                    best_trial = result

        search_run = SearchRun(
            run_id=run_id,
            strategy_id=self.strategy_id,
            search_space=search_space,
            search_type=self.search_type,
            metric_optimize=self.metric_optimize,
            n_trials=len(trials),
            best_trial=best_trial,
            trials=search_results,
            config_hash=config_hash,
            total_runtime_seconds=time.time() - total_start,
        )

        registry = RunRegistry()
        registry.register_search_run(search_run)

        return search_run


def create_default_promotion_gate() -> PromotionGate:
    """Create the default promotion gate for backtest to paper-trading."""
    return PromotionGate(
        name="default_paper_gate",
        rules=[
            PromotionRule(
                name="min_sharpe",
                metric="sharpe_ratio",
                operator=">=",
                threshold=1.0,
            ),
            PromotionRule(
                name="max_drawdown",
                metric="max_drawdown_pct",
                operator="<",
                threshold=20.0,
            ),
            PromotionRule(
                name="min_win_rate",
                metric="win_rate",
                operator=">=",
                threshold=0.40,
            ),
            PromotionRule(
                name="positive_return",
                metric="annual_return_pct",
                operator=">",
                threshold=0.0,
            ),
        ],
        min_oos_trades=30,
        require_all=True,
    )


def create_strict_promotion_gate() -> PromotionGate:
    """Create a strict promotion gate with higher thresholds."""
    return PromotionGate(
        name="strict_paper_gate",
        rules=[
            PromotionRule(
                name="min_sharpe",
                metric="sharpe_ratio",
                operator=">=",
                threshold=1.5,
            ),
            PromotionRule(
                name="max_drawdown",
                metric="max_drawdown_pct",
                operator="<",
                threshold=15.0,
            ),
            PromotionRule(
                name="min_win_rate",
                metric="win_rate",
                operator=">=",
                threshold=0.50,
            ),
            PromotionRule(
                name="positive_return",
                metric="annual_return_pct",
                operator=">",
                threshold=10.0,
            ),
        ],
        min_oos_trades=50,
        require_all=True,
    )


def apply_promotion_rules(
    artifact: BacktestResultArtifact,
    gate: Optional[PromotionGate] = None,
) -> BacktestResultArtifact:
    """
    Apply promotion rules to a backtest artifact.

    Args:
        artifact: Backtest result artifact
        gate: Promotion gate to use (uses default if not provided)

    Returns:
        Artifact with promotion_eligible and promotion_reason_code updated
    """
    gate = gate or create_default_promotion_gate()

    metrics = artifact.metrics or {}
    statistical_report = artifact.statistical_report

    passed, reason_codes = gate.evaluate(metrics, statistical_report)

    artifact.promotion_eligible = passed
    artifact.promotion_reason_code = "; ".join(reason_codes)

    artifact.validation_notes.append(
        f"Promotion gate: {'PASSED' if passed else 'FAILED'}"
    )
    artifact.validation_notes.extend([f"Rule: {r}" for r in reason_codes])

    return artifact


def create_search_runner_from_config(
    config: Dict[str, Any],
    strategy_id: str = "default",
) -> HyperparameterSearchRunner:
    """Create a hyperparameter search runner from config."""
    return HyperparameterSearchRunner(
        strategy_id=strategy_id,
        search_type=config.get("search_type", "grid"),
        metric_optimize=config.get("metric_optimize", "sharpe_ratio"),
        seed=config.get("seed", 42),
        max_trials=config.get("max_trials", 100),
    )
