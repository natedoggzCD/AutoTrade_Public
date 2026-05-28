"""
Tests for Phase 6 components: HyperparameterSearchRunner, PromotionGate, RunRegistry.
"""
from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from autotrade.backtesting.contracts import BacktestResultArtifact
from autotrade.backtesting.results import (
    HyperparameterSearchRunner,
    PromotionGate,
    PromotionRule,
    RunRegistry,
    SearchResult,
    SearchRun,
    apply_promotion_rules,
    create_default_promotion_gate,
    create_strict_promotion_gate,
    create_search_runner_from_config,
)


# ---------------------------------------------------------------------------
# PromotionRule
# ---------------------------------------------------------------------------


class TestPromotionRule:
    def test_rule_passes_gte(self):
        rule = PromotionRule(name="min_sharpe", metric="sharpe_ratio", operator=">=", threshold=1.0)
        assert rule.evaluate({"sharpe_ratio": 1.5}) is True
        assert rule.evaluate({"sharpe_ratio": 1.0}) is True
        assert rule.evaluate({"sharpe_ratio": 0.5}) is False

    def test_rule_passes_lt(self):
        rule = PromotionRule(name="max_dd", metric="max_drawdown_pct", operator="<", threshold=20.0)
        assert rule.evaluate({"max_drawdown_pct": 10.0}) is True
        assert rule.evaluate({"max_drawdown_pct": 25.0}) is False

    def test_missing_metric_fails(self):
        rule = PromotionRule(name="x", metric="nonexistent", operator=">=", threshold=0)
        assert rule.evaluate({}) is False

    def test_disabled_rule_always_passes(self):
        rule = PromotionRule(name="x", metric="sharpe_ratio", operator=">=", threshold=999.0, enabled=False)
        assert rule.evaluate({"sharpe_ratio": 0.1}) is True


# ---------------------------------------------------------------------------
# PromotionGate
# ---------------------------------------------------------------------------


class TestPromotionGate:
    def _make_gate(self) -> PromotionGate:
        return PromotionGate(
            name="test_gate",
            rules=[
                PromotionRule(name="min_sharpe", metric="sharpe_ratio", operator=">=", threshold=1.0),
                PromotionRule(name="min_wr", metric="win_rate", operator=">=", threshold=0.40),
            ],
            min_oos_trades=10,
            require_all=True,
        )

    def test_passes_all_rules(self):
        gate = self._make_gate()
        passed, reasons = gate.evaluate({"sharpe_ratio": 2.0, "win_rate": 0.60, "total_trades": 50})
        assert passed is True
        assert "min_sharpe" in reasons
        assert "min_wr" in reasons

    def test_fails_insufficient_trades(self):
        gate = self._make_gate()
        passed, reasons = gate.evaluate({"sharpe_ratio": 2.0, "win_rate": 0.60, "total_trades": 5})
        assert passed is False
        assert any("insufficient_oos_trades" in r for r in reasons)

    def test_fails_one_rule(self):
        gate = self._make_gate()
        passed, reasons = gate.evaluate({"sharpe_ratio": 0.5, "win_rate": 0.60, "total_trades": 50})
        assert passed is False
        assert "min_sharpe" in reasons

    def test_statistical_report_integration(self):
        gate = self._make_gate()
        stat_report = {"pbo_passed": True, "dsr_passed": False}
        passed, reasons = gate.evaluate(
            {"sharpe_ratio": 2.0, "win_rate": 0.60, "total_trades": 50},
            statistical_report=stat_report,
        )
        assert passed is False
        assert "dsr_failed" in reasons

    def test_to_dict(self):
        gate = self._make_gate()
        d = gate.to_dict()
        assert d["name"] == "test_gate"
        assert len(d["rules"]) == 2


# ---------------------------------------------------------------------------
# RunRegistry
# ---------------------------------------------------------------------------


class TestRunRegistry:
    def test_register_and_retrieve(self, tmp_path: Path):
        registry = RunRegistry(registry_path=tmp_path / "reg.json")
        artifact = BacktestResultArtifact(
            run_id="run_001",
            config_hash="abc123",
            strategy_id="test_strat",
            start_date="2025-01-01",
            end_date="2025-12-31",
            metrics={"sharpe_ratio": 1.5, "win_rate": 0.55},
        )
        registry.register_run(artifact)

        run = registry.get_run("run_001")
        assert run is not None
        assert run["strategy_id"] == "test_strat"

    def test_get_runs_by_strategy(self, tmp_path: Path):
        registry = RunRegistry(registry_path=tmp_path / "reg.json")
        for i in range(3):
            artifact = BacktestResultArtifact(
                run_id=f"run_{i}",
                config_hash=f"hash_{i}",
                strategy_id="strat_a" if i < 2 else "strat_b",
                start_date="2025-01-01",
                end_date="2025-12-31",
                metrics={"sharpe_ratio": float(i)},
            )
            registry.register_run(artifact)

        runs_a = registry.get_runs_by_strategy("strat_a")
        assert len(runs_a) == 2

    def test_get_best_run(self, tmp_path: Path):
        registry = RunRegistry(registry_path=tmp_path / "reg.json")
        for i, sr in enumerate([1.0, 2.5, 0.8]):
            artifact = BacktestResultArtifact(
                run_id=f"run_{i}",
                config_hash=f"h{i}",
                strategy_id="strat",
                start_date="2025-01-01",
                end_date="2025-12-31",
                metrics={"sharpe_ratio": sr},
            )
            registry.register_run(artifact)

        best = registry.get_best_run("strat", metric="sharpe_ratio")
        assert best is not None
        assert best["run_id"] == "run_1"

    def test_compare_runs(self, tmp_path: Path):
        registry = RunRegistry(registry_path=tmp_path / "reg.json")
        for i in range(2):
            artifact = BacktestResultArtifact(
                run_id=f"run_{i}",
                config_hash=f"h{i}",
                strategy_id="strat",
                start_date="2025-01-01",
                end_date="2025-12-31",
                metrics={"sharpe_ratio": float(i), "win_rate": 0.5 + i * 0.1},
            )
            registry.register_run(artifact)

        cmp = registry.compare_runs(["run_0", "run_1"])
        assert len(cmp["runs"]) == 2

    def test_persistence_round_trip(self, tmp_path: Path):
        path = tmp_path / "reg.json"
        registry = RunRegistry(registry_path=path)
        artifact = BacktestResultArtifact(
            run_id="run_persist",
            config_hash="ph",
            strategy_id="strat",
            start_date="2025-01-01",
            end_date="2025-12-31",
            metrics={"sharpe_ratio": 1.0},
        )
        registry.register_run(artifact)

        # Reload from disk
        registry2 = RunRegistry(registry_path=path)
        assert registry2.get_run("run_persist") is not None

    def test_register_search_run(self, tmp_path: Path):
        registry = RunRegistry(registry_path=tmp_path / "reg.json")
        search_run = SearchRun(
            run_id="search_001",
            strategy_id="strat",
            search_space={"param_a": [1, 2, 3]},
            search_type="grid",
            metric_optimize="sharpe_ratio",
            n_trials=3,
        )
        registry.register_search_run(search_run)

        run = registry.get_run("search_001")
        assert run is not None
        assert run["search_type"] == "grid"


# ---------------------------------------------------------------------------
# HyperparameterSearchRunner
# ---------------------------------------------------------------------------


class TestHyperparameterSearchRunner:
    def test_deterministic_grid(self):
        runner1 = HyperparameterSearchRunner(strategy_id="test", seed=42, max_trials=10)
        runner2 = HyperparameterSearchRunner(strategy_id="test", seed=42, max_trials=10)

        space = {"a": [1, 2], "b": [10, 20]}
        trials1 = runner1._generate_grid_trials(space)
        trials2 = runner2._generate_grid_trials(space)

        assert trials1 == trials2, "Same seed must produce same grid order"

    def test_random_trial_count(self):
        runner = HyperparameterSearchRunner(strategy_id="test", search_type="random", max_trials=5, seed=99)
        space = {"x": [1, 2, 3, 4, 5], "y": [10, 20, 30]}
        trials = runner._generate_random_trials(space)
        assert len(trials) == 5

    def test_create_from_config(self):
        runner = create_search_runner_from_config(
            config={"search_type": "random", "metric_optimize": "sortino_ratio", "seed": 7, "max_trials": 50},
            strategy_id="my_strat",
        )
        assert runner.strategy_id == "my_strat"
        assert runner.search_type == "random"
        assert runner.metric_optimize == "sortino_ratio"
        assert runner.seed == 7
        assert runner.max_trials == 50


# ---------------------------------------------------------------------------
# apply_promotion_rules
# ---------------------------------------------------------------------------


class TestApplyPromotionRules:
    def test_passing_artifact(self):
        artifact = BacktestResultArtifact(
            run_id="promo_pass",
            config_hash="x",
            strategy_id="strat",
            start_date="2025-01-01",
            end_date="2025-12-31",
            metrics={
                "sharpe_ratio": 2.0,
                "max_drawdown_pct": 10.0,
                "win_rate": 0.55,
                "annual_return_pct": 15.0,
                "total_trades": 100,
            },
        )
        result = apply_promotion_rules(artifact)
        assert result.promotion_eligible is True

    def test_failing_artifact(self):
        artifact = BacktestResultArtifact(
            run_id="promo_fail",
            config_hash="x",
            strategy_id="strat",
            start_date="2025-01-01",
            end_date="2025-12-31",
            metrics={
                "sharpe_ratio": 0.3,
                "max_drawdown_pct": 10.0,
                "win_rate": 0.55,
                "annual_return_pct": 15.0,
                "total_trades": 100,
            },
        )
        result = apply_promotion_rules(artifact)
        assert result.promotion_eligible is False
        assert "min_sharpe" in result.promotion_reason_code

    def test_default_vs_strict_gate(self):
        default = create_default_promotion_gate()
        strict = create_strict_promotion_gate()
        assert strict.min_oos_trades > default.min_oos_trades
        assert len(strict.rules) == len(default.rules)


# ---------------------------------------------------------------------------
# SearchResult
# ---------------------------------------------------------------------------


class TestSearchResult:
    def test_to_dict(self):
        sr = SearchResult(
            trial_id="t0",
            params={"a": 1},
            metrics={"sharpe_ratio": 1.5},
            seed=42,
        )
        d = sr.to_dict()
        assert d["trial_id"] == "t0"
        assert d["params"]["a"] == 1
        assert d["seed"] == 42

    def test_error_in_dict(self):
        sr = SearchResult(
            trial_id="t1",
            params={},
            metrics={},
            error="something broke",
        )
        d = sr.to_dict()
        assert d["error"] == "something broke"
