"""
Tests for Phase 6 - Hyperparameter search and promotion rules.
"""

import pytest
from datetime import date, timedelta

from autotrade.backtesting.contracts import (
    BacktestRequest,
    BacktestResultArtifact,
    FoldResult,
    MetricBundle,
)
from autotrade.backtesting.results import (
    SearchResult,
    SearchRun,
    PromotionRule,
    PromotionGate,
    RunRegistry,
    HyperparameterSearchRunner,
    create_default_promotion_gate,
    create_strict_promotion_gate,
    apply_promotion_rules,
    create_search_runner_from_config,
)


class TestPromotionRule:
    def test_rule_greater_than_pass(self):
        rule = PromotionRule(name="test", metric="sharpe", operator=">", threshold=1.0)
        metrics = {"sharpe": 1.5}
        assert rule.evaluate(metrics) is True

    def test_rule_greater_than_fail(self):
        rule = PromotionRule(name="test", metric="sharpe", operator=">", threshold=1.0)
        metrics = {"sharpe": 0.5}
        assert rule.evaluate(metrics) is False

    def test_rule_greater_equal_pass(self):
        rule = PromotionRule(name="test", metric="sharpe", operator=">=", threshold=1.0)
        metrics = {"sharpe": 1.0}
        assert rule.evaluate(metrics) is True

    def test_rule_less_than_pass(self):
        rule = PromotionRule(name="test", metric="dd", operator="<", threshold=20.0)
        metrics = {"dd": 15.0}
        assert rule.evaluate(metrics) is True

    def test_rule_missing_metric(self):
        rule = PromotionRule(name="test", metric="sharpe", operator=">", threshold=1.0)
        metrics = {}
        assert rule.evaluate(metrics) is False

    def test_rule_disabled(self):
        rule = PromotionRule(
            name="test", metric="sharpe", operator=">", threshold=1.0, enabled=False
        )
        metrics = {"sharpe": 0.5}
        assert rule.evaluate(metrics) is True


class TestPromotionGate:
    def test_gate_require_all_passes(self):
        gate = PromotionGate(
            name="test",
            rules=[
                PromotionRule(name="r1", metric="sharpe", operator=">=", threshold=1.0),
                PromotionRule(
                    name="r2", metric="win_rate", operator=">=", threshold=0.5
                ),
            ],
            require_all=True,
        )
        metrics = {"sharpe": 1.5, "win_rate": 0.6, "total_trades": 50}
        passed, reasons = gate.evaluate(metrics)
        assert passed is True
        assert "r1" in reasons
        assert "r2" in reasons

    def test_gate_require_all_one_fails(self):
        gate = PromotionGate(
            name="test",
            rules=[
                PromotionRule(name="r1", metric="sharpe", operator=">=", threshold=1.0),
                PromotionRule(
                    name="r2", metric="win_rate", operator=">=", threshold=0.5
                ),
            ],
            require_all=True,
        )
        metrics = {"sharpe": 1.5, "win_rate": 0.3, "total_trades": 50}
        passed, reasons = gate.evaluate(metrics)
        assert passed is False
        assert "r2" in reasons

    def test_gate_min_oos_trades_fails(self):
        gate = PromotionGate(name="test", rules=[], min_oos_trades=30)
        metrics = {"total_trades": 20}
        passed, reasons = gate.evaluate(metrics)
        assert passed is False
        assert "insufficient_oos_trades_20" in reasons

    def test_gate_statistical_report_integration(self):
        gate = PromotionGate(name="test", rules=[], min_oos_trades=30)
        metrics = {"total_trades": 50, "sharpe": 1.0}
        statistical_report = {"pbo_passed": True, "dsr_passed": True}
        passed, reasons = gate.evaluate(metrics, statistical_report)
        assert passed is True
        assert "pbo_passed" in reasons

    def test_gate_statistical_failure_blocks(self):
        gate = PromotionGate(
            name="test",
            rules=[
                PromotionRule(name="r1", metric="sharpe", operator=">=", threshold=1.0),
            ],
            require_all=True,
            min_oos_trades=30,
        )
        metrics = {"total_trades": 50, "sharpe": 1.5}
        statistical_report = {"dsr_passed": False}
        passed, reasons = gate.evaluate(metrics, statistical_report)
        assert passed is False
        assert "dsr_failed" in reasons

    def test_gate_require_any_accepts_statistical_signal(self):
        gate = PromotionGate(
            name="test",
            rules=[
                PromotionRule(name="r1", metric="sharpe", operator=">=", threshold=10.0),
            ],
            require_all=False,
            min_oos_trades=30,
        )
        metrics = {"total_trades": 50, "sharpe": 1.5}
        statistical_report = {"pbo_passed": True}
        passed, reasons = gate.evaluate(metrics, statistical_report)
        assert passed is True
        assert "pbo_passed" in reasons


class TestDefaultPromotionGates:
    def test_default_gate_creation(self):
        gate = create_default_promotion_gate()
        assert gate.name == "default_paper_gate"
        assert gate.min_oos_trades == 30
        assert len(gate.rules) == 4
        assert gate.require_all is True

    def test_strict_gate_creation(self):
        gate = create_strict_promotion_gate()
        assert gate.name == "strict_paper_gate"
        assert gate.min_oos_trades == 50
        assert len(gate.rules) == 4


class TestApplyPromotionRules:
    def test_apply_promotion_rules_passes(self):
        artifact = BacktestResultArtifact(
            strategy_id="test",
            metrics={
                "sharpe_ratio": 1.5,
                "max_drawdown_pct": 10.0,
                "win_rate": 0.55,
                "annual_return_pct": 15.0,
                "total_trades": 50,
            },
        )
        result = apply_promotion_rules(artifact)
        assert result.promotion_eligible is True

    def test_apply_promotion_rules_fails(self):
        artifact = BacktestResultArtifact(
            strategy_id="test",
            metrics={
                "sharpe_ratio": 0.5,
                "max_drawdown_pct": 10.0,
                "win_rate": 0.55,
                "annual_return_pct": 15.0,
                "total_trades": 50,
            },
        )
        result = apply_promotion_rules(artifact)
        assert result.promotion_eligible is False


class TestHyperparameterSearchRunner:
    def test_runner_creation(self):
        runner = HyperparameterSearchRunner(
            strategy_id="test",
            search_type="grid",
            metric_optimize="sharpe_ratio",
            seed=42,
            max_trials=10,
        )
        assert runner.strategy_id == "test"
        assert runner.search_type == "grid"
        assert runner.seed == 42
        assert runner.max_trials == 10

    def test_grid_search_generation(self):
        runner = HyperparameterSearchRunner(
            strategy_id="test",
            search_type="grid",
            seed=42,
            max_trials=10,
        )
        search_space = {
            "param_a": [1, 2, 3],
            "param_b": ["x", "y"],
        }
        trials = runner._generate_grid_trials(search_space)
        assert len(trials) == 6
        assert all("param_a" in t and "param_b" in t for t in trials)

    def test_random_search_generation(self):
        runner = HyperparameterSearchRunner(
            strategy_id="test",
            search_type="random",
            seed=123,
            max_trials=5,
        )
        search_space = {
            "param_a": [1, 2, 3],
            "param_b": ["x", "y", "z"],
        }
        trials = runner._generate_random_trials(search_space)
        assert len(trials) == 5
        assert all("param_a" in t and "param_b" in t for t in trials)

    def test_deterministic_seed(self):
        runner1 = HyperparameterSearchRunner(strategy_id="test", seed=42, max_trials=3)
        runner2 = HyperparameterSearchRunner(strategy_id="test", seed=42, max_trials=3)

        search_space = {"p": [1, 2, 3, 4, 5]}
        trials1 = runner1._generate_random_trials(search_space)
        trials2 = runner2._generate_random_trials(search_space)

        assert trials1 == trials2

    def test_invalid_search_type_raises(self):
        runner = HyperparameterSearchRunner(
            strategy_id="test",
            search_type="unsupported",
            seed=42,
            max_trials=3,
        )
        request = BacktestRequest(
            strategy_id="test",
            start_date=date.today() - timedelta(days=30),
            end_date=date.today(),
        )
        with pytest.raises(ValueError, match="Unsupported search_type"):
            runner.run_search({"max_hold_days": [3]}, request)


class TestSearchResult:
    def test_search_result_creation(self):
        result = SearchResult(
            trial_id="trial_1",
            params={"param_a": 1},
            metrics={"sharpe_ratio": 1.5},
            seed=42,
            runtime_seconds=1.0,
        )
        assert result.trial_id == "trial_1"
        assert result.params["param_a"] == 1
        assert result.error is None


class TestSearchRun:
    def test_search_run_creation(self):
        run = SearchRun(
            run_id="test_run",
            strategy_id="test",
            search_space={"p": [1, 2]},
            search_type="grid",
            metric_optimize="sharpe",
            n_trials=2,
        )
        assert run.run_id == "test_run"
        assert run.best_trial is None
        assert len(run.trials) == 0

    def test_search_run_to_dict(self):
        run = SearchRun(
            run_id="test_run",
            strategy_id="test",
            search_space={"p": [1, 2]},
            search_type="grid",
            metric_optimize="sharpe",
            n_trials=2,
            best_trial=SearchResult(
                trial_id="trial_1",
                params={"p": 1},
                metrics={"sharpe_ratio": 1.5},
                seed=42,
            ),
        )
        d = run.to_dict()
        assert d["run_id"] == "test_run"
        assert d["best_trial"]["trial_id"] == "trial_1"


class TestRunRegistry:
    def test_registry_creation(self, tmp_path):
        registry = RunRegistry(registry_path=tmp_path / "test_registry.json")
        assert registry.registry_path == tmp_path / "test_registry.json"

    def test_register_and_get_run(self, tmp_path):
        registry = RunRegistry(registry_path=tmp_path / "test_registry.json")
        artifact = BacktestResultArtifact(
            run_id="run_1",
            config_hash="abc123",
            strategy_id="test_strategy",
            run_timestamp="2026-01-01T00:00:00",
            start_date="2025-01-01",
            end_date="2025-12-31",
            metrics={"sharpe_ratio": 1.5},
            promotion_eligible=True,
        )
        registry.register_run(artifact)
        run = registry.get_run("run_1")
        assert run is not None
        assert run["strategy_id"] == "test_strategy"

    def test_get_runs_by_strategy(self, tmp_path):
        registry = RunRegistry(registry_path=tmp_path / "test_registry.json")
        for i in range(3):
            artifact = BacktestResultArtifact(
                run_id=f"run_{i}",
                config_hash=f"hash_{i}",
                strategy_id="my_strategy",
                run_timestamp="2026-01-01T00:00:00",
                start_date="2025-01-01",
                end_date="2025-12-31",
                metrics={},
            )
            registry.register_run(artifact)

        runs = registry.get_runs_by_strategy("my_strategy")
        assert len(runs) == 3

    def test_get_best_run(self, tmp_path):
        registry = RunRegistry(registry_path=tmp_path / "test_registry.json")
        for sharpe in [1.0, 2.5, 1.5]:
            artifact = BacktestResultArtifact(
                run_id=f"run_{sharpe}",
                config_hash=f"hash_{sharpe}",
                strategy_id="my_strategy",
                run_timestamp="2026-01-01T00:00:00",
                start_date="2025-01-01",
                end_date="2025-12-31",
                metrics={"sharpe_ratio": sharpe},
            )
            registry.register_run(artifact)

        best = registry.get_best_run("my_strategy", metric="sharpe_ratio")
        assert best is not None
        assert best["metrics"]["sharpe_ratio"] == 2.5


class TestCreateSearchRunnerFromConfig:
    def test_create_from_config(self):
        config = {
            "search_type": "random",
            "metric_optimize": "sortino_ratio",
            "seed": 999,
            "max_trials": 50,
        }
        runner = create_search_runner_from_config(config, strategy_id="my_strategy")
        assert runner.strategy_id == "my_strategy"
        assert runner.search_type == "random"
        assert runner.metric_optimize == "sortino_ratio"
        assert runner.seed == 999
        assert runner.max_trials == 50


class TestIntegration:
    def test_full_search_to_promotion_flow(self, tmp_path):
        runner = HyperparameterSearchRunner(
            strategy_id="integration_test",
            search_type="grid",
            seed=42,
            max_trials=3,
        )
        search_space = {
            "max_hold_days": [3, 5],
        }
        request = BacktestRequest(
            strategy_id="integration_test",
            start_date=date.today() - timedelta(days=90),
            end_date=date.today(),
            initial_cash=100000,
            train_days=60,
            test_days=20,
        )

        search_run = runner.run_search(search_space, request)

        assert search_run.run_id.startswith("integration_test_search_")
        assert search_run.n_trials <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
