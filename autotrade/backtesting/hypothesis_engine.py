"""
Hypothesis engine for parameter iteration loops.

This module keeps backward compatibility with the older DuckDB backtester flow,
and adds reusable sweep-grid loop logic for Strategy Lab.
"""

from __future__ import annotations

from dataclasses import asdict
from itertools import product
from typing import Any, Callable, Dict, Iterable, List, Optional

from autotrade.backtesting.comparison import StrategyComparator
from autotrade.backtesting.duckdb_backtester import BacktestResult, DuckDBBacktester


def _try_register_hypothesis_run(hypothesis: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Best-effort log of hypothesis test to RunRegistry."""
    try:
        from autotrade.backtesting.results import RunRegistry
        from autotrade.backtesting.contracts import BacktestResultArtifact
        import hashlib, json as _json

        registry = RunRegistry()
        run_id = "hypothesis_" + hashlib.sha256(
            _json.dumps(hypothesis, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
        test_metrics = result.get("test_metrics", {})
        artifact = BacktestResultArtifact(
            run_id=run_id,
            config_hash=run_id,
            strategy_id=f"hypothesis_{hypothesis.get('parameter', 'unknown')}",
            start_date="",
            end_date="",
            metrics={
                "win_rate": test_metrics.get("win_rate", 0),
                "avg_return_5d": test_metrics.get("avg_return_5d", 0),
                "profit_factor": test_metrics.get("profit_factor", 0),
            },
            validation_notes=[f"verdict={result.get('verdict', 'UNKNOWN')}"],
        )
        registry.register_run(artifact)
    except Exception:
        pass


HYPOTHESIS_TEMPLATES = {
    "slow_movers": [
        {"param": "min_atr_percent", "sweep": [1.5, 2.0, 2.5, 3.0, 3.5]},
        {"param": "min_weekly_return", "sweep": [0, 1, 2, 3, 5]},
    ],
    "missing_breakouts": [
        {"param": "min_volume_ratio", "sweep": [1.0, 1.5, 2.0, 2.5, 3.0]},
        {"param": "min_gap_percent", "sweep": [0, 1, 2, 3]},
    ],
    "bad_timing": [
        {"param": "rsi_max", "sweep": [60, 65, 70, 75, 80]},
        {"param": "rsi_min", "sweep": [25, 30, 35, 40, 45]},
    ],
}


class HypothesisEngine:
    def __init__(self, backtester: Optional[DuckDBBacktester] = None):
        self.backtester = backtester
        self.comparator = StrategyComparator()

    def diagnose(self, weekly_backtest_results: Dict[str, Any]) -> List[str]:
        diagnoses: List[str] = []
        metrics = weekly_backtest_results.get("metrics", {})
        win_rate = float(metrics.get("win_rate", 0) or 0)
        avg_ret = float(metrics.get("avg_return_5d", 0) or 0)

        if win_rate < 40.0:
            diagnoses.append("low_win_rate")
        if avg_ret < 1.5:
            diagnoses.append("low_return")
        if avg_ret < 0.5:
            diagnoses.append("slow_movers")
        return diagnoses

    def generate_hypotheses(self, diagnoses: List[str], current_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        relevant: List[str] = []
        if "low_win_rate" in diagnoses:
            relevant.extend(["bad_timing", "missing_breakouts"])
        if "low_return" in diagnoses or "slow_movers" in diagnoses:
            relevant.append("slow_movers")
        if not relevant:
            relevant = ["slow_movers", "bad_timing", "missing_breakouts"]

        hypotheses: List[Dict[str, Any]] = []
        for category in relevant:
            for tmpl in HYPOTHESIS_TEMPLATES.get(category, []):
                param = str(tmpl["param"])
                for value in list(tmpl["sweep"]):
                    current_val = current_config.get(param)
                    if current_val is not None and value == current_val:
                        continue
                    hypotheses.append(
                        {
                            "name": f"{param}={value}",
                            "diagnosis": category,
                            "parameter": param,
                            "baseline_value": current_val,
                            "test_value": value,
                        }
                    )
        return hypotheses

    def test_hypothesis(
        self,
        hypothesis: Dict[str, Any],
        base_strategy: Dict[str, Any],
        start_date: str,
        end_date: str,
    ) -> Dict[str, Any]:
        if self.backtester is None:
            raise RuntimeError("HypothesisEngine requires a DuckDBBacktester for this method")

        baseline_res = self.backtester.backtest_signal_strategy(
            base_strategy, "next_day_open", "hold_5d", start_date, end_date
        )
        test_strategy = dict(base_strategy)
        test_strategy[hypothesis["parameter"]] = hypothesis["test_value"]
        test_res = self.backtester.backtest_signal_strategy(
            test_strategy, "next_day_open", "hold_5d", start_date, end_date
        )
        comparison = self.comparator.compare(baseline_res, test_res)

        result = {
            "hypothesis": hypothesis,
            "baseline_metrics": asdict(baseline_res),
            "test_metrics": asdict(test_res),
            "improvement": comparison.deltas,
            "verdict": "ACCEPT" if comparison.is_better else "REJECT",
            "confidence": comparison.confidence,
            "details": comparison.details,
        }

        _try_register_hypothesis_run(hypothesis, result)

        return result

    @staticmethod
    def build_sweep_grid(sweep_spec: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
        keys = list(sweep_spec.keys())
        values = [list(sweep_spec[k]) for k in keys]
        if not keys:
            return []
        return [{k: v for k, v in zip(keys, combo)} for combo in product(*values)]

    def run_parameter_sweep(
        self,
        base_config: Dict[str, Any],
        sweep_spec: Dict[str, Iterable[Any]],
        evaluator: Callable[[Dict[str, Any]], Dict[str, Any]],
        sort_metric: str = "profit_factor",
    ) -> List[Dict[str, Any]]:
        """
        Run a full cartesian sweep loop and return sorted outcomes.

        `evaluator` should accept a config dict and return a result dict with metrics.
        """
        grid = self.build_sweep_grid(sweep_spec)
        rows: List[Dict[str, Any]] = []
        for patch in grid:
            cfg = dict(base_config)
            cfg.update(patch)
            result = evaluator(cfg)
            metrics = result.get("metrics", result)
            row = {"params": patch, "metrics": metrics, "raw_result": result}
            rows.append(row)

        rows.sort(key=lambda x: float((x["metrics"] or {}).get(sort_metric, 0.0) or 0.0), reverse=True)
        return rows

    def run_improvement_cycle(
        self,
        current_config: Dict[str, Any],
        start_date: str,
        end_date: str,
        max_iterations: int = 3,
    ) -> List[Dict[str, Any]]:
        if self.backtester is None:
            raise RuntimeError("HypothesisEngine requires a DuckDBBacktester for this method")

        accepted_improvements: List[Dict[str, Any]] = []
        working = dict(current_config)

        for _iteration in range(max_iterations):
            baseline = self.backtester.backtest_signal_strategy(
                working, "next_day_open", "hold_5d", start_date, end_date
            )
            diagnoses = self.diagnose(
                {
                    "metrics": {
                        "win_rate": baseline.win_rate,
                        "avg_return_5d": baseline.avg_return_5d,
                    }
                }
            )
            hypotheses = self.generate_hypotheses(diagnoses, working)
            if not hypotheses:
                break

            tested = [self.test_hypothesis(h, working, start_date, end_date) for h in hypotheses]
            accepted = [t for t in tested if t.get("verdict") == "ACCEPT"]
            if not accepted:
                break

            accepted.sort(key=lambda x: float((x.get("improvement") or {}).get("avg_return_5d", 0.0)), reverse=True)
            best = accepted[0]
            accepted_improvements.append(best)

            param = best["hypothesis"]["parameter"]
            value = best["hypothesis"]["test_value"]
            working[param] = value

        return accepted_improvements
