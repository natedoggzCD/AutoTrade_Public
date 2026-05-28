from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from autotrade.signals.strategy_pool import (
    _with_strategy_provenance,
    load_validated_strategies,
)


def test_load_validated_strategies_applies_pf_floor_and_top_n(tmp_path, monkeypatch):
    path = tmp_path / "validated_strategies.json"
    payload = [
        {"strategy_name": "a", "metrics": {"profit_factor": 1.20, "win_rate": 0.45}},
        {"strategy_name": "b", "metrics": {"profit_factor": 1.08, "win_rate": 0.42}},
        {"strategy_name": "c", "metrics": {"profit_factor": 1.32, "win_rate": 0.50}},
        {"strategy_name": "d", "metrics": {"profit_factor": 0.99, "win_rate": 0.60}},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.05,
                validated_strategy_pool_top_n=2,
            )
        ),
    )

    selected = load_validated_strategies(path=path)
    assert [row["strategy_name"] for row in selected] == ["c", "a"]


def test_load_validated_strategies_falls_back_when_none_pass_pf_floor(tmp_path, monkeypatch):
    path = tmp_path / "validated_strategies.json"
    payload = [
        {"strategy_name": "a", "metrics": {"profit_factor": 0.90}},
        {"strategy_name": "b", "metrics": {"profit_factor": 0.92}},
        {"strategy_name": "c", "metrics": {"profit_factor": 0.89}},
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.50,
                validated_strategy_pool_top_n=2,
            )
        ),
    )

    selected = load_validated_strategies(path=path)
    assert [row["strategy_name"] for row in selected] == ["b", "a"]


def test_walkforward_failure_is_advisory_when_strict_disabled(monkeypatch):
    strategy_lab = pytest.importorskip("tools.strategy_lab")
    strict_result = {
        "total_trades": 120,
        "win_rate": 0.52,
        "profit_factor": 1.22,
        "sharpe_ratio": 0.65,
        "max_drawdown": 0.18,
        "total_pnl": 1500.0,
    }
    wf_summary = {"passed": False}

    monkeypatch.setattr(
        "tools.strategy_lab.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                strict_require_walkforward_pass=False,
                strict_min_profit_factor=1.05,
                strict_min_trades=50,
                strict_min_total_pnl=100.0,
            )
        ),
    )

    passed, reasons = strategy_lab._evaluate_strict_validation(
        strict_result, wf_summary
    )

    assert passed is True
    assert "walk_forward_not_passed_advisory" in reasons


def test_load_validated_strategies_adds_provenance_fields_from_strategy_definition(
    tmp_path, monkeypatch
):
    path = tmp_path / "validated_strategies.json"
    payload = [
        {
            "strategy_name": "trend_a",
            "setup_type": "trend_follow",
            "strategy_definition": {
                "name": "trend_a",
                "entry": {"setup_type": "trend_follow"},
                "exit": {
                    "stop_atr_mult": 1.7,
                    "target_atr_mult": 3.4,
                    "trailing_stop": True,
                    "trailing_atr_mult": 1.2,
                    "max_hold_days": 9,
                    "time_stop_if_flat_days": 4,
                },
                "backtest_results": {
                    "win_rate": 0.58,
                    "profit_factor": 1.36,
                    "walk_forward_validated": True,
                },
            },
            "metrics": {"profit_factor": 1.36, "win_rate": 0.58},
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.05,
                validated_strategy_pool_top_n=5,
            )
        ),
    )

    selected = load_validated_strategies(path=path)
    assert len(selected) == 1
    row = selected[0]
    assert row["strategy_id"] == "trend_a"
    assert row["strategy_params"]["stop_atr_mult"] == 1.7
    assert row["strategy_params"]["target_atr_mult"] == 3.4
    assert row["strategy_params"]["max_hold_days"] == 9
    assert row["backtest_win_rate"] == 0.58
    assert row["backtest_profit_factor"] == 1.36
    assert row["walk_forward_validated"] is True


def test_strategy_provenance_normalizes_legacy_dataframe_runtime_alias():
    row = _with_strategy_provenance(
        {
            "strategy_name": "legacy_runtime",
            "strategy_definition": {
                "name": "legacy_runtime",
                "entry": {
                    "setup_type": "trend_follow",
                    "expression": "DataFrame({'close': close}).close > 10",
                },
                "exit": {"rule": "DataFrame({'close': close}).close < 9"},
            },
            "metrics": {"profit_factor": 1.2, "win_rate": 0.5},
        }
    )

    assert "pd.DataFrame" in row["strategy_definition"]["entry"]["expression"]
    assert "pd.DataFrame" in row["strategy_definition"]["exit"]["rule"]


def test_load_validated_strategies_adds_provenance_fields_from_config_patch(
    tmp_path, monkeypatch
):
    path = tmp_path / "validated_strategies.json"
    payload = [
        {
            "strategy_name": "meanrev_b",
            "setup_type": "mean_reversion",
            "config_patch": {
                "backtest": {
                    "stop_atr": 2.2,
                    "target_atr": 4.6,
                    "max_hold_days": 6,
                }
            },
            "metrics": {
                "profit_factor": 1.21,
                "win_rate": 0.51,
                "walk_forward_validated": False,
            },
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.05,
                validated_strategy_pool_top_n=5,
            )
        ),
    )

    selected = load_validated_strategies(path=path)
    assert len(selected) == 1
    row = selected[0]
    assert row["strategy_id"] == "meanrev_b"
    assert row["strategy_params"]["stop_atr_mult"] == 2.2
    assert row["strategy_params"]["target_atr_mult"] == 4.6
    assert row["strategy_params"]["max_hold_days"] == 6
    assert row["backtest_win_rate"] == 0.51
    assert row["backtest_profit_factor"] == 1.21
    assert row["walk_forward_validated"] is False


def test_load_validated_strategies_sanitizes_invalid_atr_multipliers(
    tmp_path, monkeypatch
):
    path = tmp_path / "validated_strategies.json"
    payload = [
        {
            "strategy_name": "broken_grid",
            "strategy_params": {
                "stop_atr_mult": 99.0,
                "target_atr_mult": 99.0,
                "max_hold_days": 5,
            },
            "metrics": {"profit_factor": 1.3, "win_rate": 0.55},
        }
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                validated_strategy_min_profit_factor=1.05,
                validated_strategy_pool_top_n=5,
            )
        ),
    )

    selected = load_validated_strategies(path=path)

    assert len(selected) == 1
    assert selected[0]["strategy_params"]["stop_atr_mult"] == 2.0
    assert selected[0]["strategy_params"]["target_atr_mult"] == 3.0
