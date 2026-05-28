"""Tests for strategy-lab ↔ live execution alignment.

Covers:
- absurd-ATR guard in autotrade.signals.strategy_pool (rejects pre-alignment
  lab artifacts at load time)
- LIVE_EXIT_POLICY / LIVE_ENTRY_GATES contract in tools/strategy_lab.py
- regime gate default (NEUTRAL when no per-date artifact)
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="module")
def strategy_lab_mod():
    spec = importlib.util.spec_from_file_location(
        "strategy_lab_under_test", str(PROJECT_ROOT / "tools" / "strategy_lab.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_live_exit_policy_contract(strategy_lab_mod):
    pol = strategy_lab_mod.LIVE_EXIT_POLICY
    assert pol["default_stop_atr_mult"] == 2.0
    assert pol["default_target_atr_mult"] == 3.0
    assert pol["absurd_mult_threshold"] == 5.0
    assert pol["trailing_enabled"] is True
    assert pol["hard_loss_pct"] == -0.05


def test_live_entry_gates_contract(strategy_lab_mod):
    g = strategy_lab_mod.LIVE_ENTRY_GATES
    assert g["max_consecutive_failures"] == 3
    assert g["max_drawdown_halt_pct"] == 0.10
    assert g["blocked_regimes"] == {"RISK_OFF", "SELLOFF"}


def test_regime_default_is_neutral(strategy_lab_mod):
    # Until per-date regime archives exist, the gate must default open.
    assert strategy_lab_mod._regime_for_date("2026-05-22") == "NEUTRAL"


def test_absurd_atr_guard_detects_99x():
    from autotrade.signals.strategy_pool import _has_absurd_atr_mult

    bad = {
        "strategy_definition": {
            "exit": {"stop_atr_mult": 99.0, "target_atr_mult": 99.0},
        },
    }
    good = {
        "strategy_definition": {
            "exit": {"stop_atr_mult": 2.0, "target_atr_mult": 3.0},
        },
    }
    assert _has_absurd_atr_mult(bad) is True
    assert _has_absurd_atr_mult(good) is False


def test_absurd_atr_guard_checks_config_patch_too():
    from autotrade.signals.strategy_pool import _has_absurd_atr_mult

    bad_patch = {
        "strategy_definition": {"exit": {}},
        "config_patch": {"backtest": {"stop_atr": 99.0, "target_atr": 3.0}},
    }
    assert _has_absurd_atr_mult(bad_patch) is True


def test_load_validated_strategies_rejects_absurd(tmp_path, monkeypatch):
    from autotrade.signals import strategy_pool

    # Two absurd entries, one sane (still below PF floor so may be dropped
    # but should at least survive the absurd-ATR filter).
    rows = [
        {
            "strategy_name": "bad_a",
            "setup_type": "trend_follow",
            "strategy_definition": {
                "exit": {"stop_atr_mult": 99.0, "target_atr_mult": 99.0},
                "backtest_results": {"profit_factor": 5.0},
            },
            "metrics": {"profit_factor": 5.0},
        },
        {
            "strategy_name": "bad_b",
            "setup_type": "trend_follow",
            "strategy_definition": {
                "exit": {"stop_atr_mult": 50.0, "target_atr_mult": 50.0},
                "backtest_results": {"profit_factor": 5.0},
            },
            "metrics": {"profit_factor": 5.0},
        },
        {
            "strategy_name": "good_a",
            "setup_type": "trend_follow",
            "strategy_definition": {
                "exit": {"stop_atr_mult": 2.0, "target_atr_mult": 3.0},
                "backtest_results": {"profit_factor": 1.5},
            },
            "metrics": {"profit_factor": 1.5},
        },
    ]
    path = tmp_path / "validated_strategies.json"
    path.write_text(json.dumps(rows), encoding="utf-8")

    selected = strategy_pool.load_validated_strategies(path=path)
    names = {row.get("strategy_name") for row in selected}
    assert "bad_a" not in names
    assert "bad_b" not in names
    # Sane row survives the absurd filter.
    assert "good_a" in names
