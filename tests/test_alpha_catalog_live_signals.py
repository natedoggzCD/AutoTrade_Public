"""Tests for autotrade.signals.alpha_catalog_runtime.

These tests use rsi_2_oversold_connors as the fixture alpha because:
- It is one of the 25 strict-eval survivors.
- It needs only ~15 daily bars to fire (RSI period = 2).
- It can be triggered deterministically with an engineered declining close series.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from autotrade.backtesting.alpha_catalog import AlphaContext
from autotrade.signals.alpha_catalog_runtime import (
    generate_alpha_catalog_signal,
    is_alpha_catalog_row,
    regime_compatible,
)
from autotrade.signals.strategy_pool import (
    VALIDATED_STRATEGIES_BY_SYMBOL_PATH,
    generate_signal_for_strategy_row,
    load_validated_strategies_by_symbol,
)


def _declining_close_bars(n: int = 30, start: float = 100.0, step: float = -1.0) -> pd.DataFrame:
    base = date(2026, 4, 1)
    closes = [start + step * i for i in range(n)]
    return pd.DataFrame(
        {
            "date": [base + timedelta(days=i) for i in range(n)],
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )


def _rally_then_drop_bars(n_rally: int = 18, drop: tuple = (-8.0, -8.0)) -> pd.DataFrame:
    """A long uptrend (RSI(2) stays high) then a sharp 2-bar drop at the end
    so RSI(2) crosses below 10 only on the final bar."""
    base = date(2026, 4, 1)
    closes = [100.0 + 1.0 * i for i in range(n_rally)]
    for d in drop:
        closes.append(closes[-1] + d)
    n = len(closes)
    return pd.DataFrame(
        {
            "date": [base + timedelta(days=i) for i in range(n)],
            "open": closes,
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1_000_000] * n,
        }
    )


def _rsi_oversold_row() -> dict:
    return {
        "strategy_name": "alpha_catalog__rsi_2_oversold_connors",
        "setup_type": "rsi_2_oversold_connors",
        "backtest_profit_factor": 2.5,
        "backtest_win_rate": 0.6,
        "walk_forward_validated": True,
        "metrics": {"profit_factor": 2.5, "win_rate": 0.6, "total_trades": 100},
        "alpha_metadata": {
            "alpha_id": "rsi_2_oversold_connors",
            "family": "mean_reversion",
            "params": {"rsi_period": 2, "low": 10.0, "exit": 70.0, "max_hold": 5},
            "regime_compatibility": ["CHOP", "NEUTRAL", "VOLATILE"],
            "data_requirements": ["daily_only"],
        },
    }


def _legacy_row() -> dict:
    return {
        "strategy_name": "ichimoku_break_v1",
        "setup_type": "ichimoku_break",
        "strategy_definition": {
            "entry": {"setup_type": "ichimoku_break"},
            "exit": {"stop_atr_mult": 2.0, "target_atr_mult": 3.0, "max_hold_days": 5},
        },
        "backtest_profit_factor": 1.4,
        "backtest_win_rate": 0.51,
    }


def test_alpha_catalog_row_detected():
    assert is_alpha_catalog_row(_rsi_oversold_row()) is True


def test_legacy_row_not_detected():
    assert is_alpha_catalog_row(_legacy_row()) is False


def test_legacy_row_with_alpha_metadata_prefers_legacy():
    """If both shapes are present, the legacy path wins (transition safety)."""
    row = _legacy_row()
    row["alpha_metadata"] = {"alpha_id": "rsi_2_oversold_connors"}
    assert is_alpha_catalog_row(row) is False


def test_regime_compatibility_match():
    row = _rsi_oversold_row()
    assert regime_compatible(row, "chop") is True
    assert regime_compatible(row, "CHOP") is True
    assert regime_compatible(row, "crisis") is True  # crisis -> VOLATILE


def test_regime_incompatibility_blocks():
    row = _rsi_oversold_row()
    # rsi_2 is CHOP/NEUTRAL/VOLATILE only — TREND is excluded
    assert regime_compatible(row, "trend") is False


def test_generate_signal_fires_on_engineered_oversold():
    row = _rsi_oversold_row()
    bars = _rally_then_drop_bars()
    signal = generate_alpha_catalog_signal(
        row, "TESTSYM", bars, AlphaContext(), "chop"
    )
    assert signal is not None
    assert signal["setup_type"] == "rsi_2_oversold_connors"
    assert signal["alpha_id"] == "rsi_2_oversold_connors"
    assert signal["side"] == "long"
    assert signal["symbol"] == "TESTSYM"
    assert signal["backtest_profit_factor"] == pytest.approx(2.5)
    assert signal["source"] == "alpha_catalog_runtime"


def test_generate_signal_does_not_fire_when_regime_excludes():
    row = _rsi_oversold_row()
    bars = _declining_close_bars(n=20, step=-2.0)
    signal = generate_alpha_catalog_signal(
        row, "TESTSYM", bars, AlphaContext(), "trend"
    )
    assert signal is None


def test_unknown_alpha_id_returns_none():
    row = _rsi_oversold_row()
    row["alpha_metadata"]["alpha_id"] = "nonexistent_xyz_999"
    bars = _declining_close_bars(n=20, step=-2.0)
    signal = generate_alpha_catalog_signal(
        row, "TESTSYM", bars, AlphaContext(), "chop"
    )
    assert signal is None


def test_generate_signal_does_not_fire_on_flat_bars():
    """Steady closes should produce no entry."""
    row = _rsi_oversold_row()
    bars = _declining_close_bars(n=20, step=0.0)
    bars["close"] = 100.0
    bars["open"] = 100.0
    bars["high"] = 100.5
    bars["low"] = 99.5
    signal = generate_alpha_catalog_signal(
        row, "TESTSYM", bars, AlphaContext(), "chop"
    )
    assert signal is None


def test_legacy_dispatcher_returns_none_for_legacy_row():
    """generate_signal_for_strategy_row must leave legacy rows to the legacy path."""
    bars = _declining_close_bars(n=20, step=-2.0)
    signal = generate_signal_for_strategy_row(
        _legacy_row(),
        symbol="TESTSYM",
        daily_bars=bars,
        alpha_context=AlphaContext(),
        current_regime="chop",
    )
    assert signal is None


def test_dispatcher_fires_for_alpha_catalog_row():
    bars = _rally_then_drop_bars()
    signal = generate_signal_for_strategy_row(
        _rsi_oversold_row(),
        symbol="TESTSYM",
        daily_bars=bars,
        alpha_context=AlphaContext(),
        current_regime="chop",
    )
    assert signal is not None
    assert signal["setup_type"] == "rsi_2_oversold_connors"


@pytest.mark.skipif(
    not VALIDATED_STRATEGIES_BY_SYMBOL_PATH.exists(),
    reason="validated_strategies_by_symbol.json missing",
)
def test_live_champion_file_yields_alpha_catalog_rows():
    """Integration test: the on-disk champion file must produce at least one
    alpha-catalog row that the dispatcher would route through the new path
    (vs. the legacy fallback that all rows used to take)."""
    symbol_map = load_validated_strategies_by_symbol()
    alpha_catalog_setups = []
    for symbol, rows in symbol_map.items():
        if symbol == "*":
            continue
        for row in rows:
            if is_alpha_catalog_row(row):
                alpha_catalog_setups.append((symbol, row.get("setup_type")))

    # The 2026-05-25 promotion put 69 alpha-catalog rows into the champion
    # file. If the file has been rolled back we still expect either the rows
    # to be present or for someone to have intentionally cleared them.
    raw = json.loads(VALIDATED_STRATEGIES_BY_SYMBOL_PATH.read_text(encoding="utf-8"))
    total_symbol_entries = len(raw.get("symbols", {}))
    if total_symbol_entries <= 1:
        pytest.skip("champion file rolled back to legacy single-entry state")

    assert alpha_catalog_setups, (
        "champion file holds non-legacy entries but dispatcher sees zero "
        "alpha-catalog rows — detection regression"
    )
    # Spot-check that the setup_type matches the alpha_id (no ichimoku_break
    # fallback leaking in)
    for symbol, setup in alpha_catalog_setups[:5]:
        assert setup != "ichimoku_break", (
            f"alpha-catalog row for {symbol} resolved to ichimoku_break — "
            "setup_type pass-through is broken"
        )
