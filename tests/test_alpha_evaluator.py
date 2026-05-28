from __future__ import annotations

import numpy as np
import pandas as pd

from autotrade.backtesting.alpha_catalog.base import AlphaDefinition
from autotrade.backtesting.alpha_evaluator import evaluate_alpha_on_symbol
from autotrade.backtesting.alpha_evaluator import SymbolAlphaEvaluation
from autotrade.backtesting.alpha_evaluator import _profit_factor
from autotrade.backtesting.cost_model import CostModel
from autotrade.backtesting.lab_orchestrator import _build_gate_diagnostics
from autotrade.backtesting.purged_walk_forward import PurgedKFoldConfig


def _trend_bars(n_days: int = 320) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    open_ = 20.0 * (1.01 ** np.arange(n_days))
    close = 20.0 * (1.0005 ** np.arange(n_days))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    return pd.DataFrame(
        {
            "date": [d.date() for d in dates],
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n_days, 1_000_000),
        }
    )


def _alpha(alpha_id: str, generator) -> AlphaDefinition:
    return AlphaDefinition(
        alpha_id=alpha_id,
        family="trend",
        hypothesis="test alpha",
        variant="test",
        params={},
        regime_compatibility=("TREND",),
        data_requirements=("daily_only",),
        generate=generator,
    )


def _short_no_edge_generator(bars: pd.DataFrame, ctx) -> pd.DataFrame:
    rows = []
    for idx in range(20, len(bars) - 12, 20):
        rows.append(
            {
                "date": bars.iloc[idx]["date"],
                "entry": True,
                "exit": False,
                "side": "short",
                "score": 1.0,
                "note": "test_short",
            }
        )
        rows.append(
            {
                "date": bars.iloc[idx + 5]["date"],
                "entry": False,
                "exit": True,
                "side": "short",
                "score": 0.0,
                "note": "test_short_exit",
            }
        )
    return pd.DataFrame(rows)


def _long_every_bar_generator(bars: pd.DataFrame, ctx) -> pd.DataFrame:
    rows = []
    for idx in range(5, len(bars) - 3, 2):
        rows.append(
            {
                "date": bars.iloc[idx]["date"],
                "entry": True,
                "exit": False,
                "side": "long",
                "score": 1.0,
                "note": "test_long",
            }
        )
        rows.append(
            {
                "date": bars.iloc[idx + 1]["date"],
                "entry": False,
                "exit": True,
                "side": "long",
                "score": 0.0,
                "note": "test_long_exit",
            }
        )
    return pd.DataFrame(rows)


def test_no_edge_alpha_deflates_and_does_not_beat_buy_and_hold():
    bars = _trend_bars()
    result = evaluate_alpha_on_symbol(
        alpha=_alpha("test_no_edge", _short_no_edge_generator),
        symbol="TEST",
        bars=bars,
        cost_model=CostModel(global_median_bps=0.0),
        wf_config=PurgedKFoldConfig(
            n_splits=3, label_horizon_days=5, embargo_days=2, min_train_days=40
        ),
        n_trials=20,
    )

    assert result.leakage_passed
    assert result.n_trades > 0
    assert result.deflated_sharpe <= result.raw_sharpe
    assert result.beats_bnh is False


def test_trivially_profitable_long_signal_beats_buy_and_hold():
    bars = _trend_bars()
    result = evaluate_alpha_on_symbol(
        alpha=_alpha("test_long_every_bar", _long_every_bar_generator),
        symbol="TEST",
        bars=bars,
        cost_model=CostModel(global_median_bps=0.0),
        wf_config=PurgedKFoldConfig(
            n_splits=3, label_horizon_days=5, embargo_days=2, min_train_days=40
        ),
        n_trials=20,
    )

    assert result.leakage_passed
    assert result.n_trades > 50
    assert result.raw_pf > 1.0
    assert result.beats_bnh is True


def test_profit_factor_uses_pseudo_loss_for_all_winner_samples():
    assert _profit_factor([2.0, 1.0]) == 2.0
    assert _profit_factor([2.0, -1.0]) == 1.0


def test_lab_gate_diagnostics_explain_rejections():
    alpha = _alpha("test_alpha", _long_every_bar_generator)
    no_edge = SymbolAlphaEvaluation(
        symbol="AAA",
        alpha_id="test_alpha",
        n_trades=20,
        raw_mean_return=0.01,
        raw_std_return=0.02,
        raw_pf=1.2,
        raw_sharpe=1.0,
        deflated_sharpe=0.5,
        buy_and_hold_return=0.2,
        beats_bnh=False,
        leakage_passed=True,
    )
    fdr_fail = SymbolAlphaEvaluation(
        symbol="BBB",
        alpha_id="test_alpha",
        n_trades=20,
        raw_mean_return=0.01,
        raw_std_return=0.02,
        raw_pf=1.3,
        raw_sharpe=1.1,
        deflated_sharpe=0.6,
        buy_and_hold_return=0.0,
        beats_bnh=True,
        leakage_passed=True,
    )

    diagnostics = _build_gate_diagnostics(
        evaluations={
            ("AAA", "test_alpha"): no_edge,
            ("BBB", "test_alpha"): fdr_fail,
        },
        alpha_lookup={"test_alpha": alpha},
        alpha_ids=["test_alpha"],
        symbols=["AAA", "BBB", "CCC"],
        significant_pairs=set(),
        shrunk_lookup={},
        promotions={},
    )

    overall = diagnostics["overall"]
    assert overall["evaluated"] == 3
    assert overall["beats_bnh_failed"] == 1
    assert overall["fdr_failed"] == 1
    assert overall["no_trades"] == 1
    assert diagnostics["top_rejections"][0]["reason"] == "fdr_failed"


def test_lab_gate_diagnostics_rejects_sparse_trade_samples():
    alpha = _alpha("test_alpha", _long_every_bar_generator)
    sparse = SymbolAlphaEvaluation(
        symbol="AAA",
        alpha_id="test_alpha",
        n_trades=2,
        raw_mean_return=1.0,
        raw_std_return=0.0,
        raw_pf=2.0,
        raw_sharpe=2.0,
        deflated_sharpe=1.0,
        buy_and_hold_return=0.0,
        beats_bnh=True,
        leakage_passed=True,
    )

    diagnostics = _build_gate_diagnostics(
        evaluations={("AAA", "test_alpha"): sparse},
        alpha_lookup={"test_alpha": alpha},
        alpha_ids=["test_alpha"],
        symbols=["AAA"],
        significant_pairs={("AAA", "test_alpha")},
        shrunk_lookup={},
        promotions={},
    )

    assert diagnostics["overall"]["min_trades_failed"] == 1
