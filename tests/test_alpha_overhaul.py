"""Tests for Phase 1-2 alpha signal generation overhaul."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import pytest

def test_signal_decision_strategy_metadata():
    from autotrade.signals.contracts import SignalDecision, SignalAction
    sd = SignalDecision(
        ticker='TEST', action=SignalAction.BUY, signal_strength=0.8,
        strategy_id='test_strat',
        strategy_params={'stop_atr_mult': 2.5, 'target_atr_mult': 4.0},
        backtest_win_rate=0.45, backtest_profit_factor=1.3,
        walk_forward_validated=True,
    )
    d = sd.to_dict()
    assert d['strategy_id'] == 'test_strat'
    assert d['strategy_params']['stop_atr_mult'] == 2.5
    assert d['backtest_win_rate'] == 0.45
    assert d['walk_forward_validated'] is True
    print('  SignalDecision strategy metadata: OK')


def test_signal_from_candidate_dict():
    from autotrade.signals.contracts import SignalDecision
    cd = SignalDecision.from_candidate_dict({
        'ticker': 'ABC', 'score': 80,
        'strategy_id': 'my_strat',
        'strategy_params': {'max_hold_days': 10},
        'backtest_win_rate': 0.5,
        'walk_forward_validated': True,
    })
    assert cd.strategy_id == 'my_strat'
    assert cd.backtest_win_rate == 0.5
    print('  from_candidate_dict strategy metadata: OK')


def test_signal_defaults_backward_compat():
    from autotrade.signals.contracts import SignalDecision
    sd = SignalDecision(ticker='OLD', action='buy', signal_strength=0.5)
    assert sd.strategy_id == ''
    assert sd.strategy_params == {}
    assert sd.backtest_win_rate == 0.0
    assert sd.walk_forward_validated is False
    print('  Backward compatibility (no strategy metadata): OK')


def test_divergence_scoring():
    from autotrade.signals.alpha_volume_divergence import score_divergence
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        'ticker': ['A'] * n,
        'date': pd.date_range('2026-01-01', periods=n),
        'close': np.cumsum(np.random.randn(n)) + 100,
        'volume': np.random.randint(100000, 500000, n),
    })
    result = score_divergence(df)
    assert 'divergence_score' in result.columns
    assert 'divergence_signal' in result.columns
    assert 'obv' in result.columns
    assert result['divergence_score'].between(0, 100).all()
    print('  Divergence scoring: OK')


def test_divergence_candidates():
    from autotrade.signals.alpha_volume_divergence import get_divergence_candidates
    np.random.seed(42)
    n = 50
    frames = []
    for ticker in ['A', 'B', 'C', 'D', 'E']:
        frames.append(pd.DataFrame({
            'ticker': [ticker] * n,
            'date': pd.date_range('2026-01-01', periods=n),
            'close': np.cumsum(np.random.randn(n) * 0.5) + 100,
            'volume': np.random.randint(100000, 500000, n),
        }))
    df = pd.concat(frames, ignore_index=True)
    candidates = get_divergence_candidates(df, min_score=0)  # low threshold for test
    assert isinstance(candidates, list)
    if candidates:
        assert 'divergence_score' in candidates[0]
        assert 'alpha_source' in candidates[0]
        assert candidates[0]['alpha_source'] == 'volume_divergence'
    print(f'  Divergence candidates: {len(candidates)} found: OK')


def test_unified_strategy_with_strategy_params():
    from autotrade.signals.unified_strategy import get_trade_levels
    levels = get_trade_levels(
        entry_price=100.0, atr=3.0,
        strategy_params={'stop_atr_mult': 1.5, 'target_atr_mult': 4.0},
    )
    assert abs(levels.stop_price - (100.0 - 3.0 * 1.5)) < 0.01
    assert abs(levels.target_price - (100.0 + 3.0 * 4.0)) < 0.01
    print('  Strategy params in calculate_levels: OK')


def test_unified_strategy_fallback():
    from autotrade.signals.unified_strategy import get_trade_levels
    levels = get_trade_levels(entry_price=100.0, atr=3.0)
    assert levels.stop_price > 0
    assert levels.target_price > 100.0
    print('  Fallback (no strategy_params): OK')


def test_risk_gate_strategy_aware():
    from autotrade.risk.risk_gate import RiskGate, RiskGateConfig
    rg = RiskGate(config=RiskGateConfig(atr_k=2.0, target_atr=3.0))
    # Without strategy params
    levels1 = rg.compute_entry_levels(entry_price=100.0, atr_14=3.0)
    assert abs(levels1['stop_price'] - (100.0 - 6.0)) < 0.01  # 2.0 * 3.0
    # With strategy params
    levels2 = rg.compute_entry_levels(
        entry_price=100.0, atr_14=3.0,
        strategy_params={'stop_atr_mult': 1.0, 'target_atr_mult': 5.0},
    )
    assert abs(levels2['stop_price'] - (100.0 - 3.0)) < 0.01  # 1.0 * 3.0
    assert abs(levels2['target_price'] - (100.0 + 15.0)) < 0.01  # 5.0 * 3.0
    print('  Risk gate strategy-aware entry levels: OK')


def test_screener_attaches_strategy_exit_params(monkeypatch):
    from autotrade.signals import screener_v2

    strategy_payload = [
        {
            'strategy_name': 'pullback_winner',
            'setup_type': 'pullback_support',
            'strategy_definition': {
                'name': 'pullback_winner',
                'entry': {'setup_type': 'pullback_support'},
                'exit': {
                    'stop_atr_mult': 1.6,
                    'target_atr_mult': 3.8,
                    'trailing_stop': True,
                    'trailing_atr_mult': 1.2,
                    'max_hold_days': 7,
                    'time_stop_if_flat_days': 3,
                },
                'backtest_results': {
                    'win_rate': 0.54,
                    'profit_factor': 1.28,
                    'walk_forward_validated': True,
                },
            },
            'metrics': {'profit_factor': 1.28, 'win_rate': 0.54},
        }
    ]

    monkeypatch.setattr(screener_v2, 'load_validated_strategies', lambda: strategy_payload)

    screener = screener_v2.ScreenerV2.__new__(screener_v2.ScreenerV2)
    screener.logger = logging.getLogger('test_screener_attach')

    candidates = [{'ticker': 'ABC', 'rsi': 45.0, 'weekly_return': 1.0}]
    out = screener._attach_strategy_metadata(candidates)

    assert len(out) == 1
    row = out[0]
    assert row['strategy_id'] == 'pullback_winner'
    assert row['setup_type'] == 'pullback_support'
    assert row['strategy_params']['stop_atr_mult'] == 1.6
    assert row['strategy_params']['target_atr_mult'] == 3.8
    assert row['strategy_params']['max_hold_days'] == 7
    assert row['backtest_win_rate'] == 0.54
    assert row['walk_forward_validated'] is True
    print('  Screener strategy param attachment: OK')


def test_day_manager_register_entry_stores_strategy_params_metadata():
    pytest.importorskip('alpaca.trading.client')
    from autotrade.core.day_manager import DayManager

    dm = DayManager.__new__(DayManager)
    dm._entry_order_lifecycle = {}
    dm._safe_float = DayManager._safe_float
    dm._safe_int = DayManager._safe_int
    dm._now_utc = lambda: datetime(2026, 2, 28, 10, 15, 0)
    dm._can_marketable_escalate_entry = lambda **kwargs: False

    dm._register_entry_order_lifecycle(
        order_id='oid-1',
        symbol='abc',
        planned_entry=101.25,
        urgency_tier='normal',
        entry_score=73.0,
        signal_data={
            'strategy_name': 'pullback_winner',
            'strategy_id': 'pullback_winner_v2',
            'setup_type': 'pullback_support',
            'strategy_params': {
                'stop_atr_mult': 1.4,
                'target_atr_mult': 3.6,
                'max_hold_days': 8,
            },
        },
    )

    row = dm._entry_order_lifecycle['oid-1']
    assert row['symbol'] == 'ABC'
    assert row['strategy_name'] == 'pullback_winner'
    assert row['strategy_id'] == 'pullback_winner_v2'
    assert row['setup_type'] == 'pullback_support'
    assert row['strategy_params']['stop_atr_mult'] == 1.4
    assert row['strategy_params']['target_atr_mult'] == 3.6
    assert row['stop_atr_mult'] == 1.4
    assert row['target_atr_mult'] == 3.6
    print('  DayManager entry metadata strategy params: OK')


def test_signal_pipeline_preserves_divergence_family_from_legacy_candidates():
    from autotrade.signals.pipeline import SignalGenerationPipeline

    legacy_candidate = {
        'ticker': 'ABC',
        'action': 'buy_open',
        'score': 82.0,
        'signal_family': 'mean_reversion',
        'alpha_source': 'volume_divergence',
        'reason': 'bullish volume-price divergence',
    }

    family = SignalGenerationPipeline._resolve_legacy_candidate_family(
        legacy_candidate
    )

    assert family is not None
    assert family.value == 'mean_reversion'
    print('  Pipeline resolves divergence family from legacy candidates: OK')


def test_screener_divergence_tags_map_to_mean_reversion():
    from autotrade.signals.screener_v2 import ScreenerV2

    tags = ScreenerV2._resolve_divergence_family_tags(
        divergence_signal='bullish',
        divergence_score=78.0,
        min_score=65.0,
    )

    assert tags['alpha_source'] == 'volume_divergence'
    assert tags['signal_family'] == 'mean_reversion'
    assert tags['alpha_family'] == 'mean_reversion'
    print('  Screener divergence tags map to mean-reversion: OK')
