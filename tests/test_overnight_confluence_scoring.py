from __future__ import annotations

from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent


def _noop_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_non_backtest_confluence_relieves_soft_backtest_penalty():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    research = {
        "symbol": "TEST",
        "final_score": 66,
        "direction": "BULLISH",
        "technical_score": 70,
        "sr_score": 64,
        "sentiment_score": 62,
        "risk_reward": 2.1,
        "news": [{"title": "Positive catalyst"}],
        "stocktwits": {"bull_bear_ratio": 1.8, "trending": True},
        "rsi_14": 54,
        "current_price": 101.0,
        "entry_price": 100.5,
        "sr_data": {"s1_price": 99.8},
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "total_trades": 42,
            "win_rate": 34.0,
            "profit_factor": 0.96,
            "walk_forward_strategy_count": 0,
            "weekly_return": 2.8,
            "volume_ratio": 1.25,
            "vol_trend_ratio": 1.05,
        },
        "signal_validation": {"historical_win_rate": 0.45, "similar_signals_found": 20},
    }

    out = agent._calculate_recommendation_multimodel(research)

    assert out.get("backtest_gated") is not True
    assert out.get("recommendation") in {"WEAK BUY", "BUY", "STRONG BUY"}
    assert out.get("non_backtest_confluence_score", 0) >= 6


def test_strong_confluence_prevents_catastrophic_backtest_hard_gate():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    research = {
        "symbol": "TEST2",
        "final_score": 82,
        "direction": "BULLISH",
        "technical_score": 78,
        "sr_score": 66,
        "sentiment_score": 63,
        "risk_reward": 2.4,
        "news": [{"title": "Strong news flow"}],
        "stocktwits": {"bull_bear_ratio": 1.4, "trending": True},
        "rsi_14": 58,
        "current_price": 55.2,
        "entry_price": 55.0,
        "sr_data": {"s1_price": 54.3},
        "backtest": {
            "source": "per_symbol_strategy_pool",
            "total_trades": 65,
            "win_rate": 21.0,
            "profit_factor": 0.82,
            "walk_forward_strategy_count": 0,
            "weekly_return": 3.5,
            "volume_ratio": 1.3,
            "vol_trend_ratio": 1.1,
        },
        "signal_validation": {"historical_win_rate": 0.42, "similar_signals_found": 30},
    }

    out = agent._calculate_recommendation_multimodel(research)

    assert out.get("backtest_gated") is not True
    assert out.get("recommendation") in {"WEAK BUY", "BUY", "STRONG BUY"}
    assert out.get("non_backtest_confluence_score", 0) >= 7


def test_strong_buy_extension_guard_downgrades_parabolic_setups():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    research = {
        "symbol": "EXTD",
        "final_score": 91,
        "direction": "BULLISH",
        "technical_score": 92,
        "sentiment_score": 72,
        "ema8_dist_pct": 3.4,
        "sma20_dist_pct": 2.0,
        "rsi_14": 61,
        "backtest": {"confidence_decision": "pending"},
    }

    out = agent._calculate_recommendation_multimodel(research)

    assert out["recommendation"] == "BUY"
    assert out["confidence"] == 78


def test_strong_buy_sma20_extension_guard_downgrades_parabolic_setups():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    research = {
        "symbol": "EXT2",
        "final_score": 91,
        "direction": "BULLISH",
        "technical_score": 92,
        "sentiment_score": 72,
        "ema8_dist_pct": 2.0,
        "sma20_dist_pct": 4.8,
        "rsi_14": 61,
        "backtest": {"confidence_decision": "pending"},
    }

    out = agent._calculate_recommendation_multimodel(research)

    assert out["recommendation"] == "BUY"
    assert out["confidence"] == 78
