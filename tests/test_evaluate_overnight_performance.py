from __future__ import annotations

from datetime import date

import pandas as pd

from tools.evaluate_overnight_performance import (
    apply_derived_metrics,
    build_correlations,
    build_report,
)


def _sample_df() -> pd.DataFrame:
    rows = [
        {
            "sig_date": "2026-04-01",
            "ticker": "AAA",
            "rank": 1,
            "ranking_score": 90.0,
            "conviction_priority_score": 91.0,
            "confidence": 95.0,
            "setup_type": "trend_follow",
            "strategy_name": "fast_trend",
            "sector": "Tech",
            "atr_percent": 8.5,
            "volume_ratio": 3.2,
            "weekly_return": 6.0,
            "rsi_14": 48.0,
            "backtest_win_rate": 52.0,
            "profit_factor": 1.3,
            "fresh_news": False,
            "catalyst_count": 0,
            "discovery_family_count": 1,
            "stale_entry_appearance_streak": 0,
            "fallback_repeat_penalty": 0.0,
            "backtest_scope": "unknown",
            "entry_source": "unknown",
            "open": 10.0,
            "high": 11.2,
            "low": 9.8,
            "close": 10.8,
            "volume": 100000,
            "open_to_high_pct": 12.0,
            "open_to_close_pct": 8.0,
            "open_to_low_pct": -2.0,
        },
        {
            "sig_date": "2026-04-01",
            "ticker": "BBB",
            "rank": 35,
            "ranking_score": 94.0,
            "conviction_priority_score": 93.0,
            "confidence": 99.0,
            "setup_type": "trend_follow",
            "strategy_name": "slow_trend",
            "sector": "Tech",
            "atr_percent": 2.5,
            "volume_ratio": 0.8,
            "weekly_return": 2.0,
            "rsi_14": 66.0,
            "backtest_win_rate": 70.0,
            "profit_factor": 2.5,
            "fresh_news": False,
            "catalyst_count": 0,
            "discovery_family_count": 0,
            "stale_entry_appearance_streak": 2,
            "fallback_repeat_penalty": 0.0,
            "backtest_scope": "unknown",
            "entry_source": "unknown",
            "open": 10.0,
            "high": 10.2,
            "low": 9.0,
            "close": 9.1,
            "volume": 50000,
            "open_to_high_pct": 2.0,
            "open_to_close_pct": -9.0,
            "open_to_low_pct": -10.0,
        },
        {
            "sig_date": "2026-04-02",
            "ticker": "CCC",
            "rank": 7,
            "ranking_score": 89.0,
            "conviction_priority_score": 88.0,
            "confidence": 92.0,
            "setup_type": "recovery_universe_fill",
            "strategy_name": "recovery_fill",
            "sector": "Energy",
            "atr_percent": 9.2,
            "volume_ratio": 4.1,
            "weekly_return": 4.0,
            "rsi_14": 44.0,
            "backtest_win_rate": 49.0,
            "profit_factor": 1.1,
            "fresh_news": False,
            "catalyst_count": 1,
            "discovery_family_count": 2,
            "stale_entry_appearance_streak": 0,
            "fallback_repeat_penalty": 0.0,
            "backtest_scope": "unknown",
            "entry_source": "unknown",
            "open": 20.0,
            "high": 24.0,
            "low": 19.5,
            "close": 23.0,
            "volume": 120000,
            "open_to_high_pct": 20.0,
            "open_to_close_pct": 15.0,
            "open_to_low_pct": -2.5,
        },
        {
            "sig_date": "2026-04-02",
            "ticker": "DDD",
            "rank": 27,
            "ranking_score": 96.0,
            "conviction_priority_score": 95.0,
            "confidence": 100.0,
            "setup_type": "new_high_breakout",
            "strategy_name": "weak_breakout",
            "sector": "Energy",
            "atr_percent": 2.2,
            "volume_ratio": 0.9,
            "weekly_return": 5.0,
            "rsi_14": 69.0,
            "backtest_win_rate": 73.0,
            "profit_factor": 3.0,
            "fresh_news": False,
            "catalyst_count": 0,
            "discovery_family_count": 0,
            "stale_entry_appearance_streak": 1,
            "fallback_repeat_penalty": 0.0,
            "backtest_scope": "unknown",
            "entry_source": "unknown",
            "open": 30.0,
            "high": 30.6,
            "low": 26.0,
            "close": 26.4,
            "volume": 65000,
            "open_to_high_pct": 2.0,
            "open_to_close_pct": -12.0,
            "open_to_low_pct": -13.3,
        },
    ]
    return pd.DataFrame(rows)


def test_apply_derived_metrics_adds_expected_fields() -> None:
    enriched = apply_derived_metrics(_sample_df())
    assert "profit_proxy" in enriched.columns
    assert "rank_bucket" in enriched.columns
    assert "atr_band" in enriched.columns
    aaa = enriched.loc[enriched["ticker"] == "AAA"].iloc[0]
    assert round(float(aaa["profit_proxy"]), 2) == 10.60
    assert str(aaa["rank_bucket"]) == "1-10"


def test_build_correlations_detects_predictive_and_anti_predictive_fields() -> None:
    df = pd.concat([_sample_df()] * 6, ignore_index=True)
    corr = build_correlations(
        apply_derived_metrics(df),
        ["atr_percent", "volume_ratio", "ranking_score", "confidence"],
    )
    assert corr["atr_percent"]["corr_profit_proxy"] is not None
    assert corr["atr_percent"]["corr_profit_proxy"] > 0
    assert corr["confidence"]["corr_profit_proxy"] < 0


def test_build_report_emits_ranking_recommendations() -> None:
    df = pd.concat([_sample_df()] * 6, ignore_index=True)
    report = build_report(
        apply_derived_metrics(df),
        start_date=date(2026, 4, 1),
        end_date=date(2026, 4, 2),
        top_n=50,
    )
    rec_text = " ".join(report["recommendations"])
    assert report["summary"]["rows_with_market_data"] == 24
    assert "atr_percent" in rec_text
    assert "confidence" in rec_text
    assert "ranking_score" in rec_text
    assert len(report["strategies"]) >= 2
    assert len(report["recommendations"]) >= 3
