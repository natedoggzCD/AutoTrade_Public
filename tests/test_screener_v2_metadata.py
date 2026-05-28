from __future__ import annotations

from datetime import datetime

import pandas as pd

from autotrade.signals.screener_v2 import (
    ScreenerV2,
    get_entry_candidates,
    get_last_screen_run_diagnostics,
)


def _make_test_parquet(path) -> None:
    rows = []
    base_date = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=39)
    for idx in range(40):
        close = 20.0 + idx * 0.5
        rows.append(
            {
                "ticker": "ABCD",
                "Date": base_date + pd.Timedelta(days=idx),
                "Open": close - 0.2,
                "High": close + 0.4,
                "Low": close - 0.5,
                "Close": close,
                "Adj Close": close,
                "Volume": 1_500_000 + idx * 1_000,
                "SMA_20": close - 1.0,
                "EMA_20": close - 0.8,
                "RSI_14": 52.0,
                "MACD": 0.5,
                "MACD_signal": 0.4,
                "MACD_hist": 0.1,
                "BB_mid": close,
                "BB_upper": close + 2.0,
                "BB_lower": close - 2.0,
                "Stoch_%K": 60.0,
                "Stoch_%D": 58.0,
                "atr_14": 1.4,
                "ROC_5": 3.0,
                "ROC_10": 5.0,
            }
        )
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_screener_v2_merges_security_metadata(tmp_path, monkeypatch):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)
    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda force_refresh=False: {
            "is_fresh": True,
            "primary_date": "2026-03-12",
            "expected_date": "2026-03-12",
            "blocking_reasons": [],
        },
    )
    screener = ScreenerV2()
    screener.daily_features_path = parquet_path
    monkeypatch.setattr(
        screener,
        "_load_security_metadata",
        lambda: pd.DataFrame(
            [
                {
                    "ticker": "ABCD",
                    "sector": "Industrials",
                    "industry": "Machinery",
                    "market_cap": 4_500_000_000,
                }
            ]
        ),
    )

    df = screener._load_price_data(symbols=["ABCD"])

    assert "sector" in df.columns
    assert "industry" in df.columns
    assert "market_cap" in df.columns
    assert df["sector"].dropna().iloc[0] == "Industrials"
    assert df["industry"].dropna().iloc[0] == "Machinery"
    assert df["market_cap"].dropna().iloc[0] == 4_500_000_000


def test_screener_v2_blocks_on_stale_core_data(tmp_path, monkeypatch):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda: {
            "is_fresh": False,
            "primary_date": "2026-03-05",
            "expected_date": "2026-03-07",
            "blocking_reasons": ["core_data_stale:2026-03-05->2026-03-07"],
        },
    )

    screener = ScreenerV2()
    screener.daily_features_path = parquet_path

    df = screener._load_price_data(symbols=["ABCD"])

    assert df.empty


def test_screener_v2_caps_extreme_composite_scores_except_strong_rally():
    screener = ScreenerV2(
        config_override={
            "min_composite_score": 75.0,
            "max_composite_score": 85.0,
            "prefer_bullish_regime": False,
        },
        scoring_mode="classic",
    )
    frame = pd.DataFrame(
        {
            "close": [20.0, 20.0, 20.0],
            "atr_pct": [4.0, 4.0, 4.0],
            "avg_volume": [500_000, 500_000, 500_000],
            "avg_dollar_volume": [10_000_000, 10_000_000, 10_000_000],
            "five_day_range_pct": [5.0, 5.0, 5.0],
            "rsi_14": [50.0, 50.0, 50.0],
            "sma5_slope_pct": [0.2, 0.2, 0.2],
            "sma5_accel": [0.2, 0.2, 0.2],
            "composite_score": [82.0, 92.0, 92.0],
            "market_regime": ["NEUTRAL", "NEUTRAL", "STRONG_RALLY"],
        }
    )

    assert screener._passes_filters(frame).tolist() == [True, False, True]


def test_screener_v2_exposes_block_diagnostics_after_stale_gate(tmp_path, monkeypatch):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda force_refresh=False: {
            "is_fresh": False,
            "primary_date": "2026-03-11",
            "expected_date": "2026-03-12",
            "blocking_reasons": ["core_data_stale:2026-03-11->2026-03-12"],
        },
    )

    screener = ScreenerV2()
    screener.daily_features_path = parquet_path

    results = get_entry_candidates(
        max_candidates=25,
        symbols=["ABCD"],
        log_samples=False,
    )
    diagnostics = get_last_screen_run_diagnostics()

    assert results == []
    assert diagnostics["status"] == "blocked"
    assert diagnostics["reason"] == "core_data_not_ready"
    assert diagnostics["blocking_reasons"] == ["core_data_stale:2026-03-11->2026-03-12"]


def test_screener_v2_reuses_prepared_history_cache_across_instances(
    tmp_path, monkeypatch
):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda force_refresh=False: {
            "is_fresh": True,
            "primary_date": "2026-03-09",
            "expected_date": "2026-03-09",
            "blocking_reasons": [],
        },
    )

    ScreenerV2._PREPARED_HISTORY_CACHE.clear()
    ScreenerV2._PREPARED_HISTORY_CACHE_KEY = None
    ScreenerV2._PREPARED_TICKER_CONTEXT_CACHE.clear()
    ScreenerV2._PREPARED_TICKER_CONTEXT_CACHE_KEY = None
    ScreenerV2._PRICE_DATA_CACHE.clear()
    ScreenerV2._PRICE_DATA_CACHE_KEY = None

    compute_calls = {"count": 0}
    original_compute_indicators = ScreenerV2._compute_indicators

    def _counting_compute(self, df):
        compute_calls["count"] += 1
        return original_compute_indicators(self, df)

    monkeypatch.setattr(ScreenerV2, "_compute_indicators", _counting_compute)

    base = ScreenerV2()
    base.daily_features_path = parquet_path
    base._get_prepared_history_session(symbols=["ABCD"])

    patched = ScreenerV2(config_override={"rsi_min": 20.0, "rsi_max": 50.0})
    patched.daily_features_path = parquet_path
    patched._get_prepared_history_session(symbols=["ABCD"])

    assert compute_calls["count"] == 1


def test_screener_v2_reuses_ticker_context_across_strategy_variants(
    tmp_path, monkeypatch
):
    parquet_path = tmp_path / "daily_features.parquet"
    _make_test_parquet(parquet_path)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda force_refresh=False: {
            "is_fresh": True,
            "primary_date": "2026-03-09",
            "expected_date": "2026-03-09",
            "blocking_reasons": [],
        },
    )

    ScreenerV2._PREPARED_HISTORY_CACHE.clear()
    ScreenerV2._PREPARED_HISTORY_CACHE_KEY = None
    ScreenerV2._PREPARED_TICKER_CONTEXT_CACHE.clear()
    ScreenerV2._PREPARED_TICKER_CONTEXT_CACHE_KEY = None
    ScreenerV2._PRICE_DATA_CACHE.clear()
    ScreenerV2._PRICE_DATA_CACHE_KEY = None

    sr_calls = {"count": 0}
    div_calls = {"count": 0}
    original_sr = ScreenerV2._load_sr_context
    original_div = ScreenerV2._compute_divergence_latest

    def _counting_sr(self, *args, **kwargs):
        sr_calls["count"] += 1
        return original_sr(self, *args, **kwargs)

    def _counting_div(self, *args, **kwargs):
        div_calls["count"] += 1
        return original_div(self, *args, **kwargs)

    monkeypatch.setattr(ScreenerV2, "_load_sr_context", _counting_sr)
    monkeypatch.setattr(ScreenerV2, "_compute_divergence_latest", _counting_div)

    base = ScreenerV2(scoring_mode="complex")
    base.daily_features_path = parquet_path
    base.screen(max_candidates=10, log_samples=False)

    patched = ScreenerV2(
        config_override={"min_atr_pct": 2.0, "max_atr_pct": 10.0},
        scoring_mode="complex",
    )
    patched.daily_features_path = parquet_path
    patched.screen(max_candidates=10, log_samples=False)

    assert sr_calls["count"] == 1
    assert div_calls["count"] == 1


def test_high_score_trap_demotion_writes_provenance_fields():
    """Demotion must record pre_demotion_score and demotion_applied so audits
    can isolate the demoted cohort from untouched candidates downstream."""
    screener = ScreenerV2.__new__(ScreenerV2)
    df = pd.DataFrame(
        [
            {"composite_score": 60.0, "rsi_14": 50.0, "volume_divergence_score": 50.0},
            {"composite_score": 82.0, "rsi_14": 50.0, "volume_divergence_score": 50.0},
            {"composite_score": 90.0, "rsi_14": 50.0, "volume_divergence_score": 50.0},
            {"composite_score": 90.0, "rsi_14": 75.0, "volume_divergence_score": 30.0},
        ]
    )

    out = screener._apply_high_score_trap_demotion(df)

    # untouched rows
    assert out.loc[0, "demotion_applied"] is False or bool(out.loc[0, "demotion_applied"]) is False
    assert out.loc[0, "final_score"] == 60.0
    assert out.loc[0, "pre_demotion_score"] == 60.0
    assert out.loc[0, "demotion_climax_penalty"] == 0.0

    # exactly at threshold — not demoted
    assert bool(out.loc[1, "demotion_applied"]) is False
    assert out.loc[1, "final_score"] == 82.0

    # composite 90 → 82 - 8*1.5 = 70.0, no climax penalty
    assert bool(out.loc[2, "demotion_applied"]) is True
    assert out.loc[2, "final_score"] == 70.0
    assert out.loc[2, "pre_demotion_score"] == 90.0
    assert out.loc[2, "demotion_climax_penalty"] == 0.0

    # composite 90 + RSI>70 + vol_div<40 → 70 - 10 = 60.0
    assert bool(out.loc[3, "demotion_applied"]) is True
    assert out.loc[3, "final_score"] == 60.0
    assert out.loc[3, "demotion_climax_penalty"] == 10.0
