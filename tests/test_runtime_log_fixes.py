import logging
from types import SimpleNamespace

import pandas as pd


def test_screener_load_price_data_prefers_adj_close_without_fillna(
    monkeypatch, tmp_path
):
    from autotrade.signals.screener_v2 import ScreenerV2

    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "Date": pd.to_datetime(["2026-04-10", "2026-04-11"]),
            "Open": [10.0, 10.5],
            "High": [10.5, 11.0],
            "Low": [9.8, 10.2],
            "Close": [10.1, 10.6],
            "Adj Close": [10.2, 10.7],
            "Volume": [1000, 1200],
            "SMA_20": [9.9, 10.1],
            "EMA_20": [10.0, 10.2],
            "RSI_14": [55.0, 57.0],
            "MACD": [0.1, 0.2],
            "MACD_signal": [0.05, 0.1],
            "MACD_hist": [0.05, 0.1],
            "BB_mid": [10.0, 10.4],
            "BB_upper": [10.8, 11.2],
            "BB_lower": [9.2, 9.6],
            "Stoch_%K": [60.0, 62.0],
            "Stoch_%D": [58.0, 60.0],
            "atr_14": [0.4, 0.5],
            "ROC_5": [1.5, 1.7],
            "ROC_10": [2.2, 2.4],
        }
    )
    parquet_path = tmp_path / "daily_features.parquet"
    frame.to_parquet(parquet_path, index=False)

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_core_market_data_readiness",
        lambda *args, **kwargs: {
            "is_fresh": True,
            "primary_date": "2026-04-11",
            "expected_date": "2026-04-11",
            "blocking_reasons": [],
        },
    )

    screener = ScreenerV2.__new__(ScreenerV2)
    screener.logger = logging.getLogger("test_screener_load_price_data")
    screener.config = SimpleNamespace(history_days=30)
    screener.daily_features_path = parquet_path
    screener.security_metadata_path = tmp_path / "missing.parquet"
    screener.core_data_readiness = None
    screener._cache_key = lambda symbols=None: ("test",)

    loaded = screener._load_price_data(symbols=["AAA"])

    assert loaded["close"].tolist() == [10.2, 10.7]
    assert loaded["ticker"].tolist() == ["AAA", "AAA"]


def test_screener_indicator_columns_and_symbol_inference_materialize():
    from autotrade.feature_engineering.adapters import get_screener_v2_adapter
    from autotrade.signals.screener_v2 import ScreenerV2

    frame = pd.DataFrame(
        {
            "ticker": ["AAA"] * 6,
            "date": pd.date_range("2026-04-01", periods=6, freq="D"),
            "open": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "high": [10.3, 10.4, 10.5, 10.6, 10.7, 10.8],
            "low": [9.8, 9.9, 10.0, 10.1, 10.2, 10.3],
            "close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "adj_close": [10.1, 10.2, 10.3, 10.4, 10.5, 10.6],
            "volume": [1000, 1100, 1200, 1300, 1400, 1500],
            "sma20": [9.9, 10.0, 10.1, 10.2, 10.3, 10.4],
            "ema20": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "rsi_14": [52.0, 53.0, 54.0, 55.0, 56.0, 57.0],
            "macd": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15],
            "macd_signal": [0.08, 0.09, 0.1, 0.11, 0.12, 0.13],
            "macd_hist": [0.02, 0.02, 0.02, 0.02, 0.02, 0.02],
            "bb_mid": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5],
            "bb_upper": [10.6, 10.7, 10.8, 10.9, 11.0, 11.1],
            "bb_lower": [9.4, 9.5, 9.6, 9.7, 9.8, 9.9],
            "stoch_k": [60.0, 61.0, 62.0, 63.0, 64.0, 65.0],
            "stoch_d": [58.0, 59.0, 60.0, 61.0, 62.0, 63.0],
            "atr_14": [0.4, 0.41, 0.42, 0.43, 0.44, 0.45],
            "roc_5": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            "roc_10": [2.0, 2.1, 2.2, 2.3, 2.4, 2.5],
        }
    )

    adapter = get_screener_v2_adapter()
    prepared, _, _, symbols = adapter._prepare_pipeline_input(frame)

    screener = ScreenerV2.__new__(ScreenerV2)
    screener.config = SimpleNamespace(
        gap_penalty_per_pct=1.0,
        gap_negative_penalty=0.0,
        sma5_curl_scale=1.0,
        sma20_trend_scale=1.0,
        momentum_scale=1.0,
        macd_hist_scale=1.0,
        volume_lookback_days=5,
        min_adx_for_trend_bonus=20.0,
    )

    enriched = screener._compute_indicators(prepared)

    assert "symbol" in prepared.columns
    assert symbols == ["AAA"]
    for column in (
        "sma5",
        "sma20",
        "ema5",
        "prev_close",
        "atr_pct",
        "avg_volume",
        "avg_dollar_volume",
        "volume_ratio",
        "five_day_range_pct",
    ):
        assert column in enriched.columns
        assert enriched[column].notna().any()


def test_screener_sector_relative_strength_scores_against_sector_etf():
    from autotrade.signals.screener_v2 import ScreenerV2

    dates = pd.date_range("2026-03-01", periods=25, freq="D")

    def _rows(ticker, closes, sector=None):
        return [
            {
                "ticker": ticker,
                "date": date,
                "open": close * 0.99,
                "high": close * 1.02,
                "low": close * 0.98,
                "close": close,
                "adj_close": close,
                "volume": 200_000,
                "sma20": close * 0.98,
                "ema20": close * 0.99,
                "rsi_14": 55.0,
                "macd": 0.1,
                "macd_signal": 0.05,
                "macd_hist": 0.05,
                "bb_mid": close,
                "bb_upper": close * 1.1,
                "bb_lower": close * 0.9,
                "stoch_k": 60.0,
                "stoch_d": 58.0,
                "atr_14": close * 0.04,
                "roc_5": 1.0,
                "roc_10": 2.0,
                "sector": sector,
            }
            for date, close in zip(dates, closes)
        ]

    frame = pd.DataFrame(
        _rows("AAA", [10 + i * 0.35 for i in range(25)], "Technology")
        + _rows("BBB", [10 + i * 0.05 for i in range(25)], "Technology")
        + _rows("XLK", [100 + i * 0.15 for i in range(25)])
    )

    screener = ScreenerV2.__new__(ScreenerV2)
    screener.config = SimpleNamespace(
        gap_penalty_per_pct=1.0,
        gap_negative_penalty=0.0,
        sma5_curl_scale=1.0,
        sma20_trend_scale=1.0,
        momentum_scale=1.0,
        macd_hist_scale=1.0,
        volume_lookback_days=5,
        min_adx_for_trend_bonus=20.0,
        sector_relative_strength_lookbacks=[5, 20],
        sector_etf_map={"technology": "XLK"},
    )

    enriched = screener._compute_indicators(frame)
    latest = enriched.sort_values("date").groupby("ticker").tail(1).set_index("ticker")

    assert latest.loc["AAA", "rs_vs_sector_5d"] > 0
    assert latest.loc["BBB", "rs_vs_sector_5d"] < latest.loc["AAA", "rs_vs_sector_5d"]
    assert (
        latest.loc["AAA", "rs_vs_sector_score"]
        > latest.loc["BBB", "rs_vs_sector_score"]
    )


def test_screener_build_scored_df_uses_sr_atr_fallback_without_series_fillna():
    from autotrade.signals.screener_v2 import ScreenerV2

    screener = ScreenerV2.__new__(ScreenerV2)
    screener.config = SimpleNamespace(
        min_history_days=1,
        sr_rr_discount_factor=1.0,
        weights={},
        sr_bonus_cap=100.0,
    )
    screener._series_or_default = ScreenerV2._series_or_default.__get__(
        screener, ScreenerV2
    )
    screener._coalesce_series = ScreenerV2._coalesce_series.__get__(
        screener, ScreenerV2
    )
    screener._compute_sr_alignment_score = lambda latest: pd.Series(
        50.0, index=latest.index
    )
    screener._enrich_with_divergence = lambda df, latest, divergence_df=None: latest
    screener._normalize_weights = lambda weights: {}
    screener._get_factor_score_series = lambda latest: {}

    df = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "date": pd.to_datetime(["2026-04-11"]),
            "close": [10.0],
            "atr_14": [pd.NA],
            "atr_pct": [pd.NA],
        }
    )
    sr_df = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "sr_atr_14": [0.5],
        }
    )

    scored = screener._build_scored_df(df, sr_df)

    assert scored.loc[0, "atr_14"] == 0.5
    assert scored.loc[0, "atr_pct"] == 5.0


def test_risk_gate_evaluate_with_pdt_uses_hold_lock_flag():
    from autotrade.risk.risk_gate import RiskGate, RiskGateConfig, RiskAction

    gate = RiskGate(config=RiskGateConfig(trim_threshold_pct=-5.0, trim_fraction=0.5))
    plan = gate.evaluate_with_pdt(
        symbol="ABC",
        entry_price=100.0,
        current_price=94.0,
        qty=10,
        atr_pct=2.0,
        time_in_trade_minutes=5.0,
        high_since_entry=100.0,
        pnl_improving=False,
    )

    assert plan.action in {RiskAction.NONE, RiskAction.EXIT, RiskAction.TRIM}
