import pandas as pd

from autotrade.signals.screener_v2 import ScreenerV2


def _ranked_frame(**overrides):
    base = {
        "ticker": "SAFE",
        "close": 10.0,
        "atr_14": 0.5,
        "final_score": 82.0,
        "composite_score": 80.0,
        "sr_bonus": 2.0,
        "rsi_14": 51.0,
        "sector": pd.NA,
        "industry": pd.NA,
        "atr_pct": pd.NA,
        "rs_vs_sector_5d": pd.NA,
        "rs_vs_sector_20d": pd.NA,
        "gap_pct": pd.NA,
        "sr_regime": pd.NA,
        "sr_defensive_flags": pd.NA,
        "sr_action_plan": pd.NA,
        "divergence_signal": pd.NA,
        "sma5_curl_score": 60.0,
        "momentum_roc_score": 62.0,
        "relative_strength_score": 64.0,
        "rs_vs_sector_score": 50.0,
        "volume_surge_score": 66.0,
        "ema_alignment_score": 68.0,
        "regime_score": 70.0,
        "rsi_pullback_score": 55.0,
        "sr_alignment_score": 50.0,
        "adx_trend_quality_score": 50.0,
        "volume_divergence_score": 50.0,
        "mean_reversion_score": 50.0,
    }
    base.update(overrides)
    return pd.DataFrame([base])


def test_format_output_handles_nullable_fields_without_truthiness_error():
    screener = ScreenerV2()

    rows = screener._format_output(_ranked_frame())

    assert len(rows) == 1
    assert rows[0]["sector"] == ""
    assert rows[0]["industry"] == ""
    assert rows[0]["atr_pct"] == 0.0


def test_format_output_rejects_zero_rsi_and_zero_atr_before_validator():
    screener = ScreenerV2()

    zero_rsi = screener._format_output(_ranked_frame(ticker="ZERORSI", rsi_14=0.0))
    zero_atr = screener._format_output(_ranked_frame(ticker="ZEROATR", atr_14=0.0))

    assert zero_rsi == []
    assert zero_atr == []
