import sys
from pathlib import Path
import pandas as pd
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import TradingConfig  # noqa: E402
from autotrade.risk.risk_gate import RiskGate, RiskGateConfig  # noqa: E402
from autotrade.signals.conviction_engine import ConvictionEngine  # noqa: E402
from autotrade.signals.screener_v2 import ScreenerV2  # noqa: E402


def test_screener_simple_sr_alias_maps_to_momentum_pullback():
    screener = ScreenerV2(scoring_mode="simple_sr")
    assert screener.scoring_mode == "momentum_pullback"


def test_screener_levels_are_atr_first_with_optional_sr_confirmation():
    screener = ScreenerV2(scoring_mode="complex")
    row = pd.Series(
        {
            "close": 100.0,
            "atr_14": 5.0,
            "sr_s1_price": 94.0,
            "sr_s1_strength": 70.0,
            "sr_r1_price": 108.0,
            "sr_r1_strength": 70.0,
        }
    )

    levels = screener._compute_levels(row)

    # Target stays ATR-based (3x ATR above entry), not capped at R1.
    assert levels["target_price"] == 115.0
    # S/R can tighten stop when S1 is near ATR stop.
    assert levels["stop_price"] > 90.0
    # Keep partial target metadata for optional R1 take-profit.
    assert levels["partial_target_price"] == 108.0
    assert levels["risk_reward"] >= 2.0


def test_screener_min_rr_is_enforced_with_atr_levels():
    screener = ScreenerV2(scoring_mode="complex")
    row = pd.Series({"close": 100.0, "atr_14": 2.0})
    levels = screener._compute_levels(row)
    assert levels["target_price"] == 108.0
    assert levels["risk_reward"] == 2.0


def test_screener_sr_weight_cap_hard_enforced_after_normalization():
    screener = ScreenerV2(scoring_mode="complex")
    normalized = screener._normalize_weights(
        {"sma5_curl": 0.7, "sr_alignment": 0.3}
    )
    assert normalized["sr_alignment"] <= screener.config.sr_max_weight + 1e-9
    assert abs(sum(normalized.values()) - 1.0) < 1e-9


def test_trading_config_sr_alignment_yaml_matches_low_sr_policy():
    config = TradingConfig.from_yaml(PROJECT_ROOT / "config" / "trading_config.yaml")
    weights = config.screener_v2.weights

    assert weights["sr_alignment"] == 0.05
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_conviction_engine_does_not_require_sr_provider_by_default():
    engine = ConvictionEngine()
    engine.config.use_sr_inputs = False

    with patch.object(
        engine,
        "_get_price_context",
        return_value={
            "current_price": 100.0,
            "atr_pct": 3.0,
            "rsi": 52.0,
            "roc_5d": 2.0,
            "macd_direction": 0.5,
            "sma5_curl": 0.2,
            "ret_5d": 3.0,
            "ret_20d": 6.0,
            "volume_ratio": 1.4,
            "volume_trend": 0.2,
            "up_down_volume_ratio": 1.4,
        },
    ), patch.object(
        engine,
        "_get_sr_context",
        side_effect=RuntimeError("SR should not be used"),
    ), patch.object(
        engine,
        "_get_catalyst_context",
        return_value={"earnings_soon": False, "event_soon": False},
    ), patch.object(
        engine,
        "_get_cached_news_sentiment",
        return_value=None,
    ), patch.object(
        engine,
        "_get_benchmark_returns",
        return_value={"ret_5d": 1.0, "ret_20d": 2.0},
    ), patch.object(
        engine,
        "_get_universe_percentile",
        return_value=70.0,
    ):
        score, factors, reasons = engine.compute_conviction(
            symbol="TEST",
            pnl_pct=1.5,
            hold_minutes=120,
            atr_pct=0.0,
            rsi=50.0,
            current_price=100.0,
        )

    assert isinstance(score, float)
    assert factors.support_score == 50.0
    assert factors.resistance_score == 50.0
    assert isinstance(reasons, list)


def test_conviction_engine_price_context_handles_datetime_index_without_recursion():
    engine = ConvictionEngine()
    price_frame = pd.DataFrame(
        {
            "Close": [100.0 + i for i in range(30)],
            "High": [100.8 + i for i in range(30)],
            "Low": [99.2 + i for i in range(30)],
            "Volume": [100_000 + (i * 1_000) for i in range(30)],
        },
        index=pd.date_range("2026-04-01", periods=30, freq="D"),
    )

    with patch.object(engine, "_get_ticker_df", return_value=price_frame):
        context = engine._get_price_context("TEST")

    assert isinstance(context, dict)
    assert context["current_price"] == price_frame["Close"].iloc[-1]
    assert "atr_pct" in context


def test_risk_gate_compute_entry_levels_atr_first():
    gate = RiskGate(RiskGateConfig(atr_k=2.0, target_atr=3.0))
    levels = gate.compute_entry_levels(entry_price=100.0, atr_14=2.0)

    assert levels["stop_price"] == 96.0
    assert levels["target_price"] == 108.0
    assert levels["risk_reward"] == 2.0


if __name__ == "__main__":
    test_screener_simple_sr_alias_maps_to_momentum_pullback()
    test_screener_levels_are_atr_first_with_optional_sr_confirmation()
    test_screener_min_rr_is_enforced_with_atr_levels()
    test_screener_sr_weight_cap_hard_enforced_after_normalization()
    test_conviction_engine_does_not_require_sr_provider_by_default()
    test_risk_gate_compute_entry_levels_atr_first()
    print("test_reduce_sr_dependency: PASS")
