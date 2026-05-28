import pytest
import pandas as pd

from autotrade.signals.contracts import RegimeLabel
from autotrade.signals.regime_router import RegimeRouter


def test_regime_router_youtube_tiebreaker_prefers_youtube_when_quant_low():
    router = RegimeRouter()
    result = router.resolve_youtube_divergence(
        quantitative_regime="trend",
        quantitative_confidence=0.45,
        youtube_regime="crisis",
        youtube_confidence=85,
    )
    assert result["regime"] == RegimeLabel.CRISIS
    assert result["method"] == "youtube_tiebreaker"
    assert pytest.approx(result["weights"]["youtube"], rel=1e-3) == 0.4


def test_regime_router_youtube_tiebreaker_keeps_quant_when_confident():
    router = RegimeRouter()
    result = router.resolve_youtube_divergence(
        quantitative_regime="trend",
        quantitative_confidence=0.8,
        youtube_regime="crisis",
        youtube_confidence=90,
    )
    assert result["regime"] == RegimeLabel.TREND
    assert result["method"] == "quantitative"


def test_regime_router_filters_strategy_families_by_regime():
    router = RegimeRouter()
    candidates = [
        {"ticker": "AAA", "family": "ts_momentum"},
        {"ticker": "BBB", "family": "mean_reversion"},
        {"ticker": "CCC"},
    ]
    allowed = router.filter_allowed_strategies("trend", candidates)
    symbols = [row.get("ticker") for row in allowed]
    assert "AAA" in symbols
    assert "BBB" not in symbols
    assert "CCC" in symbols


def test_regime_router_applies_sector_bias_scores():
    router = RegimeRouter()
    candidates = [
        {"ticker": "AAA", "sector": "Technology", "score": 50.0},
        {"ticker": "BBB", "sector": "Utilities", "final_score": 40.0},
    ]
    report = {"avoid_sectors": ["Utilities"], "favor_sectors": ["Technology"]}
    out = router.apply_sector_bias(report, candidates)
    tech = next(row for row in out if row["ticker"] == "AAA")
    util = next(row for row in out if row["ticker"] == "BBB")
    assert tech["score"] == 58.0
    assert util["final_score"] == 30.0
    assert tech["_sector_adj"] == 8.0
    assert util["_sector_adj"] == -10.0


def test_regime_router_intraday_trigger_detects_crisis():
    router = RegimeRouter()
    spy_bars = [{"open": 100.0, "close": 96.0}]
    result = router.get_current_regime(price_data=spy_bars)
    assert result.regime == RegimeLabel.CRISIS
    assert result.method == "intraday_trigger"


def test_regime_router_feature_label_takes_priority():
    router = RegimeRouter()
    feature_df = pd.DataFrame([{"regime_label": "chop"}])
    result = router.get_current_regime(feature_data=feature_df)
    assert result.regime == RegimeLabel.CHOP
    assert result.method == "feature_label"


def test_regime_router_fallback_without_price_data():
    router = RegimeRouter()
    result = router.get_current_regime(price_data=None)
    assert result.regime == RegimeLabel.NEUTRAL
    assert result.method == "router_fallback"


def test_regime_router_dataframe_symbol_query_detects_chop():
    router = RegimeRouter()
    df = pd.DataFrame(
        [
            {"symbol": "SPY", "open": 100.0, "close": 98.0, "high": 101.0, "low": 97.0},
            {"symbol": "SPY", "open": 100.0, "close": 98.0, "high": 101.0, "low": 97.0},
        ]
    )
    result = router.get_current_regime(price_data=df)
    assert result.regime == RegimeLabel.CHOP
    assert result.method == "intraday_trigger"


def test_regime_router_filter_signals_by_regime_preserves_order():
    router = RegimeRouter()

    class _Family:
        def __init__(self, value: str):
            self.value = value

    class _Signal:
        def __init__(self, symbol: str, family: str):
            self.symbol = symbol
            self.family = _Family(family)

    alpha = _Signal("AAA", "ts_momentum")
    meanrev = _Signal("BBB", "mean_reversion")
    accepted, filtered = router.filter_signals_by_regime(
        [alpha, meanrev], regime=RegimeLabel.CHOP
    )
    assert accepted == [meanrev]
    assert filtered == [alpha]


def test_regime_router_apply_sector_bias_object_candidate():
    router = RegimeRouter()

    class _Cand:
        def __init__(self):
            self.sector = "Healthcare"
            self.score = 45.0

    cand = _Cand()
    report = {"favor_sectors": ["Health"], "avoid_sectors": []}
    out = router.apply_sector_bias(report, [cand])
    assert out[0].score == 53.0
    assert out[0]._sector_adj == 8.0
