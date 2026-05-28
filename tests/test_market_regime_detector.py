from datetime import datetime

from autotrade.analysis.market_regime import (
    MarketRegime,
    MarketRegimeDetector,
    RegimeAnalysis,
    SectorAnalyzer,
)


def test_detect_regime_ignores_recent_cache_from_prior_session():
    detector = MarketRegimeDetector.__new__(MarketRegimeDetector)
    detector._last_analysis = RegimeAnalysis(
        regime=MarketRegime.NEUTRAL,
        confidence=0.0,
        breadth_pct_positive=50.0,
        breadth_5d_trend="stable",
        avg_universe_return_1d=0.0,
        avg_universe_return_5d=0.0,
        top_decile_return=0.0,
        bottom_decile_return=0.0,
        dispersion=0.0,
        consecutive_down_days=0,
        consecutive_up_days=0,
        volume_trend="stable",
        recommended_strategy={},
    )
    detector._last_analysis_time = datetime.now()
    detector._last_analysis_session_key = "2026-03-30"
    detector._cache_ttl_minutes = 5

    calls = {"compute": 0}

    def _compute(_as_of_date=None):
        calls["compute"] += 1
        return {
            "pct_advancing": 72.0,
            "breadth_trend": "rising",
            "avg_return_1d": 1.5,
            "avg_return_5d": 2.2,
            "top_decile": 4.0,
            "bottom_decile": -1.0,
            "return_dispersion": 1.1,
            "consecutive_down_days": 0,
            "consecutive_up_days": 2,
            "volume_trend": "expanding",
            "leading_sectors": ["technology"],
            "lagging_sectors": ["utilities"],
            "total_stocks": 100,
        }

    detector._compute_breadth_metrics = _compute
    detector._classify_regime = lambda metrics: (
        MarketRegime.STRONG_RALLY,
        0.8,
        "breakout",
    )
    detector.get_strategy_adjustments = lambda regime, metrics: {"max_positions": 12}

    analysis = detector.detect_regime(as_of_date="2026-03-31", use_cache=True)

    assert calls["compute"] == 1
    assert analysis.regime == MarketRegime.STRONG_RALLY
    assert detector._last_analysis_session_key == "2026-03-31"


def test_analyze_sector_bias_uses_btc_proxy_without_btc_usd():
    analyzer = SectorAnalyzer(symbols={"XLB": "materials"})
    calls = []

    def _compute(ticker):
        calls.append(ticker)
        payloads = {
            "GLD": {"ret_1h": 1.0, "ret_1d": 2.0, "momentum": 1.4},
            "IBIT": None,
            "GBTC": {"ret_1h": 3.0, "ret_1d": 4.0, "momentum": 3.4},
            "XLB": {"ret_1h": 0.5, "ret_1d": 0.8, "momentum": 0.62},
        }
        return payloads.get(ticker)

    analyzer._compute_momentum = _compute

    result = analyzer.analyze_sector_bias()

    assert calls[:4] == ["GLD", "IBIT", "GBTC", "XLB"]
    assert "BTC-USD" not in calls
    assert "GLD" in result["macro_references"]
    assert "BTC" in result["macro_references"]
    assert result["macro_references"]["BTC"]["momentum"] == 3.4
