from types import SimpleNamespace

from autotrade.core.day_manager import DayManager
from autotrade.signals.contracts import RegimeLabel


def test_day_manager_regime_router_context_uses_youtube_tiebreaker():
    dm = DayManager.__new__(DayManager)
    dm.regime_router_active = True
    dm.regime_router_context = {}
    dm.youtube_context = {"regime": "CRISIS", "regime_confidence": 90}
    dm._feature_cache = {}
    dm._build_signal_pipeline_price_data = lambda tickers: None

    class _Router:
        def get_current_regime(self, *args, **kwargs):
            return SimpleNamespace(regime=RegimeLabel.TREND, confidence=0.4, method="router")

        def resolve_youtube_divergence(
            self,
            quantitative_regime,
            quantitative_confidence,
            youtube_regime,
            youtube_confidence,
        ):
            return {
                "regime": RegimeLabel.CRISIS,
                "confidence": 0.9,
                "method": "youtube_tiebreaker",
                "weights": {"quantitative": 0.6, "youtube": 0.4},
            }

    dm.regime_router = _Router()

    out = DayManager._refresh_regime_router_context(dm)

    assert out["regime"] == "crisis"
    assert out["method"] == "youtube_tiebreaker"
    assert out["weights"]["youtube"] == 0.4
