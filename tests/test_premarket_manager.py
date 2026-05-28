from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from autotrade.core.premarket_manager import (
    ET,
    PHASE_MARKET_CONTEXT,
    PHASE_OPEN_PREP,
    PHASE_OUTSIDE_WINDOW,
    PHASE_POSITION_CHECKS,
    PHASE_WATCHLIST_SCAN,
    PremarketManager,
)
from autotrade.core.research_artifacts import (
    ResearchArtifactBundle,
    save_research_artifact_bundle,
)
from autotrade.utils.premarket_vwap import (
    PremarketVWAPTracker,
    STATE_STRONG_ABOVE,
    STATE_STRONG_BELOW,
)


class DummyAnalyzer:
    def analyze_ticker(self, symbol: str):
        symbol = symbol.upper()
        mapping = {
            "AAA": SimpleNamespace(
                has_data=True,
                gap_pct=3.2,
                volume_ratio=2.1,
                premarket_volume=250_000,
                premarket_trend="bullish",
            ),
            "BBB": SimpleNamespace(
                has_data=True,
                gap_pct=-2.8,
                volume_ratio=0.7,
                premarket_volume=90_000,
                premarket_trend="bearish",
            ),
            "SPY": SimpleNamespace(
                has_data=True,
                gap_pct=0.6,
                volume_ratio=1.0,
                premarket_volume=1_000_000,
                premarket_trend="bullish",
            ),
            "QQQ": SimpleNamespace(
                has_data=True,
                gap_pct=0.4,
                volume_ratio=1.0,
                premarket_volume=900_000,
                premarket_trend="bullish",
            ),
            "XLK": SimpleNamespace(
                has_data=True,
                gap_pct=1.1,
                volume_ratio=1.0,
                premarket_volume=120_000,
                premarket_trend="bullish",
            ),
            "XLF": SimpleNamespace(
                has_data=True,
                gap_pct=-0.7,
                volume_ratio=1.0,
                premarket_volume=110_000,
                premarket_trend="bearish",
            ),
        }
        return mapping.get(
            symbol,
            SimpleNamespace(
                has_data=False,
                gap_pct=0.0,
                volume_ratio=0.0,
                premarket_volume=0,
                premarket_trend="flat",
            ),
        )

    def get_premarket_bars(self, symbol: str):
        if symbol.upper() == "AAA":
            return [
                {"open": 10.0, "high": 10.4, "low": 9.9, "close": 10.3, "volume": 1000},
                {"open": 10.3, "high": 10.5, "low": 10.2, "close": 10.45, "volume": 1200},
                {"open": 10.45, "high": 10.6, "low": 10.4, "close": 10.55, "volume": 1400},
            ]
        if symbol.upper() == "BBB":
            return [
                {"open": 8.0, "high": 8.05, "low": 7.8, "close": 7.85, "volume": 900},
                {"open": 7.85, "high": 7.9, "low": 7.7, "close": 7.75, "volume": 1100},
                {"open": 7.75, "high": 7.8, "low": 7.6, "close": 7.65, "volume": 1300},
            ]
        return []


class DummyNews:
    def collect(self, symbol: str):
        return {
            "symbol": symbol,
            "available": True,
            "sentiment_score": 0.3 if symbol == "AAA" else -0.2,
            "headline_count": 3,
            "coverage": "full",
            "confidence": 0.8,
            "headlines": [{"title": f"{symbol} catalyst"}],
            "source_status": {"news_sentiment": "ok"},
            "has_catalyst": symbol == "AAA",
            "catalyst_score": 0.7 if symbol == "AAA" else 0.0,
            "catalyst_tags": ["earnings"] if symbol == "AAA" else [],
            "catalyst_note": "Earnings beat" if symbol == "AAA" else "",
        }


class DummyStocktwits:
    def fetch(self, symbol: str):
        return {
            "symbol": symbol,
            "available": True,
            "sentiment_score": 0.2 if symbol == "AAA" else -0.1,
            "bull_bear_ratio": 1.8,
            "message_velocity": 12.0,
            "is_trending": symbol == "AAA",
            "coverage": "partial",
            "confidence": 0.6,
            "source_status": "ok",
        }


class DeterministicPremarketManager(PremarketManager):
    def _safe_vix_level(self):
        return 18.5


def _build_manager(tmp_path: Path) -> PremarketManager:
    return DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=DummyAnalyzer(),
        news_aggregator=DummyNews(),
        stocktwits_scraper=DummyStocktwits(),
        max_watchlist_symbols=10,
    )


@pytest.fixture(autouse=True)
def mock_external_state(monkeypatch):
    """Ensure tests are independent of real project state by mocking gates."""
    # Mock overnight workflow gate
    monkeypatch.setattr(
        "autotrade.core.premarket_manager.check_research_freshness",
        lambda **kwargs: {
            "workflow_complete": True,
            "is_fresh": True,
            "workflow_reason": "mocked",
        },
    )
    # Mock YouTube intelligence context
    monkeypatch.setattr(
        "autotrade.utils.youtube_readiness.get_intelligence_context",
        lambda **kwargs: {
            "available": True,
            "regime": "RISK-ON",
            "sizing_multiplier": 0.6,
            "avoid_sectors": [],
            "favor_sectors": ["Technology"],
        },
    )


def test_phase_transitions_et_windows(tmp_path):
    manager = _build_manager(tmp_path)
    assert manager.resolve_phase(datetime(2026, 2, 11, 4, 30, tzinfo=ET)) == PHASE_MARKET_CONTEXT
    assert manager.resolve_phase(datetime(2026, 2, 11, 6, 15, tzinfo=ET)) == PHASE_POSITION_CHECKS
    assert manager.resolve_phase(datetime(2026, 2, 11, 7, 45, tzinfo=ET)) == PHASE_WATCHLIST_SCAN
    assert manager.resolve_phase(datetime(2026, 2, 11, 9, 0, tzinfo=ET)) == PHASE_OPEN_PREP
    assert manager.resolve_phase(datetime(2026, 2, 11, 10, 0, tzinfo=ET)) == PHASE_OUTSIDE_WINDOW


def test_ranked_watchlist_schema_and_artifact_integrity(tmp_path):
    manager = _build_manager(tmp_path)
    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        holdings=["AAA"],
        now_et=datetime(2026, 2, 11, 8, 45, tzinfo=ET),
    )

    assert handoff["phase"] == PHASE_OPEN_PREP
    assert handoff["ranked_watchlist"]
    assert handoff["ranked_watchlist"][0]["symbol"] == "AAA"
    assert "score" in handoff["ranked_watchlist"][0]
    assert "rationale" in handoff["ranked_watchlist"][0]
    assert "has_catalyst" in handoff["ranked_watchlist"][0]
    assert "catalyst_score" in handoff["ranked_watchlist"][0]
    assert "catalyst_tags" in handoff["ranked_watchlist"][0]
    assert "s1_price" in handoff["ranked_watchlist"][0]
    assert "r1_price" in handoff["ranked_watchlist"][0]
    assert "sr_quality_score" in handoff["ranked_watchlist"][0]
    assert "open_alerts" in handoff
    assert handoff["artifact_path"]

    artifact = Path(handoff["artifact_path"])
    assert artifact.exists()
    latest = tmp_path / "morning_intelligence_latest.json"
    assert latest.exists()
    assert handoff["market_context"]["youtube_regime"] in {"RISK-ON", "LEAN-BULLISH"}
    assert handoff["market_context"]["position_sizing_multiplier"] == 0.6


def test_persist_handoff_uses_pm_plan_date_for_after_close_artifacts(tmp_path):
    manager = _build_manager(tmp_path)
    stamped = manager._persist_handoff(
        {
            "phase": PHASE_OPEN_PREP,
            "ranked_watchlist": [{"symbol": "AAA", "score": 88.0}],
            "scalp_watchlist": [],
            "adjust_plan_reasons": [],
        },
        now_et=datetime(2026, 4, 27, 17, 5, tzinfo=ET),
    )

    assert stamped.name.startswith("morning_intelligence_20260428_1705")
    latest = json.loads((tmp_path / "morning_intelligence_latest.json").read_text())
    stamped_payload = json.loads(stamped.read_text())
    assert latest["trade_date"] == "2026-04-28"
    assert latest["report_date"] == "2026-04-28"
    assert stamped_payload["trade_date"] == "2026-04-28"


def test_vwap_state_updates_from_synthetic_one_minute_bars():
    tracker = PremarketVWAPTracker(min_bars_for_signal=3, strong_threshold_pct=0.10, strong_streak_bars=2)
    strong_above = tracker.update_many(
        [
            {"high": 10.2, "low": 9.9, "close": 10.1, "volume": 1000},
            {"high": 10.4, "low": 10.1, "close": 10.35, "volume": 1100},
            {"high": 10.6, "low": 10.3, "close": 10.55, "volume": 1200},
        ]
    )
    assert strong_above.state == STATE_STRONG_ABOVE

    tracker.reset()
    strong_below = tracker.update_many(
        [
            {"high": 10.2, "low": 9.9, "close": 10.0, "volume": 1000},
            {"high": 10.0, "low": 9.7, "close": 9.75, "volume": 1100},
            {"high": 9.8, "low": 9.5, "close": 9.55, "volume": 1200},
        ]
    )
    assert strong_below.state == STATE_STRONG_BELOW


def test_partial_data_source_failures_degrade_gracefully(tmp_path):
    class BrokenNews:
        def collect(self, _symbol: str):
            raise RuntimeError("news down")

    class BrokenStocktwits:
        def fetch(self, _symbol: str):
            raise RuntimeError("stocktwits down")

    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=DummyAnalyzer(),
        news_aggregator=BrokenNews(),
        stocktwits_scraper=BrokenStocktwits(),
    )
    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 8, 45, tzinfo=ET),
    )

    assert handoff["ranked_watchlist"]
    candidate = handoff["ranked_watchlist"][0]
    assert candidate["news"]["available"] is False
    assert candidate["stocktwits"]["available"] is False
    assert isinstance(handoff["coverage"]["degraded_mode"], bool)


def test_premarket_cycle_waits_for_overnight_completion_before_fallback_deadline(
    tmp_path,
):
    manager = _build_manager(tmp_path)
    manager._overnight_gate_status = lambda timestamp: {
        "workflow_complete": False,
        "workflow_reason": "research_complete_false",
        "age_hours": 0.5,
        "max_age_hours": 18.0,
        "is_fresh": False,
        "should_wait": True,
        "fallback_ready": False,
        "fallback_deadline_et": "08:15",
        "retry_after_seconds": 300,
        "freshness": {"workflow_complete": False},
    }

    result = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 15, tzinfo=ET),
    )

    assert result["status"] == "waiting_for_overnight"
    assert result["error"] == "overnight_workflow_incomplete"
    assert result["retry_after_seconds"] == 300
    assert result["overnight_gate"]["should_wait"] is True


def test_premarket_cycle_uses_fallback_after_deadline_when_overnight_incomplete(
    tmp_path,
):
    manager = _build_manager(tmp_path)
    manager._overnight_gate_status = lambda timestamp: {
        "workflow_complete": False,
        "workflow_reason": "research_complete_false",
        "age_hours": 1.5,
        "max_age_hours": 18.0,
        "is_fresh": False,
        "should_wait": False,
        "fallback_ready": True,
        "fallback_deadline_et": "08:15",
        "retry_after_seconds": 0,
        "freshness": {"workflow_complete": False},
    }

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        holdings=["AAA"],
        now_et=datetime(2026, 2, 11, 8, 20, tzinfo=ET),
    )

    assert handoff["phase"] == PHASE_WATCHLIST_SCAN
    assert handoff["overnight_fallback_used"] is True
    assert handoff["overnight_gate"]["fallback_ready"] is True
    assert handoff["degraded_mode"] is True
    assert handoff["fallback_reason"] == "overnight_incomplete:research_complete_false"
    assert handoff["ranked_watchlist"]


def _get_row(symbol: str, handoff: dict) -> dict:
    return next(r for r in handoff["ranked_watchlist"] if r["symbol"] == symbol)


def test_rvol_window_boosts_high_volume_with_momentum(tmp_path):
    """High RVOL + bullish momentum inside 6:30-8:30 ET should get a boost."""
    manager = _build_manager(tmp_path)
    inside = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 15, tzinfo=ET),
    )
    outside = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 9, 5, tzinfo=ET),
    )

    aaa_inside = _get_row("AAA", inside)
    aaa_outside = _get_row("AAA", outside)
    bbb_inside = _get_row("BBB", inside)

    assert aaa_inside["high_rvol"] is True
    assert aaa_inside["rvol_window"] is True
    assert aaa_inside["momentum_signal"] == "bullish"
    assert aaa_inside["score"] > aaa_outside["score"]
    assert bbb_inside["high_rvol"] is False


def test_high_rvol_bearish_penalized(tmp_path):
    """High RVOL with bearish momentum should be detected and penalized."""

    class BearishHighRvolAnalyzer(DummyAnalyzer):
        def analyze_ticker(self, symbol: str):
            if symbol.upper() == "CCC":
                return SimpleNamespace(
                    has_data=True,
                    gap_pct=-3.0,
                    volume_ratio=2.5,
                    premarket_volume=220_000,
                    premarket_trend="bearish",
                )
            return super().analyze_ticker(symbol)

        def get_premarket_bars(self, symbol: str):
            if symbol.upper() == "CCC":
                return [
                    {"open": 9.0, "high": 9.05, "low": 8.8, "close": 8.85, "volume": 2000},
                    {"open": 8.85, "high": 8.9, "low": 8.6, "close": 8.65, "volume": 2400},
                    {"open": 8.65, "high": 8.7, "low": 8.4, "close": 8.45, "volume": 2600},
                ]
            return super().get_premarket_bars(symbol)

    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=BearishHighRvolAnalyzer(),
        news_aggregator=DummyNews(),
        stocktwits_scraper=DummyStocktwits(),
        max_watchlist_symbols=5,
    )

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "CCC"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 40, tzinfo=ET),
    )
    row = _get_row("CCC", handoff)

    assert row["high_rvol"] is True
    assert row["momentum_signal"] == "bearish"
    assert row["rvol_window"] is True
    assert any("bearish momentum" in r.lower() for r in row["rationale"])


class MutableAnalyzer(DummyAnalyzer):
    def __init__(self, high_vol: bool = False):
        super().__init__()
        self.high_vol = high_vol

    def analyze_ticker(self, symbol: str):
        base = super().analyze_ticker(symbol)
        if symbol.upper() == "AAA":
            vr = 2.4 if self.high_vol else 1.4
            return SimpleNamespace(
                has_data=True,
                gap_pct=2.0,
                volume_ratio=vr,
                premarket_volume=220_000,
                premarket_trend="bullish",
            )
        return base


class CatalystNews(DummyNews):
    def __init__(self, with_catalyst: bool = True):
        self.with_catalyst = with_catalyst

    def collect(self, symbol: str):
        base = super().collect(symbol)
        base["has_catalyst"] = self.with_catalyst and symbol.upper() == "AAA"
        base["catalyst_score"] = 0.9 if base["has_catalyst"] else 0.0
        base["catalyst_tags"] = ["contract"] if base["has_catalyst"] else []
        return base


def test_adjust_plan_triggers_on_rvol_delta(tmp_path):
    analyzer = MutableAnalyzer(high_vol=False)
    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=analyzer,
        news_aggregator=CatalystNews(with_catalyst=False),
        stocktwits_scraper=DummyStocktwits(),
    )

    first = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 5, tzinfo=ET),
    )
    assert first["adjust_plan_triggered"] is True
    assert first["adjust_plan_reasons"]

    second = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 10, tzinfo=ET),
    )
    assert second["adjust_plan_triggered"] is False

    analyzer.high_vol = True
    third = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 15, tzinfo=ET),
    )
    assert third["adjust_plan_triggered"] is True
    assert any("RVOL delta" in r for r in third["adjust_plan_reasons"])


def test_scalp_watchlist_filters_high_rvol_with_catalyst(tmp_path):
    analyzer = MutableAnalyzer(high_vol=True)
    news = CatalystNews(with_catalyst=True)
    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=analyzer,
        news_aggregator=news,
        stocktwits_scraper=DummyStocktwits(),
    )

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}, {"ticker": "BBB"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 7, 20, tzinfo=ET),
    )

    assert handoff["scalp_watchlist"]
    symbols = [row["symbol"] for row in handoff["scalp_watchlist"]]
    assert "AAA" in symbols
    assert "BBB" not in symbols


def test_premarket_manager_prefers_research_artifact_bundle(tmp_path):
    bundle = ResearchArtifactBundle(
        trade_date="2026-02-11",
        generated_at_et="2026-02-11T04:00:00-05:00",
        full_watchlist=[
            {"symbol": "AAA", "final_score": 91.0},
            {"symbol": "BBB", "final_score": 78.0},
        ],
        top_picks=[{"symbol": "BBB", "final_score": 78.0}],
        catalysts={
            "BBB": {"score": 0.8, "tags": ["earnings"], "note": "Overnight catalyst"}
        },
        support_resistance={"BBB": {"s1_price": 7.6, "r1_price": 8.4}},
    )
    save_research_artifact_bundle(bundle, output_dir=tmp_path)

    manager = DeterministicPremarketManager(
        output_dir=tmp_path,
        premarket_analyzer=DummyAnalyzer(),
        news_aggregator=DummyNews(),
        stocktwits_scraper=DummyStocktwits(),
        max_watchlist_symbols=10,
    )

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 8, 45, tzinfo=ET),
    )

    symbols = [row["symbol"] for row in handoff["ranked_watchlist"]]
    assert "BBB" in symbols
    assert handoff["research_bundle"]["loaded"] is True
    assert handoff["research_bundle"]["top_pick_symbols"] == ["BBB"]


def test_premarket_manager_merges_live_momentum_watchlist(tmp_path):
    artifact_path = tmp_path / "momentum_watchlist_live.json"
    artifact_path.write_text(
        json.dumps(
            {
                "generated_at_et": "2026-02-11T08:40:00-05:00",
                "session": "premarket",
                "scan_count": 1,
                "symbols": [
                    {
                        "ticker": "LMND",
                        "score": 84.0,
                        "entry_source": "momentum_scanner",
                        "source_bucket": "watchlist",
                        "intraday_reserve": True,
                        "catalyst_summary": "News-led continuation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manager = _build_manager(tmp_path)
    manager.momentum_scanner_cfg = SimpleNamespace(enabled=True, artifact_path=artifact_path)

    handoff = manager.run_cycle(
        watchlist=[{"ticker": "AAA"}],
        holdings=[],
        now_et=datetime(2026, 2, 11, 8, 45, tzinfo=ET),
    )

    symbols = [row["symbol"] for row in handoff["ranked_watchlist"]]
    assert "LMND" in symbols
    lmnd = _get_row("LMND", handoff)
    assert lmnd["entry_source"] == "momentum_scanner"
    assert handoff["momentum_scanner"]["loaded"] is True


def test_load_market_intelligence_reads_utf8_report(tmp_path, monkeypatch):
    rag_dir = tmp_path / "data" / "youtube" / "rag" / "daily_reports"
    rag_dir.mkdir(parents=True)
    report_path = rag_dir / "2026-02-11_consolidated.json"
    report_path.write_text(
        json.dumps(
            {
                "regime": {"classification": "RISK_ON"},
                "trading_signals": {"headline": "Momentum \u201cconfirmed\u201d"},
            }
        ),
        encoding="utf-8",
    )

    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = datetime(2026, 2, 11, 8, 45)
            return base if tz is None else base.replace(tzinfo=tz)

    manager = _build_manager(tmp_path)
    monkeypatch.setattr("autotrade.core.premarket_manager.PROJECT_DIR", tmp_path)
    monkeypatch.setattr("autotrade.core.premarket_manager.datetime", FixedDateTime)

    report = manager._load_market_intelligence()

    assert report is not None
    assert report["regime"]["classification"] == "RISK_ON"
    assert report["trading_signals"]["headline"] == "Momentum “confirmed”"
