import json
from pathlib import Path

import pandas as pd

from autotrade.analysis.sequential_outcome_scorer import (
    _find_trade_by_id,
    _future_profile,
    _future_return_pct,
    _load_journal_trades,
    build_comparative_report,
    score_event_outcome,
)


def test_score_entry_event_basic(monkeypatch):
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._find_trade_by_id",
        lambda trade_id: {"id": trade_id, "outcome": "loss"},
    )
    event = {
        "event_id": "evt1",
        "symbol": "AAPL",
        "event_type": "buy",
        "trade_id": "t1",
        "fill_price": 100.0,
        "fill_time": "2026-02-26T10:00:00",
    }
    pred = {"recommended_action": "avoid"}
    out = score_event_outcome(event, pred, horizon_minutes=120)
    assert out.event_id == "evt1"
    assert out.sequential_more_accurate is True


def test_score_exit_event_hold_counterfactual(monkeypatch):
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._future_return_pct",
        lambda symbol, start_time_iso, start_price, horizon_minutes: 1.2,
    )
    event = {
        "event_id": "evt2",
        "symbol": "AAPL",
        "event_type": "exit",
        "fill_price": 100.0,
        "fill_time": "2026-02-26T10:00:00",
    }
    pred = {"recommended_action": "hold"}
    out = score_event_outcome(event, pred, horizon_minutes=120)
    assert out.event_id == "evt2"
    assert out.sequential_more_accurate is True


def test_score_event_includes_metric_breakdown(monkeypatch):
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._find_trade_by_id",
        lambda trade_id: None,
    )
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._future_profile",
        lambda symbol, start_time_iso, start_price, horizon_minutes: {
            "ret_pct": 1.1,
            "vol_pct": 0.3,
            "max_favorable_pct": 1.5,
            "max_adverse_pct": -0.4,
        },
    )
    event = {
        "event_id": "evt3",
        "symbol": "AAPL",
        "event_type": "buy",
        "fill_price": 100.0,
        "fill_time": "2026-02-26T10:00:00",
    }
    pred = {"recommended_action": "buy"}
    out = score_event_outcome(event, pred, horizon_minutes=120)
    assert "profitability_actual" in out.metric_breakdown
    assert "risk_adjusted_actual" in out.metric_breakdown
    assert "timing_actual" in out.metric_breakdown
    assert "total_actual" in out.metric_breakdown


def test_build_comparative_report_groups_by_event_type():
    report = build_comparative_report(
        [
            {
                "event_id": "e1",
                "event_type": "buy",
                "score_delta": 0.5,
                "sequential_more_accurate": True,
                "baseline_action": "buy",
                "sequential_action": "avoid",
            },
            {
                "event_id": "e2",
                "event_type": "exit",
                "score_delta": -0.2,
                "sequential_more_accurate": False,
                "baseline_action": "exit",
                "sequential_action": "hold",
            },
        ]
    )
    assert report["total_events"] == 2
    assert report["by_event_type"]["buy"]["count"] == 1
    assert report["by_event_type"]["exit"]["count"] == 1
    assert report["action_pairs"]["buy->avoid"]["count"] == 1


def test_build_comparative_report_empty():
    report = build_comparative_report([])
    assert report["total_events"] == 0
    assert report["by_event_type"] == {}


def test_load_journal_trades_and_find_trade_by_id(tmp_path: Path, monkeypatch):
    journal_path = tmp_path / "trade_journal.json"
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer.JOURNAL_PATH", journal_path
    )

    assert _load_journal_trades() == []
    journal_path.write_text("{bad json", encoding="utf-8")
    assert _load_journal_trades() == []

    journal_path.write_text(
        json.dumps({"trades": [{"id": "t1", "outcome": "win"}]}), encoding="utf-8"
    )
    trades = _load_journal_trades()
    assert len(trades) == 1
    assert _find_trade_by_id("t1")["outcome"] == "win"
    assert _find_trade_by_id("missing") is None


def test_future_return_and_profile_from_history(monkeypatch):
    idx = pd.date_range("2026-02-27 10:00:00", periods=6, freq="5min", tz="UTC")
    hist = pd.DataFrame({"Close": [100, 100.5, 101, 100.8, 101.2, 101.5]}, index=idx)

    class _Ticker:
        def history(self, period="5d", interval="5m"):
            return hist

    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer.yf.Ticker",
        lambda symbol: _Ticker(),
    )

    ret = _future_return_pct("AAPL", "2026-02-27T10:00:00", 100.0, horizon_minutes=60)
    assert ret > 0

    profile = _future_profile("AAPL", "bad-time", 100.0, horizon_minutes=60)
    assert "ret_pct" in profile
    assert "vol_pct" in profile
    assert "max_favorable_pct" in profile


def test_future_profile_graceful_on_history_exception(monkeypatch):
    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer._future_return_pct",
        lambda symbol, start_time_iso, start_price, horizon_minutes: -0.8,
    )

    class _Ticker:
        def history(self, period="5d", interval="5m"):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        "autotrade.analysis.sequential_outcome_scorer.yf.Ticker",
        lambda symbol: _Ticker(),
    )
    profile = _future_profile("AAPL", "2026-02-27T10:00:00", 100.0, horizon_minutes=60)
    assert profile["ret_pct"] == -0.8
    assert profile["max_adverse_pct"] <= 0
