from __future__ import annotations

import json
from types import SimpleNamespace

from tools.intraday_market_analyzer import (
    build_intraday_analysis,
    calculate_dispersion_score,
    classify_market_adaptation_regime,
    extract_execution_diagnostics_from_lines,
    market_adaptation_sizing_multiplier,
    write_intraday_analysis,
)


def test_dispersion_score_flags_flat_index_high_vix():
    score = calculate_dispersion_score(spy_pct=-0.10, vix=27.1)

    assert score > 1.0
    regime = classify_market_adaptation_regime(spy_pct=-0.10, vix=27.1)
    assert regime == "DISPERSION"
    assert (
        market_adaptation_sizing_multiplier(regime_label=regime, dispersion_score=score)
        > 1.0
    )


def test_build_intraday_analysis_labels_dispersion_and_book_rankings():
    payload = build_intraday_analysis(
        market_context={"spy_pct": -0.10, "vix": 27.1, "regime_label": "NEUTRAL"},
        positions=[
            SimpleNamespace(
                symbol="WIN",
                unrealized_pl=125.0,
                unrealized_plpc=0.06,
                market_value=2000.0,
                sector="technology",
            ),
            SimpleNamespace(
                symbol="LOSE",
                unrealized_pl=-75.0,
                unrealized_plpc=-0.03,
                market_value=2500.0,
                sector="consumer_cyclical",
            ),
        ],
        execution_diagnostics={"short_engine_ran": True},
        timestamp="2026-05-15T12:00:00",
    )

    assert payload["market_context"]["regime_label"] == "DISPERSION"
    assert payload["market_context"]["sizing_multiplier"] > 1.0
    assert payload["book_performance"]["total_unrealized_pnl"] == 50.0
    assert payload["book_performance"]["top_winners"][0]["sym"] == "WIN"
    assert payload["book_performance"]["top_losers"][0]["sym"] == "LOSE"
    assert payload["market_context"]["sector_leaders"] == ["technology"]
    assert payload["market_context"]["sector_laggards"] == ["consumer_cyclical"]


def test_write_intraday_analysis_writes_snapshot_and_history(tmp_path):
    payload = build_intraday_analysis(
        market_context={"spy_pct": 0.0, "vix": 20.0, "regime_label": "NEUTRAL"},
        positions=[],
        execution_diagnostics={},
        timestamp="2026-05-15T12:00:00",
    )
    snapshot = tmp_path / "data" / "intraday_analysis.json"
    history = tmp_path / "logs" / "intraday_analysis_2026-05-15.jsonl"

    write_intraday_analysis(payload, snapshot_path=snapshot, history_path=history)

    assert json.loads(snapshot.read_text(encoding="utf-8"))["timestamp"].startswith(
        "2026-05-15"
    )
    assert len(history.read_text(encoding="utf-8").strip().splitlines()) == 1


def test_replayed_logs_surface_replacement_narrowness_anomaly():
    diagnostics = extract_execution_diagnostics_from_lines(
        [
            "09:30 [REPLACEMENT REJECTED] CYTK->NCNO reason=l2_bearish_imbalance",
            "09:37 [REPLACEMENT REJECTED] CYTK->PCT reason=vwap_overextended",
            "09:44 [REPLACEMENT REJECTED] CYTK->MIGI reason=entry_gap_too_large",
            "09:54 [REPLACEMENT] CYTK exhausted 3 candidate attempts this cycle",
        ]
    )
    payload = build_intraday_analysis(
        market_context={"spy_pct": -0.10, "vix": 27.1, "regime_label": "NEUTRAL"},
        positions=[
            {
                "symbol": "ASAN",
                "unrealized_pl": 923.0,
                "unrealized_plpc": 0.0644,
                "market_value": 14332.0,
                "sector": "technology",
            },
            {
                "symbol": "NCLH",
                "unrealized_pl": -303.0,
                "unrealized_plpc": -0.0381,
                "market_value": 7952.0,
                "sector": "consumer_cyclical",
            },
        ],
        execution_diagnostics=diagnostics,
        timestamp="2026-05-15T12:00:00",
    )

    assert payload["market_context"]["regime_label"] == "DISPERSION"
    assert payload["market_context"]["sector_leaders"] == ["technology"]
    assert payload["market_context"]["sector_laggards"] == ["consumer_cyclical"]
    assert "replacement_engine_narrowness" in payload["anomalies"]
    assert (
        payload["execution_diagnostics"]["replacement_rejection_reasons"][
            "entry_gap_too_large"
        ]
        == 1
    )


def test_build_intraday_analysis_default_partial_flags_are_false():
    """day-manager 2026-05-19: when the writer has all inputs, the payload
    must carry partial=False, missing_fields=[], and a non-empty
    generated_at alongside timestamp."""
    payload = build_intraday_analysis(
        market_context={"spy_pct": -0.50, "vix": 18.0, "regime_label": "ROTATION"},
        positions=[],
        execution_diagnostics={},
        timestamp="2026-05-19T14:00:00",
    )
    assert payload["partial"] is False
    assert payload["missing_fields"] == []
    assert payload["generated_at"] == "2026-05-19T14:00:00"
    assert payload["timestamp"] == payload["generated_at"]


def test_build_intraday_analysis_records_partial_and_missing_fields():
    """day-manager 2026-05-19: when an upstream input fails, the writer
    must still emit the snapshot with partial=True and the exact list of
    missing fields recorded — never silently fall back to zeros without
    saying so."""
    payload = build_intraday_analysis(
        market_context={"spy_pct": 0.0, "vix": 0.0, "regime_label": "NEUTRAL"},
        positions=[],
        execution_diagnostics={},
        timestamp="2026-05-19T14:30:00",
        partial=True,
        missing_fields=["spy_pct", "vix"],
    )
    assert payload["partial"] is True
    assert payload["missing_fields"] == ["spy_pct", "vix"]
    assert payload["generated_at"] == "2026-05-19T14:30:00"
