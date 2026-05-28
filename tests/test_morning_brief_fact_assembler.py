import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from autotrade.reporting.morning_brief.fact_assembler import assemble_facts


NOW = datetime(2026, 5, 21, 11, 0, tzinfo=UTC)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_assemble_facts_is_deterministic_with_mock_sources(tmp_path):
    _write_json(
        tmp_path / "logs" / "signals_2026-05-21.json",
        {
            "regime": "neutral",
            "breadth_pct": 58,
            "signals": [
                {
                    "symbol": "AAA",
                    "entry_price": 10,
                    "stop_loss": 9,
                    "target": 12,
                    "atr_14": 0.7,
                    "weekly_return": 4.2,
                    "final_score": 85,
                    "ranking_score": 90,
                    "fresh_news": True,
                },
                {
                    "symbol": "BBB",
                    "entry_price": 11,
                    "stop_loss": 10,
                    "target": 13,
                    "atr_14": 0.8,
                    "weekly_return": 1.2,
                    "final_score": 95,
                    "ranking_score": 80,
                },
            ],
        },
    )
    _write_json(
        tmp_path / "plans" / "decision_claw_final_watchlist_2026-05-20.json",
        {"selected_symbols": ["AAA"]},
    )
    _write_json(tmp_path / "plans" / "pm_plan_2026-05-20.json", {"symbols": ["AAA"]})
    _write_json(
        tmp_path / "reports" / "historical_bullish_revisit_2026-05-20.json",
        {"rows": [{"symbol": "AAA"}]},
    )
    _write_json(
        tmp_path / "data" / "position_thesis_cache.json",
        [{"symbol": "AAA", "state": "intact"}, {"symbol": "BBB", "state": "broken"}],
    )
    _write_json(
        tmp_path / "logs" / "trade_journal.json",
        [
            {"symbol": "BBB", "date": "2026-05-20", "pnl": -12},
            {"symbol": "BBB", "date": "2026-05-19", "pnl": -3},
        ],
    )
    db_path = tmp_path / "data" / "financial.db"
    conn = sqlite3.connect(db_path)
    conn.execute("create table earnings_calendar(symbol text, report_date text)")
    conn.execute("insert into earnings_calendar values('AAA', '2026-05-23')")
    conn.commit()
    conn.close()

    first = assemble_facts(tmp_path, NOW)
    second = assemble_facts(tmp_path, NOW)

    assert first == second
    assert [row.symbol for row in first.facts] == ["BBB", "AAA"]
    aaa = first.facts[1]
    assert aaa.source_flags == frozenset(
        {
            "overnight_pick",
            "claw_yesterday",
            "pm_plan",
            "strength_reentry",
            "thesis_intact",
        }
    )
    assert "earnings_within_5d" in aaa.catalyst_flags
    assert aaa.thesis_state == "intact"
    assert first.facts[0].recent_loss_count == 2
    assert first.facts[0].score == 95


def test_missing_signals_returns_empty_facts_and_freshness_gap(tmp_path):
    result = assemble_facts(tmp_path, NOW)

    assert result.facts == []
    assert result.freshness[0].name == "overnight_signals"
    assert result.freshness[0].fresh is False
    assert result.freshness[0].message == "missing source"


def test_latest_signal_prefers_filename_date_over_modified_time(tmp_path):
    older_by_name = tmp_path / "logs" / "signals_2026-05-20.json"
    newer_by_mtime = tmp_path / "logs" / "signals_2026-04-17.json"
    _write_json(
        older_by_name,
        {
            "signals": [
                {
                    "symbol": "CURR",
                    "entry_price": 5,
                    "stop_loss": 4.5,
                    "target": 6,
                    "ranking_score": 2,
                }
            ]
        },
    )
    _write_json(
        newer_by_mtime,
        {
            "signals": [
                {
                    "symbol": "OLD",
                    "entry_price": 5,
                    "stop_loss": 4.5,
                    "target": 6,
                    "ranking_score": 9,
                }
            ]
        },
    )
    os.utime(newer_by_mtime, (NOW.timestamp() + 100, NOW.timestamp() + 100))

    result = assemble_facts(tmp_path, NOW)

    assert [fact.symbol for fact in result.facts] == ["CURR"]


def test_regime_dict_is_normalized_for_phone_tape(tmp_path):
    _write_json(
        tmp_path / "logs" / "signals_2026-05-21.json",
        {
            "regime": {"available": True, "regime": "SELLOFF"},
            "signals": [
                {
                    "symbol": "AAA",
                    "entry_price": 10,
                    "stop_loss": 9,
                    "target": 12,
                    "ranking_score": 1,
                }
            ],
        },
    )

    result = assemble_facts(tmp_path, NOW)

    assert result.tape.regime_label == "SELLOFF"
    assert result.tape.breadth_pct is None
    assert result.tape.is_live is False
    assert "not live tape" in result.tape.source_label


def test_same_day_intraday_context_overrides_plan_regime(tmp_path):
    _write_json(
        tmp_path / "logs" / "signals_2026-05-21.json",
        {
            "regime": {"available": True, "regime": "SELLOFF"},
            "signals": [
                {
                    "symbol": "AAA",
                    "entry_price": 10,
                    "stop_loss": 9,
                    "target": 12,
                    "ranking_score": 1,
                }
            ],
        },
    )
    _write_json(
        tmp_path / "data" / "intraday_analysis.json",
        {
            "timestamp": "2026-05-21T09:11:59",
            "market_context": {
                "regime_label": "RISK_ON",
                "breadth_pct": 61.5,
            },
            "tape_inference": "Index and breadth are constructive.",
        },
    )

    result = assemble_facts(tmp_path, NOW)

    assert result.tape.regime_label == "RISK_ON"
    assert result.tape.breadth_pct == 61.5
    assert result.tape.is_live is True


def test_fact_assembler_rejects_known_20260520_broken_rows(tmp_path):
    audited = json.loads(
        Path("logs/signals_2026-05-20.json").read_text(encoding="utf-8")
    )
    broken_symbols = {
        "RVYL",
        "VG",
        "DHC",
        "ATHE",
        "RKLB",
        "FLEX",
        "TSEM",
        "VICR",
        "DOCN",
        "OKLO",
        "SMR",
        "VOYG",
        "RDW",
        "POET",
        "EOSE",
        "MLYS",
        "RYTM",
        "ENPH",
        "CWK",
    }
    rows = [row for row in audited["signals"] if row["symbol"] in broken_symbols]
    _write_json(
        tmp_path / "logs" / "signals_2026-05-20.json",
        {"date": "2026-05-20", "signals": rows},
    )

    result = assemble_facts(tmp_path, NOW)

    gate = next(
        item for item in result.freshness if item.name == "morning_brief_input_gate"
    )
    reject_log = json.loads(Path(gate.reject_path).read_text(encoding="utf-8"))
    assert gate.rejected_count == 19
    assert len(reject_log) == 19
    reasons = {row["symbol"]: row["reason"] for row in reject_log}
    assert reasons["DHC"] == "zero_or_missing_stop"
    assert reasons["ATHE"] == "zero_or_missing_stop"
    assert reasons["RVYL"] == "target_at_or_below_entry"
    assert reasons["VG"] == "rr_below_1.2"
    assert reasons["RKLB"] == "target_at_or_below_entry"
    assert reasons["CWK"] == "target_at_or_below_entry"
    assert result.facts == []


def test_fact_assembler_orders_by_final_score_before_ranking_score(tmp_path):
    _write_json(
        tmp_path / "logs" / "signals_2026-05-21.json",
        {
            "signals": [
                {
                    "symbol": "RANK",
                    "entry_price": 10,
                    "stop_loss": 9,
                    "target": 12,
                    "ranking_score": 100,
                    "final_score": 70,
                },
                {
                    "symbol": "FINAL",
                    "entry_price": 10,
                    "stop_loss": 9,
                    "target": 12,
                    "ranking_score": 80,
                    "final_score": 90,
                },
            ]
        },
    )

    result = assemble_facts(tmp_path, NOW)

    assert [row.symbol for row in result.facts] == ["FINAL", "RANK"]
