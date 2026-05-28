import json
from pathlib import Path

from tools.decision_claw_effectiveness_report import build_report


def test_decision_claw_effectiveness_report_uses_local_fixtures(tmp_path):
    date = "2026-05-21"
    logs = Path(tmp_path) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / f"decision_claw_decisions_{date}.jsonl").write_text(
        json.dumps({"phase_agent": "market_state", "decision": "deploy"}) + "\n",
        encoding="utf-8",
    )
    (logs / f"decision_claw_actions_{date}.jsonl").write_text(
        json.dumps(
            {
                "phase_agent": "market_state",
                "actions": [{"action_type": "submit_entry", "symbol": "GLNG"}],
                "rejected_actions": [
                    {
                        "action": "trim_position",
                        "symbol": "YPF",
                        "reason": "confidence_floor_blocked",
                        "confidence": 0.3,
                        "confidence_floor": 0.6,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / f"decision_claw_parse_failures_{date}.jsonl").write_text(
        json.dumps({"reason": "json_decode"}) + "\n",
        encoding="utf-8",
    )
    (logs / "trade_journal.json").write_text(
        json.dumps(
            [
                {"timestamp": f"{date}T15:00:00", "symbol": "GLNG"},
                {"timestamp": "2026-05-20T15:00:00", "symbol": "OLD"},
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(date=date, root=Path(tmp_path))

    assert "Decisions: 1" in report
    assert "submit_entry: 1" in report
    assert "confidence_floor_blocked: 1" in report
    assert "GLNG" in report
    assert "json_decode: 1" in report


def test_decision_claw_effectiveness_report_filters_window_and_matches_outcomes(
    tmp_path,
):
    date = "2026-05-21"
    logs = Path(tmp_path) / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    (logs / f"decision_claw_decisions_{date}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": f"{date}T09:00:00",
                        "phase_agent": "market_state",
                    }
                ),
                json.dumps(
                    {
                        "timestamp": f"{date}T10:00:00",
                        "phase_agent": "market_state",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / f"decision_claw_actions_{date}.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "timestamp": f"{date}T09:00:00",
                        "phase_agent": "market_state",
                        "actions": [
                            {"action_type": "submit_entry", "symbol": "OLD"}
                        ],
                    }
                ),
                json.dumps(
                    {
                        "timestamp": f"{date}T10:00:00",
                        "phase_agent": "market_state",
                        "actions": [
                            {"action_type": "submit_entry", "symbol": "GLNG"},
                            {"action_type": "trim_position", "symbol": "YPF"},
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (logs / "trade_journal.json").write_text(
        json.dumps(
            [
                {
                    "entry_time": f"{date}T09:05:00",
                    "symbol": "OLD",
                    "trade_type": "entry",
                    "pnl_percent": -2.0,
                },
                {
                    "entry_time": f"{date}T10:05:00+00:00",
                    "symbol": "GLNG",
                    "trade_type": "entry",
                    "pnl_percent": 3.0,
                },
                {
                    "timestamp": f"{date}T10:10:00",
                    "symbol": "YPF",
                    "trade_type": "trim",
                },
            ]
        ),
        encoding="utf-8",
    )

    report = build_report(
        date=date,
        root=Path(tmp_path),
        since=f"{date}T09:30:00",
        max_match_hours=1,
    )

    assert "Since: 2026-05-21T09:30:00" in report
    assert "Decisions: 1" in report
    assert "Submit-entry signals matched to journal entries/exits: 1 / 1 (100.0%)" in report
    assert "Trim signals matched to journal trims: 1 / 1 (100.0%)" in report
    assert "Matched closed-trade average P&L: 3.00%" in report
    assert "OLD" not in report
