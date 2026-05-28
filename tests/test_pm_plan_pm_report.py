from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent


def test_pm_report_top_entries_populate_full_watchlist(tmp_path):
    plan_path = tmp_path / "pm_plan_test.json"
    plan_path.write_text('{"signals": []}', encoding="utf-8")

    pm_report = {
        "date": "2026-03-10",
        "top_entries": [
            {"symbol": "AAA", "score": 88.5, "has_catalyst": True},
            {"symbol": "BBB", "score": 72.1, "has_catalyst": False},
        ],
    }

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent._coerce_float = lambda value, default=0.0: float(value if value is not None else default)
    agent.logger = SimpleNamespace(debug=lambda *args, **kwargs: None)
    agent._persist_pm_report_to_plan(pm_report, plan_path)

    updated = plan_path.read_text(encoding="utf-8")
    plan = __import__("json").loads(updated)

    assert plan["pm_report"]["top_entries"][0]["symbol"] == "AAA"
    assert [row["symbol"] for row in plan.get("signals", [])] == ["AAA", "BBB"]
    assert [row["symbol"] for row in plan.get("actionable_top50", [])] == ["AAA", "BBB"]
    symbols = [row["symbol"] for row in plan.get("full_watchlist", [])]
    assert symbols == ["AAA", "BBB"]
    assert plan["full_watchlist"][0]["final_score"] == 88.5
    assert plan["full_watchlist"][0]["source"] == "pm_report"
    assert plan["signals"][0]["entry_source"] == "pm_report"
    assert plan["pm_report_status"]["status"] == "seeded_top_entries"
    assert plan["pm_report_status"]["signals_count"] == 2
