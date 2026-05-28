import json
from pathlib import Path
from types import SimpleNamespace

import requests

from autotrade.core import daily_lessons_analyzer as analyzer_mod
from autotrade.core.daily_lessons_analyzer import DailyLessonsAnalyzer


def test_daily_lessons_analyzer_writes_fallback_report(tmp_path, monkeypatch):
    logs_dir = Path(tmp_path) / "logs"
    plans_dir = Path(tmp_path) / "plans"
    reports_dir = Path(tmp_path) / "reports"
    prompts_dir = Path(tmp_path) / "prompts" / "llm"
    logs_dir.mkdir(parents=True)
    plans_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)

    monkeypatch.setattr(analyzer_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(analyzer_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(analyzer_mod, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(analyzer_mod, "PROMPTS_DIR", prompts_dir)

    (logs_dir / "daily_review_2026-03-25.json").write_text(
        json.dumps(
            {
                "grade": "B",
                "net_pnl": 317.16,
                "critical_issues": ["Wave capacity too low"],
                "summary": "Session improved but left capital unused.",
            }
        ),
        encoding="utf-8",
    )
    (logs_dir / "workflow_journal_2026-03-25.jsonl").write_text(
        json.dumps({"symbol": "AAA", "action": "BUY", "reasoning": "breakout"}) + "\n",
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260325.json").write_text(
        json.dumps({"signals": [{"symbol": "AAA", "score": 82.0, "setup": "orb"}]}),
        encoding="utf-8",
    )
    (prompts_dir / "daily_lessons.txt").write_text(
        "Date: {date}\nReview: {daily_review}\nJournal: {workflow_journal}\nPlan: {morning_plan}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analyzer_mod, "get_config", lambda: SimpleNamespace(llm=SimpleNamespace(model_decision="qwen-test"))
    )
    monkeypatch.setattr(
        requests,
        "post",
        lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError("connection refused")),
    )

    analyzer = DailyLessonsAnalyzer()
    result = analyzer.run("2026-03-25")

    report_path = reports_dir / "daily_lessons_2026-03-25.md"
    legacy_path = reports_dir / "post_market_lesson_2026-03-25.json"
    assert result["success"] is True
    assert result["mode"] == "fallback"
    assert report_path.exists()
    assert legacy_path.exists()
    content = report_path.read_text(encoding="utf-8")
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert "Fallback report generated" in content
    assert "Wave capacity too low" in content
    assert legacy_payload["artifact_type"] == "daily_lessons_compatibility_export"
    assert legacy_payload["mode"] == "fallback"
    assert legacy_payload["source_report_path"].endswith("daily_lessons_2026-03-25.md")
    assert "Fallback report generated" in legacy_payload["lesson"]


def test_daily_lessons_analyzer_writes_legacy_compatibility_export_on_llm_success(
    tmp_path, monkeypatch
):
    logs_dir = Path(tmp_path) / "logs"
    plans_dir = Path(tmp_path) / "plans"
    reports_dir = Path(tmp_path) / "reports"
    prompts_dir = Path(tmp_path) / "prompts" / "llm"
    logs_dir.mkdir(parents=True)
    plans_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)

    monkeypatch.setattr(analyzer_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(analyzer_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(analyzer_mod, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(analyzer_mod, "PROMPTS_DIR", prompts_dir)

    (logs_dir / "daily_review_2026-03-26.json").write_text(
        json.dumps({"grade": "A", "net_pnl": 1200.5}),
        encoding="utf-8",
    )
    (logs_dir / "workflow_journal_2026-03-26.jsonl").write_text(
        json.dumps({"symbol": "BBB", "action": "BUY", "reasoning": "trend day"}) + "\n",
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260326.json").write_text(
        json.dumps({"signals": [{"symbol": "BBB", "score": 88.0, "setup": "breakout"}]}),
        encoding="utf-8",
    )
    (prompts_dir / "daily_lessons.txt").write_text(
        "Date: {date}\nReview: {daily_review}\nJournal: {workflow_journal}\nPlan: {morning_plan}",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analyzer_mod, "get_config", lambda: SimpleNamespace(llm=SimpleNamespace(model_decision="qwen-test"))
    )

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"response": "# Daily Lessons - 2026-03-26\n\n## Session Summary\n- Strong day"}

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Response())

    analyzer = DailyLessonsAnalyzer()
    result = analyzer.run("2026-03-26")

    report_path = reports_dir / "daily_lessons_2026-03-26.md"
    legacy_path = reports_dir / "post_market_lesson_2026-03-26.json"
    assert result["success"] is True
    assert result["mode"] == "llm"
    assert report_path.exists()
    assert legacy_path.exists()
    legacy_payload = json.loads(legacy_path.read_text(encoding="utf-8"))
    assert legacy_payload["artifact_type"] == "daily_lessons_compatibility_export"
    assert legacy_payload["mode"] == "llm"
    assert legacy_payload["model"] == "qwen-test"
    assert legacy_payload["metrics"]["key_decisions_count"] == 1
    assert legacy_payload["metrics"]["top_signals_count"] == 1
    assert legacy_payload["source_report_path"].endswith("daily_lessons_2026-03-26.md")


def test_daily_lessons_analyzer_writes_fallback_report_when_daily_review_missing(
    tmp_path, monkeypatch
):
    logs_dir = Path(tmp_path) / "logs"
    plans_dir = Path(tmp_path) / "plans"
    reports_dir = Path(tmp_path) / "reports"
    prompts_dir = Path(tmp_path) / "prompts" / "llm"
    logs_dir.mkdir(parents=True)
    plans_dir.mkdir(parents=True)
    reports_dir.mkdir(parents=True)
    prompts_dir.mkdir(parents=True)

    monkeypatch.setattr(analyzer_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(analyzer_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(analyzer_mod, "REPORTS_DIR", reports_dir)
    monkeypatch.setattr(analyzer_mod, "PROMPTS_DIR", prompts_dir)

    (logs_dir / "workflow_journal_2026-03-27.jsonl").write_text(
        json.dumps({"symbol": "CCC", "action": "BUY", "reasoning": "gap hold"}) + "\n",
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260327.json").write_text(
        json.dumps({"signals": [{"symbol": "CCC", "score": 77.0, "setup": "gap"}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        analyzer_mod,
        "get_config",
        lambda: SimpleNamespace(llm=SimpleNamespace(model_decision="qwen-test")),
    )

    analyzer = DailyLessonsAnalyzer()
    result = analyzer.run("2026-03-27")

    report_path = reports_dir / "daily_lessons_2026-03-27.md"
    legacy_path = reports_dir / "post_market_lesson_2026-03-27.json"
    assert result["success"] is True
    assert result["mode"] == "fallback"
    assert result["error"] == "daily_review_missing"
    assert report_path.exists()
    assert legacy_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "daily_review_missing" in content
    assert "CCC" in content
