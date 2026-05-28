import json
from datetime import datetime, timedelta

from tools.session_hygiene_check import inspect_recent_pm_plan_health


def test_recent_watch_only_pm_plan_is_reported(tmp_path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    plan_path = plans_dir / "pm_plan_2026-04-23.json"
    plan_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-23T00:15:47",
                "signals": [
                    {
                        "symbol": "ABC",
                        "recommendation": "WATCH",
                        "entry_score": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    failures = inspect_recent_pm_plan_health(
        tmp_path,
        now=datetime(2026, 4, 23, 1, 0, 0),
    )

    assert failures == ["plans/pm_plan_2026-04-23.json is watch-only/non-actionable"]


def test_old_watch_only_pm_plan_is_ignored(tmp_path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-23.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-23T00:15:47",
                "signals": [{"symbol": "ABC", "recommendation": "WATCH"}],
            }
        ),
        encoding="utf-8",
    )

    failures = inspect_recent_pm_plan_health(
        tmp_path,
        now=datetime(2026, 4, 23, 4, 16, 0) + timedelta(hours=1),
    )

    assert failures == []


def test_recent_actionable_pm_plan_passes(tmp_path):
    plans_dir = tmp_path / "plans"
    plans_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-23.json").write_text(
        json.dumps(
            {
                "generated_at": "2026-04-23T00:15:47",
                "signals": [{"symbol": "ABC", "recommendation": "BUY"}],
            }
        ),
        encoding="utf-8",
    )

    failures = inspect_recent_pm_plan_health(
        tmp_path,
        now=datetime(2026, 4, 23, 1, 0, 0),
    )

    assert failures == []
