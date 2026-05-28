from __future__ import annotations

import json
from types import SimpleNamespace

from autotrade.core.autonomous_agent import AutonomousAgent


def _noop_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_persist_research_artifact_bundle_from_latest_plan(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    research_dir = tmp_path / "research"
    plans_dir.mkdir()
    research_dir.mkdir()

    monkeypatch.setattr("autotrade.core.autonomous_agent.PLANS_DIR", plans_dir)
    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)

    state = {
        "watchlist": [
            {
                "symbol": "AAA",
                "final_score": 87.0,
                "catalyst_score": 0.7,
                "catalyst_tags": ["earnings"],
                "catalyst_note": "Beat and raise",
                "s1_price": 9.7,
                "r1_price": 11.0,
                "sr_quality_score": 71.0,
            }
        ],
        "workflow_completion": {"watchlist_selected": True},
    }
    game_plan = {
        "date": "2026-02-11",
        "signals": [{"symbol": "AAA", "final_score": 87.0}],
        "full_watchlist": [{"symbol": "AAA", "final_score": 87.0}],
    }

    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260211.json").write_text(
        json.dumps(game_plan),
        encoding="utf-8",
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    bundle_path = agent._persist_research_artifact_bundle_from_latest_plan()

    assert bundle_path is not None
    assert bundle_path.exists()

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    assert payload["trade_date"] == "2026-02-11"
    assert payload["catalysts"]["AAA"]["score"] == 0.7
    assert payload["support_resistance"]["AAA"]["s1_price"] == 9.7
