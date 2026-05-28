from datetime import datetime, timezone
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent


def _noop_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )


def test_persist_fresh_run_state_reset_clears_completion_state():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    captured = {}

    def fake_save(state):
        captured.update(state)

    agent._save_overnight_state = fake_save
    agent._persist_fresh_run_state_reset()

    assert captured["research_complete"] is False
    assert captured["watchlist"] == []
    assert captured["discovery_queue"] == []
    assert captured["researched"] == {}
    assert captured["all_candidates"] == []

    completion = captured["workflow_completion"]
    assert completion["watchlist_selected"] is False
    assert completion["game_plan_generated"] is False
    assert completion["youtube_ready"] is False
    assert completion["reset_reason"] == "fresh_run_cli"
    assert completion["reset_at"]


def test_load_overnight_state_returns_fresh_when_research_complete(tmp_path, monkeypatch):
    """When research_complete is true and hour >= 17, should start fresh, not resume."""
    import json

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    state_file = tmp_path / "research" / "overnight_state.json"
    state_file.parent.mkdir(parents=True, exist_ok=True)

    # Write a recently-updated state with research_complete=True
    now = datetime.now(timezone.utc)
    completed_state = {
        "session_date": now.strftime("%Y-%m-%d"),
        "updated_at": now.isoformat(),
        "research_complete": True,
        "watchlist": [{"symbol": "AAPL"}],
        "discovery_queue": [],
        "researched": {"AAPL": {}},
        "all_candidates": [],
    }
    state_file.write_text(json.dumps(completed_state), encoding="utf-8")

    monkeypatch.setattr("autotrade.core.autonomous_agent.PROJECT_DIR", tmp_path)
    agent.RESEARCH_DIR = state_file.parent

    # Mock scheduler to return hour >= 17
    mock_time = now.replace(hour=20, minute=0)
    agent.scheduler = SimpleNamespace(get_current_time=lambda: mock_time)

    result = agent._load_overnight_state()

    # Should get fresh state, not the completed one
    assert result.get("research_complete") is False
    # Fresh state resets research tracking — the completed AAPL entry must not carry over
    assert "AAPL" not in result.get("researched", {})


def test_load_overnight_state_resets_stale_yesterday_state_after_0630_et(
    tmp_path, monkeypatch
):
    import json

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 4, 14, 11, 5, 0, tzinfo=timezone.utc)

    class _Logger:
        def __init__(self):
            self.info_messages = []
            self.warning_messages = []
            self.debug_messages = []
            self.error_messages = []

        def info(self, *args, **kwargs):
            self.info_messages.append(args[0] % args[1:] if len(args) > 1 else args[0])

        def warning(self, *args, **kwargs):
            self.warning_messages.append(
                args[0] % args[1:] if len(args) > 1 else args[0]
            )

        def debug(self, *args, **kwargs):
            self.debug_messages.append(args[0] % args[1:] if len(args) > 1 else args[0])

        def error(self, *args, **kwargs):
            self.error_messages.append(args[0] % args[1:] if len(args) > 1 else args[0])

    agent = AutonomousAgent.__new__(AutonomousAgent)
    logger = _Logger()
    agent.logger = logger

    research_dir = tmp_path / "research"
    plans_dir = tmp_path / "plans"
    research_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    stale_state = {
        "date": "2026-04-13",
        "updated_at": "2026-04-13T20:00:00-05:00",
        "research_complete": True,
        "watchlist": [{"symbol": "AAPL"}],
        "discovery_queue": [],
        "researched": {"AAPL": {}},
        "all_candidates": [{"symbol": "AAPL"}],
        "workflow_completion": {
            "watchlist_selected": True,
            "game_plan_generated": True,
            "youtube_ready": True,
        },
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(stale_state),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)

    result = agent._load_overnight_state()

    assert result["research_complete"] is False
    assert result["watchlist"] == []
    assert result["workflow_completion"]["reset_reason"] == (
        "overnight_leaked_into_premarket"
    )
    assert not any(
        "Running overnight mode during market hours" in message
        for message in logger.warning_messages
    )
    assert any(
        "overnight_leaked_into_premarket" in message
        for message in logger.warning_messages
    )


def test_load_overnight_state_keeps_next_trading_day_state_on_sunday_night(
    tmp_path, monkeypatch
):
    import json

    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            # Sunday May 3, 2026 22:42 ET. The valid overnight target is
            # Monday May 4, not the current calendar date.
            return cls(2026, 5, 4, 2, 42, 0, tzinfo=timezone.utc)

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _noop_logger()

    research_dir = tmp_path / "research"
    plans_dir = tmp_path / "plans"
    research_dir.mkdir(parents=True, exist_ok=True)
    plans_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "date": "2026-05-04",
        "updated_at": "2026-05-03T22:40:00-04:00",
        "research_complete": False,
        "watchlist": [{"symbol": "AAPL"}],
        "discovery_queue": [{"symbol": "MSFT"}],
        "researched": {},
        "all_candidates": [{"symbol": "AAPL"}],
    }
    (research_dir / "overnight_state.json").write_text(
        json.dumps(state),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "datetime", _FixedDateTime)

    result = agent._load_overnight_state()

    assert result["date"] == "2026-05-04"
    assert result["watchlist"] == [{"symbol": "AAPL"}]
    assert result["discovery_queue"] == [{"symbol": "MSFT"}]
