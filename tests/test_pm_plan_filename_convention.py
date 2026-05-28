from __future__ import annotations

import json
import sys
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent
from autotrade.execution import post_market_workflow as post_market_workflow_mod
from autotrade.execution.post_market_workflow import PostMarketWorkflow
from autotrade.utils.market_time import get_pm_plan_date


def test_get_pm_plan_date_rolls_to_next_trading_day_after_close():
    plan_date = get_pm_plan_date(datetime(2026, 4, 14, 17, 50))

    assert plan_date == date(2026, 4, 15)
    assert Path("plans") / f"pm_plan_{plan_date.strftime('%Y-%m-%d')}.json" == Path(
        "plans/pm_plan_2026-04-15.json"
    )


def test_pm_writer_uses_target_trading_day_filename(tmp_path, monkeypatch):
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )
    workflow._rescale_plan_scores = lambda signals: list(signals or [])
    workflow._log_rescaled_score_diagnostics = lambda *args, **kwargs: None

    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: date(2026, 4, 15),
    )
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_core_market_data_readiness",
        lambda: {
            "is_fresh": True,
            "primary_date": "2026-04-14",
            "expected_date": "2026-04-14",
            "blocking_reasons": [],
        },
    )

    plan_path = workflow._save_plan(
        {
            "signals": [{"symbol": "ABC", "score": 72.0}],
            "summary": {},
        }
    )

    assert plan_path == tmp_path / "pm_plan_2026-04-15.json"
    assert plan_path.exists()


def test_pm_reader_resolves_same_filename(tmp_path, monkeypatch):
    plans_dir = tmp_path / "plans"
    logs_dir = tmp_path / "logs"
    plans_dir.mkdir()
    logs_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-15.json").write_text(
        json.dumps({"signals": [{"symbol": "ABC", "score": 72.0}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setitem(
        sys.modules,
        "autotrade.utils.market_time",
        SimpleNamespace(get_pm_plan_date=lambda *_args, **_kwargs: date(2026, 4, 15)),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda plan_path: {"score": 80, "path": plan_path.name},
        optimize_with_openai=lambda signals: {},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(
        get_missing_tickers=lambda tickers: [],
        available=False,
        bulk_fetch=lambda tickers: [],
    )

    result = agent.validate_pm_workflow()

    assert result["plan_sources"]["pm_plan"] == "pm_plan_2026-04-15.json"
    assert result["analysis_plan"] == "pm_plan_2026-04-15.json"


def test_validate_pm_workflow_promotes_empty_pm_plan_from_morning_plan(
    tmp_path, monkeypatch
):
    plans_dir = tmp_path / "plans"
    logs_dir = tmp_path / "logs"
    plans_dir.mkdir()
    logs_dir.mkdir()
    (plans_dir / "pm_plan_2026-04-15.json").write_text(
        json.dumps({"plan_date": "2026-04-15", "signals": []}),
        encoding="utf-8",
    )
    (plans_dir / "morning_game_plan_20260415.json").write_text(
        json.dumps({"signals": [{"symbol": "ABC", "score": 72.0}]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", plans_dir)
    monkeypatch.setattr(autonomous_agent_mod, "LOG_DIR", logs_dir)
    monkeypatch.setattr(
        autonomous_agent_mod,
        "_pm_plan_date",
        lambda now=None: date(2026, 4, 15),
    )

    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )
    agent.watchlist_optimizer = SimpleNamespace(
        analyze_current_picks=lambda plan_path: {"score": 80, "path": plan_path.name},
        optimize_with_openai=lambda signals: {},
        update_watchlist=lambda *args, **kwargs: None,
    )
    agent.web_researcher = SimpleNamespace(openai_client=None)
    agent.data_fetcher = SimpleNamespace(
        get_missing_tickers=lambda tickers: [],
        available=False,
        bulk_fetch=lambda tickers: [],
    )

    result = agent.validate_pm_workflow()

    promoted_plan = json.loads(
        (plans_dir / "pm_plan_2026-04-15.json").read_text(encoding="utf-8")
    )
    assert result["pm_plan_exists"] is True
    assert result["pm_plan_authority_dead"] is False
    assert result["analysis_plan"] == "pm_plan_2026-04-15.json"
    assert result["actions_taken"][0].startswith("Promoted 1 morning signals")
    assert len(promoted_plan["signals"]) == 1
    assert promoted_plan["pm_plan_source"] == "morning_game_plan_20260415.json"
