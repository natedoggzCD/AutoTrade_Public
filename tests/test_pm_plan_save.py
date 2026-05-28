from __future__ import annotations

import json
from datetime import date, datetime

import pytest

from autotrade.execution import post_market_workflow as post_market_workflow_mod
from autotrade.execution.post_market_workflow import PostMarketWorkflow
from autotrade.utils.market_time import get_pm_plan_date


def _workflow_stub() -> PostMarketWorkflow:
    workflow = PostMarketWorkflow.__new__(PostMarketWorkflow)
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )
    workflow._rescale_plan_scores = lambda signals: list(signals or [])
    workflow._log_rescaled_score_diagnostics = lambda *args, **kwargs: None
    return workflow


def _save_minimal_plan(
    monkeypatch,
    tmp_path,
    *,
    plan_date: date,
):
    workflow = _workflow_stub()
    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: plan_date,
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
    return workflow._save_plan({"signals": [{"symbol": "ABC", "score": 80.0}], "summary": {}})


def test_pm_plan_save_after_close_targets_next_trading_day(tmp_path, monkeypatch):
    as_of = datetime(2026, 4, 14, 18, 30)
    target_plan_date = get_pm_plan_date(as_of)

    assert target_plan_date == date(2026, 4, 15)

    path = _save_minimal_plan(monkeypatch, tmp_path, plan_date=target_plan_date)
    assert path == tmp_path / "pm_plan_2026-04-15.json"
    assert path.exists()


def test_pm_plan_save_midnight_window_targets_active_session_date(tmp_path, monkeypatch):
    # 00:04 on Tuesday (after Monday close) should key to Tuesday's session date.
    as_of = datetime(2026, 4, 14, 0, 4)
    target_plan_date = get_pm_plan_date(as_of)

    assert target_plan_date == date(2026, 4, 14)

    path = _save_minimal_plan(monkeypatch, tmp_path, plan_date=target_plan_date)
    assert path == tmp_path / "pm_plan_2026-04-14.json"
    assert path.exists()


def test_pm_plan_save_overwrites_stale_plan_date(tmp_path, monkeypatch):
    plan_date = date(2026, 4, 15)
    workflow = _workflow_stub()
    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: plan_date,
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

    path = workflow._save_plan(
        {
            "plan_date": "unknown (fallback)",
            "signals": [{"symbol": "ABC", "score": 80.0}],
            "summary": {},
        }
    )

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["plan_date"] == "2026-04-15"
    assert len(saved["signals"]) == 1


def test_pm_plan_save_rejects_unresolvable_plan_date(tmp_path, monkeypatch):
    workflow = _workflow_stub()
    monkeypatch.setattr(post_market_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        post_market_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="PM plan save cannot resolve target date"):
        workflow._save_plan(
            {
                "signals": [{"symbol": "ABC", "score": 80.0}],
                "summary": {},
            }
        )
