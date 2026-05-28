from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

import pytest

from autotrade.core import pm_workflow as pm_workflow_mod
from autotrade.core.pm_workflow import PMWorkflow


class _LogCapture:
    def __init__(self):
        self.infos = []
        self.warnings = []
        self.debugs = []

    def info(self, msg, *args, **kwargs):
        self.infos.append(str(msg) % args if args else str(msg))

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(str(msg) % args if args else str(msg))

    def debug(self, msg, *args, **kwargs):
        self.debugs.append(str(msg) % args if args else str(msg))


def _make_workflow():
    workflow = PMWorkflow.__new__(PMWorkflow)
    workflow._last_validated_strategy_count = 0
    workflow._last_merged_candidate_count = 0
    workflow._last_strategy_routing_diagnostics = {}
    return workflow


def _as_dict(row):
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_dict"):
        data = row.to_dict()
        # Enrich with routing attributes stored on EntryCandidate instances.
        for key in ("strategy_name", "setup_type", "symbol_strategy_rank", "symbol_strategy_source"):
            if key not in data and hasattr(row, key):
                data[key] = getattr(row, key)
        return data
    return getattr(row, "__dict__", {"value": row})


def test_merge_keeps_only_symbol_eligible_strategy_pairs(monkeypatch):
    workflow = _make_workflow()
    log = _LogCapture()

    monkeypatch.setattr(
        "autotrade.core.pm_workflow.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                per_symbol_strategy_enabled=True,
                per_symbol_fallback_to_global=True,
            )
        ),
    )

    strategies = [
        {
            "strategy_name": "strat_a",
            "setup_type": "momentum",
            "config_patch": {"screener_v2": {"mock_id": "a"}, "backtest": {}},
        },
        {
            "strategy_name": "strat_b",
            "setup_type": "reversion",
            "config_patch": {"screener_v2": {"mock_id": "b"}, "backtest": {}},
        },
    ]
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies",
        lambda: strategies,
    )
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies_by_symbol",
        lambda fallback_to_global=False: {
            "AAPL": [
                {
                    "strategy_name": "strat_a",
                    "setup_type": "momentum",
                    "symbol_score": 90.0,
                }
            ],
            "MSFT": [
                {
                    "strategy_name": "strat_b",
                    "setup_type": "reversion",
                    "symbol_score": 88.0,
                }
            ],
        },
    )

    def fake_get_entry_candidates(*args, **kwargs):
        patch = kwargs.get("config_override") or {}
        mock_id = patch.get("mock_id")
        if mock_id == "a":
            return [
                {"symbol": "AAPL", "composite_score": 70.0},
                {"symbol": "MSFT", "composite_score": 69.0},
                {"symbol": "TSLA", "composite_score": 66.0},
            ]
        if mock_id == "b":
            return [
                {"symbol": "AAPL", "composite_score": 60.0},
                {"symbol": "MSFT", "composite_score": 75.0},
            ]
        return []

    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_entry_candidates",
        fake_get_entry_candidates,
    )

    baseline = [
        {"symbol": "AAPL", "composite_score": 45.0},
        {"symbol": "MSFT", "composite_score": 40.0},
    ]
    merged = workflow._merge_multi_strategy_signals(
        baseline_candidates=baseline,
        open_slots=5,
        log=log,
    )

    by_symbol = {_as_dict(row)["ticker"]: _as_dict(row) for row in merged}
    assert set(by_symbol.keys()) == {"AAPL", "MSFT", "TSLA"}
    assert by_symbol["AAPL"]["strategy_name"] == "strat_a"
    assert by_symbol["AAPL"]["symbol_strategy_rank"] == 1
    assert by_symbol["AAPL"]["symbol_strategy_source"] == "per_symbol"
    assert by_symbol["MSFT"]["strategy_name"] == "strat_b"
    assert by_symbol["MSFT"]["symbol_strategy_rank"] == 1
    assert by_symbol["MSFT"]["symbol_strategy_source"] == "per_symbol"
    assert by_symbol["TSLA"]["symbol_strategy_source"] == "global_fallback"
    assert by_symbol["TSLA"]["symbol_strategy_rank"] is None

    diag = workflow._last_strategy_routing_diagnostics
    assert diag["dropped_not_in_top_k"] >= 2
    assert diag["kept_global_fallback"] >= 1


def test_merge_uses_global_fallback_when_symbol_map_missing(monkeypatch):
    workflow = _make_workflow()
    log = _LogCapture()

    monkeypatch.setattr(
        "autotrade.core.pm_workflow.get_config",
        lambda: SimpleNamespace(
            strategy_lab=SimpleNamespace(
                per_symbol_strategy_enabled=True,
                per_symbol_fallback_to_global=True,
            )
        ),
    )

    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies",
        lambda: [
            {
                "strategy_name": "strat_only",
                "setup_type": "breakout",
                "config_patch": {"screener_v2": {"mock_id": "x"}, "backtest": {}},
            }
        ],
    )
    monkeypatch.setattr(
        "autotrade.signals.strategy_pool.load_validated_strategies_by_symbol",
        lambda fallback_to_global=False: {},
    )
    monkeypatch.setattr(
        "autotrade.signals.screener_v2.get_entry_candidates",
        lambda *args, **kwargs: [{"symbol": "NVDA", "composite_score": 81.0}],
    )

    merged = workflow._merge_multi_strategy_signals(
        baseline_candidates=[],
        open_slots=3,
        log=log,
    )

    assert len(merged) == 1
    first = _as_dict(merged[0])
    assert first["ticker"] == "NVDA"
    assert first["symbol_strategy_source"] == "global_fallback"
    assert first["symbol_strategy_rank"] is None
    assert workflow._last_strategy_routing_diagnostics["kept_global_fallback"] >= 1


def test_filter_plan_signals_drops_inactive_and_missing_assets(monkeypatch):
    workflow = _make_workflow()
    workflow._asset_eligibility_cache = {}

    class _Client:
        def get_asset(self, symbol):
            if symbol == "ACTIVE":
                return SimpleNamespace(status="active", tradable=True)
            if symbol == "HALTED":
                return SimpleNamespace(status="inactive", tradable=False)
            raise RuntimeError("404 asset not found")

    workflow.client = _Client()
    monkeypatch.setattr(
        pm_workflow_mod,
        "get_market_now",
        lambda: datetime(2026, 3, 24, 10, 30),
    )
    filtered, dropped = workflow._filter_plan_signals_by_asset_status(
        [{"symbol": "ACTIVE"}, {"symbol": "HALTED"}, {"symbol": "DELISTED"}]
    )

    assert [row["symbol"] for row in filtered] == ["ACTIVE"]
    assert dropped == ["HALTED", "DELISTED"]


def test_save_plan_preserves_existing_non_empty_plan_when_new_signals_empty(
    tmp_path, monkeypatch
):
    workflow = _make_workflow()
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(pm_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        pm_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 3, 24, 18, 30),
    )

    plan_path = tmp_path / "pm_plan_2026-03-24.json"
    existing_plan = {
        "generated_at": "2026-03-24T00:00:00",
        "signals": [{"symbol": "KEEP", "score": 88.0}],
        "entry_candidates": [{"symbol": "KEEP", "score": 88.0}],
        "summary": {},
    }
    plan_path.write_text(json.dumps(existing_plan), encoding="utf-8")

    result = workflow._save_plan(
        {
            "generated_at": "2026-03-24T06:00:00",
            "signals": [],
            "entry_candidates": [],
            "summary": {},
            "positions": [{"symbol": "CAVA"}],
        }
    )

    saved = json.loads(plan_path.read_text(encoding="utf-8"))
    assert result == plan_path
    assert len(saved["signals"]) == 1
    assert saved["signals"][0]["symbol"] == "KEEP"


def test_save_plan_overwrites_stale_plan_date(tmp_path, monkeypatch):
    workflow = _make_workflow()
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )

    monkeypatch.setattr(pm_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        pm_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: datetime(2026, 3, 24, 18, 30),
    )

    result = workflow._save_plan(
        {
            "generated_at": "2026-03-24T06:00:00",
            "plan_date": "unknown (fallback)",
            "signals": [{"symbol": "KEEP", "score": 88.0}],
            "entry_candidates": [{"symbol": "KEEP", "score": 88.0}],
            "summary": {},
        }
    )

    saved = json.loads((tmp_path / "pm_plan_2026-03-24.json").read_text(encoding="utf-8"))
    assert result == tmp_path / "pm_plan_2026-03-24.json"
    assert saved["plan_date"] == "2026-03-24"


def test_save_plan_rejects_unresolvable_plan_date(tmp_path, monkeypatch):
    workflow = _make_workflow()
    workflow._filter_plan_signals_by_asset_status = (
        lambda signals, log=None: (list(signals or []), [])
    )
    monkeypatch.setattr(pm_workflow_mod, "PLANS_DIR", tmp_path)
    monkeypatch.setattr(
        pm_workflow_mod,
        "get_pm_plan_date",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="PM plan save cannot resolve target date"):
        workflow._save_plan(
            {
                "generated_at": "2026-03-24T06:00:00",
                "plan_date": "unknown (fallback)",
                "signals": [{"symbol": "KEEP", "score": 88.0}],
                "entry_candidates": [{"symbol": "KEEP", "score": 88.0}],
                "summary": {},
            }
        )
