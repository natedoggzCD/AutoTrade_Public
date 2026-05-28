import json
from datetime import datetime as real_datetime
from pathlib import Path

import requests

from autotrade.core import autonomous_agent as autonomous_agent_mod
from autotrade.core.autonomous_agent import AutonomousAgent
from types import SimpleNamespace

from autotrade.core.decision_claw import DecisionClaw, DecisionClawPositionAdvisor
from config.config_loader import DecisionClawConfig


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def test_decision_claw_fallback_promotes_top_overnight_symbols(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.decisions_log_path = "logs/decision_claw_decisions_{date}.jsonl"
    cfg.actions_log_path = "logs/decision_claw_actions_{date}.jsonl"
    cfg.cost_log_path = "logs/decision_claw_cost_{date}.jsonl"
    cfg.phase_snapshot_path = "plans/decision_claw_phase_snapshot_{date}.json"
    cfg.overnight.paid_enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="overnight",
        trigger="morning_plan_finalization",
        payload={
            "phase": "overnight",
            "candidate_count": 3,
            "candidates": [
                {"symbol": "GLNG", "score": 81.0},
                {"symbol": "YPF", "score": 79.0},
                {"symbol": "VIAV", "score": 74.0},
            ],
        },
        legacy_recommendation={"legacy_choice": ["GLNG", "YPF"]},
    )

    assert result.decision == "finalize_watchlist"
    assert [action.symbol for action in result.actions[:2]] == ["GLNG", "YPF"]

    logs = list((Path(tmp_path) / "logs").glob("decision_claw_decisions_*.jsonl"))
    assert logs
    payload = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[0])
    assert payload["phase_agent"] == "overnight"


def test_decision_claw_disabled_phase_bypasses_cooldown_budget(tmp_path, monkeypatch):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.decisions_log_path = "logs/decision_claw_decisions_{date}.jsonl"
    cfg.actions_log_path = "logs/decision_claw_actions_{date}.jsonl"
    cfg.cost_log_path = "logs/decision_claw_cost_{date}.jsonl"
    cfg.phase_snapshot_path = "plans/decision_claw_phase_snapshot_{date}.json"
    cfg.market_state.enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    monkeypatch.setattr(
        claw,
        "_should_review",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("budget gate called")),
    )

    result = claw.review(
        phase_agent="market_state",
        trigger="cycle",
        payload={"phase": "market_state", "positions": [{"symbol": "TEST"}]},
        legacy_recommendation={"legacy_choice": ["TEST"]},
    )

    assert result.provider == "legacy"
    assert result.reasoning_summary == "fallback:phase_disabled"

    logs = list((Path(tmp_path) / "logs").glob("decision_claw_decisions_*.jsonl"))
    assert logs
    payload = json.loads(logs[0].read_text(encoding="utf-8").splitlines()[0])
    assert payload["phase_agent"] == "market_state"
    assert payload["provider"] == "legacy"
    assert payload["reasoning_summary"] == "fallback:phase_disabled"


def test_decision_claw_overnight_compaction_screens_full_watchlist(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.overnight.max_symbols = 48

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    candidates = []
    for idx in range(60):
        candidates.append(
            {
                "symbol": f"S{idx:03d}",
                "score": float(95 - idx),
                "entry_source": "overnight_full_watchlist"
                if idx < 40
                else "momentum_scanner",
                "sector": "Energy" if idx % 2 == 0 else "Industrials",
                "strategy": "breakout" if idx % 3 == 0 else "pullback",
                "catalyst": "earnings" if idx < 10 else "news",
                "reason": "stale_research" if idx >= 55 else "",
            }
        )

    packet = claw._compact_payload(
        {
            "phase": "overnight",
            "candidate_count": len(candidates),
            "candidates": candidates,
            "full_watchlist_size": len(candidates),
            "watchlist_target_size": 200,
            "watchlist_completion_min_size": 200,
        },
        phase_agent="overnight",
        phase_cfg=cfg.overnight,
    )

    screen = packet["watchlist_screen"]
    assert screen["watchlist_size"] == 60
    assert screen["watchlist_target_size"] == 200
    assert len(screen["top_candidates"]) >= 12
    assert len(screen["flagged_candidates"]) >= 8
    assert screen["source_breakdown"]["overnight_full_watchlist"] == 40
    assert screen["stale_name_count"] >= 5
    assert len(packet["symbols_considered"]) >= 20


def test_decision_claw_persists_final_watchlist_artifact_for_overnight(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.decisions_log_path = "logs/decision_claw_decisions_{date}.jsonl"
    cfg.actions_log_path = "logs/decision_claw_actions_{date}.jsonl"
    cfg.cost_log_path = "logs/decision_claw_cost_{date}.jsonl"
    cfg.phase_snapshot_path = "plans/decision_claw_phase_snapshot_{date}.json"
    cfg.final_watchlist_path = "plans/decision_claw_final_watchlist_{date}.json"
    cfg.overnight.paid_enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="overnight",
        trigger="morning_plan_finalization",
        payload={
            "phase": "overnight",
            "candidate_count": 3,
            "candidates": [
                {"symbol": "GLNG", "score": 81.0},
                {"symbol": "YPF", "score": 79.0},
                {"symbol": "VIAV", "score": 74.0},
            ],
            "full_watchlist_size": 3,
            "watchlist_target_size": 200,
            "watchlist_completion_min_size": 200,
        },
        legacy_recommendation={"legacy_choice": ["GLNG", "YPF"]},
    )

    assert result.decision == "finalize_watchlist"
    artifact = list(
        (Path(tmp_path) / "plans").glob("decision_claw_final_watchlist_*.json")
    )
    assert artifact
    payload = json.loads(artifact[0].read_text(encoding="utf-8"))
    assert payload["phase_agent"] == "overnight"
    assert payload["selected_symbols"][:2] == ["GLNG", "YPF"]
    assert payload["watchlist_screen"]["watchlist_target_size"] == 200


def test_decision_claw_fallback_respects_top_n_floor_over_repeat_names(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.decisions_log_path = "logs/decision_claw_decisions_{date}.jsonl"
    cfg.actions_log_path = "logs/decision_claw_actions_{date}.jsonl"
    cfg.cost_log_path = "logs/decision_claw_cost_{date}.jsonl"
    cfg.phase_snapshot_path = "plans/decision_claw_phase_snapshot_{date}.json"
    cfg.final_watchlist_path = "plans/decision_claw_final_watchlist_{date}.json"
    cfg.premarket.paid_enabled = False
    cfg.top_n_floor = 3

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="premarket",
        trigger="premarket_finalization",
        payload={
            "phase": "premarket",
            "candidate_count": 6,
            "open_slots": 50,
            "candidates": [
                {
                    "symbol": "HOLO",
                    "score": 100.0,
                    "final_score": 65.25,
                    "ranking_score": 100.0,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 2,
                    "setup_type": "recovery_universe_fill",
                },
                {
                    "symbol": "SGML",
                    "score": 100.0,
                    "final_score": 67.75,
                    "ranking_score": 100.0,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 2,
                    "setup_type": "recovery_universe_fill",
                },
                {
                    "symbol": "BMNR",
                    "score": 81.25,
                    "final_score": 81.25,
                    "ranking_score": 57.38,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 0,
                },
                {
                    "symbol": "SRRK",
                    "score": 77.75,
                    "final_score": 77.75,
                    "ranking_score": 60.15,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 0,
                },
                {
                    "symbol": "LTM",
                    "score": 77.75,
                    "final_score": 77.75,
                    "ranking_score": 57.48,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 0,
                },
                {
                    "symbol": "TPC",
                    "score": 77.75,
                    "final_score": 77.75,
                    "ranking_score": 55.0,
                    "recommendation": "STRONG BUY",
                    "fallback_repeat_count": 0,
                },
            ],
        },
        legacy_recommendation={"legacy_choice": ["HOLO", "SGML", "BMNR"]},
    )

    selected_actions = [
        action.symbol
        for action in result.actions
        if action.action_type
        in {"submit_entry", "promote_symbol", "force_signal_recheck"}
    ]
    assert result.decision == "finalize_watchlist"
    assert selected_actions[:3] == ["BMNR", "SRRK", "LTM"]

    artifact = list(
        (Path(tmp_path) / "plans").glob("decision_claw_final_watchlist_*.json")
    )
    assert artifact
    payload = json.loads(artifact[0].read_text(encoding="utf-8"))
    assert payload["selected_symbols"][:3] == ["BMNR", "SRRK", "LTM"]


def test_decision_claw_query_local_sets_think_false_for_qwen3_family(
    monkeypatch, tmp_path
):
    seen = {}

    class _Resp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "message": {
                    "content": '{"decision":"hold","confidence":0.7,"actions":[]}'
                }
            }

    def _fake_post(url, json=None, timeout=None):
        seen["url"] = url
        seen["json"] = json or {}
        return _Resp()

    monkeypatch.setattr(requests, "post", _fake_post)
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw._query_local(
        prompt="test",
        system_prompt="system",
        model="qwen3.5:9b-q4_K_M",
        max_tokens=300,
    )

    assert result["success"] is True
    assert seen["json"].get("think") is False


def test_decision_claw_query_local_strips_think_tags(monkeypatch, tmp_path):
    class _Resp:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "message": {
                    "content": '<think>internal reasoning</think>{"decision":"hold","confidence":0.7,"actions":[]}'
                }
            }

    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: _Resp())
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw._query_local(
        prompt="test",
        system_prompt="system",
        model="qwen3:14b-q4_K_M",
        max_tokens=300,
    )

    assert result["success"] is True
    assert result["content"].startswith('{"decision"')


def test_apply_decision_claw_to_plan_payload_reorders_signals():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    plan_payload = {
        "signals": [
            {"symbol": "VIAV", "score": 70},
            {"symbol": "GLNG", "score": 80},
            {"symbol": "YPF", "score": 79},
        ],
        "full_watchlist": [
            {"symbol": "VIAV", "score": 70},
            {"symbol": "GLNG", "score": 80},
            {"symbol": "YPF", "score": 79},
        ],
    }
    review_result = {
        "actions": [
            {"action_type": "promote_symbol", "symbol": "YPF"},
            {"action_type": "promote_symbol", "symbol": "GLNG"},
        ]
    }

    updated = agent._apply_decision_claw_to_plan_payload(plan_payload, review_result)

    assert [row["symbol"] for row in updated["signals"][:3]] == ["YPF", "GLNG", "VIAV"]
    assert updated["decision_claw"]["selected_symbols"] == ["YPF", "GLNG"]


def test_apply_decision_claw_symbols_to_rows_prioritizes_selected_symbols():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    rows = [
        {"symbol": "AAA", "score": 60},
        {"symbol": "BBB", "score": 50},
        {"symbol": "CCC", "score": 40},
    ]
    review_result = {
        "actions": [
            {"action_type": "force_signal_recheck", "symbol": "CCC"},
            {"action_type": "promote_symbol", "symbol": "AAA"},
        ]
    }

    prioritized = agent._apply_decision_claw_symbols_to_rows(rows, review_result)

    assert [row["symbol"] for row in prioritized] == ["CCC", "AAA", "BBB"]


def test_decision_claw_review_morning_plan_uses_full_watchlist_for_overnight():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.overnight_workflow_cfg = SimpleNamespace(
        target_watchlist_size=200,
        completion_min_watchlist_size=200,
    )
    agent._load_raw_morning_plan_payload = lambda target_date=None: (
        None,
        {
            "date": "2026-03-24",
            "signals": [{"symbol": "TOP1", "score": 95.0}],
            "full_watchlist": [
                {"symbol": "TOP1", "score": 95.0},
                {"symbol": "TOP2", "score": 92.0},
                {"symbol": "TOP3", "score": 90.0},
                {"symbol": "TAIL1", "score": 41.0},
            ],
        },
    )
    captured = {}
    agent._run_decision_claw_review = lambda **kwargs: captured.update(kwargs) or None

    review = agent._decision_claw_review_morning_plan(
        phase_agent="overnight",
        trigger="overnight_finalization",
    )

    assert review is None
    assert captured["payload"]["candidate_count"] == 4
    assert captured["payload"]["full_watchlist_size"] == 4
    assert captured["payload"]["watchlist_target_size"] == 200
    assert captured["legacy_recommendation"]["legacy_choice"][:3] == [
        "TOP1",
        "TOP2",
        "TOP3",
    ]


def test_decision_claw_position_advisor_defaults_to_legacy_then_overrides_trim():
    class _LegacyAdvisor:
        def build_context(self, position, entry_time=None):
            return {
                "symbol": position.symbol,
                "pnl_pct": -3.0,
                "hold_duration_minutes": 120.0,
            }

        def get_advice(self, context):
            return {
                "action": "hold",
                "confidence": 0.4,
                "reasoning": "legacy hold",
                "risk_level": "medium",
                "flags": [],
            }

    class _Controller:
        def review(self, **kwargs):
            from autotrade.core.decision_claw import (
                DecisionClawResult,
                DecisionClawAction,
            )

            return DecisionClawResult(
                phase_agent="market_state",
                trigger="position_ABC",
                decision="manage_positions",
                confidence=0.8,
                actions=[
                    DecisionClawAction(
                        action_type="trim_position",
                        symbol="ABC",
                        reason="weak",
                        metadata={"trim_fraction": 0.25},
                    )
                ],
                reasoning_summary="trim weak position",
            )

    advisor = DecisionClawPositionAdvisor(
        _Controller(), legacy_advisor=_LegacyAdvisor()
    )
    advice = advisor.get_advice(
        {"symbol": "ABC", "pnl_pct": -3.0, "hold_duration_minutes": 120.0}
    )

    assert advice["action"] == "trim"
    assert advice["trim_fraction"] == 0.25
    assert advice["legacy_advice"]["action"] == "hold"


def test_decision_claw_position_advisor_caches_trim_during_cooldown():
    class _LegacyAdvisor:
        def get_advice(self, context):
            return {
                "action": "trim",
                "confidence": 0.6,
                "reasoning": "legacy trim",
                "risk_level": "medium",
                "flags": [],
            }

    class _Controller:
        def __init__(self):
            self.calls = 0

        def review(self, **kwargs):
            from autotrade.core.decision_claw import (
                DecisionClawResult,
                DecisionClawAction,
            )

            self.calls += 1
            return DecisionClawResult(
                phase_agent="market_state",
                trigger="position_ABC",
                decision="manage_positions",
                confidence=0.8,
                actions=[
                    DecisionClawAction(
                        action_type="trim_position",
                        symbol="ABC",
                        reason="weak",
                        metadata={"trim_fraction": 0.25},
                    )
                ],
                reasoning_summary="trim weak position",
            )

    controller = _Controller()
    advisor = DecisionClawPositionAdvisor(controller, legacy_advisor=_LegacyAdvisor())
    context = {
        "symbol": "ABC",
        "pnl_pct": -3.0,
        "hold_duration_minutes": 120.0,
        "qty": 100,
        "current_price": 10.0,
        "atr_14": 0.50,
        "failsafe_level": "normal",
        "critical_news_signature": "none",
    }

    first = advisor.get_advice(dict(context))
    second = advisor.get_advice(dict(context))

    assert first["action"] == "trim"
    assert second["action"] == "trim"
    assert controller.calls == 1
    assert "position_cooldown_hit" in second["reasoning"]


def test_decision_claw_position_advisor_invalidates_cooldown_on_price_move():
    class _LegacyAdvisor:
        def get_advice(self, context):
            return {"action": "exit", "confidence": 0.6, "reasoning": "legacy exit"}

    class _Controller:
        def __init__(self):
            self.calls = 0

        def review(self, **kwargs):
            from autotrade.core.decision_claw import (
                DecisionClawResult,
                DecisionClawAction,
            )

            self.calls += 1
            return DecisionClawResult(
                phase_agent="market_state",
                trigger="position_ABC",
                decision="manage_positions",
                confidence=0.8,
                actions=[
                    DecisionClawAction(
                        action_type="exit_position",
                        symbol="ABC",
                        reason="breakdown",
                    )
                ],
                reasoning_summary=f"exit call {self.calls}",
            )

    controller = _Controller()
    advisor = DecisionClawPositionAdvisor(controller, legacy_advisor=_LegacyAdvisor())
    context = {
        "symbol": "ABC",
        "qty": 100,
        "current_price": 10.0,
        "atr_14": 0.50,
        "failsafe_level": "normal",
    }

    advisor.get_advice(dict(context))
    moved = dict(context)
    moved["current_price"] = 11.0
    advisor.get_advice(moved)

    assert controller.calls == 2


def test_decision_claw_market_state_fallback_skips_fresh_zero_health_trim(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.market_state.paid_enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="market_state",
        trigger="market_state_10_0",
        payload={
            "phase": "market_hours",
            "open_slots": 1,
            "candidates": [{"symbol": "BTU", "score": 81.0}],
            "weak_positions": [
                {
                    "symbol": "BTU",
                    "pnl_pct": 0.0,
                    "health_score": 0.0,
                    "hold_minutes": 5.0,
                }
            ],
            "positions": [
                {
                    "symbol": "BTU",
                    "pnl_pct": 0.0,
                    "health_score": 0.0,
                    "hold_minutes": 5.0,
                }
            ],
            "hold_minutes_by_symbol": {"BTU": 5.0},
        },
        legacy_recommendation={"legacy_choice": ["BTU"]},
    )

    assert not any(action.action_type == "trim_position" for action in result.actions)


def test_decision_claw_market_state_fallback_uses_position_context_for_trim(tmp_path):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.market_state.paid_enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="market_state",
        trigger="market_state_11_0",
        payload={
            "phase": "market_hours",
            "open_slots": 1,
            "candidates": [{"symbol": "BTU", "score": 81.0}],
            "weak_positions": [
                {
                    "symbol": "BTU",
                    "pnl_pct": 0.0,
                    "health_score": 0.0,
                    "hold_minutes": 45.0,
                }
            ],
            "positions": [
                {
                    "symbol": "BTU",
                    "pnl_pct": -1.8,
                    "health_score": -25.0,
                    "hold_minutes": 45.0,
                }
            ],
            "hold_minutes_by_symbol": {"BTU": 45.0},
        },
        legacy_recommendation={"legacy_choice": ["BTU"]},
    )

    trim_actions = [
        action for action in result.actions if action.action_type == "trim_position"
    ]
    assert len(trim_actions) == 1
    assert trim_actions[0].symbol == "BTU"
    assert trim_actions[0].metadata["hold_minutes"] == 45.0


def test_decision_claw_market_state_fallback_defaults_wind_down_oversized_trim(
    tmp_path,
):
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.market_state.paid_enabled = False

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    result = claw.review(
        phase_agent="market_state",
        trigger="wind_down_overnight_size_review",
        payload={
            "phase": "wind_down",
            "open_slots": 0,
            "candidates": [],
            "weak_positions": [
                {
                    "symbol": "WEAK",
                    "pnl_pct": -3.0,
                    "health_score": -30.0,
                    "hold_minutes": 90.0,
                }
            ],
            "wind_down_oversized_winners": [
                {
                    "symbol": "APA",
                    "position_size_multiple": 2.25,
                    "pnl_pct": 5.0,
                    "current_price": 43.0,
                    "headroom_pct": 8.0,
                }
            ],
        },
        legacy_recommendation={"legacy_choice": []},
    )

    trim_actions = [
        action for action in result.actions if action.action_type == "trim_position"
    ]
    assert len(trim_actions) == 1
    assert trim_actions[0].symbol == "APA"
    assert trim_actions[0].reason == "wind_down_default_trim_to_target"


def test_decision_claw_prompt_focuses_wind_down_review(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    prompt = claw._user_prompt(
        phase_agent="market_state",
        trigger="wind_down_overnight_size_review",
        packet={
            "phase": "wind_down",
            "wind_down_oversized_winners": [{"symbol": "APA"}, {"symbol": "EC"}],
        },
        legacy_recommendation={},
    )

    assert "The only symbols under overnight size review are: APA, EC." in prompt
    assert "Do not emit trim_position or exit_position for symbols outside" in prompt
    assert (
        "Use overnight_hold_bias on each reviewed symbol as a first-pass guide"
        in prompt
    )


def test_decision_claw_wind_down_oversized_review_uses_compact_openai_model(
    monkeypatch, tmp_path
):
    cfg = DecisionClawConfig()
    cfg.market_state.paid_enabled = True
    cfg.market_state.provider = "openai"
    cfg.market_state.model = "gpt-5"
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    seen = {}

    def _fake_query_provider(**kwargs):
        seen.update(kwargs)
        return {
            "success": True,
            "content": json.dumps(
                {
                    "decision": "trim_position",
                    "confidence": 0.9,
                    "reasoning_summary": "compact model emitted parseable JSON",
                    "legacy_comparison": {
                        "legacy_choice": [],
                        "agent_choice": ["APA"],
                        "agreement_level": "partial",
                        "override_reason": "default trim",
                    },
                    "actions": [
                        {
                            "action_type": "trim_position",
                            "symbol": "APA",
                            "reason": "trim oversized winner",
                            "metadata": {},
                        }
                    ],
                }
            ),
            "provider": kwargs["provider"],
            "model": kwargs["model"],
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)
    result = claw.review(
        phase_agent="market_state",
        trigger="wind_down_overnight_size_review",
        payload={
            "phase": "wind_down",
            "wind_down_oversized_winners": [
                {
                    "symbol": "APA",
                    "position_size_multiple": 2.25,
                    "pnl_pct": 5.0,
                    "current_price": 43.0,
                    "headroom_pct": 8.0,
                }
            ],
        },
        legacy_recommendation={"legacy_choice": []},
    )

    assert seen["provider"] == "openai"
    assert seen["model"] == "gpt-4.1-mini"
    assert result.model == "gpt-4.1-mini"
    assert result.actions[0].symbol == "APA"


def test_decision_claw_normalizes_favorable_wind_down_hold_bias(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    parsed = claw._parse_response(
        content=json.dumps(
            {
                "decision": "trim_position",
                "confidence": 0.9,
                "reasoning_summary": "default trim",
                "legacy_comparison": {
                    "legacy_choice": [],
                    "agent_choice": [],
                    "agreement_level": "partial",
                    "override_reason": "default trim",
                },
                "actions": [
                    {
                        "action_type": "trim_position",
                        "symbol": "BTSG",
                        "reason": "default trim",
                        "metadata": {},
                    }
                ],
            }
        ),
        phase_agent="market_state",
        trigger="wind_down_overnight_size_review",
        packet={
            "phase": "wind_down",
            "wind_down_oversized_winners": [
                {
                    "symbol": "BTSG",
                    "overnight_hold_bias": "favorable",
                }
            ],
        },
        legacy_recommendation={},
    )

    normalized = claw._normalize_wind_down_overnight_result(
        parsed=parsed,
        packet={
            "wind_down_oversized_winners": [
                {
                    "symbol": "BTSG",
                    "overnight_hold_bias": "favorable",
                }
            ]
        },
    )

    hold_actions = [
        action for action in normalized.actions if action.action_type == "hold_position"
    ]
    assert normalized.decision == "trim_and_hold"
    assert len(hold_actions) == 1
    assert hold_actions[0].symbol == "BTSG"
    assert hold_actions[0].metadata["approve_overnight_oversize"] is True


def test_decision_claw_normalizes_unfavorable_wind_down_hold_bias(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    parsed = claw._parse_response(
        content=json.dumps(
            {
                "decision": "hold_position",
                "confidence": 0.9,
                "reasoning_summary": "hold it",
                "legacy_comparison": {
                    "legacy_choice": [],
                    "agent_choice": ["MUR"],
                    "agreement_level": "override",
                    "override_reason": "model wanted hold",
                },
                "actions": [
                    {
                        "action_type": "hold_position",
                        "symbol": "MUR",
                        "reason": "carry it",
                        "metadata": {
                            "approve_overnight_oversize": True,
                            "max_size_multiplier": 2.0,
                        },
                    }
                ],
            }
        ),
        phase_agent="market_state",
        trigger="wind_down_overnight_size_review",
        packet={
            "phase": "wind_down",
            "wind_down_oversized_winners": [
                {
                    "symbol": "MUR",
                    "overnight_hold_bias": "unfavorable",
                }
            ],
        },
        legacy_recommendation={},
    )

    normalized = claw._normalize_wind_down_overnight_result(
        parsed=parsed,
        packet={
            "wind_down_oversized_winners": [
                {
                    "symbol": "MUR",
                    "overnight_hold_bias": "unfavorable",
                }
            ]
        },
    )

    hold_actions = [
        action for action in normalized.actions if action.action_type == "hold_position"
    ]
    assert hold_actions == []


def test_run_day_manager_cycle_derives_wind_down_risk_reward_when_missing():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()

    class _DM:
        @staticmethod
        def _find_signal_data(symbol):
            assert symbol == "BTSG"
            return {
                "symbol": symbol,
                "target": 47.5923,
                "stop_loss": 40.4858,
                "risk_reward": 0.0,
                "atr_14": 0.0,
                "relative_strength": 0.0,
            }

    method = AutonomousAgent._decision_claw_wind_down_oversized_winners.__get__(
        agent, AutonomousAgent
    )
    rows = method(
        dm=_DM(),
        positions=[
            {
                "symbol": "BTSG",
                "qty": 99,
                "pnl_pct": 0.807,
                "market_value": 4287.69,
                "current_price": 43.31,
            }
        ],
        target_value=2000.0,
        max_value=4500.0,
    )

    assert len(rows) == 1
    assert rows[0]["symbol"] == "BTSG"
    assert rows[0]["risk_reward"] > 1.5
    assert rows[0]["overnight_hold_bias"] == "favorable"


def test_benchmark_providers_returns_per_provider_rows(monkeypatch, tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    def _fake_query_provider(**kwargs):
        return {
            "success": True,
            "content": '{"decision":"hold","confidence":0.6,"reasoning_summary":"ok","legacy_comparison":{"legacy_choice":[],"agent_choice":[],"agreement_level":"full","override_reason":""},"actions":[]}',
            "provider": kwargs["provider"],
            "model": kwargs["model"],
            "tokens_used": 10,
            "cost_usd": 0.01 if kwargs["provider"] != "local" else 0.0,
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)
    results = claw.benchmark_providers(
        phase_agent="market_state",
        trigger="benchmark",
        payload={
            "phase": "market_hours",
            "candidates": [{"symbol": "GLNG", "score": 80.0}],
        },
    )

    assert [row["provider"] for row in results] == ["local", "openai", "openrouter"]


def test_decision_claw_loads_phase_prompt_profile(tmp_path):
    profile_path = Path(tmp_path) / "config" / "decision_claw_prompt_profiles.json"
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(
        json.dumps(
            {
                "global": {
                    "output_contract": "Strict JSON only.",
                },
                "phases": {
                    "market_open": {
                        "mission": "Own the first hour aggressively.",
                        "review_task": "Review the opening board.",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = DecisionClawConfig()
    cfg.prompt_profile_path = "config/decision_claw_prompt_profiles.json"

    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    assert "Own the first hour aggressively." in claw._system_prompt("market_open")
    assert "Review the opening board." in claw._user_prompt(
        phase_agent="market_open",
        trigger="open",
        packet={"phase": "market_open", "candidates": []},
        legacy_recommendation={},
    )


def test_build_decision_claw_execution_override_maps_actions():
    agent = AutonomousAgent.__new__(AutonomousAgent)

    override = agent._build_decision_claw_execution_override(
        {
            "decision": "manage_positions",
            "deployment_request": {
                "mode": "long_exception",
                "symbols": ["GLNG"],
                "max_new_entries": 1,
                "reason": "dead capital",
            },
            "symbol_reviews": {
                "GLNG": {
                    "decision": "approve",
                    "accepted": True,
                    "thesis": "clean tape",
                },
            },
            "actions": [
                {"action_type": "submit_entry", "symbol": "GLNG"},
                {"action_type": "demote_symbol", "symbol": "WEAK"},
                {
                    "action_type": "trim_position",
                    "symbol": "VIAV",
                    "reason": "free capital",
                    "metadata": {"trim_fraction": 0.4},
                },
                {
                    "action_type": "hold_position",
                    "symbol": "APA",
                    "reason": "carry winner overnight",
                    "metadata": {
                        "approve_overnight_oversize": True,
                        "max_size_multiplier": 2.0,
                    },
                },
                {"action_type": "exit_position", "symbol": "DVN"},
                {"action_type": "hold_wave"},
            ],
        }
    )

    assert override["entry_priority_symbols"] == ["GLNG"]
    assert override["blocked_entry_symbols"] == ["WEAK"]
    assert override["trim_map"]["VIAV"]["trim_fraction"] == 0.4
    assert override["hold_overnight_map"]["APA"]["approve_overnight_oversize"] is True
    assert override["exit_symbols"] == ["DVN"]
    assert override["wave_action"] == "hold_wave"
    assert override["deployment_request"]["mode"] == "long_exception"
    assert override["symbol_reviews"]["GLNG"]["accepted"] is True
    assert override["reason"] == "manage_positions"


def test_parse_response_preserves_deployment_request(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw._parse_response(
        content=json.dumps(
            {
                "decision": "deploy",
                "confidence": 0.9,
                "reasoning_summary": "dead capital",
                "deployment_request": {
                    "mode": "long_exception",
                    "symbols": ["glng", "ypf"],
                    "max_new_entries": 2,
                    "reason": "no fills with qualified names",
                },
                "legacy_comparison": {
                    "legacy_choice": [],
                    "agent_choice": ["GLNG", "YPF"],
                    "agreement_level": "override",
                    "override_reason": "idle capital",
                },
                "actions": [{"action_type": "submit_entry", "symbol": "GLNG"}],
            }
        ),
        phase_agent="market_state",
        trigger="market_state_cycle",
        packet={
            "phase": "market_hours",
            "candidates": [{"symbol": "GLNG", "score": 80.0}],
        },
        legacy_recommendation={},
    )

    assert result.deployment_request["mode"] == "long_exception"
    assert result.deployment_request["symbols"] == ["GLNG", "YPF"]


def test_parse_response_normalizes_safe_action_aliases(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw._parse_response(
        content=json.dumps(
            {
                "decision": "act",
                "confidence": 0.9,
                "actions": [
                    {"action_type": "submit_entry_queue", "symbol": "glng"},
                    {"action_type": "recheck_signal", "symbol": "ypf"},
                    {"action_type": "recheck_symbol", "symbol": "viav"},
                    {"action_type": "flag_for_premarket_recheck", "symbol": "cenx"},
                    {"action_type": "update_watchlist"},
                ],
            }
        ),
        phase_agent="market_state",
        trigger="cycle",
        packet={},
        legacy_recommendation={},
    )

    assert [action.action_type for action in result.actions] == [
        "submit_entry",
        "force_signal_recheck",
        "force_signal_recheck",
        "force_signal_recheck",
        "refresh_watchlist",
    ]
    assert result.rejected_actions == []


def test_parse_response_rejects_forbidden_pseudo_actions(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    forbidden = [
        "request_research",
        "flag_research_needed",
        "flag_for_research",
        "note",
        "monitor_board",
        "monitor_candidates",
        "request_capital_allocation",
        "finalize_watchlist_top",
        "archive_from_watchlist",
        "redeploy_watchlist",
    ]
    result = claw._parse_response(
        content=json.dumps(
            {
                "decision": "act",
                "confidence": 0.9,
                "actions": [{"action_type": action} for action in forbidden],
            }
        ),
        phase_agent="market_state",
        trigger="cycle",
        packet={},
        legacy_recommendation={},
    )

    assert result.actions == []
    assert [row["reason"] for row in result.rejected_actions] == [
        "unsupported_action"
    ] * len(forbidden)


def test_confidence_floor_downgrades_low_confidence_position_actions(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    parsed = DecisionClaw(
        cfg, logger_=_LoggerStub(), project_root=Path(tmp_path)
    )._parse_response(
        content=json.dumps(
            {
                "decision": "manage_positions",
                "confidence": 0.3,
                "actions": [
                    {"action_type": "trim_position", "symbol": "GLNG"},
                    {"action_type": "exit_position", "symbol": "YPF"},
                ],
            }
        ),
        phase_agent="market_state",
        trigger="cycle",
        packet={},
        legacy_recommendation={},
    )

    result = claw._apply_confidence_floor(
        parsed=parsed,
        packet={"positions": [{"symbol": "GLNG"}, {"symbol": "YPF"}]},
        confidence_floor=0.60,
    )
    claw._persist_result(result)

    assert [action.action_type for action in result.actions] == [
        "hold_position",
        "hold_position",
    ]
    assert all(
        action.metadata["confidence_floor_blocked"] is True for action in result.actions
    )
    assert [row["reason"] for row in result.rejected_actions] == [
        "confidence_floor_blocked",
        "confidence_floor_blocked",
    ]
    action_log = next((Path(tmp_path) / "logs").glob("decision_claw_actions_*.jsonl"))
    payload = json.loads(action_log.read_text(encoding="utf-8").splitlines()[0])
    assert payload["rejected_actions"][0]["reason"] == "confidence_floor_blocked"


def test_confidence_floor_removes_low_confidence_submit_entry(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    parsed = claw._parse_response(
        content=json.dumps(
            {
                "decision": "deploy",
                "confidence": 0.3,
                "deployment_request": {
                    "mode": "long_exception",
                    "symbols": ["GLNG"],
                    "max_new_entries": 1,
                    "reason": "idle capital",
                },
                "actions": [{"action_type": "submit_entry", "symbol": "GLNG"}],
            }
        ),
        phase_agent="market_state",
        trigger="cycle",
        packet={},
        legacy_recommendation={},
    )

    result = claw._apply_confidence_floor(
        parsed=parsed,
        packet={"positions": []},
        confidence_floor=0.60,
    )
    claw._persist_result(result)

    assert result.actions == []
    assert result.deployment_request["mode"] == "long_exception"
    assert result.rejected_actions == [
        {
            "action": "submit_entry",
            "symbol": "GLNG",
            "reason": "confidence_floor_blocked",
            "confidence": 0.3,
            "confidence_floor": 0.6,
        }
    ]
    action_log = next((Path(tmp_path) / "logs").glob("decision_claw_actions_*.jsonl"))
    payload = json.loads(action_log.read_text(encoding="utf-8").splitlines()[0])
    assert payload["actions"] == []
    assert payload["rejected_actions"][0]["reason"] == "confidence_floor_blocked"


def test_review_symbol_validation_uses_local_phase_config(monkeypatch, tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    def _fake_query_provider(**kwargs):
        assert kwargs["provider"] == "local"
        return {
            "success": True,
            "content": json.dumps(
                {
                    "decision": "approve",
                    "entry_type": "long_exception",
                    "thesis": "clean tape",
                    "why_now": "reclaim is holding",
                    "failure_mode": "",
                    "confidence": 0.82,
                    "warning_flags": [],
                }
            ),
            "provider": "local",
            "model": "qwen3:14b-q4_K_M",
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)
    review = claw.review_symbol_validation(
        symbol="GLNG",
        technical_context={"above_vwap": True, "volume_ratio": 1.6},
    )

    assert review["accepted"] is True
    assert review["decision"] == "approve"
    assert review["thesis"] == "clean tape"
    assert review["provider"] == "local"


def test_review_symbol_validation_falls_back_to_openai_on_invalid_local_response(
    monkeypatch, tmp_path
):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))
    calls = []

    def _fake_query_provider(**kwargs):
        calls.append((kwargs["provider"], kwargs["model"]))
        if kwargs["provider"] == "local":
            return {
                "success": True,
                "content": json.dumps({"decision": "maybe"}),
                "provider": "local",
                "model": "qwen3:14b-q4_K_M",
            }
        return {
            "success": True,
            "content": json.dumps(
                {
                    "decision": "approve",
                    "entry_type": "long_exception",
                    "thesis": "cloud fallback clean",
                    "why_now": "fresh redeploy window",
                    "failure_mode": "",
                    "confidence": 0.74,
                    "warning_flags": [],
                }
            ),
            "provider": "openai",
            "model": "gpt-5",
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)
    review = claw.review_symbol_validation(
        symbol="GLNG",
        technical_context={"volume_ratio": 1.6, "current_price": 11.0},
    )

    assert calls[0][0] == "local"
    assert calls[1][0] == "openai"
    assert review["accepted"] is True
    assert review["provider"] == "openai"


def test_run_day_manager_cycle_applies_decision_claw_execution_override():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.state = None
    agent.market_cycle_count = 0
    agent._data_gateway = None
    agent.max_positions = 5
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [
            {
                "symbol": "OLD",
                "qty": 10,
                "pnl_pct": -4.5,
                "market_value": 900.0,
            }
        ],
        get_account_info=lambda: {"buying_power": 25000.0},
    )
    agent._ensure_ollama_for_phase = lambda phase: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._upsert_signals_into_dm = lambda dm, rows, **kwargs: (
        len(rows),
        [row["symbol"] for row in rows],
    )
    agent._decision_claw_market_rows = (
        AutonomousAgent._decision_claw_market_rows.__get__(agent, AutonomousAgent)
    )
    agent._decision_claw_weak_positions = (
        AutonomousAgent._decision_claw_weak_positions.__get__(agent, AutonomousAgent)
    )
    agent._run_decision_claw_review = lambda **kwargs: {
        "decision": "manage_positions",
        "actions": [
            {"action_type": "promote_symbol", "symbol": "GLNG"},
            {"action_type": "demote_symbol", "symbol": "WEAK"},
            {
                "action_type": "trim_position",
                "symbol": "OLD",
                "reason": "free capital",
                "metadata": {"trim_fraction": 0.5},
            },
            {"action_type": "hold_wave"},
        ],
    }
    agent._apply_decision_claw_symbols_to_rows = (
        AutonomousAgent._apply_decision_claw_symbols_to_rows.__get__(
            agent, AutonomousAgent
        )
    )
    agent._build_decision_claw_execution_override = (
        AutonomousAgent._build_decision_claw_execution_override.__get__(
            agent, AutonomousAgent
        )
    )

    capture = {
        "candidate_override": None,
        "execution_override": None,
        "execution_seen_in_cycle": None,
        "cleared_execution_override": False,
    }

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {"streak": 2}
        _watchlist_stale_streak = 1

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="market_hours")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 1.0

        def set_candidate_universe_override(self, symbols, **kwargs):
            capture["candidate_override"] = {"symbols": list(symbols), **kwargs}

        def clear_candidate_universe_override(self):
            capture["candidate_override_cleared"] = True

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = {"plan": dict(plan), **kwargs}
            self._execution_override_plan = dict(plan)

        def clear_execution_override_plan(self):
            capture["cleared_execution_override"] = True
            self._execution_override_plan = {}

        def run_cycle(self):
            capture["execution_seen_in_cycle"] = dict(
                getattr(self, "_execution_override_plan", {}) or {}
            )
            return {"entries": 0, "exits": 1}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[
            {"symbol": "WEAK", "score": 55.0},
            {"symbol": "GLNG", "score": 81.0},
        ],
        override_reason="market_state_review",
    )

    assert result == {"entries": 0, "exits": 1}
    assert capture["candidate_override"]["symbols"] == ["GLNG", "WEAK"]
    assert capture["execution_override"]["plan"]["entry_priority_symbols"] == ["GLNG"]
    assert capture["execution_override"]["plan"]["blocked_entry_symbols"] == ["WEAK"]
    assert (
        capture["execution_override"]["plan"]["trim_map"]["OLD"]["trim_fraction"] == 0.5
    )
    assert capture["execution_override"]["plan"]["wave_action"] == "hold_wave"
    assert capture["execution_seen_in_cycle"]["trim_map"]["OLD"]["trim_fraction"] == 0.5
    assert capture["cleared_execution_override"] is True


def test_run_day_manager_cycle_routes_wind_down_overnight_hold_override():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.logger = _LoggerStub()
    agent.state = None
    agent.market_cycle_count = 0
    agent._data_gateway = None
    agent.max_positions = 5
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [
            {
                "symbol": "APA",
                "qty": 45,
                "pnl_pct": 5.0,
                "market_value": 4500.0,
                "current_price": 100.0,
            }
        ],
        get_account_info=lambda: {"buying_power": 25000.0},
    )
    agent._ensure_ollama_for_phase = lambda phase: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._upsert_signals_into_dm = lambda dm, rows, **kwargs: (
        len(rows),
        [row["symbol"] for row in rows],
    )
    agent._decision_claw_market_rows = (
        AutonomousAgent._decision_claw_market_rows.__get__(agent, AutonomousAgent)
    )
    agent._decision_claw_weak_positions = (
        AutonomousAgent._decision_claw_weak_positions.__get__(agent, AutonomousAgent)
    )
    agent._decision_claw_wind_down_oversized_winners = (
        AutonomousAgent._decision_claw_wind_down_oversized_winners.__get__(
            agent, AutonomousAgent
        )
    )
    seen = {"payload": None}

    def _review(**kwargs):
        seen["payload"] = kwargs["payload"]
        return {
            "decision": "manage_positions",
            "actions": [
                {
                    "action_type": "hold_position",
                    "symbol": "APA",
                    "reason": "carry winner overnight",
                    "metadata": {
                        "approve_overnight_oversize": True,
                        "max_size_multiplier": 2.0,
                    },
                }
            ],
        }

    agent._run_decision_claw_review = _review
    agent._apply_decision_claw_symbols_to_rows = (
        AutonomousAgent._apply_decision_claw_symbols_to_rows.__get__(
            agent, AutonomousAgent
        )
    )
    agent._build_decision_claw_execution_override = (
        AutonomousAgent._build_decision_claw_execution_override.__get__(
            agent, AutonomousAgent
        )
    )

    capture = {"execution_override": None}

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {"streak": 0}
        _watchlist_stale_streak = 0

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="wind_down")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 1.0

        @staticmethod
        def _find_signal_data(symbol):
            return {
                "symbol": symbol,
                "target": 112.0,
                "stop_loss": 96.0,
                "risk_reward": 2.0,
                "atr_14": 3.5,
                "relative_strength": 1.6,
            }

        def set_candidate_universe_override(self, symbols, **kwargs):
            return None

        def clear_candidate_universe_override(self):
            return None

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = {"plan": dict(plan), **kwargs}
            self._execution_override_plan = dict(plan)

        def clear_execution_override_plan(self):
            self._execution_override_plan = {}

        def run_cycle(self):
            return {"entries": 0, "exits": 0}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[],
        override_reason="market_state_review",
    )

    assert result == {"entries": 0, "exits": 0}
    assert seen["payload"]["wind_down_oversized_winners"][0]["symbol"] == "APA"
    assert (
        capture["execution_override"]["plan"]["hold_overnight_map"]["APA"][
            "approve_overnight_oversize"
        ]
        is True
    )


def test_run_day_manager_cycle_preserves_override_entry_context(monkeypatch):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.state = None
    agent.market_cycle_count = 0
    agent.max_positions = 10
    agent.logger = _LoggerStub()
    agent._data_gateway = None
    agent._ensure_ollama_for_phase = lambda *args, **kwargs: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._decision_claw_market_rows = lambda rows, limit=12: rows or []
    agent._decision_claw_weak_positions = lambda positions, limit=5: []
    agent._run_decision_claw_review = lambda **kwargs: None
    agent._build_decision_claw_execution_override = lambda review_result: {}
    agent._apply_decision_claw_symbols_to_rows = (
        lambda rows, review_result, max_rows=0: rows
    )
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [],
        get_account_info=lambda: {"buying_power": 10000.0},
    )

    capture = {"marked": []}

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {}
        _watchlist_stale_streak = 0

        def __init__(self):
            self.signals = []
            self.signal_status = {}

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="market_hours")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 0.0

        def _mark_watchlist_symbol(self, ticker):
            capture["marked"].append(ticker)

        def set_candidate_universe_override(self, symbols, **kwargs):
            capture["candidate_override"] = {"symbols": list(symbols), **kwargs}

        def clear_candidate_universe_override(self):
            capture["candidate_override_cleared"] = True

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = {"plan": dict(plan), **kwargs}

        def clear_execution_override_plan(self):
            capture["execution_override_cleared"] = True

        def run_cycle(self):
            capture["signals_seen_in_cycle"] = [dict(sig) for sig in self.signals]
            return {"entries": 0, "exits": 0}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[
            {
                "symbol": "GLNG",
                "score": 81.0,
                "entry_source": "overnight_full_watchlist",
                "plan_score_source": "adjusted_plan_20260326_0829.json",
            }
        ],
        override_reason="overnight_first_hour_recheck",
    )

    assert result == {"entries": 0, "exits": 0}
    assert capture["candidate_override"]["symbols"] == ["GLNG"]
    assert capture["marked"] == ["GLNG"]
    signal = capture["signals_seen_in_cycle"][0]
    assert signal["entry_source"] == "overnight_first_hour_recheck"
    assert signal["origin_entry_source"] == "overnight_full_watchlist"
    assert signal["runtime_entry_context"] == "overnight_first_hour_recheck"
    assert signal["override_reason"] == "overnight_first_hour_recheck"
    assert signal["plan_score_source"] == "adjusted_plan_20260326_0829.json"
    assert signal["regime"] == "NEUTRAL"


def test_run_day_manager_cycle_excludes_only_executed_symbols_from_opening_state(
    monkeypatch, tmp_path: Path
):
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.state = None
    agent.market_cycle_count = 0
    agent.max_positions = 10
    agent.logger = _LoggerStub()
    agent._data_gateway = None
    agent._ensure_ollama_for_phase = lambda *args, **kwargs: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._decision_claw_market_rows = lambda rows, limit=12: rows or []
    agent._decision_claw_weak_positions = lambda positions, limit=5: []
    agent._run_decision_claw_review = lambda **kwargs: None
    agent._build_decision_claw_execution_override = lambda review_result: {}
    agent._apply_decision_claw_symbols_to_rows = (
        lambda rows, review_result, max_rows=0: rows
    )
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [],
        get_account_info=lambda: {"buying_power": 10000.0},
    )

    monkeypatch.setattr(autonomous_agent_mod, "PLANS_DIR", tmp_path)
    state_path = (
        tmp_path / f".execution_state_{real_datetime.now().strftime('%Y%m%d')}.json"
    )
    state_path.write_text(
        json.dumps(
            {
                "executed_symbols": ["EXEC"],
                "skipped_symbols": ["SKIP_A", "SKIP_B"],
            }
        ),
        encoding="utf-8",
    )

    capture = {}

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {}
        _watchlist_stale_streak = 0

        def __init__(self):
            self.signals = []
            self.signal_status = {}

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="market_hours")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 0.0

        def set_candidate_universe_override(self, symbols, **kwargs):
            capture["candidate_override"] = {"symbols": list(symbols), **kwargs}

        def clear_candidate_universe_override(self):
            capture["candidate_override_cleared"] = True

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = {"plan": dict(plan), **kwargs}

        def clear_execution_override_plan(self):
            capture["execution_override_cleared"] = True

        def set_external_exclusion_list(self, symbols):
            capture["external_exclusion"] = sorted(symbols)

        def clear_external_exclusion_list(self):
            capture["external_exclusion_cleared"] = True

        def run_cycle(self):
            return {"entries": 0, "exits": 0}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[{"symbol": "SKIP_A", "score": 81.0}],
        override_reason="overnight_first_hour_recheck",
    )

    assert result == {"entries": 0, "exits": 0}
    assert capture["external_exclusion"] == ["EXEC"]
    assert capture["external_exclusion_cleared"] is True


def test_run_day_manager_cycle_suppresses_deployment_request_until_redeployment_due():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.state = None
    agent.market_cycle_count = 0
    agent.max_positions = 5
    agent.logger = _LoggerStub()
    agent._data_gateway = None
    agent._ensure_ollama_for_phase = lambda *args, **kwargs: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._upsert_signals_into_dm = lambda dm, rows, **kwargs: (
        len(rows),
        [row["symbol"] for row in rows],
    )
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [{"symbol": "OLD", "market_value": 2000.0}],
        get_account_info=lambda: {"buying_power": 8000.0},
    )
    agent._run_decision_claw_review = lambda **kwargs: {
        "decision": "deploy",
        "deployment_request": {
            "mode": "long_exception",
            "symbols": ["GLNG"],
            "max_new_entries": 1,
            "reason": "idle capital",
        },
        "actions": [{"action_type": "promote_symbol", "symbol": "GLNG"}],
    }

    capture = {}

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {"streak": 0}
        _watchlist_stale_streak = 0

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="market_hours")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 1.0

        def set_candidate_universe_override(self, symbols, **kwargs):
            capture["candidate_override"] = {"symbols": list(symbols), **kwargs}

        def clear_candidate_universe_override(self):
            return None

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = dict(plan)

        def clear_execution_override_plan(self):
            return None

        def run_cycle(self):
            return {"entries": 0, "exits": 0}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[
            {"symbol": "GLNG", "score": 81.0, "current_price": 11.0}
        ],
    )

    assert result == {"entries": 0, "exits": 0}
    assert capture["execution_override"]["deployment_request"] == {}
    assert capture["execution_override"]["entry_priority_symbols"] == ["GLNG"]


def test_run_day_manager_cycle_reviews_due_redeployment_symbols():
    agent = AutonomousAgent.__new__(AutonomousAgent)
    agent.state = None
    agent.market_cycle_count = 0
    agent.max_positions = 5
    agent.logger = _LoggerStub()
    agent._data_gateway = None
    agent._ensure_ollama_for_phase = lambda *args, **kwargs: None
    agent._monitor_ollama_health = lambda *args, **kwargs: None
    agent._record_ollama_result = lambda *args, **kwargs: None
    agent._update_workflow_state_flag = lambda *args, **kwargs: None
    agent._inject_full_watchlist_into_dm = lambda dm: None
    agent._upsert_signals_into_dm = lambda dm, rows, **kwargs: (
        len(rows),
        [row["symbol"] for row in rows],
    )
    agent.plan_generator = SimpleNamespace(
        get_current_positions=lambda: [{"symbol": "OLD", "market_value": 2000.0}],
        get_account_info=lambda: {"buying_power": 8000.0},
    )
    agent._run_decision_claw_review = lambda **kwargs: {
        "decision": "deploy",
        "deployment_request": {
            "mode": "long_exception",
            "symbols": ["GLNG"],
            "max_new_entries": 1,
            "reason": "idle capital",
        },
        "actions": [{"action_type": "submit_entry", "symbol": "GLNG"}],
    }
    agent.decision_claw = SimpleNamespace(
        review_symbol_validation=lambda **kwargs: {
            "symbol": kwargs["symbol"],
            "decision": "approve",
            "accepted": True,
            "thesis": "good tape and catalyst",
            "why_now": "hourly redeploy review",
            "failure_mode": "",
            "confidence": 0.81,
            "warning_flags": [],
            "provider": "local",
            "model": "qwen3:14b-q4_K_M",
        }
    )

    capture = {}

    class _DM:
        entry_wave = 2
        _no_fill_watchdog_state = {"streak": 2}
        _watchlist_stale_streak = 1

        @staticmethod
        def get_current_phase():
            return SimpleNamespace(value="market_hours")

        @staticmethod
        def _live_execution_mode():
            return {"resolved_regime": {"regime": "NEUTRAL"}}

        @staticmethod
        def _get_fractional_position_count(_positions):
            return 1.0

        @staticmethod
        def _build_candidate_validation_report(signal_data):
            return {"allowed": True, "entry_source": "momentum_scanner"}

        @staticmethod
        def _find_signal_data(_symbol):
            return {
                "ticker": "GLNG",
                "current_price": 11.2,
                "entry_price": 11.0,
                "risk_reward": 2.1,
                "volume_ratio": 1.8,
                "news_sentiment": 0.35,
                "backtest_score": 62.0,
                "similar_signals_found": 18,
            }

        @staticmethod
        def _check_intraday_momentum(_symbol, avg_volume=0.0):
            return {"above_vwap": True, "volume_ratio": 1.8, "trend_up": True}

        @staticmethod
        def _candidate_relative_strength_value(_signal_data, current_price=0.0):
            return 1.9

        def set_candidate_universe_override(self, symbols, **kwargs):
            capture["candidate_override"] = {"symbols": list(symbols), **kwargs}

        def clear_candidate_universe_override(self):
            return None

        def set_execution_override_plan(self, plan, **kwargs):
            capture["execution_override"] = dict(plan)

        def clear_execution_override_plan(self):
            return None

        def run_cycle(self):
            return {"entries": 0, "exits": 0}

        @staticmethod
        def _save_state():
            return None

    dm = _DM()
    agent._get_day_manager = lambda dry_run=True: dm

    result = agent.run_day_manager_cycle(
        dry_run=True,
        candidate_universe_rows=[
            {
                "symbol": "GLNG",
                "score": 81.0,
                "current_price": 11.2,
                "entry_source": "momentum_scanner",
                "source_bucket": "watchlist",
                "setup_type": "breakout",
                "strategy_name": "news_momentum",
                "risk_reward": 2.1,
                "volume_ratio": 1.8,
                "news_sentiment": 0.35,
                "backtest_score": 62.0,
                "similar_signals_found": 18,
            }
        ],
    )

    assert result == {"entries": 0, "exits": 0}
    assert capture["execution_override"]["deployment_request"]["symbols"] == ["GLNG"]
    assert capture["execution_override"]["deployment_request"]["max_new_entries"] == 1
    assert (
        capture["execution_override"]["symbol_reviews"]["GLNG"]["decision"] == "approve"
    )


def test_decision_claw_compact_payload_preserves_daily_learning_advisory(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    payload = {
        "candidates": [{"symbol": "GLNG", "score": 80}],
        "daily_learning_advisory": {
            "as_of_date": "2026-03-24",
            "artifact_status": {
                "status": "fresh",
                "usable_for_bias": True,
                "warnings": [],
            },
            "blocked_symbols": ["BAD1", "BAD2"],
            "preferred_symbols": ["GOOD2"],
            "blocked_setup_types": ["weak_breakout"],
            "blocked_sectors": ["Retail"],
            "boosted_symbols": ["GOOD1"],
            "preferred_setup_keywords": ["high_volume"],
            "workflow_flags": ["tight_stops"],
        },
    }

    compact = claw._compact_payload(payload, phase_agent="market_open", phase_cfg=None)

    assert "daily_learning_advisory" in compact
    advisory = compact["daily_learning_advisory"]
    assert advisory["as_of_date"] == "2026-03-24"
    assert advisory["artifact_status"]["usable_for_bias"] is True
    assert advisory["blocked_symbols"] == ["BAD1", "BAD2"]
    assert advisory["preferred_symbols"] == ["GOOD2"]
    assert advisory["blocked_setup_types"] == ["weak_breakout"]


def test_decision_claw_overnight_prompt_surfaces_learning_status(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    packet = claw._compact_payload(
        {
            "candidates": [{"symbol": "GLNG", "score": 80}],
            "daily_learning_advisory": {
                "as_of_date": "2026-03-24",
                "artifact_status": {
                    "status": "degraded",
                    "usable_for_bias": False,
                    "warnings": ["learning_status:stale_daily_lessons"],
                },
                "boosted_symbols": ["GLNG"],
            },
        },
        phase_agent="overnight",
        phase_cfg=None,
    )

    prompt = claw._user_prompt(
        phase_agent="overnight",
        trigger="test",
        packet=packet,
        legacy_recommendation={},
    )

    assert "Daily learning advisory:" in prompt
    assert "usable_for_bias=False" in prompt
    assert "learning_status:stale_daily_lessons" in prompt
    assert "treat learning as neutral" not in prompt.lower()
    assert "do not promote or demote symbols solely from this advisory" in prompt


def test_decision_claw_parse_response_rejects_bad_schema_fail_closed(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw.interpret_provider_response(
        phase_agent="overnight",
        trigger="test",
        payload={"candidates": [{"symbol": "GLNG", "score": 80}]},
        legacy_recommendation={"legacy_choice": ["GLNG"]},
        response={"success": True, "content": "[]"},
    )

    assert result.provider == "legacy"
    assert result.reasoning_summary == "fallback:invalid_response_schema"


def test_decision_claw_parse_response_handles_bad_action_types(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    result = claw.interpret_provider_response(
        phase_agent="overnight",
        trigger="test",
        payload={"candidates": [{"symbol": "GLNG", "score": 80}]},
        legacy_recommendation={"legacy_choice": ["GLNG"]},
        response={
            "success": True,
            "content": json.dumps(
                {
                    "decision": "finalize_watchlist",
                    "confidence": "not-a-number",
                    "legacy_comparison": [],
                    "actions": [
                        "bad",
                        {
                            "action_type": "promote_symbol",
                            "symbol": "GLNG",
                            "size_hint": "bad",
                            "metadata": "bad",
                        },
                    ],
                }
            ),
        },
    )

    assert result.confidence == 0.5
    assert [action.symbol for action in result.actions] == ["GLNG"]
    assert result.actions[0].size_hint is None
    assert {"action": "bad", "reason": "action_not_object"} in result.rejected_actions
    assert {
        "action": "promote_symbol",
        "reason": "invalid_size_hint",
    } in result.rejected_actions


def test_decision_claw_compact_payload_preserves_entry_constraints(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    payload = {
        "entry_constraints": {
            "max_positions": 35,
            "core_max_positions": 25,
            "reserve_max_positions": 10,
            "weak_day": True,
            "source": "pm_workflow",
            "regime": "SELL_OFF",
        }
    }

    compact = claw._compact_payload(payload, phase_agent="market_open", phase_cfg=None)

    assert compact["entry_constraints"] == {
        "max_positions": 35,
        "core_max_positions": 25,
        "reserve_max_positions": 10,
        "weak_day": True,
        "source": "pm_workflow",
        "regime": "SELL_OFF",
    }


def test_decision_claw_position_advisor_includes_thesis_context(tmp_path):
    class _ThesisCache:
        def get_prompt_context(self, symbol):
            return f"Bull case for {symbol}: high relative volume."

    class _Controller:
        def __init__(self):
            self.captured_payload = None

        def review(self, **kwargs):
            from autotrade.core.decision_claw import DecisionClawResult

            self.captured_payload = kwargs.get("payload")
            return DecisionClawResult(
                phase_agent="market_state",
                trigger="test",
                decision="hold",
                confidence=0.9,
                actions=[],
                reasoning_summary="ok",
            )

    controller = _Controller()
    advisor = DecisionClawPositionAdvisor(controller, thesis_cache=_ThesisCache())

    context = {"symbol": "GLNG", "pnl_pct": 2.0, "hold_duration_minutes": 30.0}
    advisor.get_advice(context)

    assert "thesis_context" in controller.captured_payload
    assert "Bull case for GLNG" in controller.captured_payload["thesis_context"]


def test_decision_claw_updates_phase_memo_after_review(tmp_path):
    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    # Mocking review results
    from autotrade.core.decision_claw import DecisionClawResult, DecisionClawAction

    result = DecisionClawResult(
        phase_agent="overnight",
        trigger="morning_plan_finalization",
        decision="finalize_watchlist",
        confidence=0.8,
        actions=[
            DecisionClawAction(action_type="promote_symbol", symbol="GLNG"),
            DecisionClawAction(action_type="demote_symbol", symbol="WEAK"),
        ],
        reasoning_summary="overnight promote GLNG",
    )

    claw._update_phase_memo(result)

    assert "GLNG" in claw._phase_memo["overnight_promoted"]
    assert "WEAK" in claw._phase_memo["overnight_demoted"]
    assert claw._phase_memo["override_count"] == 2
    assert claw._phase_memo["last_decision_summary"] == "overnight promote GLNG"

    # Check compaction
    compact = claw._compact_payload(
        {"candidates": []}, phase_agent="market_open", phase_cfg=None
    )
    assert "cross_phase_memo" in compact
    assert "GLNG" in compact["cross_phase_memo"]["overnight_promoted"]


def test_decision_claw_score_session_effectiveness(tmp_path):
    from datetime import datetime

    cfg = DecisionClawConfig()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    date_str = datetime.now().strftime("%Y-%m-%d")
    log_dir = Path(tmp_path) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    actions_log = log_dir / f"decision_claw_actions_{date_str}.jsonl"

    actions_log.write_text(
        json.dumps(
            {"symbol": "GLNG", "action_type": "promote_symbol", "reason": "strong"}
        )
        + "\n",
        encoding="utf-8",
    )

    result = claw.score_session_effectiveness()

    assert result["success"] is True
    assert result["scorecard"]["total_overrides"] == 1

    eff_file = log_dir / f"decision_claw_effectiveness_{date_str}.json"
    assert eff_file.exists()
