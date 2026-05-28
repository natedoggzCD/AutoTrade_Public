import json
from pathlib import Path

from autotrade.core.decision_claw import DecisionClaw, DecisionClawResult
from config.config_loader import DecisionClawConfig


class _LoggerStub:
    def info(self, *args, **kwargs):
        return None

    def debug(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None


def _build_cfg() -> DecisionClawConfig:
    cfg = DecisionClawConfig()
    cfg.state_path = "data/decision_claw_state.json"
    cfg.decisions_log_path = "logs/decision_claw_decisions_{date}.jsonl"
    cfg.actions_log_path = "logs/decision_claw_actions_{date}.jsonl"
    cfg.cost_log_path = "logs/decision_claw_cost_{date}.jsonl"
    cfg.phase_snapshot_path = "plans/decision_claw_phase_snapshot_{date}.json"
    cfg.final_watchlist_path = "plans/decision_claw_final_watchlist_{date}.json"
    cfg.premarket.enabled = True
    cfg.premarket.paid_enabled = True
    return cfg


def test_premarket_hold_on_unheld_symbols_is_rejected_and_coerced(
    tmp_path, monkeypatch
):
    cfg = _build_cfg()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    plans_dir = Path(tmp_path) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = plans_dir / "morning_intelligence_latest.json"
    handoff_path.write_text(
        json.dumps(
            {
                "ranked_watchlist": [],
                "scalp_watchlist": [],
                "coverage": {"watchlist": "none", "degraded_mode": True},
                "degraded_mode": True,
            }
        ),
        encoding="utf-8",
    )

    llm_payload = {
        "decision": "hold_position",
        "confidence": 0.85,
        "reasoning_summary": "hold weak names",
        "legacy_comparison": {
            "legacy_choice": [],
            "agent_choice": [],
            "agreement_level": "override",
            "override_reason": "model_output",
        },
        "actions": [
            {"action_type": "hold_position", "symbol": "DFTX", "reason": "wait"},
            {"action_type": "hold_position", "symbol": "FLEX", "reason": "wait"},
            {"action_type": "hold_position", "symbol": "TVTX", "reason": "wait"},
        ],
    }

    def _fake_query_provider(**kwargs):
        return {
            "success": True,
            "content": json.dumps(llm_payload),
            "provider": "openai",
            "model": "gpt-5",
            "tokens_used": 32,
            "cost_usd": 0.001,
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)

    result = claw.review(
        phase_agent="premarket",
        trigger="premarket_finalization",
        payload={
            "phase": "premarket",
            "positions_count": 0,
            "open_slots": 50,
            "candidate_count": 48,
            "positions": [],
            "candidates": [
                {"symbol": "CENX", "score": 105.7, "claw_quality_score": 139.7},
                {"symbol": "TBI", "score": 98.2, "claw_quality_score": 128.4},
                {"symbol": "FGI", "score": 96.5, "claw_quality_score": 122.2},
                {"symbol": "RKLB", "score": 93.1, "claw_quality_score": 119.0},
                {"symbol": "SA", "score": 91.4, "claw_quality_score": 117.3},
            ],
        },
        legacy_recommendation={},
    )

    selected_actions = [
        action.symbol
        for action in result.actions
        if action.action_type
        in {"submit_entry", "promote_symbol", "force_signal_recheck"}
    ]
    assert result.decision != "hold_position"
    assert selected_actions[:3] == ["CENX", "TBI", "FGI"]
    assert all(action.action_type != "hold_position" for action in result.actions)
    assert any(
        item.get("reason") == "hold_on_unheld_symbol"
        for item in (result.rejected_actions or [])
    )
    assert result.legacy_comparison.legacy_choice[:3] == ["CENX", "TBI", "FGI"]

    final_watchlist = list(plans_dir.glob("decision_claw_final_watchlist_*.json"))
    assert final_watchlist
    final_payload = json.loads(final_watchlist[0].read_text(encoding="utf-8"))
    assert final_payload["selected_symbols"][:3] == ["CENX", "TBI", "FGI"]
    assert final_payload["decision"] != "hold_position"

    updated_handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    ranked_symbols = [
        str(row.get("symbol") or row.get("ticker") or "").upper()
        for row in (updated_handoff.get("ranked_watchlist") or [])
    ]
    scalp_symbols = [
        str(row.get("symbol") or row.get("ticker") or "").upper()
        for row in (updated_handoff.get("scalp_watchlist") or [])
    ]
    assert ranked_symbols[:3] == ["CENX", "TBI", "FGI"]
    assert scalp_symbols[:3] == ["CENX", "TBI", "FGI"]
    assert updated_handoff.get("coverage", {}).get("watchlist") == "full"
    assert bool(updated_handoff.get("degraded_mode", True)) is False


def test_premarket_selected_symbol_sync_refuses_stale_latest_handoff(tmp_path):
    cfg = _build_cfg()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    plans_dir = Path(tmp_path) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    handoff_path = plans_dir / "morning_intelligence_latest.json"
    handoff_path.write_text(
        json.dumps(
            {
                "generated_at_et": "2026-04-24T15:02:31.274104-04:00",
                "ranked_watchlist": [{"symbol": "OLD", "score": 99.0}],
                "scalp_watchlist": [{"symbol": "OLD", "score": 99.0}],
                "coverage": {"watchlist": "full", "degraded_mode": False},
                "degraded_mode": False,
            }
        ),
        encoding="utf-8",
    )

    claw._sync_premarket_handoff_from_selected_symbols(
        selected_symbols=["BMNR", "BLSH"],
        result=DecisionClawResult(
            phase_agent="premarket",
            trigger="premarket_finalization",
            decision="finalize_watchlist",
            confidence=0.8,
            evidence_summary={"candidates": [{"symbol": "BMNR", "score": 81.25}]},
        ),
    )

    updated = json.loads(handoff_path.read_text(encoding="utf-8"))
    assert [row["symbol"] for row in updated["ranked_watchlist"]] == ["OLD"]
    assert updated["degraded_mode"] is True
    assert (
        updated["decision_claw"]["reason"] == "premarket_selected_symbols_sync_skipped"
    )
    assert "generated_at_et_stale" in updated["decision_claw"]["stale_reason"]


def test_premarket_floor_inserts_top_final_score_names_when_llm_misses_them(
    tmp_path, monkeypatch
):
    cfg = _build_cfg()
    claw = DecisionClaw(cfg, logger_=_LoggerStub(), project_root=Path(tmp_path))

    llm_payload = {
        "decision": "finalize_watchlist",
        "confidence": 0.72,
        "reasoning_summary": "repeat names preferred",
        "legacy_comparison": {
            "legacy_choice": ["HOLO", "SGML", "CENX"],
            "agent_choice": ["HOLO", "SGML", "CENX"],
            "agreement_level": "partial",
            "override_reason": "model_output",
        },
        "actions": [
            {
                "action_type": "promote_symbol",
                "symbol": "HOLO",
                "reason": "premarket_priority",
            },
            {
                "action_type": "promote_symbol",
                "symbol": "SGML",
                "reason": "premarket_priority",
            },
            {
                "action_type": "promote_symbol",
                "symbol": "CENX",
                "reason": "premarket_priority",
            },
        ],
    }

    def _fake_query_provider(**kwargs):
        return {
            "success": True,
            "content": json.dumps(llm_payload),
            "provider": "local",
            "model": "qwen3:14b-q4_K_M",
            "tokens_used": 64,
            "cost_usd": 0.0,
        }

    monkeypatch.setattr(claw, "_query_provider", _fake_query_provider)

    result = claw.review(
        phase_agent="premarket",
        trigger="premarket_finalization",
        payload={
            "phase": "premarket",
            "positions_count": 0,
            "open_slots": 50,
            "candidate_count": 6,
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
                    "symbol": "CENX",
                    "score": 76.5,
                    "final_score": 76.5,
                    "ranking_score": 100.0,
                    "recommendation": "BUY",
                    "fallback_repeat_count": 2,
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
            ],
        },
        legacy_recommendation={},
    )

    selected_actions = [
        action.symbol
        for action in result.actions
        if action.action_type
        in {"submit_entry", "promote_symbol", "force_signal_recheck"}
    ]
    assert selected_actions[:3] == ["BMNR", "SRRK", "LTM"]
