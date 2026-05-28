from __future__ import annotations

from datetime import datetime

from autotrade.core.research_artifacts import (
    build_research_artifact_bundle,
    load_latest_research_artifact_bundle,
    save_research_artifact_bundle,
)


def test_research_artifact_bundle_roundtrip(tmp_path):
    now_et = datetime(2026, 2, 11, 4, 0)
    state = {
        "watchlist": [
            {
                "symbol": "AAA",
                "final_score": 88.5,
                "sector": "Technology",
                "recommendation": "BUY",
                "confidence": 82,
                "has_catalyst": True,
                "catalyst_score": 0.9,
                "catalyst_tags": ["earnings", "guidance"],
                "catalyst_note": "Raised forward guidance",
                "s1_price": 9.8,
                "r1_price": 11.2,
                "sr_quality_score": 74.0,
            }
        ],
        "workflow_completion": {
            "watchlist_selected": True,
            "game_plan_generated": True,
        },
        "quantitative_regime": {
            "regime": "risk_on",
            "scan_mode": "normal",
        },
    }
    game_plan = {
        "date": "2026-02-11",
        "generated_at": "2026-02-11T04:00:00",
        "signals": [{"symbol": "AAA", "final_score": 88.5}],
        "full_watchlist": [{"symbol": "AAA", "final_score": 88.5}],
        "youtube_intelligence": {"regime": "RISK-ON"},
        "resolved_regime": {
            "regime": "CRISIS",
            "allow_new_longs": False,
            "sizing_multiplier": 0.0,
        },
    }

    bundle = build_research_artifact_bundle(
        state=state,
        game_plan=game_plan,
        now_et=now_et,
    )
    save_research_artifact_bundle(bundle, output_dir=tmp_path)
    loaded = load_latest_research_artifact_bundle(output_dir=tmp_path)

    assert loaded is not None
    assert loaded.trade_date == "2026-02-11"
    assert loaded.full_watchlist[0]["symbol"] == "AAA"
    assert loaded.top_picks[0]["symbol"] == "AAA"
    assert loaded.catalysts["AAA"]["score"] == 0.9
    assert loaded.catalysts["AAA"]["tags"] == ["earnings", "guidance"]
    assert loaded.support_resistance["AAA"]["s1_price"] == 9.8
    assert loaded.support_resistance["AAA"]["r1_price"] == 11.2
    assert loaded.resolved_regime["regime"] == "CRISIS"
    assert loaded.resolved_regime["allow_new_longs"] is False
    assert (tmp_path / "overnight_research_bundle_latest.json").exists()
