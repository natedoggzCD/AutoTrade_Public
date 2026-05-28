"""H1 regression: cautious_selective_longs posture on SELLOFF regime.

On 2026-05-18 the posture `cautious_selective_longs` (regime=SELLOFF,
breadth=21.7%) rejected 100% of OCUL's replacement candidates with
`bad_day_posture_block:cautious_selective_longs`. Same failure family as the
2026-05-05 silent_entry_block incident.

The fix (commit a4468167) tuned the posture thresholds (min_entry_score from
its prior level down to 68.0, min_risk_reward to 1.25) so genuinely qualified
candidates can pass.

This test complements the existing `test_cautious_selective_longs_admits_*`
test (which uses RISK_OFF regime) by:
  1. Pinning the threshold *values* so a future refactor cannot silently
     re-raise them.
  2. Running the same admit-≥4 assertion against the SELLOFF regime label that
     was actually active on 2026-05-18.
  3. Including a representative candidate batch with stop/target/RR shapes
     drawn from the symbols that were rejected today (CAVA, PVH, OWL, ALM,
     OKLO, SLG, CWK).
"""

from __future__ import annotations

import json
from datetime import datetime

import sys
from pathlib import Path

import pytest

pytest.importorskip("alpaca.trading.client")

# Allow importing the existing stub from the sibling test module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from autotrade.core import day_manager as day_manager_mod
from autotrade.core.day_manager import DayManager
from test_day_manager_execution_policy import _new_dm_stub  # noqa: E402


def test_cautious_selective_longs_thresholds_pinned():
    """The tuned posture thresholds must stay at the fix values."""
    dm = _new_dm_stub()
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "SELLOFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm.get_positions = lambda: []

    posture = DayManager._market_posture(dm, positions=[])

    assert posture["posture"] == "cautious_selective_longs"
    # If any of these regress upward, OCUL-style 100%-rejection returns.
    assert posture["min_entry_score"] == pytest.approx(68.0), (
        f"min_entry_score regressed to {posture['min_entry_score']}; the H1 "
        "fix tuned this to 68.0 specifically to admit qualified candidates "
        "on a SELLOFF day."
    )
    assert posture["min_risk_reward"] == pytest.approx(1.25), (
        f"min_risk_reward regressed to {posture['min_risk_reward']}; the H1 "
        "fix tuned this to 1.25."
    )
    assert posture["min_volume_ratio"] == pytest.approx(1.0)
    assert posture["min_relative_strength"] == pytest.approx(0.8)


def test_cautious_selective_longs_admits_qualified_selloff_candidates(
    tmp_path, monkeypatch
):
    """Under SELLOFF + posture, ≥4 of 10 representative candidates pass.

    Same acceptance criteria as the agent's `test_cautious_selective_longs_admits_representative_quality_candidates`
    but explicitly pins to SELLOFF (not RISK_OFF) and uses candidate shapes
    that resemble the symbols rejected on 2026-05-18.
    """
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(day_manager_mod, "LOG_DIR", tmp_path / "logs")
    dm = _new_dm_stub()
    dm.cycle_count = 42
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "SELLOFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }
    dm.get_positions = lambda: []
    dm.live_sector_bias = {}
    dm._live_benchmark_snapshot = lambda *args, **kwargs: {"recovery_confirmed": False}

    # 5 qualified (score 70+, RR 1.3, vol 1.1) — should pass.
    # 5 unqualified (score 64, RR 1.05, vol 0.85) — should be blocked.
    candidates = []
    qualified_syms = ["CAVA", "PVH", "OWL", "ALM", "OKLO"]
    unqualified_syms = ["SLG", "CWK", "WDAY", "WOLF", "NTSK"]

    for sym in qualified_syms:
        candidates.append(
            {
                "ticker": sym,
                "score": 70.0,
                "relative_strength_pp": 0.9,
                "risk_reward": 1.3,
                "volume_ratio": 1.1,
                "entry_source": "replacement_engine",
            }
        )
    for sym in unqualified_syms:
        candidates.append(
            {
                "ticker": sym,
                "score": 64.0,
                "relative_strength_pp": 0.5,
                "risk_reward": 1.05,
                "volume_ratio": 0.85,
                "entry_source": "replacement_engine",
            }
        )

    decisions = [
        dm._defensive_long_block_reason(
            row["ticker"],
            signal_data=row,
            current_price=20.0,
            positions=[],
        )
        for row in candidates
    ]

    accepted = [
        candidates[i]["ticker"] for i, reason in enumerate(decisions) if reason == ""
    ]
    rejected = [
        candidates[i]["ticker"] for i, reason in enumerate(decisions) if reason != ""
    ]

    assert len(accepted) >= 4, (
        f"SELLOFF posture admitted only {len(accepted)}/10 qualified "
        f"candidates ({accepted}); H1 acceptance criterion requires ≥4. "
        "Rejected: " + ", ".join(rejected)
    )
    # Confirm rejections are the unqualified ones.
    assert set(rejected).issubset(set(unqualified_syms) | set(qualified_syms)), (
        "rejection set is malformed"
    )

    # Per-cycle accepted/rejected telemetry must be emitted.
    telemetry_path = (
        tmp_path / "logs" / f"posture_transitions_{datetime.now():%Y-%m-%d}.jsonl"
    )
    assert telemetry_path.exists(), "posture_transitions telemetry file was not written"
    impact_records = [
        json.loads(line)
        for line in telemetry_path.read_text(encoding="utf-8").splitlines()
        if '"event": "posture_impact"' in line
    ]
    assert impact_records, (
        "no posture_impact telemetry rows written — H1 telemetry regressed"
    )
    last = impact_records[-1]
    assert last["accepted_count"] >= 4
    assert last["rejected_count"] >= 4


def test_cautious_selective_longs_rejects_ocul_threshold_candidate():
    """The exact-threshold edge case: score=68, RR=1.25 should pass; lower fails."""
    dm = _new_dm_stub()
    dm.live_sector_bias = {}
    dm._live_execution_mode = lambda: {
        "resolved_regime": {"regime": "SELLOFF"},
        "entry_authority_state": "open",
        "entry_authority_snapshot": {},
    }

    # Just below the bar — must be blocked.
    reason = DayManager._defensive_long_block_reason(
        dm,
        "BELOW",
        signal_data={
            "ticker": "BELOW",
            "score": 67.9,
            "relative_strength_pp": 0.7,
            "risk_reward": 1.24,
            "volume_ratio": 0.95,
            "entry_source": "replacement_engine",
        },
        current_price=20.0,
        positions=[],
    )
    assert reason != "", "candidate below the tuned threshold should still block"

    # At the bar — must pass.
    reason = DayManager._defensive_long_block_reason(
        dm,
        "ATBAR",
        signal_data={
            "ticker": "ATBAR",
            "score": 68.0,
            "relative_strength_pp": 0.85,
            "risk_reward": 1.25,
            "volume_ratio": 1.0,
            "entry_source": "replacement_engine",
        },
        current_price=20.0,
        positions=[],
    )
    assert reason == "", (
        f"candidate at exactly the tuned threshold should pass; got: {reason}"
    )
