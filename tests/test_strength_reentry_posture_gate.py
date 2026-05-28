"""Tests for the H6 durable fix to _strength_reentry_mode_active().

day-manager 2026-05-19 (H6, option d): the bearish-session strength_reentry
gate used to activate when ANY of 4 regime sources carried a bearish
label, OR'd together. On 2026-05-19 a single stale CHOP/SELLOFF label
poisoned the entry funnel for ~90 min while live posture was
normal_risk. The durable fix requires posture to actually be defensive
before honoring a bearish regime label.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _DM():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from autotrade.core import day_manager as dm_mod

    return dm_mod


def _make_dm(*, posture: str, regime_labels: dict):
    """Build a minimal DayManager stub for _strength_reentry_mode_active.

    regime_labels keys: current_regime, effective, youtube, router.
    """
    dm_mod = _DM()
    dm = dm_mod.DayManager.__new__(dm_mod.DayManager)
    dm.entry_quality_cfg = SimpleNamespace(
        strength_reentry_enabled=True,
        strength_reentry_active_regimes=["CHOP", "SELLOFF", "BEAR"],
    )
    dm.get_current_phase = lambda: dm_mod.TradingPhase.CORE_TRADING
    dm._get_entry_authority_state = lambda: {"state": "open"}
    dm._normalize_strength_reentry_regime_label = (
        lambda x: str(x or "").upper().strip()
    )
    dm.current_regime = regime_labels.get("current_regime", "ROTATION")
    dm._effective_market_regime = lambda: regime_labels.get("effective", "ROTATION")
    dm.youtube_context = {"regime": regime_labels.get("youtube", "ROTATION")}
    dm.regime_router_context = {"regime": regime_labels.get("router", "ROTATION")}
    dm._market_posture = lambda **kwargs: {"posture": posture}
    return dm


def test_2026_05_19_repro_normal_risk_with_one_stale_bearish_source_returns_false():
    """The exact 2026-05-19 failure mode: posture=normal_risk, 3/4 regime
    sources say ROTATION, 1 stale source says CHOP → mode must NOT
    activate. Without the H6 fix, the OR-across-4 returned True and the
    bearish-session gate blocked every add candidate."""
    dm = _make_dm(
        posture="normal_risk",
        regime_labels={
            "current_regime": "ROTATION",
            "effective": "ROTATION",
            "youtube": "ROTATION",
            "router": "CHOP",  # stale label
        },
    )
    assert dm._strength_reentry_mode_active() is False


def test_defensive_posture_plus_bearish_label_activates_mode():
    """When live posture is actually defensive AND a regime source agrees,
    the gate should still arm — that's its job."""
    for posture in (
        "defensive_risk_off",
        "capital_preservation",
        "cautious_selective_longs",
    ):
        dm = _make_dm(
            posture=posture,
            regime_labels={
                "current_regime": "ROTATION",
                "effective": "SELLOFF",
                "youtube": "ROTATION",
                "router": "ROTATION",
            },
        )
        assert (
            dm._strength_reentry_mode_active() is True
        ), f"mode should activate when posture={posture} AND a bearish label exists"


def test_normal_risk_with_all_sources_bearish_still_returns_false():
    """Strong assertion of option (d): even if all 4 sources unanimously
    say bearish, normal_risk posture overrides. (Real bearish days will
    move posture defensive on their own; if posture is still normal_risk,
    the labels are stale or wrong.)"""
    dm = _make_dm(
        posture="normal_risk",
        regime_labels={
            "current_regime": "SELLOFF",
            "effective": "SELLOFF",
            "youtube": "SELLOFF",
            "router": "SELLOFF",
        },
    )
    assert dm._strength_reentry_mode_active() is False


def test_no_bearish_labels_returns_false_regardless_of_posture():
    """Unchanged behavior: zero bearish labels → mode never arms."""
    dm = _make_dm(
        posture="capital_preservation",
        regime_labels={
            "current_regime": "ROTATION",
            "effective": "ROTATION",
            "youtube": "ROTATION",
            "router": "ROTATION",
        },
    )
    assert dm._strength_reentry_mode_active() is False


def test_bull_lock_and_inverse_fast_short_circuit_posture_check():
    """Authority states bull_lock/inverse_fast bypass the regime+posture
    logic entirely — they're already explicit gate states."""
    dm_mod = _DM()
    for state in ("bull_lock", "inverse_fast"):
        dm = _make_dm(posture="normal_risk", regime_labels={})
        dm._get_entry_authority_state = lambda s=state: {"state": s}
        assert dm._strength_reentry_mode_active() is True
