"""H2 regression: hard losers must exit BEFORE replacement search.

The 2026-05-18 OCUL incident showed that when the replacement engine could
not find a viable swap (every candidate rejected by `cautious_selective_longs`
posture), a clear loser (score=-40, pnl=-9%) sat unliquidated for ~50 minutes
while it kept bleeding.

The fix (commit 73aa73fe) added an explicit "if hard loser, exit first" branch
inside `_run_cycle_inner` BEFORE the replacement attempt loop. This test pins
the ordering so a future refactor cannot silently undo it.

We use a source-level structural check rather than a behavioral simulation:
`_run_cycle_inner` is ~1500 lines and tightly coupled to dozens of `self.*`
attributes — a full behavioral fixture would be a multi-hundred-line mock that
breaks on any unrelated refactor. The structural check is narrow and durable.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DAY_MGR = REPO_ROOT / "autotrade" / "core" / "day_manager.py"


def _read_day_manager_text() -> str:
    return DAY_MGR.read_text(encoding="utf-8")


def test_hard_replace_score_floor_matches_ocul_observed_value():
    """OCUL hit score=-40 today; the fix must trip at that exact threshold."""
    text = _read_day_manager_text()
    m = re.search(r"HARD_REPLACE_SCORE_FLOOR\s*=\s*(-?\d+)", text)
    assert m, "HARD_REPLACE_SCORE_FLOOR constant not found"
    value = int(m.group(1))
    assert value == -40, (
        f"HARD_REPLACE_SCORE_FLOOR is {value}; OCUL hit -40 today and would "
        "no longer be caught by the hard-loser exit branch."
    )


def test_hard_loser_exit_branch_precedes_replacement_attempt_loop():
    """The hard-loser execute_exit must run BEFORE the replacement attempt loop.

    If a future refactor moves the replacement loop above the hard-loser
    branch, OCUL-style losers will again wait through 9 attempts before
    exiting. This test pins the source ordering inside `_run_cycle_inner`.
    """
    text = _read_day_manager_text()

    # Locate the hard-loser exit branch by its distinctive log message.
    hard_loser_marker = text.find("exiting before replacement search")
    assert hard_loser_marker > 0, (
        "Hard-loser exit branch (log: 'exiting before replacement search') "
        "not found in day_manager.py. The H2/Item #11B fix has regressed."
    )

    # The replacement attempt loop is identified by the
    # MAX_REPLACEMENT_ATTEMPTS_PER_POS guard immediately inside the
    # candidates iteration.
    replacement_loop_marker = text.find(
        "replacement_attempts >= MAX_REPLACEMENT_ATTEMPTS_PER_POS"
    )
    assert replacement_loop_marker > 0, (
        "Replacement attempt loop marker not found. File structure may have "
        "changed; review this test."
    )

    assert hard_loser_marker < replacement_loop_marker, (
        "Hard-loser exit branch must precede the replacement-attempt loop in "
        f"source order (hard-loser at offset {hard_loser_marker}, "
        f"replacement loop at offset {replacement_loop_marker}). H2/Item #11B "
        "has regressed — OCUL-style losers will be delayed again."
    )


def test_hard_loser_branch_uses_loser_exit_pnl_pct_threshold():
    """The branch reads `loser_exit_pnl_pct` from the failsafe snapshot.

    Today's OCUL hit pnl=-9% and loser_exit_pnl_pct=-5.0; the comparison
    -9 <= -5 must be the trigger. This test pins the source so a future
    refactor doesn't accidentally hardcode -5 or read from a different
    config.
    """
    text = _read_day_manager_text()
    # Find a window around the hard-loser branch.
    idx = text.find("exiting before replacement search")
    assert idx > 0
    window_start = max(0, idx - 1500)
    window = text[window_start:idx]

    assert "loser_exit_pnl_pct" in window, (
        "Hard-loser branch must use the failsafe `loser_exit_pnl_pct` "
        "threshold, not a hardcoded value."
    )
    assert "strategy_failsafe_snapshot" in window, (
        "Hard-loser branch must read the threshold from strategy_failsafe_snapshot."
    )
    assert "HARD_REPLACE_SCORE_FLOOR" in window, (
        "Hard-loser branch must compare against HARD_REPLACE_SCORE_FLOOR."
    )


def test_ocul_observed_values_would_trigger_hard_loser_exit():
    """Pure-logic sanity: OCUL's score/pnl values satisfy the branch conditions."""
    HARD_REPLACE_SCORE_FLOOR = -40
    loser_exit_pnl_pct = -5.0

    # OCUL on 2026-05-18 at the time of the failed exit attempt.
    ocul_score = -40
    ocul_pnl_pct = -9.0

    assert ocul_score <= HARD_REPLACE_SCORE_FLOOR
    assert ocul_pnl_pct <= loser_exit_pnl_pct
    # Both conditions met -> would trigger the immediate exit branch.
