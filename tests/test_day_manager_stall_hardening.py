"""Regression test for H1 day_manager scoring-loop stall hardening.

Validates that the per-symbol budget timer (shipped in day_manager.py
~L19592) abandons a hung worker rather than blocking the entire batch
forever. We invoke the on-disk repro harness as a subprocess (so the
abandoned non-daemon worker thread dies with the subprocess) and parse
its exit code + stdout for the expected PASS / STALL log lines.

If the hardening regresses (timeout removed, context-manager re-added,
cancel_futures dropped), this test will fail when the subprocess
either hits the 120s faulthandler dump or exceeds the test timeout.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent
HARNESS_PATH = PROJECT_DIR / "tools" / "day_manager_stall_repro.py"


def test_repro_harness_passes_with_tight_budget():
    """Tight 5s budget → overall 45s → must abandon VST and PASS."""
    assert HARNESS_PATH.exists(), f"missing repro harness: {HARNESS_PATH}"
    env = dict(os.environ)
    env["AUTOTRADE_CANDIDATE_EVAL_BUDGET_S"] = "5"
    proc = subprocess.run(
        [sys.executable, str(HARNESS_PATH)],
        cwd=str(PROJECT_DIR),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=80,
    )
    combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
    assert proc.returncode == 0, (
        f"repro harness exit={proc.returncode}\n--- output ---\n{combined}"
    )
    assert "PASS" in combined, f"PASS marker missing:\n{combined}"
    assert "STALL" in combined, f"abandon path did not fire:\n{combined}"
    assert "Stuck=VST" in combined or "stuck" in combined.lower(), (
        f"hung ticker not surfaced in abandon log:\n{combined}"
    )
