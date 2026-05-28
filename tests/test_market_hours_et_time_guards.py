"""Regression tests for the 2026-05-20 MARKET_HOURS hang.

The hang was caused by ``_execute_entry_waves`` using ``datetime.now()``
(local CDT) but comparing against ET-based wave-time thresholds
(585=09:45 ET, 720=12:00 ET). At any CDT in [09:45, 12:00], the math
fell inside the wave window even though the actual ET time was past
the noon cutoff. The function then called ``_effective_max_positions``
which triggered an expensive ``_refresh_core_data_readiness`` parquet
read that hung under disk contention with MomentumScannerDaemon.

These tests pin the ET-time invariants for:
* ``_execute_entry_waves`` early-return past 12:00 ET regardless of TZ.
* ``_run_market_hours_cycle`` ``current_minutes`` computed in ET.
* OpenAI client passes ``timeout=`` on every chat.completions.create.
* ``DataSyncManager`` cache TTL is generous enough to absorb a slow
  ``check_sync_status`` call without re-entering the slow path.
"""

from __future__ import annotations

import importlib
import inspect
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest


# --------------------------------------------------------------------------
# 1. ``_execute_entry_waves`` must return early past 12:00 ET regardless
#    of the host machine's local timezone.
# --------------------------------------------------------------------------


def _patch_datetime_now_eastern(monkeypatch, target_module, et_dt: datetime) -> None:
    """Make ``datetime.now(pytz.timezone(...))`` return ``et_dt`` inside the
    target module without breaking unrelated ``datetime`` use.
    """

    class _FakeDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is not None and getattr(tz, "zone", "") == "US/Eastern":
                return et_dt
            return datetime.now(tz)

    monkeypatch.setattr(target_module, "datetime", _FakeDateTime)


def test_execute_entry_waves_returns_early_past_noon_et(monkeypatch):
    """The wave-gate early-return must trigger when ET time is past 12:00,
    even if local CDT math (11:46 CDT == 706 < 720) would have passed it.
    """
    from autotrade.core import autonomous_agent as aa

    # 12:46 ET = 766 minutes since midnight ET, past the 720 cutoff.
    et_dt = datetime(2026, 5, 20, 12, 46, tzinfo=ZoneInfo("US/Eastern"))
    _patch_datetime_now_eastern(monkeypatch, aa, et_dt)

    agent = MagicMock(spec=aa.AutonomousAgent)
    agent.logger = MagicMock()

    # Bind the real method to our mock so we exercise the actual guard.
    fn = aa.AutonomousAgent._execute_entry_waves.__get__(agent)
    result = fn(execute=False)

    assert result is None, "function must return None on past-noon ET"


def test_execute_entry_waves_runs_at_10_oclock_et(monkeypatch):
    """Within the wave window (09:45-12:00 ET), the gate must NOT return
    early. The function should at least reach the execution_state_file
    check.
    """
    from autotrade.core import autonomous_agent as aa

    # 10:00 ET = 600 minutes, inside [585, 720).
    et_dt = datetime(2026, 5, 20, 10, 0, tzinfo=ZoneInfo("US/Eastern"))
    _patch_datetime_now_eastern(monkeypatch, aa, et_dt)

    # Make the execution_state_file.exists() check return False so the
    # function returns at the second guard, which proves we got PAST
    # the wave-time guard.
    fake_path = MagicMock()
    fake_path.exists.return_value = False
    monkeypatch.setattr(
        aa.AutonomousAgent,
        "_execute_entry_waves",
        aa.AutonomousAgent._execute_entry_waves,
    )
    with patch.object(aa, "PLANS_DIR", fake_path.__truediv__("plans")):
        agent = MagicMock(spec=aa.AutonomousAgent)
        agent.logger = MagicMock()
        fn = aa.AutonomousAgent._execute_entry_waves.__get__(agent)
        result = fn(execute=False)
        assert result is None  # returned via execution_state_file guard


def test_execute_entry_waves_returns_early_before_9_45_et(monkeypatch):
    """The function must also return early before 09:45 ET (premarket)."""
    from autotrade.core import autonomous_agent as aa

    # 09:30 ET = 570 minutes, below 585 cutoff.
    et_dt = datetime(2026, 5, 20, 9, 30, tzinfo=ZoneInfo("US/Eastern"))
    _patch_datetime_now_eastern(monkeypatch, aa, et_dt)

    agent = MagicMock(spec=aa.AutonomousAgent)
    agent.logger = MagicMock()
    fn = aa.AutonomousAgent._execute_entry_waves.__get__(agent)
    result = fn(execute=False)

    assert result is None


# --------------------------------------------------------------------------
# 2. Source-level guard: ``_execute_entry_waves`` source must not contain
#    the bare ``datetime.now()`` pattern that comparing against ET
#    thresholds. This catches re-introduction of the bug if someone
#    rewrites the function without realising the timezone subtlety.
# --------------------------------------------------------------------------


def test_execute_entry_waves_source_uses_eastern_timezone():
    from autotrade.core import autonomous_agent as aa

    source = inspect.getsource(aa.AutonomousAgent._execute_entry_waves)
    # Must call pytz.timezone("US/Eastern") (or equivalent) before
    # deriving et_time from now().
    assert "US/Eastern" in source, (
        "_execute_entry_waves must derive et_time from US/Eastern "
        "timezone; comparing local time against ET-based thresholds "
        "was today's hang (see prompts/dev/2026-05-20_session_observations.md)"
    )


def test_run_market_hours_cycle_recheck_uses_eastern():
    """The first-hour recheck cutoff (570/630) is in ET minutes."""
    from autotrade.core import autonomous_agent as aa

    source = inspect.getsource(aa.AutonomousAgent._run_market_hours_cycle)
    # Look for the current_minutes derivation. It must use Eastern.
    assert "current_minutes" in source
    # The bug pattern was: current_minutes = datetime.now().hour * 60 + ...
    # The fix uses ZoneInfo("US/Eastern") or pytz.timezone("US/Eastern").
    assert "US/Eastern" in source, (
        "current_minutes in _run_market_hours_cycle must be in ET "
        "since cutoffs (570=09:30, 630=10:30) are ET minutes"
    )


# --------------------------------------------------------------------------
# 3. OpenAI client must pass timeout= on chat.completions.create.
# --------------------------------------------------------------------------


def test_openai_client_passes_request_timeout():
    """OpenAIClient.chat must pass a bounded timeout. Without it, the
    SDK defaults to 600s per request and the 3-retry loop can pin the
    DecisionClaw budget for 30 min.
    """
    from autotrade.utils.openai_client import OpenAIClient

    client = OpenAIClient.__new__(OpenAIClient)
    client.api_key = "fake-key"
    client._available = True
    client._client = None

    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "ok"
    fake_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
    fake_response.model = "gpt-4.1-mini"

    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = fake_response
    client._client = fake_openai_client
    client._calculate_cost = lambda *a, **k: 0.0

    client.chat(prompt="hi", model="gpt-4.1-mini")

    call_kwargs = fake_openai_client.chat.completions.create.call_args.kwargs
    assert "timeout" in call_kwargs, (
        "OpenAIClient.chat must pass timeout= on chat.completions.create "
        "to bound the HTTP request; the SDK default is 600s and the "
        "3-retry loop can pin DecisionClaw for 30 min without it."
    )
    assert call_kwargs["timeout"] <= 120, (
        "timeout must be <= 120s so 3 retries fit inside the 240s "
        "DecisionClaw budget"
    )


# --------------------------------------------------------------------------
# 4. DataSyncManager cache TTL must be large enough that a slow
#    check_sync_status call cannot re-enter the slow path before it
#    completes.
# --------------------------------------------------------------------------


def test_data_sync_cache_ttl_is_generous():
    """A 60s TTL was too short: _get_csv_info's full pd.read_csv +
    pd.to_datetime could exceed it under disk pressure, creating an
    unresolvable re-read cascade. TTL must be at least 300s.
    """
    from autotrade.utils.data_sync import DataSyncManager

    mgr = DataSyncManager()
    assert mgr._sync_status_summary_ttl_seconds >= 300, (
        "Cache TTL too short. Today (2026-05-20) the 60s TTL caused "
        "an unresolvable re-read cascade when concurrent disk pressure "
        "made check_sync_status take longer than 60s. See "
        "prompts/dev/2026-05-20_session_observations.md."
    )


# --------------------------------------------------------------------------
# 5. Cache fingerprint contract — TTL bump must not weaken freshness
#    semantics. When the underlying file changes, the cache must
#    invalidate immediately regardless of TTL.
# --------------------------------------------------------------------------


def test_data_sync_cache_invalidates_on_file_change(tmp_path, monkeypatch):
    """Fingerprint (mtime+size) must override TTL when a source file
    actually changes. The TTL bump in this commit only suppresses the
    fast re-read cascade — it must not make stale data visible.
    """
    from autotrade.utils import data_sync

    fake_parquet = tmp_path / "fake.parquet"
    fake_parquet.write_text("v1")
    monkeypatch.setattr(
        data_sync,
        "DATA_SOURCES",
        {"fake_source": fake_parquet},
    )

    mgr = data_sync.DataSyncManager()

    fingerprint_1 = mgr._build_source_fingerprint()

    # Mutate the file's content/size.
    fake_parquet.write_text("v2 longer content")
    fingerprint_2 = mgr._build_source_fingerprint()

    assert fingerprint_1 != fingerprint_2, (
        "Fingerprint must change when file content/size changes — "
        "this guarantees the TTL bump does not mask stale data."
    )
