"""Signal throughput monitor.

day-manager 2026-05-19: 2026-05-19 session ran with 393 signals accepted
and 0 executed for 90 minutes before a human noticed. The
[signal.count] event stream had the conversion_rate=0.0 metric all
along; nothing tied it to an alarm.

This module is the safety net. It consumes every aggregate_signal_lifecycle
emission, computes rolling per-family conversion rate, and:

1. Logs `[SIGNAL THROUGHPUT BROKEN]` at CRITICAL whenever a family's
   signals_accepted >= MIN_ACCEPTED_FOR_ALARM and conversion_rate <
   CONVERSION_FLOOR.
2. After CONSECUTIVE_BAD_CYCLES_FOR_FLAG consecutive bad cycles for a
   family, writes flags/signal_throughput_broken.flag with the current
   counts and the top rejection reasons (visibility forcing function;
   NOT a kill switch — trading continues).
3. On recovery (conversion back above floor for one cycle), emits
   `[SIGNAL THROUGHPUT RECOVERED]` and removes the flag.
4. Appends one JSONL row per observation to
   logs/signal_throughput_<YYYY-MM-DD>.jsonl for the missed-opportunities
   report (H9) to cross-reference against.

Thresholds are intentionally simple constants. Tune from the telemetry
data after a few sessions of real-world operation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Dict, Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - py<3.9 fallback
    ZoneInfo = None  # type: ignore[assignment]

logger = logging.getLogger("signal_throughput_monitor")

# day-manager 2026-05-19: tuning targets. Suggested defaults per the
# evening directive in prompts/dev/2026-05-19_session_observations.md
# item #11. Override via SignalThroughputMonitor constructor for tests.
MIN_ACCEPTED_FOR_ALARM = 50
CONVERSION_FLOOR = 0.05
CONSECUTIVE_BAD_CYCLES_FOR_FLAG = 3

DEFAULT_FLAG_PATH = Path("flags/signal_throughput_broken.flag")
DEFAULT_TELEMETRY_DIR = Path("logs")

# H7 (2026-05-21): the monitor only makes sense during the trading window.
# executed=0 during PREMARKET / OVERNIGHT / POST_MARKET is the expected
# state (no orders can fire) and was producing false-positive
# [SIGNAL THROUGHPUT BROKEN] CRITICAL lines plus spurious flag writes.
# Accept any string the caller's phase enum yields; match case-insensitively.
TRADING_PHASES = frozenset(
    {
        "market_open",
        "market_hours",
        "observation",
        "research",
        "core_trading",
        "wind_down",
    }
)
NON_TRADING_PHASES = frozenset(
    {
        "early_premarket",
        "premarket",
        "overnight",
        "post_market",
        "after_hours",
        "pm_workflow",
    }
)


def _is_trading_window_now() -> bool:
    """Clock-based fallback when no phase string is supplied.

    Returns True iff wall-clock is within the US equity regular session
    (09:30-16:00 ET, Mon-Fri). Used when callers don't pass `phase=`.
    """
    if ZoneInfo is None:
        return True  # be permissive if tz support missing
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    return dt_time(9, 30) <= now_et.time() < dt_time(16, 0)


@dataclass
class _FamilyState:
    """Per-family rolling state for the throughput monitor."""

    consecutive_bad: int = 0
    last_observation_was_bad: bool = False
    total_observations: int = 0
    last_top_reasons: Dict[str, int] = field(default_factory=dict)


class SignalThroughputMonitor:
    """Computes per-family conversion rate and alarms on collapse."""

    def __init__(
        self,
        *,
        min_accepted_for_alarm: int = MIN_ACCEPTED_FOR_ALARM,
        conversion_floor: float = CONVERSION_FLOOR,
        consecutive_bad_for_flag: int = CONSECUTIVE_BAD_CYCLES_FOR_FLAG,
        flag_path: Optional[Path] = None,
        telemetry_dir: Optional[Path] = None,
    ) -> None:
        self.min_accepted_for_alarm = max(1, int(min_accepted_for_alarm))
        self.conversion_floor = max(0.0, float(conversion_floor))
        self.consecutive_bad_for_flag = max(1, int(consecutive_bad_for_flag))
        self._flag_path = flag_path or DEFAULT_FLAG_PATH
        self._telemetry_dir = telemetry_dir or DEFAULT_TELEMETRY_DIR
        self._state: Dict[str, _FamilyState] = {}

    def observe(
        self,
        family: str,
        *,
        signals_accepted: int,
        signals_executed: int,
        skip_reasons: Optional[Dict[str, int]] = None,
        cycle_n: Optional[int] = None,
        phase: Optional[str] = None,
    ) -> Dict[str, object]:
        """Record one cycle's outcome for a family. Returns a small summary
        dict describing what (if anything) the monitor did. Never raises.

        Bad cycle: signals_accepted >= min_accepted_for_alarm AND
        conversion_rate < conversion_floor. Cycles below the minimum
        sample size never count as bad and never reset the consecutive
        counter (the family was simply quiet that cycle).
        """
        accepted = max(0, int(signals_accepted or 0))
        executed = max(0, int(signals_executed or 0))
        conversion = (executed / accepted) if accepted > 0 else 0.0

        # H7 (2026-05-21): phase-gate. Outside the trading window, executed=0
        # is expected and should not raise the alarm or write the flag.
        phase_norm = str(phase).lower() if phase else None
        if phase_norm in NON_TRADING_PHASES:
            return {
                "family": family,
                "signals_accepted": accepted,
                "signals_executed": executed,
                "conversion_rate": conversion,
                "broken": False,
                "recovered": False,
                "flag_written": False,
                "flag_removed": False,
                "skipped_phase": phase_norm,
            }
        if phase_norm is None and not _is_trading_window_now():
            return {
                "family": family,
                "signals_accepted": accepted,
                "signals_executed": executed,
                "conversion_rate": conversion,
                "broken": False,
                "recovered": False,
                "flag_written": False,
                "flag_removed": False,
                "skipped_phase": "clock_outside_session",
            }

        state = self._state.setdefault(str(family), _FamilyState())
        state.total_observations += 1
        result: Dict[str, object] = {
            "family": family,
            "signals_accepted": accepted,
            "signals_executed": executed,
            "conversion_rate": conversion,
            "broken": False,
            "recovered": False,
            "flag_written": False,
            "flag_removed": False,
        }

        # Cycles below the sample-size floor are noise: don't alarm and
        # don't reset the streak. Just record telemetry.
        if accepted < self.min_accepted_for_alarm:
            self._append_telemetry(
                family=family,
                cycle_n=cycle_n,
                phase=phase,
                accepted=accepted,
                executed=executed,
                conversion=conversion,
                skip_reasons=skip_reasons or {},
                bad=False,
                consecutive_bad=state.consecutive_bad,
                action="below_sample_floor",
            )
            return result

        is_bad = conversion < self.conversion_floor

        if is_bad:
            state.consecutive_bad += 1
            state.last_observation_was_bad = True
            if skip_reasons:
                state.last_top_reasons = self._top_reasons(skip_reasons)
            self._emit_broken_log(
                family=family,
                accepted=accepted,
                executed=executed,
                conversion=conversion,
                consecutive_bad=state.consecutive_bad,
                top_reasons=state.last_top_reasons,
            )
            result["broken"] = True
            if state.consecutive_bad >= self.consecutive_bad_for_flag:
                wrote = self._write_flag(
                    family=family,
                    cycle_n=cycle_n,
                    phase=phase,
                    accepted=accepted,
                    executed=executed,
                    conversion=conversion,
                    top_reasons=state.last_top_reasons,
                    consecutive_bad=state.consecutive_bad,
                )
                result["flag_written"] = wrote
        else:
            recovered = state.last_observation_was_bad
            state.last_observation_was_bad = False
            state.consecutive_bad = 0
            if recovered:
                self._emit_recovered_log(
                    family=family,
                    accepted=accepted,
                    executed=executed,
                    conversion=conversion,
                )
                result["recovered"] = True
                removed = self._remove_flag()
                result["flag_removed"] = removed

        self._append_telemetry(
            family=family,
            cycle_n=cycle_n,
            phase=phase,
            accepted=accepted,
            executed=executed,
            conversion=conversion,
            skip_reasons=skip_reasons or {},
            bad=is_bad,
            consecutive_bad=state.consecutive_bad,
            action="broken" if is_bad else ("recovered" if result["recovered"] else "ok"),
        )
        return result

    @staticmethod
    def _top_reasons(skip_reasons: Dict[str, int], n: int = 5) -> Dict[str, int]:
        if not skip_reasons:
            return {}
        sorted_items = sorted(
            skip_reasons.items(), key=lambda kv: (-int(kv[1] or 0), str(kv[0]))
        )
        return {str(k): int(v) for k, v in sorted_items[:n]}

    def _emit_broken_log(
        self,
        *,
        family: str,
        accepted: int,
        executed: int,
        conversion: float,
        consecutive_bad: int,
        top_reasons: Dict[str, int],
    ) -> None:
        reason_str = (
            ", ".join(f"{k}={v}" for k, v in top_reasons.items())
            if top_reasons
            else "no_skip_reasons_attached"
        )
        logger.critical(
            "[SIGNAL THROUGHPUT BROKEN] family=%s accepted=%d executed=%d "
            "conversion=%.2f%% consecutive_bad_cycles=%d top_reasons=[%s]",
            family,
            accepted,
            executed,
            100.0 * conversion,
            consecutive_bad,
            reason_str,
        )

    def _emit_recovered_log(
        self,
        *,
        family: str,
        accepted: int,
        executed: int,
        conversion: float,
    ) -> None:
        logger.warning(
            "[SIGNAL THROUGHPUT RECOVERED] family=%s accepted=%d executed=%d "
            "conversion=%.2f%%",
            family,
            accepted,
            executed,
            100.0 * conversion,
        )

    def _write_flag(
        self,
        *,
        family: str,
        cycle_n: Optional[int],
        phase: Optional[str],
        accepted: int,
        executed: int,
        conversion: float,
        top_reasons: Dict[str, int],
        consecutive_bad: int,
    ) -> bool:
        try:
            self._flag_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "ts": datetime.now().isoformat(),
                "family": family,
                "cycle_n": cycle_n,
                "phase": phase,
                "signals_accepted": accepted,
                "signals_executed": executed,
                "conversion_rate": conversion,
                "consecutive_bad_cycles": consecutive_bad,
                "top_rejection_reasons": top_reasons,
            }
            self._flag_path.write_text(
                json.dumps(payload, indent=2), encoding="utf-8"
            )
            return True
        except Exception as exc:
            logger.warning(
                "[SIGNAL THROUGHPUT MONITOR] flag write failed: %s", exc
            )
            return False

    def _remove_flag(self) -> bool:
        try:
            if self._flag_path.exists():
                self._flag_path.unlink()
                return True
            return False
        except Exception as exc:
            logger.warning(
                "[SIGNAL THROUGHPUT MONITOR] flag remove failed: %s", exc
            )
            return False

    def _append_telemetry(
        self,
        *,
        family: str,
        cycle_n: Optional[int],
        phase: Optional[str],
        accepted: int,
        executed: int,
        conversion: float,
        skip_reasons: Dict[str, int],
        bad: bool,
        consecutive_bad: int,
        action: str,
    ) -> None:
        try:
            now = datetime.now()
            path = (
                self._telemetry_dir
                / f"signal_throughput_{now.strftime('%Y-%m-%d')}.jsonl"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            row = {
                "ts": now.isoformat(),
                "family": family,
                "cycle_n": cycle_n,
                "phase": phase,
                "signals_accepted": accepted,
                "signals_executed": executed,
                "conversion_rate": conversion,
                "bad": bad,
                "consecutive_bad": consecutive_bad,
                "action": action,
                "top_rejection_reasons": self._top_reasons(skip_reasons),
            }
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        except Exception as exc:
            logger.debug(
                "[SIGNAL THROUGHPUT MONITOR] telemetry write skipped: %s", exc
            )


_singleton: Optional[SignalThroughputMonitor] = None


def get_signal_throughput_monitor() -> SignalThroughputMonitor:
    """Return the process-wide singleton.

    Singleton is fine here because per-family state is decoupled and the
    monitor is read-only from the rest of the code. Tests should
    instantiate SignalThroughputMonitor directly with isolated paths.
    """
    global _singleton
    if _singleton is None:
        _singleton = SignalThroughputMonitor()
    return _singleton
