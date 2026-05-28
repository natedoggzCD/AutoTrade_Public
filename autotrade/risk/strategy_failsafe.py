"""
Strategy Failsafe Manager
=========================
Central state manager for strategy-health based risk throttling.

This module persists strategy health and exposes a deterministic runtime profile:
- normal
- degraded
- failing
- critical
"""

from __future__ import annotations

import json
import logging
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from config.config_loader import get_config
from autotrade.utils.market_time import is_trading_day

logger = logging.getLogger(__name__)


@dataclass
class StrategyFailsafeSnapshot:
    """Persisted failsafe state shared across PM workflow, autonomous agent, and day manager."""

    level: str = "normal"
    previous_level: str = "normal"
    reason: str = "init"
    source: str = "unknown"
    updated_at: str = ""

    validation_status: str = "UNKNOWN"
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sample_size: int = 0

    current_equity: float = 0.0
    peak_equity: float = 0.0
    peak_drawdown_pct: float = 0.0
    drawdown_pct: float = 0.0
    session_drawdown_pct: float = 0.0
    drawdown_anchor_date: str = ""
    drawdown_anchor_equity: float = 0.0

    recovery_streak_days: int = 0
    recovery_last_date: str = ""

    # Active profile (flattened for convenience in consumers)
    halt_new_entries: bool = False
    max_positions: int = 25
    min_conviction_hold: float = 50.0
    min_conviction_exit: float = 55.0
    position_size_pct: float = 3.0
    position_size_multiplier: float = 1.0
    stop_multiplier: float = 2.0

    # Emergency triage thresholds
    loser_exit_pnl_pct: float = -1.0
    profit_take_pnl_pct: float = 10.0

    # Session-scoped override lineage for market-hours no-trade intervention.
    operator_override_active: bool = False
    original_halt_reason: str = ""
    override_reason: str = ""
    override_timestamp: str = ""
    override_source: str = ""
    override_session_key: str = ""
    halt_lineage_cause: str = ""
    halt_lineage_set_at: str = ""
    halt_lineage_cleared_at: str = ""
    halt_lineage_clear_reason: str = ""
    error_cascade_last_error_at: str = ""
    error_cascade_quiet_cycles: int = 0
    error_cascade_recovery_stage: str = "idle"
    error_cascade_last_stepdown_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StrategyFailsafeSnapshot":
        if not isinstance(data, dict):
            return cls()
        base = cls()
        allowed = set(base.to_dict().keys())
        filtered = {k: v for k, v in data.items() if k in allowed}
        for key, default_val in base.to_dict().items():
            filtered.setdefault(key, default_val)
        return cls(**filtered)


class StrategyFailsafeManager:
    """Determines and persists failsafe level from strategy validation health."""

    def __init__(self, state_path: Optional[Path] = None):
        self.config = get_config().strategy_failsafe
        project_root = get_config().project_root
        configured_path = Path(self.config.state_file)
        if configured_path.is_absolute():
            self.state_path = configured_path
        else:
            self.state_path = project_root / configured_path
        if state_path:
            self.state_path = state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    def load_snapshot(self) -> StrategyFailsafeSnapshot:
        """Load current snapshot from disk (or return defaults)."""
        if not self.state_path.exists():
            return self._snapshot_with_profile(
                StrategyFailsafeSnapshot(level="normal", previous_level="normal")
            )
        try:
            with open(self.state_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            snap = StrategyFailsafeSnapshot.from_dict(payload)
            snap = self._normalize_drawdown_fields(snap)
            snap = self._normalize_legacy_drawdown_state(snap)
            if snap.reason == "legacy_drawdown_peak_reset":
                snap = self._snapshot_with_profile(snap)
                with open(self.state_path, "w", encoding="utf-8") as f:
                    json.dump(snap.to_dict(), f, indent=2)
                return snap
            snap = self._apply_drawdown_gate(snap)
            snap = self._apply_error_cascade_recovery(snap)
            return self._snapshot_with_profile(snap)
        except Exception as e:
            logger.warning(f"Failed to load strategy failsafe state: {e}")
            return self._snapshot_with_profile(
                StrategyFailsafeSnapshot(level="normal", previous_level="normal")
            )

    def save_snapshot(self, snapshot: StrategyFailsafeSnapshot) -> None:
        """Persist snapshot to disk."""
        snapshot.updated_at = datetime.now().isoformat()
        snapshot = self._normalize_drawdown_fields(snapshot)
        snapshot = self._apply_drawdown_gate(snapshot)
        snapshot = self._apply_error_cascade_recovery(snapshot)
        snapshot = self._snapshot_with_profile(snapshot)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, indent=2)

    def apply_operator_override(
        self,
        *,
        reason: str,
        source: str = "autonomous_agent.market_hours",
        session_key: Optional[str] = None,
    ) -> StrategyFailsafeSnapshot:
        """
        Clear a stale entry halt for the active session while preserving lineage.

        Catastrophic states (critical level or critical validation) are left untouched.
        """
        snapshot = self.load_snapshot()
        level_norm = str(snapshot.level or "normal").lower()
        validation_status = str(snapshot.validation_status or "UNKNOWN").upper()
        if level_norm == "critical" or validation_status in {"CRITICAL", "BROKEN"}:
            return snapshot

        snapshot.operator_override_active = True
        snapshot.original_halt_reason = (
            str(snapshot.reason or "").strip()
            if str(snapshot.reason or "").strip()
            else str(snapshot.original_halt_reason or "").strip()
        )
        snapshot.override_reason = str(reason or "operator_override").strip()
        snapshot.override_source = str(
            source or "autonomous_agent.market_hours"
        ).strip()
        snapshot.override_timestamp = datetime.now().isoformat()
        snapshot.override_session_key = str(
            session_key or datetime.now().strftime("%Y-%m-%d")
        ).strip()
        snapshot.halt_new_entries = False
        self.save_snapshot(snapshot)
        return self.load_snapshot()

    def mark_error_cascade_halt(
        self,
        *,
        reason: str = "runtime_error_cascade",
        source: str = "runtime_error_monitor",
        observed_at: Optional[datetime] = None,
    ) -> StrategyFailsafeSnapshot:
        """Escalate to critical failsafe for runtime error cascades with lineage."""
        snapshot = self.load_snapshot()
        event_ts = observed_at or datetime.now()
        event_iso = event_ts.isoformat()
        snapshot.previous_level = snapshot.level
        snapshot.level = "critical"
        snapshot.reason = str(reason or "runtime_error_cascade")
        snapshot.source = str(source or "runtime_error_monitor")
        snapshot.halt_lineage_cause = "error_cascade"
        snapshot.halt_lineage_set_at = event_iso
        snapshot.halt_lineage_cleared_at = ""
        snapshot.halt_lineage_clear_reason = ""
        snapshot.error_cascade_last_error_at = event_iso
        snapshot.error_cascade_quiet_cycles = 0
        snapshot.error_cascade_recovery_stage = "critical"
        snapshot.error_cascade_last_stepdown_at = event_iso
        self.save_snapshot(snapshot)
        return self.load_snapshot()

    def get_profile(self, level: str):
        level_norm = (level or "normal").lower()
        if level_norm == "critical":
            return self.config.critical
        if level_norm == "failing":
            return self.config.failing
        if level_norm == "degraded":
            return self.config.degraded
        return self.config.normal

    def update_from_strategy_validation(
        self,
        strategy_validation: Optional[Any],
        equity: float,
        source: str = "pm_workflow",
        as_of_date: Optional[str] = None,
    ) -> StrategyFailsafeSnapshot:
        """Update full state from latest strategy validation + account equity."""
        snapshot = self.load_snapshot()
        snapshot.source = source
        snapshot.previous_level = snapshot.level
        snapshot.current_equity = float(equity or 0.0)
        date_key = as_of_date or datetime.now().strftime("%Y-%m-%d")

        if snapshot.current_equity > snapshot.peak_equity:
            snapshot.peak_equity = snapshot.current_equity
        metrics = self._extract_metrics(strategy_validation)
        metrics_available = not (
            metrics["status"] == "UNKNOWN" and metrics["sample_size"] == 0
        )
        if metrics_available:
            snapshot.validation_status = metrics["status"]
            snapshot.win_rate = metrics["win_rate"]
            snapshot.profit_factor = metrics["profit_factor"]
            snapshot.sample_size = metrics["sample_size"]

            raw_level, reason = self._classify_level(
                status=snapshot.validation_status,
                win_rate=snapshot.win_rate,
                profit_factor=snapshot.profit_factor,
            )
        else:
            snapshot.validation_status = "UNKNOWN"
            raw_level = snapshot.level or "normal"
            reason = "validation_unavailable"

        peak_rebased = False
        if metrics_available and self._should_rebase_stale_drawdown_peak(
            snapshot, raw_level
        ):
            self._rebase_stale_drawdown_peak(
                snapshot,
                reason="stale_drawdown_peak_rebased",
            )
            peak_rebased = True

        self._refresh_daily_drawdown(
            snapshot=snapshot,
            date_key=date_key,
            update_session_anchor=self._source_updates_session_drawdown(source),
        )

        raw_level, reason = self._combine_with_drawdown_gate(
            raw_level,
            reason,
            self._drawdown_gate_pct(snapshot),
            apply_drawdown_gate=not self._validation_snapshot_is_healthy(snapshot),
        )
        if peak_rebased and raw_level == "normal":
            reason = "stale_drawdown_peak_rebased"
        if raw_level == "normal" and self._validation_snapshot_is_healthy(snapshot):
            level = "normal"
        else:
            level = self._apply_recovery_gate(snapshot, raw_level, date_key)
        snapshot.level = level
        snapshot.reason = reason

        self.save_snapshot(snapshot)
        if snapshot.previous_level != snapshot.level:
            logger.warning(
                "[FAILSAFE] Level transition %s -> %s | WR=%.1f%% PF=%.2f DD=%.1f%% reason=%s",
                snapshot.previous_level,
                snapshot.level,
                snapshot.win_rate * 100.0,
                snapshot.profit_factor,
                self._drawdown_gate_pct(snapshot),
                snapshot.reason,
            )
        return snapshot

    def update_equity_only(
        self, equity: float, source: str = "runtime"
    ) -> StrategyFailsafeSnapshot:
        """Refresh equity telemetry intraday when only equity is available."""
        snapshot = self.load_snapshot()
        snapshot.source = source
        snapshot.previous_level = snapshot.level
        snapshot.current_equity = float(equity or 0.0)
        if snapshot.current_equity > snapshot.peak_equity:
            snapshot.peak_equity = snapshot.current_equity
        self._refresh_daily_drawdown(
            snapshot=snapshot,
            date_key=datetime.now().strftime("%Y-%m-%d"),
            update_session_anchor=self._source_updates_session_drawdown(source),
        )
        snapshot = self._apply_drawdown_gate(snapshot)

        self.save_snapshot(snapshot)
        return snapshot

    def _refresh_daily_drawdown(
        self,
        *,
        snapshot: StrategyFailsafeSnapshot,
        date_key: str,
        update_session_anchor: bool = True,
    ) -> None:
        """Refresh peak telemetry and, only during live-session updates, daily drawdown."""
        current_equity = float(snapshot.current_equity or 0.0)
        if current_equity <= 0.0:
            snapshot.peak_drawdown_pct = 0.0
            snapshot.drawdown_pct = 0.0
            snapshot.session_drawdown_pct = 0.0
            return

        peak_equity = float(snapshot.peak_equity or 0.0)
        if peak_equity <= 0.0:
            snapshot.peak_equity = current_equity
            peak_equity = current_equity
        if current_equity > peak_equity:
            snapshot.peak_equity = current_equity
            peak_equity = current_equity

        peak_drawdown_pct = max(
            0.0,
            (peak_equity - current_equity) / peak_equity * 100.0,
        )
        snapshot.peak_drawdown_pct = peak_drawdown_pct
        snapshot.drawdown_pct = peak_drawdown_pct

        anchor_date = str(snapshot.drawdown_anchor_date or "").strip()
        anchor_equity = float(snapshot.drawdown_anchor_equity or 0.0)
        if not self._date_key_is_trading_day(anchor_date):
            snapshot.drawdown_anchor_date = ""
            snapshot.drawdown_anchor_equity = 0.0
            snapshot.session_drawdown_pct = 0.0
            anchor_date = ""
            anchor_equity = 0.0

        if not update_session_anchor or not self._date_key_is_trading_day(date_key):
            return

        if anchor_date != date_key or anchor_equity <= 0.0:
            snapshot.drawdown_anchor_date = date_key
            snapshot.drawdown_anchor_equity = current_equity
            snapshot.session_drawdown_pct = 0.0
            return

        snapshot.session_drawdown_pct = max(
            0.0,
            (anchor_equity - current_equity) / anchor_equity * 100.0,
        )

    @staticmethod
    def _source_updates_session_drawdown(source: str) -> bool:
        source_norm = str(source or "").strip().lower()
        return source_norm in {
            "runtime",
            "day_manager",
            "market_open",
            "market_hours",
        }

    @staticmethod
    def _date_key_is_trading_day(date_key: str) -> bool:
        text = str(date_key or "").strip()
        if not text:
            return False
        try:
            return bool(is_trading_day(datetime.fromisoformat(text).date()))
        except Exception:
            return False

    def _normalize_drawdown_fields(
        self, snap: StrategyFailsafeSnapshot
    ) -> StrategyFailsafeSnapshot:
        peak_drawdown = float(
            snap.peak_drawdown_pct
            if float(snap.peak_drawdown_pct or 0.0) > 0.0
            else snap.drawdown_pct
            or 0.0
        )
        snap.peak_drawdown_pct = peak_drawdown
        snap.drawdown_pct = peak_drawdown
        if not self._date_key_is_trading_day(snap.drawdown_anchor_date):
            snap.drawdown_anchor_date = ""
            snap.drawdown_anchor_equity = 0.0
            snap.session_drawdown_pct = 0.0
        return snap

    def _extract_metrics(self, strategy_validation: Optional[Any]) -> Dict[str, Any]:
        if strategy_validation is None:
            return {
                "status": "UNKNOWN",
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "sample_size": 0,
            }
        if isinstance(strategy_validation, dict):
            status = str(strategy_validation.get("status", "UNKNOWN")).upper()
            win_rate = float(
                strategy_validation.get("recent_win_rate")
                or strategy_validation.get("win_rate")
                or 0.0
            )
            profit_factor = float(
                strategy_validation.get("recent_profit_factor")
                or strategy_validation.get("profit_factor")
                or 0.0
            )
            sample_size = int(strategy_validation.get("sample_size") or 0)
            return {
                "status": status,
                "win_rate": max(0.0, win_rate),
                "profit_factor": max(0.0, profit_factor),
                "sample_size": max(0, sample_size),
            }

        status = str(getattr(strategy_validation, "status", "UNKNOWN")).upper()
        win_rate = float(getattr(strategy_validation, "recent_win_rate", 0.0) or 0.0)
        profit_factor = float(
            getattr(strategy_validation, "recent_profit_factor", 0.0) or 0.0
        )
        sample_size = int(getattr(strategy_validation, "sample_size", 0) or 0)
        return {
            "status": status,
            "win_rate": max(0.0, win_rate),
            "profit_factor": max(0.0, profit_factor),
            "sample_size": max(0, sample_size),
        }

    def _classify_level(
        self, status: str, win_rate: float, profit_factor: float
    ) -> tuple[str, str]:
        """Classify failsafe level from validation health only."""
        status_norm = str(status or "UNKNOWN").upper()
        critical_wr = float(getattr(self.config, "critical_win_rate", 0.2) or 0.2)
        failing_wr = float(getattr(self.config, "failing_win_rate", 0.25) or 0.25)
        degraded_wr = float(getattr(self.config, "degraded_win_rate", 0.30) or 0.30)
        failing_pf = float(getattr(self.config, "failing_profit_factor", 0.5) or 0.5)
        degraded_pf = float(getattr(self.config, "degraded_profit_factor", 0.7) or 0.7)

        if status_norm in {"CRITICAL", "BROKEN"} or win_rate < critical_wr:
            return "critical", (
                f"validation critical: status={status_norm} "
                f"wr={win_rate:.2f} pf={profit_factor:.2f}"
            )
        if (
            status_norm in {"FAILING", "FAIL"}
            or win_rate < failing_wr
            or profit_factor < failing_pf
        ):
            return "failing", (
                f"validation failing: status={status_norm} "
                f"wr={win_rate:.2f} pf={profit_factor:.2f}"
            )
        if (
            status_norm in {"DEGRADED", "WARNING"}
            or win_rate < degraded_wr
            or profit_factor < degraded_pf
        ):
            return "degraded", (
                f"validation degraded: status={status_norm} "
                f"wr={win_rate:.2f} pf={profit_factor:.2f}"
            )
        return "normal", "validation healthy"

    @staticmethod
    def _level_rank(level: str) -> int:
        order = {"normal": 0, "degraded": 1, "failing": 2, "critical": 3}
        return order.get(str(level or "normal").lower(), 0)

    def _classify_drawdown_level(self, drawdown_pct: float) -> tuple[str, str]:
        """Classify session-daily drawdown using configured thresholds."""
        try:
            dd = max(0.0, float(drawdown_pct or 0.0))
        except Exception:
            dd = 0.0
        critical_dd = float(getattr(self.config, "critical_drawdown_pct", 10.0) or 10.0)
        failing_dd = float(getattr(self.config, "failing_drawdown_pct", 7.0) or 7.0)
        degraded_dd = float(getattr(self.config, "degraded_drawdown_pct", 5.0) or 5.0)
        if dd >= critical_dd:
            return "critical", f"drawdown critical: {dd:.2f}% >= {critical_dd:.2f}%"
        if dd >= failing_dd:
            return "failing", f"drawdown failing: {dd:.2f}% >= {failing_dd:.2f}%"
        if dd >= degraded_dd:
            return "degraded", f"drawdown degraded: {dd:.2f}% >= {degraded_dd:.2f}%"
        return "normal", "drawdown within limits"

    def _combine_with_drawdown_gate(
        self,
        validation_level: str,
        validation_reason: str,
        drawdown_pct: float,
        *,
        apply_drawdown_gate: bool = True,
    ) -> tuple[str, str]:
        if not apply_drawdown_gate:
            return validation_level, validation_reason
        drawdown_level, drawdown_reason = self._classify_drawdown_level(drawdown_pct)
        if self._level_rank(drawdown_level) > self._level_rank(validation_level):
            return drawdown_level, drawdown_reason
        return validation_level, validation_reason

    @staticmethod
    def _drawdown_gate_pct(snapshot: StrategyFailsafeSnapshot) -> float:
        """Use session drawdown for entry halts; keep peak drawdown as telemetry only."""
        return float(snapshot.session_drawdown_pct or 0.0)

    def _apply_drawdown_gate(
        self, snapshot: StrategyFailsafeSnapshot
    ) -> StrategyFailsafeSnapshot:
        """Apply drawdown thresholds to persisted/runtime snapshots."""
        if self._validation_snapshot_is_healthy(snapshot):
            if str(snapshot.reason or "").lower().startswith("drawdown"):
                snapshot.previous_level = snapshot.level
                snapshot.level = "normal"
                snapshot.reason = "validation healthy"
            if str(snapshot.halt_lineage_cause or "").lower() == "drawdown":
                snapshot.halt_lineage_cleared_at = datetime.now().isoformat()
                snapshot.halt_lineage_clear_reason = "validation_healthy_drawdown_halt_cleared"
                snapshot.halt_lineage_cause = ""
            return snapshot
        level, reason = self._combine_with_drawdown_gate(
            str(snapshot.level or "normal"),
            str(snapshot.reason or "snapshot_loaded"),
            self._drawdown_gate_pct(snapshot),
            apply_drawdown_gate=True,
        )
        if level != snapshot.level:
            snapshot.previous_level = snapshot.level
            snapshot.level = level
            snapshot.reason = reason
        if str(level or "normal").lower() == "critical" and str(
            reason or ""
        ).lower().startswith("drawdown"):
            if snapshot.halt_lineage_cause != "drawdown":
                snapshot.halt_lineage_set_at = datetime.now().isoformat()
            snapshot.halt_lineage_cause = "drawdown"
        return snapshot

    @staticmethod
    def _parse_iso_datetime(value: str) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text)
        except Exception:
            return None

    def _apply_error_cascade_recovery(
        self, snapshot: StrategyFailsafeSnapshot
    ) -> StrategyFailsafeSnapshot:
        """
        Deterministic de-latch path for runtime error-cascade halts only.

        Drawdown-triggered critical states remain fail-closed and are never
        auto-cleared by this path.
        """
        if str(snapshot.halt_lineage_cause or "").lower() != "error_cascade":
            return snapshot

        drawdown_level, _ = self._classify_drawdown_level(
            self._drawdown_gate_pct(snapshot)
        )
        if drawdown_level == "critical":
            snapshot.halt_lineage_cause = "drawdown"
            if not snapshot.halt_lineage_set_at:
                snapshot.halt_lineage_set_at = datetime.now().isoformat()
            return snapshot

        now = datetime.now()
        quiet_minutes = max(
            1, int(getattr(self.config, "error_cascade_cooldown_minutes", 15) or 15)
        )
        stepdown_minutes = max(
            1, int(getattr(self.config, "error_cascade_stepdown_minutes", 15) or 15)
        )
        min_quiet_cycles = max(
            1, int(getattr(self.config, "error_cascade_min_quiet_cycles", 2) or 2)
        )

        last_error_ts = self._parse_iso_datetime(snapshot.error_cascade_last_error_at)
        if last_error_ts is None:
            last_error_ts = self._parse_iso_datetime(snapshot.halt_lineage_set_at)
        if last_error_ts is None:
            last_error_ts = now

        quiet_delta = now - last_error_ts
        if quiet_delta < timedelta(minutes=quiet_minutes):
            snapshot.error_cascade_quiet_cycles = 0
            snapshot.error_cascade_recovery_stage = str(
                snapshot.level or "critical"
            ).lower()
            return snapshot

        snapshot.error_cascade_quiet_cycles = (
            int(snapshot.error_cascade_quiet_cycles or 0) + 1
        )
        if snapshot.error_cascade_quiet_cycles < min_quiet_cycles:
            snapshot.error_cascade_recovery_stage = str(
                snapshot.level or "critical"
            ).lower()
            return snapshot

        stage_anchor = self._parse_iso_datetime(snapshot.error_cascade_last_stepdown_at)
        if stage_anchor is None:
            stage_anchor = self._parse_iso_datetime(snapshot.halt_lineage_set_at)
        if stage_anchor is None:
            stage_anchor = now
        if now - stage_anchor < timedelta(minutes=stepdown_minutes):
            snapshot.error_cascade_recovery_stage = str(
                snapshot.level or "critical"
            ).lower()
            return snapshot

        level_norm = str(snapshot.level or "normal").lower()
        if level_norm == "critical":
            snapshot.previous_level = snapshot.level
            snapshot.level = "failing"
            snapshot.reason = "error_cascade_stepdown:critical_to_failing"
            snapshot.error_cascade_recovery_stage = "failing"
            snapshot.error_cascade_last_stepdown_at = now.isoformat()
            return snapshot

        if level_norm == "failing":
            snapshot.previous_level = snapshot.level
            snapshot.level = "normal"
            snapshot.reason = "error_cascade_stepdown:failing_to_normal"
            snapshot.error_cascade_recovery_stage = "normal"
            snapshot.error_cascade_last_stepdown_at = now.isoformat()
            snapshot.halt_lineage_cleared_at = now.isoformat()
            snapshot.halt_lineage_clear_reason = "error_cascade_auto_recovered"
            snapshot.halt_lineage_cause = ""
            snapshot.error_cascade_quiet_cycles = 0
            return snapshot

        return snapshot

    def _validation_snapshot_is_healthy(
        self, snapshot: StrategyFailsafeSnapshot
    ) -> bool:
        """Return true when persisted validation is strong enough to avoid drawdown-only hard halt."""
        status = str(snapshot.validation_status or "UNKNOWN").upper()
        if status not in {"HEALTHY", "PASS", "OK"}:
            return False
        if int(snapshot.sample_size or 0) < 30:
            return False
        win_rate = float(snapshot.win_rate or 0.0)
        profit_factor = float(snapshot.profit_factor or 0.0)
        recovery_pf = float(getattr(self.config, "recovery_profit_factor", 1.0) or 1.0)
        critical_wr = float(getattr(self.config, "critical_win_rate", 0.2) or 0.2)
        return win_rate > critical_wr and profit_factor >= recovery_pf

    def _apply_recovery_gate(
        self, snapshot: StrategyFailsafeSnapshot, raw_level: str, as_of_date: str
    ) -> str:
        """Keep conservative mode until validation recovers for N distinct trading days."""
        was_stressed = snapshot.previous_level in {
            "degraded",
            "failing",
            "critical",
        } or snapshot.level in {"degraded", "failing", "critical"}
        meets_recovery = raw_level == "normal"

        if meets_recovery:
            if snapshot.recovery_last_date != as_of_date:
                snapshot.recovery_streak_days += 1
                snapshot.recovery_last_date = as_of_date
        else:
            snapshot.recovery_streak_days = 0
            snapshot.recovery_last_date = ""

        if (
            raw_level == "normal"
            and was_stressed
            and snapshot.recovery_streak_days < self.config.recovery_days
        ):
            return "degraded"
        return raw_level

    def _normalize_legacy_drawdown_state(
        self, snap: StrategyFailsafeSnapshot
    ) -> StrategyFailsafeSnapshot:
        """Downgrade legacy drawdown-triggered states to validation-based levels."""
        reason_text = str(snap.reason or "").lower()
        legacy_drawdown_state = (
            "drawdown" in reason_text or "auto-escalated" in reason_text
        )
        if not legacy_drawdown_state:
            return snap

        metrics_available = not (
            str(snap.validation_status or "UNKNOWN").upper() == "UNKNOWN"
            and int(snap.sample_size or 0) == 0
        )
        normalized_level = "normal"
        normalized_reason = "legacy_drawdown_state_cleared"
        if metrics_available:
            normalized_level, normalized_reason = self._classify_level(
                status=snap.validation_status,
                win_rate=snap.win_rate,
                profit_factor=snap.profit_factor,
            )
            if self._should_rebase_stale_drawdown_peak(snap, normalized_level):
                logger.warning(
                    "[FAILSAFE] Resetting stale legacy drawdown peak %.2f -> %.2f "
                    "(drawdown %.1f%%, validation=%s, sample=%s).",
                    float(snap.peak_equity or 0.0),
                    float(snap.current_equity or 0.0),
                    float(snap.drawdown_pct or 0.0),
                    snap.validation_status,
                    snap.sample_size,
                )
                self._rebase_stale_drawdown_peak(
                    snap,
                    reason="legacy_drawdown_peak_reset",
                    level=normalized_level,
                )
                return snap

        if snap.level != normalized_level or snap.reason != normalized_reason:
            logger.info(
                "[FAILSAFE] Normalized legacy drawdown state %s -> %s.",
                snap.level,
                normalized_level,
            )
            snap.previous_level = snap.level
            snap.level = normalized_level
            snap.reason = normalized_reason
        return snap

    def _should_rebase_stale_drawdown_peak(
        self, snap: StrategyFailsafeSnapshot, normalized_level: str
    ) -> bool:
        """Allow recovery from stale account peaks without weakening live drawdown gates."""
        if str(normalized_level or "").lower() != "normal":
            return False
        if str(snap.validation_status or "").upper() not in {"HEALTHY", "PASS", "OK"}:
            return False
        if int(snap.sample_size or 0) < 30:
            return False
        if float(snap.win_rate or 0.0) <= 0.0 or float(snap.profit_factor or 0.0) < 1.0:
            return False
        current_equity = float(snap.current_equity or 0.0)
        peak_equity = float(snap.peak_equity or 0.0)
        if current_equity <= 0.0 or peak_equity <= current_equity:
            return False
        critical_dd = float(getattr(self.config, "critical_drawdown_pct", 10.0) or 10.0)
        stale_peak_dd = max(critical_dd * 2.0, critical_dd + 10.0)
        implied_drawdown_pct = ((peak_equity - current_equity) / peak_equity) * 100.0
        return (
            max(float(snap.drawdown_pct or 0.0), implied_drawdown_pct) >= stale_peak_dd
        )

    def _rebase_stale_drawdown_peak(
        self,
        snap: StrategyFailsafeSnapshot,
        *,
        reason: str,
        level: str = "normal",
    ) -> None:
        """Rebase a clearly stale peak to the current account equity."""
        logger.warning(
            "[FAILSAFE] Rebase stale drawdown peak %.2f -> %.2f | reason=%s "
            "level=%s validation=%s sample=%s WR=%.3f PF=%.3f caller=%s",
            float(snap.peak_equity or 0.0),
            float(snap.current_equity or 0.0),
            reason,
            level,
            snap.validation_status,
            snap.sample_size,
            float(snap.win_rate or 0.0),
            float(snap.profit_factor or 0.0),
            " > ".join(frame.name for frame in traceback.extract_stack(limit=5)[:-1]),
        )
        snap.previous_level = snap.level
        snap.level = level
        snap.reason = reason
        snap.peak_equity = float(snap.current_equity or 0.0)
        snap.peak_drawdown_pct = 0.0
        snap.drawdown_pct = 0.0
        snap.session_drawdown_pct = 0.0

    def _snapshot_with_profile(
        self, snapshot: StrategyFailsafeSnapshot
    ) -> StrategyFailsafeSnapshot:
        profile = self.get_profile(snapshot.level)
        snapshot.halt_new_entries = bool(profile.halt_new_entries)
        snapshot.max_positions = int(profile.max_positions)
        snapshot.min_conviction_hold = float(profile.min_conviction_hold)
        snapshot.min_conviction_exit = float(profile.min_conviction_exit)
        snapshot.position_size_pct = float(profile.position_size_pct)
        snapshot.position_size_multiplier = float(profile.position_size_multiplier)
        snapshot.stop_multiplier = float(profile.stop_multiplier)
        snapshot.loser_exit_pnl_pct = float(self.config.loser_exit_pnl_pct)
        snapshot.profit_take_pnl_pct = float(self.config.profit_take_pnl_pct)
        if bool(getattr(snapshot, "operator_override_active", False)):
            validation_status = str(snapshot.validation_status or "UNKNOWN").upper()
            if str(
                snapshot.level or "normal"
            ).lower() == "critical" or validation_status in {
                "CRITICAL",
                "BROKEN",
            }:
                snapshot.operator_override_active = False
            else:
                snapshot.halt_new_entries = False
                if snapshot.override_timestamp and not snapshot.override_reason:
                    snapshot.override_reason = "operator_override"
        return snapshot
