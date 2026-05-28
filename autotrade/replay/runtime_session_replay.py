"""
Runtime session replay harness.

Replays one completed market session through the real DayManager gating path,
captures inverse-entry attempts without live side effects, and compares replay
verdicts against the recorded trade-decision ledger.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import copy
import json
import logging
import math
import shutil
import statistics
import subprocess
import tempfile
import threading
import time as wall_time
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)
from zoneinfo import ZoneInfo

import pandas as pd
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from autotrade.replay.minute_bar_archive import (
    load_session_archive,
    resolve_archive_path,
)
from autotrade.risk.inverse_etf_manager import InverseETFManager, is_inverse_etf
from autotrade.utils.alpaca_client_factory import create_data_client
from autotrade.utils.execution_accounting import ExecutionAccounting

from autotrade.core.day_manager import DayManager, TradingPhase
import autotrade.core.day_manager as day_manager_mod
import autotrade.core.autonomous_agent as autonomous_agent_mod
import autotrade.utils.intraday_data_provider as intraday_provider_mod
from autotrade.analysis.market_regime import MarketRegime, RegimeAnalysis
from config.config_loader import get_config

logger = logging.getLogger("runtime_session_replay")

PROJECT_DIR = Path(__file__).resolve().parents[2]
CT = ZoneInfo("America/Chicago")
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
DEFAULT_BENCHMARKS: Sequence[str] = ("SPY", "QQQ", "IWM")
DEFAULT_BENCHMARK_DATE_SETS: Dict[str, Sequence[str]] = {
    "strict_core": (
        "2026-03-09",
        "2026-03-10",
        "2026-03-12",
        "2026-03-17",
        "2026-03-18",
    ),
    "this_week_focus": (
        "2026-03-16",
        "2026-03-17",
        "2026-03-18",
    ),
    "expanded_march": (
        "2026-03-06",
        "2026-03-09",
        "2026-03-10",
        "2026-03-12",
        "2026-03-16",
        "2026-03-17",
        "2026-03-18",
    ),
    "strict_core_plus_mar19": (
        "2026-03-09",
        "2026-03-10",
        "2026-03-12",
        "2026-03-17",
        "2026-03-18",
        "2026-03-19",
    ),
}
NON_GATING_BENCHMARK_DATES = {"2026-03-16"}
DEFAULT_DISCOVERY_VARIANT = "discovery_v1"
DEFAULT_DISCOVERY_BENCHMARK_SET = "strict_core_plus_mar19"
REPLAY_MODE_DAYMANAGER_CORE = "daymanager_core"
REPLAY_MODE_AGENT_WORKFLOW = "agent_workflow"
REPLAY_MODE_CHOICES = (
    REPLAY_MODE_DAYMANAGER_CORE,
    REPLAY_MODE_AGENT_WORKFLOW,
)
REPLAY_PROFILE_FAST = "fast"
REPLAY_PROFILE_FULL = "full"
REPLAY_PROFILE_CHOICES = (
    REPLAY_PROFILE_FAST,
    REPLAY_PROFILE_FULL,
)


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _benchmark_completion_status_path(benchmark_output: Path) -> Path:
    return benchmark_output.with_name(f"{benchmark_output.stem}.status.json")


def _write_benchmark_status(path: Path, payload: Dict[str, Any]) -> None:
    _write_json(path, payload)


def wait_for_runtime_replay_benchmark_completion(
    status_path: Path | str,
    *,
    timeout_seconds: Optional[float] = 3600.0,
    poll_interval_seconds: float = 2.0,
) -> Dict[str, Any]:
    """Poll a benchmark status file until it reaches a terminal state."""
    path = Path(status_path)
    start = wall_time.monotonic()
    while True:
        payload: Dict[str, Any] = {}
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    loaded = json.load(handle)
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}

        status = str(payload.get("status") or "").strip().lower()
        if status == "completed":
            return payload
        if status == "failed":
            error = str(payload.get("error") or "benchmark failed")
            raise RuntimeError(f"runtime replay benchmark failed: {error}")

        if timeout_seconds is not None and (wall_time.monotonic() - start) > float(
            timeout_seconds
        ):
            raise TimeoutError(
                f"Timed out waiting for benchmark completion marker: {path}"
            )
        wall_time.sleep(max(0.1, float(poll_interval_seconds)))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = str(line or "").strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _coerce_local_timestamp(raw_value: str) -> datetime:
    text = str(raw_value or "").strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=CT)
    return parsed.astimezone(CT)


def _session_minutes(session_date: str) -> List[datetime]:
    current = datetime.combine(date.fromisoformat(session_date), time(8, 30), tzinfo=CT)
    close = datetime.combine(date.fromisoformat(session_date), time(15, 0), tzinfo=CT)
    minutes: List[datetime] = []
    while current <= close:
        minutes.append(current)
        current += timedelta(minutes=1)
    return minutes


def _sample_symbols(values: Iterable[str], *, limit: int = 5) -> List[str]:
    unique = sorted(
        {str(value).upper().strip() for value in values if str(value).strip()}
    )
    return unique[:limit]


def _pipeline_row_by_symbol(
    rows: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    indexed: Dict[str, Dict[str, Any]] = {}
    for raw in rows or []:
        if not isinstance(raw, dict):
            continue
        symbol = str(raw.get("ticker") or raw.get("symbol") or "").upper().strip()
        if not symbol:
            continue
        indexed[symbol] = dict(raw)
    return indexed


def _first_nonempty_value(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            if value.strip():
                return value
            continue
        return value
    return ""


class _ReplayScheduler:
    def __init__(self, replay: "RuntimeSessionReplay") -> None:
        self.replay = replay

    def get_market_phase(self):
        replay_time = (
            self.replay._current_replay_time
            or _session_minutes(self.replay.session_date)[0]
        )
        phase = self.replay._phase_for_time(replay_time)
        return SimpleNamespace(value=str(getattr(phase, "value", phase)))

    def get_current_time(self) -> datetime:
        return (
            self.replay._current_replay_time
            or _session_minutes(self.replay.session_date)[0]
        )


class _ReplayPlanGenerator:
    def __init__(
        self,
        replay: "RuntimeSessionReplay",
        *,
        plan_payload: Dict[str, Any],
        logs_dir: Path,
        positions: Sequence[Dict[str, Any]],
        initial_deployable_capital: float = 0.0,
    ) -> None:
        self._replay = replay
        self._plan_payload = copy.deepcopy(plan_payload or {})
        self._logs_dir = logs_dir
        self._seed_positions = [dict(row) for row in positions if isinstance(row, dict)]
        self._capital_reserve = 25_000.0
        self._deployable_capital = max(float(initial_deployable_capital or 0.0), 0.0)
        self.logger = logger
        self.alpaca_client = None
        self.market_data_client = None

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except Exception:
            return float(default)

    @staticmethod
    def _coerce_int(value: Any, default: int = 0) -> int:
        try:
            return int(float(value))
        except Exception:
            return int(default)

    def _load_latest_plan(self) -> Dict[str, Any]:
        return copy.deepcopy(self._plan_payload)

    def _get_current_price_quick(self, symbol: str) -> Optional[float]:
        return self._replay._current_price_for_symbol(symbol)

    def get_account_info(self) -> Dict[str, Any]:
        positions = self.get_current_positions()
        positions_value = sum(
            float(row.get("market_value", 0.0) or 0.0)
            for row in positions
            if isinstance(row, dict)
        )
        buying_power = self._capital_reserve + max(self._deployable_capital, 0.0)
        return {
            "cash": buying_power,
            "buying_power": buying_power,
            "equity": positions_value + buying_power,
            "day_trade_count": 0,
            "pattern_day_trader": False,
            "day_pnl": 0.0,
        }

    def get_current_positions(self) -> List[Dict[str, Any]]:
        hydrated: List[Dict[str, Any]] = []
        for row in self._seed_positions:
            symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            if not symbol:
                continue
            qty = self._coerce_int(row.get("qty"), 0)
            avg_entry = self._coerce_float(
                row.get(
                    "avg_entry", row.get("entry_price", row.get("avg_entry_price", 0.0))
                ),
                0.0,
            )
            if qty <= 0 or avg_entry <= 0:
                continue
            current_price = self._coerce_float(
                self._replay._current_price_for_symbol(symbol)
                or row.get("current_price"),
                avg_entry,
            )
            market_value = current_price * qty
            cost_basis = avg_entry * qty
            unrealized_pnl = market_value - cost_basis
            pnl_pct = (
                ((current_price - avg_entry) / avg_entry) * 100.0
                if avg_entry > 0
                else 0.0
            )
            payload = dict(row)
            payload.update(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "side": str(payload.get("side") or "long"),
                    "avg_entry": avg_entry,
                    "avg_entry_price": avg_entry,
                    "entry_price": payload.get("entry_price", avg_entry),
                    "current_price": current_price,
                    "market_value": market_value,
                    "unrealized_pnl": unrealized_pnl,
                    "unrealized_pl": unrealized_pnl,
                    "pnl_pct": pnl_pct,
                    "unrealized_plpc": pnl_pct / 100.0,
                    "cost_basis": payload.get("cost_basis", cost_basis),
                }
            )
            hydrated.append(payload)
        return hydrated

    def apply_replay_entry(
        self,
        *,
        symbol: str,
        qty: int,
        price: float,
        entry_context: str = "",
    ) -> None:
        symbol_key = str(symbol or "").upper().strip()
        qty_value = self._coerce_int(qty, 0)
        price_value = self._coerce_float(price, 0.0)
        if not symbol_key or qty_value <= 0 or price_value <= 0:
            return
        entry_notional = qty_value * price_value
        if entry_notional > 0:
            self._deployable_capital = max(
                0.0, self._deployable_capital - entry_notional
            )
        for row in self._seed_positions:
            row_symbol = (
                str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            )
            if row_symbol != symbol_key:
                continue
            existing_qty = self._coerce_int(row.get("qty"), 0)
            existing_avg = self._coerce_float(
                row.get(
                    "avg_entry",
                    row.get("entry_price", row.get("avg_entry_price", price_value)),
                ),
                price_value,
            )
            total_qty = existing_qty + qty_value
            if total_qty <= 0:
                return
            blended_avg = (
                (existing_qty * existing_avg) + (qty_value * price_value)
            ) / total_qty
            row["qty"] = total_qty
            row["avg_entry"] = blended_avg
            row["entry_price"] = blended_avg
            row["avg_entry_price"] = blended_avg
            if entry_context:
                row["last_entry_context"] = str(entry_context)
            return
        self._seed_positions.append(
            {
                "symbol": symbol_key,
                "qty": qty_value,
                "avg_entry": price_value,
                "entry_price": price_value,
                "avg_entry_price": price_value,
                "current_price": price_value,
                "market_value": qty_value * price_value,
                "unrealized_pnl": 0.0,
                "pnl_pct": 0.0,
                "cost_basis": qty_value * price_value,
                "side": "long",
                "last_entry_context": str(entry_context or ""),
            }
        )

    def apply_replay_exit(
        self,
        *,
        symbol: str,
        qty: int,
        price: float,
        exit_context: str = "",
    ) -> int:
        """Apply a sell fill to seed positions; return shares actually closed.

        Mirrors apply_replay_entry but in reverse. Without this, every DM-issued
        exit drains nothing from the plan_generator's view, so DM keeps re-issuing
        the same exit every minute.
        """
        symbol_key = str(symbol or "").upper().strip()
        remaining = self._coerce_int(qty, 0)
        price_value = self._coerce_float(price, 0.0)
        if not symbol_key or remaining <= 0:
            return 0
        closed_total = 0
        for row in list(self._seed_positions):
            if remaining <= 0:
                break
            row_symbol = (
                str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            )
            if row_symbol != symbol_key:
                continue
            existing_qty = self._coerce_int(row.get("qty"), 0)
            if existing_qty <= 0:
                self._seed_positions.remove(row)
                continue
            closed = min(existing_qty, remaining)
            remaining -= closed
            closed_total += closed
            new_qty = existing_qty - closed
            avg_entry = self._coerce_float(
                row.get(
                    "avg_entry",
                    row.get("entry_price", row.get("avg_entry_price", price_value)),
                ),
                price_value,
            )
            if new_qty <= 0:
                self._seed_positions.remove(row)
            else:
                row["qty"] = new_qty
                row["market_value"] = new_qty * (price_value or avg_entry)
                row["cost_basis"] = new_qty * avg_entry
                if exit_context:
                    row["last_exit_context"] = str(exit_context)
            if price_value > 0 and closed > 0:
                self._deployable_capital += closed * price_value
        return closed_total

    def detect_odd_lots(self, min_position_size: int = 5) -> List[Dict[str, Any]]:
        odd_lots: List[Dict[str, Any]] = []
        for pos in self.get_current_positions():
            if int(pos.get("qty", 0) or 0) >= int(min_position_size):
                continue
            odd_lots.append(
                {
                    "symbol": pos["symbol"],
                    "qty": int(pos["qty"]),
                    "reason": f"Position too small ({int(pos['qty'])} shares)",
                    "action": "CLOSE_AT_OPEN",
                    "current_value": float(pos.get("market_value", 0.0) or 0.0),
                    "pnl": float(pos.get("unrealized_pnl", 0.0) or 0.0),
                    "pnl_pct": float(pos.get("pnl_pct", 0.0) or 0.0),
                }
            )
        return odd_lots

    def detect_losers_to_cut(self, max_loss_pct: float = -5.0) -> List[Dict[str, Any]]:
        losers: List[Dict[str, Any]] = []
        threshold = float(max_loss_pct or -5.0)
        for pos in self.get_current_positions():
            pnl_pct = float(pos.get("pnl_pct", 0.0) or 0.0)
            if pnl_pct >= threshold:
                continue
            losers.append(
                {
                    "symbol": pos["symbol"],
                    "qty": int(pos["qty"]),
                    "reason": f"Loss too big ({pnl_pct:.1f}%)",
                    "action": "CLOSE_AT_OPEN",
                    "loss": float(pos.get("unrealized_pnl", 0.0) or 0.0),
                    "loss_pct": pnl_pct,
                }
            )
            return losers

    def _log_trade_decision(self, decision: Dict, order: Dict, executed: bool) -> None:
        today = self._replay.session_date.replace("-", "")
        log_file = self._logs_dir / f"trade_decisions_{today}.json"
        entry = {
            "timestamp": (
                self._replay._current_replay_time
                or _session_minutes(self._replay.session_date)[0]
            ).isoformat(),
            "symbol": order.get("symbol"),
            "planned_price": order.get("entry_price"),
            "current_price": decision.get("current_price"),
            "gap_pct": decision.get("gap_pct"),
            "planned_qty": order.get("qty"),
            "adjusted_qty": decision.get("adjusted_qty"),
            "score": order.get("score"),
            "llm_execute": decision.get("execute"),
            "llm_confidence": decision.get("confidence"),
            "llm_reasoning": decision.get("reasoning"),
            "actually_executed": bool(executed),
            "entry_source": str(
                order.get("entry_source") or decision.get("entry_source") or ""
            ),
            "plan_score_source": str(
                order.get("plan_score_source")
                or decision.get("plan_score_source")
                or ""
            ),
            "source_bucket": str(
                order.get("source_bucket") or decision.get("source_bucket") or ""
            ),
            "strategy_name": str(order.get("strategy_name") or ""),
            "setup_type": str(order.get("setup_type") or ""),
            "strategy_id": str(order.get("strategy_id") or ""),
        }
        if log_file.exists():
            existing = _load_json(log_file)
            if not isinstance(existing, list):
                existing = []
        else:
            existing = []
        existing.append(entry)
        _write_json(log_file, existing)


def _scaled_win_rate(raw_value: Any) -> float:
    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    if value <= 1.0:
        value *= 100.0
    return round(value, 2)


def _safe_float(raw_value: Any) -> Optional[float]:
    try:
        if raw_value in (None, ""):
            return None
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _load_production_trade_journal_counts(
    project_dir: Path, session_date: str
) -> Dict[str, Any]:
    """Count production trade-journal events on a given session date.

    The decisions ledger (`trade_decisions_<date>.json`) only records the
    Wave-1 entry decisions Day Manager scored explicitly — it misses trims,
    exits, and any entries that took alternative paths. The trade journal is a
    closer-to-truth view of what actually happened in production.
    """
    journal_path = project_dir / "logs" / "trade_journal.json"
    counts = {
        "trade_journal_file": "",
        "entries_journaled": 0,
        "exits_journaled": 0,
        "trims_journaled": 0,
        "other_journaled": 0,
        "total_journaled": 0,
        "unique_symbols_journaled": 0,
    }
    if not journal_path.exists():
        return counts
    payload = _load_json(journal_path) or {}
    trades = payload.get("trades") if isinstance(payload, dict) else None
    if not isinstance(trades, list):
        return counts
    counts["trade_journal_file"] = journal_path.name
    target = str(session_date)
    unique_symbols: set[str] = set()
    for row in trades:
        if not isinstance(row, dict):
            continue
        # Count the row under the action timestamp. Management events prefer
        # exit_time so overnight entries are not miscounted as same-day trims.
        event_timestamp = _trade_journal_event_timestamp(row)
        if str(event_timestamp)[:10] != target:
            continue
        trade_type = str(row.get("trade_type") or "").lower().strip()
        if trade_type in {"trim", "trim_oversized_position"}:
            counts["trims_journaled"] += 1
        elif trade_type in {"exit", "stop", "sell", "scale_out"}:
            counts["exits_journaled"] += 1
        elif trade_type in {"entry", "buy", "scale_in"}:
            counts["entries_journaled"] += 1
        elif row.get("exit_time"):
            # Closed trade with no explicit type — counts as one round-trip.
            counts["exits_journaled"] += 1
        else:
            counts["other_journaled"] += 1
        symbol = str(row.get("symbol") or "").upper().strip()
        if symbol:
            unique_symbols.add(symbol)
    counts["total_journaled"] = (
        counts["entries_journaled"]
        + counts["exits_journaled"]
        + counts["trims_journaled"]
        + counts["other_journaled"]
    )
    counts["unique_symbols_journaled"] = len(unique_symbols)
    return counts


def _trade_journal_event_timestamp(row: Dict[str, Any]) -> str:
    trade_type = str(row.get("trade_type") or "").lower().strip()
    if trade_type in {
        "trim",
        "trim_oversized_position",
        "exit",
        "stop",
        "sell",
        "scale_out",
    }:
        preferred = ("exit_time", "filled_at", "timestamp", "time", "entry_time")
    else:
        preferred = ("timestamp", "filled_at", "time", "entry_time", "exit_time")
    for key in preferred:
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return ""


def _load_actual_day_context(project_dir: Path, session_date: str) -> Dict[str, Any]:
    journal_counts = _load_production_trade_journal_counts(project_dir, session_date)
    review_path = project_dir / "data" / f"eod_review_{session_date}.json"
    if not review_path.exists():
        return {
            "eod_review_file": "",
            "total_trades": 0,
            "avg_pnl": 0.0,
            "win_rate_pct": 0.0,
            "net_pnl": 0.0,
            "net_pnl_estimated": False,
            **journal_counts,
        }

    payload = _load_json(review_path) or {}
    if not isinstance(payload, dict):
        payload = {}

    net_pnl = payload.get("net_pnl")
    estimated = False
    if net_pnl is None:
        broker_day_pnl = payload.get("broker_day_pnl")
        if broker_day_pnl is not None:
            try:
                net_pnl = round(float(broker_day_pnl or 0.0), 2)
            except (TypeError, ValueError):
                net_pnl = None
        if net_pnl is None:
            realized = payload.get("realized_day_pnl")
            unrealized = payload.get("open_position_unrealized_pnl")
            if realized is not None or unrealized is not None:
                try:
                    net_pnl = round(
                        float(realized or 0.0) + float(unrealized or 0.0),
                        2,
                    )
                except (TypeError, ValueError):
                    net_pnl = None
        if net_pnl is None and payload.get("total_pnl") is not None:
            try:
                net_pnl = round(float(payload.get("total_pnl") or 0.0), 2)
            except (TypeError, ValueError):
                net_pnl = None
        score_buckets = payload.get("score_buckets")
        if net_pnl is None and isinstance(score_buckets, dict):
            bucket_sum = 0.0
            found_bucket_pnl = False
            for bucket in score_buckets.values():
                if not isinstance(bucket, dict) or bucket.get("pnl_sum") is None:
                    continue
                try:
                    bucket_sum += float(bucket.get("pnl_sum") or 0.0)
                    found_bucket_pnl = True
                except (TypeError, ValueError):
                    continue
            if found_bucket_pnl:
                net_pnl = round(bucket_sum, 2)
        if (
            net_pnl is None
            and payload.get("avg_pnl") is not None
            and payload.get("total_trades") is not None
        ):
            try:
                net_pnl = round(
                    float(payload.get("avg_pnl") or 0.0)
                    * int(payload.get("total_trades") or 0),
                    2,
                )
                estimated = True
            except (TypeError, ValueError):
                net_pnl = 0.0
                estimated = False

    return {
        "eod_review_file": review_path.name,
        "total_trades": int(payload.get("total_trades", 0) or 0),
        "avg_pnl": round(float(payload.get("avg_pnl", 0.0) or 0.0), 2),
        "win_rate_pct": _scaled_win_rate(payload.get("win_rate", 0.0)),
        "net_pnl": round(float(net_pnl or 0.0), 2),
        "net_pnl_estimated": estimated,
        "realized_day_pnl": round(
            float(payload.get("realized_day_pnl", 0.0) or 0.0), 2
        ),
        "open_position_unrealized_pnl": round(
            float(payload.get("open_position_unrealized_pnl", 0.0) or 0.0), 2
        ),
        "broker_day_pnl": round(float(payload.get("broker_day_pnl", 0.0) or 0.0), 2),
        "best_trade": str(payload.get("best_trade") or ""),
        "worst_trade": str(payload.get("worst_trade") or ""),
        **journal_counts,
    }


def _git_output(project_dir: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(project_dir), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        return ""
    if completed.returncode != 0:
        return ""
    return (completed.stdout or "").strip()


def _file_mtime_iso(path: Optional[Path]) -> str:
    if not path or not path.exists():
        return ""
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=CT).isoformat()
    except Exception:
        return ""


def _summarize_divergences(divergences: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_type = Counter()
    authority_reasons = Counter()
    regime_reasons = Counter()
    for row in divergences:
        if not isinstance(row, dict):
            continue
        by_type[str(row.get("type") or "unknown")] += 1
        authority = str(row.get("replay_authority_reason") or "").strip()
        regime = str(row.get("replay_regime_reason") or "").strip()
        if authority:
            authority_reasons[authority] += 1
        if regime:
            regime_reasons[regime] += 1

    examples: List[Dict[str, Any]] = []
    for row in list(divergences)[:5]:
        if not isinstance(row, dict):
            continue
        examples.append(
            {
                "timestamp": str(row.get("timestamp") or ""),
                "symbol": str(row.get("symbol") or ""),
                "type": str(row.get("type") or ""),
                "replay_regime_reason": str(row.get("replay_regime_reason") or ""),
                "replay_authority_reason": str(
                    row.get("replay_authority_reason") or ""
                ),
            }
        )

    top_reasons: List[str] = []
    for reason, count in authority_reasons.most_common(3):
        top_reasons.append(f"authority:{reason} ({count})")
    for reason, count in regime_reasons.most_common(3):
        top_reasons.append(f"regime:{reason} ({count})")

    return {
        "by_type": dict(by_type),
        "authority_reasons": dict(authority_reasons),
        "regime_reasons": dict(regime_reasons),
        "top_reasons": top_reasons[:4],
        "examples": examples,
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _entry_replay_order_count(summary: Mapping[str, Any]) -> int:
    buckets = summary.get("replay_orders_by_bucket", {}) or {}
    if not isinstance(buckets, Mapping):
        buckets = {}
    if "replay_entries_executed" in summary:
        return _safe_int(summary.get("replay_entries_executed"), 0)
    return _safe_int(buckets.get("equity_entry"), 0) + _safe_int(
        buckets.get("inverse_etf"), 0
    )


def _build_inverse_absence_diagnostic(
    authority_timeline: Sequence[Mapping[str, Any]],
    resolved_regime: Mapping[str, Any],
) -> Dict[str, Any]:
    """Explain why a bearish persisted regime did not produce inverse-fast entries."""

    first_inverse_fast_at = ""
    recovered_snapshots = 0
    bearish_snapshots = 0
    for row in authority_timeline:
        if str(row.get("state") or "") == "inverse_fast" and not first_inverse_fast_at:
            first_inverse_fast_at = str(row.get("timestamp") or "")
        snapshot = row.get("snapshot")
        if not isinstance(snapshot, Mapping):
            continue
        if bool(snapshot.get("recovery_confirmed", False)):
            recovered_snapshots += 1
        red_ratio = _safe_float(snapshot.get("red_ratio")) or 0.0
        avg_pct_change = _safe_float(snapshot.get("avg_pct_change")) or 0.0
        if red_ratio >= 0.67 and avg_pct_change <= -0.25:
            bearish_snapshots += 1

    persisted_regime = (
        str(
            resolved_regime.get("regime")
            or resolved_regime.get("resolved_regime")
            or ""
        )
        .upper()
        .replace(" ", "_")
    )
    persisted_bearish = persisted_regime in {
        "SELLOFF",
        "SELL_OFF",
        "RISK_OFF",
        "CRASH",
        "CAPITULATION",
        "CRISIS",
    }
    if first_inverse_fast_at:
        reason = "inverse_fast_reached"
    elif persisted_bearish and recovered_snapshots > 0 and bearish_snapshots == 0:
        reason = "recovered_benchmarks_no_inverse_fast"
    elif persisted_bearish:
        reason = "persisted_bearish_without_live_inverse_trigger"
    else:
        reason = "no_bearish_inverse_context"
    return {
        "reason": reason,
        "persisted_regime": persisted_regime,
        "first_inverse_fast_at": first_inverse_fast_at,
        "recovered_snapshots": recovered_snapshots,
        "bearish_snapshots": bearish_snapshots,
    }


def _cycle_block_reasons(cycle: Mapping[str, Any]) -> List[str]:
    reasons: List[str] = []
    block_reason = str(cycle.get("block_reason") or "").strip()
    if block_reason:
        reasons.append(block_reason)

    blocked_by_reason = cycle.get("blocked_by_reason")
    if isinstance(blocked_by_reason, Mapping):
        reasons.extend(
            str(reason) for reason in blocked_by_reason if str(reason).strip()
        )

    entry_audit = cycle.get("entry_audit")
    if isinstance(entry_audit, Mapping):
        audit_block_reason = str(entry_audit.get("block_reason") or "").strip()
        if audit_block_reason:
            reasons.append(audit_block_reason)
        audit_blocked = entry_audit.get("blocked_by_reason")
        if isinstance(audit_blocked, Mapping):
            reasons.extend(
                str(reason) for reason in audit_blocked if str(reason).strip()
            )
        for row in entry_audit.get("candidates", []) or []:
            if isinstance(row, Mapping):
                skip_reason = str(row.get("skip_reason") or "").strip()
                if skip_reason:
                    reasons.append(skip_reason)

    for row in cycle.get("blocked_symbols", []) or []:
        if not isinstance(row, Mapping):
            continue
        reason = str(
            row.get("authority_reason")
            or row.get("regime_reason")
            or row.get("reason")
            or ""
        ).strip()
        if reason:
            reasons.append(reason)

    return sorted(set(reasons))


def build_replay_entry_diagnostic_gate(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Classify replay entry behavior so no-entry incidents cannot pass silently."""
    summary = dict(report.get("summary", {}) or {})
    entry_orders = _entry_replay_order_count(summary)
    date_value = str(summary.get("date") or "")

    cycles: List[Dict[str, Any]] = []
    for row in report.get("daymanager_cycle_stats", []) or []:
        if isinstance(row, Mapping):
            cycles.append({"source": "daymanager_cycle_stats", **dict(row)})
    workflow = report.get("workflow_replay", {}) or {}
    if isinstance(workflow, Mapping):
        for row in workflow.get("action_log", []) or []:
            if isinstance(row, Mapping):
                cycles.append({"source": "workflow_action_log", **dict(row)})

    actionable_cycles = []
    silent_cycles = []
    explained_cycles = []
    for cycle in cycles:
        candidate_count = _safe_int(cycle.get("candidate_count"), 0)
        open_slots = _safe_int(cycle.get("open_slots"), 0)
        orders_submitted = _safe_int(
            cycle.get(
                "orders_submitted",
                cycle.get("entries_submitted", cycle.get("submitted", 0)),
            ),
            0,
        )
        if candidate_count <= 0 or open_slots <= 0 or orders_submitted > 0:
            continue
        reasons = _cycle_block_reasons(cycle)
        diagnostic_row = {
            "source": str(cycle.get("source") or ""),
            "timestamp": str(cycle.get("timestamp") or ""),
            "candidate_count": candidate_count,
            "open_slots": open_slots,
            "orders_submitted": orders_submitted,
            "block_reasons": reasons[:10],
        }
        actionable_cycles.append(diagnostic_row)
        if reasons:
            explained_cycles.append(diagnostic_row)
        else:
            silent_cycles.append(diagnostic_row)

    divergences_count = _safe_int(summary.get("divergences_count"), 0)
    zero_divergence_zero_entry = divergences_count == 0 and entry_orders == 0
    if entry_orders > 0:
        diagnostic_status = "pass"
        diagnostic_reason = "entry_orders_replayed"
        diagnostic_passed = True
    elif silent_cycles:
        diagnostic_status = "fail"
        diagnostic_reason = "silent_no_entry_replay"
        diagnostic_passed = False
    else:
        diagnostic_status = "no_entry_explained"
        diagnostic_reason = "zero_entries_with_concrete_block_reasons"
        diagnostic_passed = True

    status = diagnostic_status
    reason = diagnostic_reason
    passed = diagnostic_passed

    incident_acceptance = ""
    if date_value == "2026-05-01":
        if entry_orders > 0:
            incident_acceptance = "pass_entry_replayed"
        elif zero_divergence_zero_entry:
            incident_acceptance = "fail_reproduced_production_failure"
            status = "fail"
            if silent_cycles:
                reason = "reproduced_production_failure_silent_no_entry"
            else:
                reason = "reproduced_production_failure_no_entry_explained"
            passed = False
        else:
            incident_acceptance = "no_entry_explained"

    return {
        "status": status,
        "passed": passed,
        "reason": reason,
        "diagnostic_status": diagnostic_status,
        "diagnostic_passed": diagnostic_passed,
        "diagnostic_reason": diagnostic_reason,
        "date": date_value,
        "entry_orders": entry_orders,
        "divergences_count": divergences_count,
        "zero_divergence_zero_entry_reproduces_failure": bool(
            date_value == "2026-05-01" and zero_divergence_zero_entry
        ),
        "incident_acceptance": incident_acceptance,
        "actionable_no_submit_cycles": len(actionable_cycles),
        "silent_no_submit_cycles": len(silent_cycles),
        "explained_no_submit_cycles": len(explained_cycles),
        "silent_examples": silent_cycles[:5],
        "explained_examples": explained_cycles[:5],
    }


class RuntimeSessionReplay:
    """Replay one session through the DayManager runtime decision path."""

    def __init__(
        self,
        *,
        session_date: str,
        mode: str = REPLAY_MODE_DAYMANAGER_CORE,
        profile: str = REPLAY_PROFILE_FULL,
        project_dir: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None,
        strategy_eval: Optional[str] = None,
        strategy_variant: str = DEFAULT_DISCOVERY_VARIANT,
        force_modern_crash_protection: bool = False,
        data_client: Any = None,
        benchmark_symbols: Optional[Sequence[str]] = None,
        market_bars: Optional[Dict[str, pd.DataFrame]] = None,
        previous_closes: Optional[Dict[str, float]] = None,
        inverse_screen_provider: Optional[
            Callable[[datetime, str, Optional[List[str]]], List[Dict[str, Any]]]
        ] = None,
    ) -> None:
        self.session_date = session_date
        normalized_mode = str(mode or REPLAY_MODE_DAYMANAGER_CORE).strip().lower()
        if normalized_mode not in REPLAY_MODE_CHOICES:
            raise ValueError(f"Unsupported replay mode: {mode}")
        self.mode = normalized_mode
        normalized_profile = str(profile or REPLAY_PROFILE_FULL).strip().lower()
        if normalized_profile not in REPLAY_PROFILE_CHOICES:
            raise ValueError(f"Unsupported replay profile: {profile}")
        self.profile = normalized_profile
        self.skip_position_advisor = self.profile == REPLAY_PROFILE_FAST
        self.project_dir = Path(project_dir) if project_dir else PROJECT_DIR
        self.logs_dir = self.project_dir / "logs"
        self.plans_dir = self.project_dir / "plans"
        output_suffix = "_modern_crash" if force_modern_crash_protection else ""
        default_output = (
            self.logs_dir
            / f"runtime_replay_{session_date.replace('-', '')}{output_suffix}.json"
        )
        self.output_path = Path(output_path) if output_path else default_output
        self.strategy_eval = str(strategy_eval or "").strip().lower() or None
        self.strategy_variant = str(
            strategy_variant or DEFAULT_DISCOVERY_VARIANT
        ).strip()
        self.force_modern_crash_protection = bool(force_modern_crash_protection)
        self.data_client = data_client
        self.benchmark_symbols = tuple(benchmark_symbols or DEFAULT_BENCHMARKS)
        self.market_bars = {
            str(symbol).upper(): self._normalize_bars_frame(df)
            for symbol, df in (market_bars or {}).items()
        }
        self.previous_closes = {
            str(symbol).upper(): float(value)
            for symbol, value in (previous_closes or {}).items()
        }
        self.inverse_screen_provider = inverse_screen_provider
        self.inverse_etf_manager = InverseETFManager()
        self._current_replay_time: Optional[datetime] = None
        # Inverse-ETF positions tracked here have their own evaluation loop
        # (_evaluate_inverse_positions). Equity positions live in the
        # plan_generator's _seed_positions only. Do NOT cross the streams.
        self._positions: List[SimpleNamespace] = []
        # Backward-compat alias; new code should read _replay_orders.
        self._replay_orders: List[Dict[str, Any]] = []
        self._inverse_orders = self._replay_orders
        # Orders DM tried to issue against symbols with no minute-bar data.
        # Tracked separately so they don't pollute the equity_entry bucket.
        self._replay_data_gap_skips: List[Dict[str, Any]] = []
        self._inverse_trade_results: List[Dict[str, Any]] = []
        self._workflow_action_log: List[Dict[str, Any]] = []
        self._workflow_selected_symbols: set[str] = set()
        self._workflow_add_selected_symbols: set[str] = set()
        self._workflow_phase_cycles: Dict[str, int] = {
            "market_open": 0,
            "market_hours": 0,
        }
        self._workflow_bullish_actions: List[Dict[str, Any]] = []
        self._workflow_last_strength_trace_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._workflow_deployment_samples: List[Dict[str, Any]] = []
        self._workflow_dm: Optional[DayManager] = None
        self._workflow_agent: Any = None
        self._workflow_runtime_signals: List[Dict[str, Any]] = []
        self._workflow_plan_selected_symbols: set[str] = set()
        self._workflow_plan_promoted_symbols: set[str] = set()
        self._workflow_plan_trimmed_symbols: set[str] = set()
        self._inverse_universe_rows: Optional[List[Dict[str, Any]]] = None
        self._replay_notes: List[str] = []
        self._heartbeat_interval_seconds = 300.0
        self._heartbeat_started_at = wall_time.monotonic()
        self._heartbeat_last_logged_at = self._heartbeat_started_at
        archive_cfg = getattr(get_config().data, "replay_minute_archive", None)
        self._archive_cfg = archive_cfg
        self._archive_db_path = (
            resolve_archive_path(
                getattr(archive_cfg, "duckdb_path", ""),
                project_dir=self.project_dir,
            )
            if archive_cfg is not None
            else None
        )
        self._archive_manifest_by_symbol: Dict[str, Dict[str, Any]] = {}
        self._archive_missing_symbols: set[str] = set()
        self._archive_diagnostics: Dict[str, Any] = {
            "enabled": bool(getattr(archive_cfg, "enabled", False)),
            "prefer_local_archive": bool(
                getattr(archive_cfg, "prefer_local_archive", False)
            ),
            "allow_live_fallback_if_incomplete": bool(
                getattr(archive_cfg, "allow_live_fallback_if_incomplete", True)
            ),
            "db_path": str(self._archive_db_path or ""),
            "archive_found": False,
            "loaded_symbols": [],
            "missing_symbols": [],
            "live_fetch_symbols": [],
            "premarket_missing_symbols": [],
        }

    def run(self, *, persist: bool = True) -> Dict[str, Any]:
        if self.mode == REPLAY_MODE_AGENT_WORKFLOW:
            return self._run_agent_workflow(persist=persist)
        return self._run_daymanager_core(persist=persist)

    def _maybe_log_progress(
        self,
        *,
        replay_minute: datetime,
        step_index: int,
        total_steps: int,
        phase: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = wall_time.monotonic()
        if step_index == 1:
            self._heartbeat_started_at = now
            self._heartbeat_last_logged_at = now
            logger.info(
                "[REPLAY START] profile=%s mode=%s date=%s phase=%s total_steps=%d start_minute=%s advisor=%s",
                self.profile,
                self.mode,
                self.session_date,
                phase,
                total_steps,
                replay_minute.strftime("%H:%M"),
                "skipped" if self.skip_position_advisor else "enabled",
            )
            return

        if (now - self._heartbeat_last_logged_at) < self._heartbeat_interval_seconds:
            return

        elapsed = max(0.0, now - self._heartbeat_started_at)
        pct_complete = (
            (float(step_index) / float(total_steps)) * 100.0 if total_steps > 0 else 0.0
        )
        extras = dict(extra or {})
        extra_text = (
            " | " + ", ".join(f"{key}={value}" for key, value in extras.items())
            if extras
            else ""
        )
        logger.info(
            "[REPLAY HEARTBEAT] profile=%s mode=%s date=%s progress=%d/%d (%.1f%%) replay_minute=%s phase=%s elapsed=%.1fm%s",
            self.profile,
            self.mode,
            self.session_date,
            step_index,
            total_steps,
            pct_complete,
            replay_minute.strftime("%H:%M"),
            phase,
            elapsed / 60.0,
            extra_text,
        )
        self._heartbeat_last_logged_at = now

    def _run_daymanager_core(self, *, persist: bool = True) -> Dict[str, Any]:
        artifacts = self._resolve_artifacts()
        decisions = self._load_decisions(artifacts)
        workflow_journal = self._load_workflow_journal(artifacts)
        trade_journal = self._load_trade_journal(artifacts)
        raw_signals = self._load_signals(artifacts)
        enriched_signals = self._augment_signals_with_plan_metadata(
            raw_signals,
            artifacts=artifacts,
        )
        signals = self._augment_signals_with_recorded_metadata(
            enriched_signals,
            decisions=decisions,
            trade_journal=trade_journal,
        )
        resolved_regime = dict(artifacts["plan_payload"].get("resolved_regime") or {})
        resolved_regime.setdefault(
            "plan_source", artifacts["plan_path"].name if artifacts["plan_path"] else ""
        )
        self._hydrate_archive_market_data(
            signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
        )
        self._ensure_market_data()

        dm = self._build_replay_day_manager(
            signals=signals,
            resolved_regime=resolved_regime,
            artifacts=artifacts,
        )
        decisions_by_minute = defaultdict(list)
        for decision in decisions:
            decision_minute = _coerce_local_timestamp(decision["timestamp"]).replace(
                second=0,
                microsecond=0,
            )
            decisions_by_minute[decision_minute].append(decision)

        authority_timeline: List[Dict[str, Any]] = []
        long_checks: List[Dict[str, Any]] = []
        divergences: List[Dict[str, Any]] = []
        first_inverse_fast_at = ""
        daymanager_cycle_runs = 0
        daymanager_cycle_stats: List[Dict[str, Any]] = []

        session_minutes = _session_minutes(self.session_date)
        total_steps = len(session_minutes)
        for step_index, replay_minute in enumerate(session_minutes, start=1):
            self._current_replay_time = replay_minute
            self._maybe_log_progress(
                replay_minute=replay_minute,
                step_index=step_index,
                total_steps=total_steps,
                phase="daymanager_core",
                extra={
                    "inverse_orders": len(self._inverse_orders),
                    "positions": len(self._positions),
                },
            )
            with self._patched_runtime_clock(), self._patched_inverse_screen():
                marked_positions = self._mark_to_market_positions()
                authority = dm._refresh_entry_authority_state(marked_positions)
                if (
                    not first_inverse_fast_at
                    and authority.get("state") == "inverse_fast"
                ):
                    first_inverse_fast_at = replay_minute.isoformat()
                dm._run_defensive_screen(marked_positions)
                self._evaluate_inverse_positions(replay_minute)
                authority_timeline.append(
                    {
                        "timestamp": replay_minute.isoformat(),
                        "state": authority.get("state", "open"),
                        "reason": authority.get("reason", ""),
                        "inverse_fast_entries_taken": int(
                            dm._get_entry_authority_state().get(
                                "inverse_fast_entries_taken", 0
                            )
                            or 0
                        ),
                        "snapshot": dict(authority.get("snapshot", {}) or {}),
                    }
                )

                if self._should_run_replay_daymanager_cycle(replay_minute):
                    cycle_stats = self._run_replay_daymanager_cycle(
                        dm=dm,
                        replay_minute=replay_minute,
                    )
                    daymanager_cycle_runs += 1
                    if isinstance(cycle_stats, dict):
                        daymanager_cycle_stats.append(
                            {
                                "timestamp": replay_minute.isoformat(),
                                "phase": str(
                                    getattr(
                                        self._phase_for_time(replay_minute),
                                        "value",
                                        self._phase_for_time(replay_minute),
                                    )
                                ),
                                "candidate_count": int(
                                    cycle_stats.get("candidate_count", 0) or 0
                                ),
                                "entries": int(cycle_stats.get("entries", 0) or 0),
                                "orders_submitted": int(
                                    cycle_stats.get(
                                        "orders_submitted",
                                        cycle_stats.get("entries_submitted", 0),
                                    )
                                    or 0
                                ),
                                "open_slots": int(
                                    cycle_stats.get("open_slots", 0) or 0
                                ),
                                "max_new_entries": int(
                                    cycle_stats.get("max_new_entries", 0) or 0
                                ),
                                "block_reason": str(
                                    cycle_stats.get("block_reason", "") or ""
                                ),
                                "blocked_by_reason": dict(
                                    cycle_stats.get("blocked_by_reason", {}) or {}
                                ),
                                "entry_audit": dict(
                                    cycle_stats.get("entry_audit", {}) or {}
                                ),
                            }
                        )

                for decision in decisions_by_minute.get(replay_minute, []):
                    symbol = str(decision.get("symbol") or "").upper()
                    signal_data = self._build_replay_signal_data(
                        dm=dm,
                        symbol=symbol,
                        decision=decision,
                        trade_journal=trade_journal,
                    )
                    regime_blocked, regime_reason = dm._entries_blocked_by_regime(
                        symbol
                    )
                    authority_contract = dm._resolve_entry_authority(signal_data)
                    replay_long_allowed = bool(
                        (not regime_blocked)
                        and authority_contract.get("eligible", False)
                    )
                    check = {
                        "timestamp": decision.get("timestamp"),
                        "symbol": symbol,
                        "recorded_executed": bool(decision.get("actually_executed")),
                        "recorded_score": decision.get("score"),
                        "entry_source": str(signal_data.get("entry_source") or ""),
                        "origin_entry_source": str(
                            signal_data.get("origin_entry_source") or ""
                        ),
                        "runtime_entry_context": str(
                            signal_data.get("runtime_entry_context") or ""
                        ),
                        "override_reason": str(
                            signal_data.get("override_reason") or ""
                        ),
                        "plan_score_source": str(
                            signal_data.get("plan_score_source") or ""
                        ),
                        "source_bucket": str(signal_data.get("source_bucket") or ""),
                        "replay_regime_blocked": regime_blocked,
                        "replay_regime_reason": regime_reason,
                        "replay_authority_eligible": bool(
                            authority_contract.get("eligible", False)
                        ),
                        "replay_authority_reason": str(
                            authority_contract.get("reason", "") or ""
                        ),
                        "replay_long_allowed": replay_long_allowed,
                        "entry_authority_state": authority_contract.get(
                            "execution_mode", {}
                        ).get(
                            "entry_authority_state",
                            "open",
                        ),
                        "persisted_resolved_regime": str(
                            (
                                authority_contract.get("execution_mode", {})
                                .get("resolved_regime", {})
                                .get("regime")
                            )
                            or ""
                        ),
                        "persisted_allow_new_longs": bool(
                            (
                                authority_contract.get("execution_mode", {})
                                .get("resolved_regime", {})
                                .get("allow_new_longs", True)
                            )
                        ),
                        "live_router_regime": str(
                            (getattr(dm, "regime_router_context", {}) or {}).get(
                                "regime", ""
                            )
                            or ""
                        ),
                        "youtube_resolved_regime": str(
                            (
                                (getattr(dm, "youtube_context", {}) or {})
                                .get("resolved_regime", {})
                                .get("regime")
                            )
                            or ""
                        ),
                        "youtube_allow_new_longs": bool(
                            (
                                (getattr(dm, "youtube_context", {}) or {})
                                .get("resolved_regime", {})
                                .get("allow_new_longs", True)
                            )
                        ),
                        "effective_market_regime": str(
                            dm._effective_market_regime() or ""
                        ),
                        "metadata_backfilled": bool(
                            signal_data.get("_replay_metadata_backfilled", False)
                        ),
                        "metadata_inferred": bool(
                            signal_data.get("_replay_metadata_inferred", False)
                        ),
                    }
                    long_checks.append(check)
                    if check["recorded_executed"] and not replay_long_allowed:
                        divergences.append(
                            {
                                "type": "recorded_long_executed_while_replay_blocked",
                                "timestamp": decision.get("timestamp"),
                                "symbol": symbol,
                                "replay_regime_reason": regime_reason,
                                "replay_authority_reason": check[
                                    "replay_authority_reason"
                                ],
                            }
                        )

        self._finalize_inverse_positions()
        inverse_performance = self._summarize_inverse_trade_results()
        order_breakdown = self._summarize_replay_orders()
        trade_journal_management_events = self._trade_journal_management_events(
            trade_journal
        )
        management_replay = self._replay_position_management(
            dm=dm,
            journal_entries=list(workflow_journal) + trade_journal_management_events,
        )
        actual_day_context = _load_actual_day_context(
            self.project_dir, self.session_date
        )
        management_coverage = self._build_management_coverage_diagnostics(
            management_replay=management_replay,
            actual_day_context=actual_day_context,
            workflow_journal=workflow_journal,
            trade_journal_management_events=trade_journal_management_events,
        )
        if management_coverage["status"] in {"incomplete", "degraded"}:
            divergences.append(
                {
                    "type": "management_replay_coverage_gap",
                    "timestamp": "",
                    "symbol": "",
                    "replay_regime_reason": str(
                        management_coverage.get("problem") or ""
                    ),
                    "replay_authority_reason": str(
                        management_coverage.get("status") or ""
                    ),
                }
            )
        if first_inverse_fast_at and not self._inverse_orders:
            divergences.append(
                {
                    "type": "inverse_fast_without_inverse_entry",
                    "timestamp": first_inverse_fast_at,
                    "symbol": "",
                    "replay_regime_reason": "inverse_fast_active",
                    "replay_authority_reason": "no_inverse_orders_captured",
                }
            )
        inverse_absence = _build_inverse_absence_diagnostic(
            authority_timeline,
            resolved_regime,
        )
        divergence_summary = _summarize_divergences(divergences)
        handoff_diagnostics = self._build_handoff_diagnostics(
            artifacts=artifacts,
            raw_signals=raw_signals,
            replay_signals=signals,
            decisions=decisions,
        )
        counterfactual_long_eval = self._evaluate_counterfactual_longs(
            dm=dm,
            signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            actual_day_net_pnl=float(actual_day_context.get("net_pnl", 0.0) or 0.0),
            capacity_snapshot=self._extract_counterfactual_capacity_snapshot(
                artifacts.get("plan_payload") or {},
                artifacts.get("pm_plan_payload") or {},
            ),
        )
        discovery_strategy_eval: Optional[Dict[str, Any]] = None
        if self.strategy_eval == "discovery":
            discovery_strategy_eval = self._evaluate_discovery_strategy(
                artifacts=artifacts,
                raw_signals=raw_signals,
                decisions=decisions,
            )
        watchlist_causality_snapshot: Dict[str, Any] = {}
        watchlist_causality_path = ""
        try:
            watchlist_causality_snapshot = dm._persist_watchlist_causality_snapshot(
                phase=TradingPhase.CORE_TRADING
            )
            watchlist_causality_path = str(dm._watchlist_causality_path())
        except Exception as exc:
            self._replay_notes.append(
                f"Watchlist causality snapshot persistence failed: {exc}"
            )
        signal_pipeline_audit = self._build_signal_pipeline_audit(
            artifacts=artifacts,
            raw_signals=raw_signals,
            replay_signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            watchlist_causality_snapshot=watchlist_causality_snapshot,
        )
        signal_pipeline_audit_path = str(self._signal_pipeline_audit_path())
        if persist:
            _write_json(Path(signal_pipeline_audit_path), signal_pipeline_audit)
        stop_loss_diagnostics = self._build_stop_loss_diagnostics(management_replay)
        intraday_bar_coverage = self._build_intraday_bar_coverage(
            signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            workflow_actions=self._workflow_action_log,
        )
        replay_order_accounting = self._build_replay_order_accounting(
            order_breakdown=order_breakdown,
            counterfactual_candidates=int(
                counterfactual_long_eval.get("eligible_candidates_total", 0) or 0
            ),
        )
        replay_environment = self._build_replay_environment_snapshot(artifacts)

        report = {
            "summary": {
                "date": self.session_date,
                "replay_profile": self.profile,
                "position_advisor_mode": (
                    "skipped" if self.skip_position_advisor else "enabled"
                ),
                "daymanager_cycles_run": daymanager_cycle_runs,
                "force_modern_crash_protection": self.force_modern_crash_protection,
                "signals_total": len(signals),
                "recorded_decisions_total": len(decisions),
                "recorded_longs_executed": sum(
                    1 for row in decisions if row.get("actually_executed")
                ),
                "replay_longs_allowed": sum(
                    1 for row in long_checks if row.get("replay_long_allowed")
                ),
                # True inverse-ETF order count only. See replay_orders_by_bucket
                # for the full per-bucket breakdown.
                "inverse_orders_captured": int(
                    order_breakdown["by_bucket"].get("inverse_etf", 0)
                ),
                "replay_orders_total": int(order_breakdown.get("total", 0) or 0),
                "replay_orders_by_bucket": dict(order_breakdown.get("by_bucket", {})),
                "replay_orders_unique_symbols_by_bucket": dict(
                    order_breakdown.get("unique_symbol_count_by_bucket", {})
                ),
                "replay_orders_top_repeated_exits": list(
                    order_breakdown.get("top_repeated_exit_symbols", [])
                ),
                "replay_data_gap_skipped": int(
                    order_breakdown.get("data_gap_skipped_total", 0) or 0
                ),
                "replay_data_gap_skipped_unique_symbols": int(
                    order_breakdown.get("data_gap_skipped_unique_symbols", 0) or 0
                ),
                "replay_data_gap_skipped_examples": list(
                    order_breakdown.get("data_gap_skipped_unique_symbols_examples", [])
                ),
                "data_quality": self._build_data_quality_status(),
                "inverse_trades_closed": int(
                    inverse_performance.get("closed_trades", 0) or 0
                ),
                "inverse_net_pnl": float(
                    inverse_performance.get("net_pnl", 0.0) or 0.0
                ),
                "inverse_avg_return_pct": float(
                    inverse_performance.get("avg_return_pct", 0.0) or 0.0
                ),
                "inverse_win_rate": float(
                    inverse_performance.get("win_rate", 0.0) or 0.0
                ),
                "management_replay_events": int(
                    management_replay.get("summary", {}).get("events_replayed", 0) or 0
                ),
                "management_replay_differences": int(
                    management_replay.get("summary", {}).get("action_differences", 0)
                    or 0
                ),
                "management_replay_exit_signals": int(
                    management_replay.get("summary", {}).get("exit_like_signals", 0)
                    or 0
                ),
                "management_hard_stop_exit_signals": int(
                    management_replay.get("summary", {}).get(
                        "hard_stop_exit_signals", 0
                    )
                    or 0
                ),
                "divergences_count": len(divergences),
                "first_inverse_fast_at": first_inverse_fast_at,
                "inverse_absence_reason": str(inverse_absence.get("reason") or ""),
                "inverse_absence_persisted_regime": str(
                    inverse_absence.get("persisted_regime") or ""
                ),
                "inverse_absence_recovered_snapshots": int(
                    inverse_absence.get("recovered_snapshots", 0) or 0
                ),
                "inverse_absence_bearish_snapshots": int(
                    inverse_absence.get("bearish_snapshots", 0) or 0
                ),
                "metadata_backfilled_checks": sum(
                    1 for row in long_checks if row.get("metadata_backfilled")
                ),
                "metadata_inferred_checks": sum(
                    1 for row in long_checks if row.get("metadata_inferred")
                ),
                "decision_artifact_missing": bool(
                    artifacts.get("decision_path") is None
                ),
                "actual_net_pnl": float(actual_day_context.get("net_pnl", 0.0) or 0.0),
                "actual_total_trades": int(
                    actual_day_context.get("total_trades", 0) or 0
                ),
                "handoff_status": str(handoff_diagnostics.get("status") or "healthy"),
                "handoff_problem": str(handoff_diagnostics.get("problem") or ""),
                "handoff_recoverable_candidates": int(
                    handoff_diagnostics.get("recoverable_candidates_count", 0) or 0
                ),
                "signal_alignment_status": str(
                    handoff_diagnostics.get("signal_alignment_status") or "aligned"
                ),
                "signal_alignment_problem": str(
                    handoff_diagnostics.get("signal_alignment_problem") or ""
                ),
                "discovery_strategy_eval_enabled": bool(
                    discovery_strategy_eval is not None
                ),
                "market_data_coverage_mode": self._market_data_coverage_mode(),
                "archive_found": bool(self._archive_diagnostics.get("archive_found")),
                "archive_symbols_loaded": len(
                    self._archive_diagnostics.get("loaded_symbols", [])
                ),
                "archive_symbols_missing": len(
                    self._archive_diagnostics.get("missing_symbols", [])
                ),
                "archive_live_fetch_symbols": len(
                    self._archive_diagnostics.get("live_fetch_symbols", [])
                ),
                "archive_premarket_missing_symbols": len(
                    self._archive_diagnostics.get("premarket_missing_symbols", [])
                ),
                "counterfactual_long_candidates": int(
                    counterfactual_long_eval.get("eligible_candidates_total", 0) or 0
                ),
                "counterfactual_long_selected": int(
                    counterfactual_long_eval.get("selected_candidates_total", 0) or 0
                ),
                "counterfactual_long_net_pnl_selected": float(
                    counterfactual_long_eval.get("net_synthetic_pnl_selected", 0.0)
                    or 0.0
                ),
                "counterfactual_estimated_day_net_pnl_selected": float(
                    counterfactual_long_eval.get("estimated_day_net_pnl_selected", 0.0)
                    or 0.0
                ),
                "management_replay_coverage_status": str(
                    management_coverage.get("status") or ""
                ),
                "unresolved_stop_loss_warnings": int(
                    stop_loss_diagnostics.get("unresolved_stop_loss_warnings", 0) or 0
                ),
                "intraday_zero_bar_symbols": int(
                    intraday_bar_coverage.get("symbols_with_zero_bars", 0) or 0
                ),
            },
            "artifacts": {
                "signals_file": artifacts["signals_path"].name
                if artifacts["signals_path"]
                else "",
                "signals_file_today": (
                    artifacts["signals_today_path"].name
                    if artifacts["signals_today_path"]
                    else ""
                ),
                "signals_file_yesterday": (
                    artifacts["signals_yesterday_path"].name
                    if artifacts["signals_yesterday_path"]
                    else ""
                ),
                "decision_file": artifacts["decision_path"].name
                if artifacts["decision_path"]
                else "",
                "plan_file": artifacts["plan_path"].name
                if artifacts["plan_path"]
                else "",
                "pm_plan_file": artifacts["pm_plan_path"].name
                if artifacts["pm_plan_path"]
                else "",
                "morning_plan_file": (
                    artifacts["morning_plan_path"].name
                    if artifacts["morning_plan_path"]
                    else ""
                ),
                "adjusted_plan_file": (
                    artifacts["adjusted_plan_path"].name
                    if artifacts["adjusted_plan_path"]
                    else ""
                ),
                "workflow_journal_file": (
                    artifacts["workflow_journal_path"].name
                    if artifacts["workflow_journal_path"]
                    else ""
                ),
                "trade_journal_file": (
                    artifacts["trade_journal_path"].name
                    if artifacts["trade_journal_path"]
                    else ""
                ),
                "eod_review_file": str(actual_day_context.get("eod_review_file") or ""),
                "watchlist_causality_file": watchlist_causality_path,
                "signal_pipeline_audit_file": signal_pipeline_audit_path,
            },
            "resolved_regime": resolved_regime,
            "actual_day_context": actual_day_context,
            "replay_environment": replay_environment,
            "daymanager_cycle_stats": daymanager_cycle_stats,
            "handoff_diagnostics": handoff_diagnostics,
            "watchlist_causality_snapshot": watchlist_causality_snapshot,
            "signal_pipeline_audit": signal_pipeline_audit,
            "market_data_archive": dict(self._archive_diagnostics),
            "intraday_bar_coverage": intraday_bar_coverage,
            "replay_order_accounting": replay_order_accounting,
            "replay_notes": list(self._replay_notes),
            "inverse_absence_diagnostic": inverse_absence,
            "authority_timeline": authority_timeline,
            "long_decision_checks": long_checks,
            "inverse_orders": self._inverse_orders,
            "inverse_trade_results": self._inverse_trade_results,
            "inverse_performance": inverse_performance,
            "position_management_replay": management_replay,
            "management_replay_coverage": management_coverage,
            "stop_loss_diagnostics": stop_loss_diagnostics,
            "counterfactual_long_eval": counterfactual_long_eval,
            "divergences": divergences,
            "divergence_summary": divergence_summary,
        }
        if discovery_strategy_eval is not None:
            report["discovery_strategy_eval"] = discovery_strategy_eval
            report["summary"].update(
                {
                    "discovery_variant": str(
                        discovery_strategy_eval.get("variant") or ""
                    ),
                    "discovery_candidate_pool_size": int(
                        discovery_strategy_eval.get("candidate_pool_size", 0) or 0
                    ),
                    "discovery_candidates_evaluated": int(
                        discovery_strategy_eval.get("synthetic_candidates_evaluated", 0)
                        or 0
                    ),
                    "discovery_added_winners": int(
                        discovery_strategy_eval.get("added_winners", 0) or 0
                    ),
                    "discovery_added_losers": int(
                        discovery_strategy_eval.get("added_losers", 0) or 0
                    ),
                    "discovery_net_synthetic_pnl": float(
                        discovery_strategy_eval.get("net_synthetic_pnl", 0.0) or 0.0
                    ),
                    "discovery_slot_fill_delta": int(
                        discovery_strategy_eval.get("slot_fill_delta", 0) or 0
                    ),
                }
            )
        replay_entry_gate = build_replay_entry_diagnostic_gate(report)
        report["replay_entry_diagnostic_gate"] = replay_entry_gate
        report["summary"]["replay_entry_gate_status"] = replay_entry_gate["status"]
        report["summary"]["replay_entry_gate_reason"] = replay_entry_gate["reason"]
        report["summary"]["replay_entry_gate_passed"] = bool(
            replay_entry_gate["passed"]
        )
        if persist:
            _write_json(self.output_path, report)
        return report

    def _resolve_artifacts(self) -> Dict[str, Any]:
        compact = self.session_date.replace("-", "")
        session_dt = date.fromisoformat(self.session_date)
        previous_date = (session_dt - timedelta(days=1)).isoformat()
        signals_today_path = self.logs_dir / f"signals_{self.session_date}.json"
        signals_yesterday_path = self.logs_dir / f"signals_{previous_date}.json"
        signals_path = signals_today_path if signals_today_path.exists() else None
        if signals_path is None and signals_yesterday_path.exists():
            signals_path = signals_yesterday_path
        dashed_decision_path = (
            self.logs_dir / f"trade_decisions_{self.session_date}.json"
        )
        decision_candidates = [
            self.logs_dir / f"trade_decisions_{compact}.json",
            dashed_decision_path,
        ]
        decision_path = next(
            (path for path in decision_candidates if path.exists()), None
        )
        workflow_journal_path = (
            self.logs_dir / f"workflow_journal_{self.session_date}.jsonl"
        )
        trade_journal_path = self.logs_dir / "trade_journal.json"
        adjusted_plan_candidates = sorted(
            self.plans_dir.glob(f"adjusted_plan_{compact}_*.json"),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        adjusted_plan_path = next(
            (path for path in adjusted_plan_candidates if path.exists()), None
        )
        morning_plan_candidates = [
            self.plans_dir / f"morning_game_plan_{compact}.json",
            self.plans_dir / f"morning_game_plan_{self.session_date}.json",
        ]
        morning_plan_path = next(
            (path for path in morning_plan_candidates if path.exists()), None
        )
        pm_plan_candidates = [
            self.plans_dir / f"pm_plan_{self.session_date}.json",
            self.plans_dir / f"pm_plan_{compact}.json",
        ]
        pm_plan_path = next(
            (path for path in pm_plan_candidates if path.exists()), None
        )
        candidate_plans = list(morning_plan_candidates)
        candidate_plans.extend(pm_plan_candidates)
        candidate_plans.extend(adjusted_plan_candidates)
        plan_path = next((path for path in candidate_plans if path.exists()), None)
        if plan_path is None:
            raise FileNotFoundError(f"No plan artifact found for {self.session_date}")
        artifacts = {
            "signals_path": signals_path,
            "signals_today_path": signals_today_path
            if signals_today_path.exists()
            else None,
            "signals_yesterday_path": (
                signals_yesterday_path if signals_yesterday_path.exists() else None
            ),
            "decision_path": decision_path,
            "workflow_journal_path": workflow_journal_path
            if workflow_journal_path.exists()
            else None,
            "trade_journal_path": trade_journal_path
            if trade_journal_path.exists()
            else None,
            "adjusted_plan_path": adjusted_plan_path,
            "morning_plan_path": morning_plan_path,
            "pm_plan_path": pm_plan_path,
            "plan_path": plan_path,
            "plan_payload": _load_json(plan_path) or {},
            "pm_plan_payload": _load_json(pm_plan_path)
            if pm_plan_path and pm_plan_path.exists()
            else {},
        }
        self._refresh_raw_signals_ledger_from_authoritative_plan(artifacts)
        if decision_path is None:
            self._replay_notes.append(
                f"Decision artifact missing for {self.session_date}; replay continues with an empty recorded-decision ledger."
            )
        if artifacts.get("signals_path") == signals_yesterday_path:
            self._replay_notes.append(
                f"Using yesterday's signals file for {self.session_date}: {signals_yesterday_path.name}"
            )
        return artifacts

    def _build_replay_environment_snapshot(
        self, artifacts: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        artifacts = artifacts or {}
        status_short = _git_output(self.project_dir, "status", "--short")
        dirty_lines = [line for line in status_short.splitlines() if line.strip()]
        commit = _git_output(self.project_dir, "rev-parse", "HEAD")
        commit_short = _git_output(self.project_dir, "rev-parse", "--short", "HEAD")
        commit_time = _git_output(
            self.project_dir, "show", "-s", "--format=%cI", "HEAD"
        )

        cfg = get_config()
        day_manager_cfg = getattr(cfg, "day_manager", None)
        screener_cfg = getattr(cfg, "screener_v2", None)
        drift_pct = getattr(
            day_manager_cfg, "entry_anchor_prev_close_max_drift_pct", 5.0
        )
        cooldown_minutes = getattr(
            day_manager_cfg, "same_day_reentry_cooldown_minutes", 60
        )

        artifact_paths = {
            "signals": artifacts.get("signals_path"),
            "decisions": artifacts.get("decision_path"),
            "plan": artifacts.get("plan_path"),
            "pm_plan": artifacts.get("pm_plan_path"),
            "morning_plan": artifacts.get("morning_plan_path"),
            "adjusted_plan": artifacts.get("adjusted_plan_path"),
            "workflow_journal": artifacts.get("workflow_journal_path"),
            "trade_journal": artifacts.get("trade_journal_path"),
        }
        artifact_snapshot = {
            name: {
                "path": str(path) if path else "",
                "mtime": _file_mtime_iso(path if isinstance(path, Path) else None),
            }
            for name, path in artifact_paths.items()
        }

        return {
            "code_revision": {
                "commit": commit,
                "commit_short": commit_short,
                "commit_time": commit_time,
                "dirty": bool(dirty_lines),
                "dirty_entries": len(dirty_lines),
            },
            "replay_scope": {
                "uses_current_python_code": True,
                "runtime_daymanager_gates_replayed": True,
                "position_management_replayed": True,
                "actual_day_context_loaded": True,
                "signal_generation_artifacts_regenerated": False,
                "signal_generation_scope_note": (
                    "Fast/daymanager_core replay applies current runtime policy to "
                    "recorded plan/signal artifacts. Rebuild the source plan or "
                    "rerun the signal-generation pipeline before replay when "
                    "evaluating upstream candidate-generation changes."
                ),
            },
            "active_remediation_policies": {
                "prev_close_entry_drift_guard": {
                    "enabled": True,
                    "max_drift_pct": float(drift_pct or 5.0),
                    "block_reason": "prev_close_entry_drift",
                },
                "same_day_sell_trim_lockout": {
                    "enabled": True,
                    "cooldown_minutes": int(cooldown_minutes or 0),
                    "block_reason_prefix": "same_day_sell_lockout",
                },
                "screener_high_score_trap_veto": {
                    "enabled": float(
                        getattr(screener_cfg, "max_composite_score", 0.0) or 0.0
                    )
                    > 0.0,
                    "max_composite_score": float(
                        getattr(screener_cfg, "max_composite_score", 0.0) or 0.0
                    ),
                    "exempt_regimes": list(
                        getattr(
                            screener_cfg,
                            "max_composite_score_exempt_regimes",
                            [],
                        )
                        or []
                    ),
                },
                "news_cache_symbol_scoped_identity": {
                    "enabled": True,
                    "note": (
                        "Covered by current code path when news is fetched during "
                        "workflow replay; existing recorded news artifacts are not "
                        "rewritten by fast replay."
                    ),
                },
            },
            "source_artifacts": artifact_snapshot,
        }

    def _load_signals(self, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        signals_path = artifacts["signals_path"]
        if signals_path is not None:
            signals = self._normalize_signals(_load_json(signals_path))
            if signals:
                return signals
        return self._normalize_signals(artifacts["plan_payload"])

    @staticmethod
    def _signal_symbol_set(rows: Sequence[Dict[str, Any]]) -> set[str]:
        symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            for row in rows or []
            if isinstance(row, dict)
        }
        symbols.discard("")
        return symbols

    @staticmethod
    def _is_plan_origin_signal_row(row: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(row, dict):
            return False
        entry_source = str(row.get("entry_source") or "").strip().lower()
        if entry_source in {
            "overnight_plan",
            "premarket_adjusted",
            "overnight_full_watchlist",
        }:
            return True
        plan_score_source = str(row.get("plan_score_source") or "").strip().lower()
        return any(
            token in plan_score_source
            for token in ("morning_game_plan_", "adjusted_plan_", "pm_plan_")
        )

    @classmethod
    def _is_runtime_augmented_signal_row(cls, row: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(row, dict):
            return False
        if cls._is_plan_origin_signal_row(row):
            return False
        entry_source = str(row.get("entry_source") or "").strip().lower()
        if entry_source in {
            "momentum_scanner",
            "intraday_reserve_scan",
            "vwap_universe_scanner",
            "wave_entry",
            "early_runner_market_open",
            "early_runner_actionable",
            "early_runner_watchlist",
            "early_runner_deep_research",
            "overnight_recheck",
            "overnight_first_hour_recheck",
            "power_hour_volume",
            "watchlist_batch_rotate",
            "watchlist_rotation_scheduler",
        }:
            return True
        return bool(str(row.get("origin_entry_source") or "").strip())

    def _authoritative_signal_plan(
        self, artifacts: Dict[str, Any]
    ) -> Tuple[str, Optional[Path], List[Dict[str, Any]]]:
        for source_label, path in (
            ("adjusted_plan", artifacts.get("adjusted_plan_path")),
            ("pm_plan", artifacts.get("pm_plan_path")),
            ("morning_game_plan", artifacts.get("morning_plan_path")),
        ):
            signals = self._load_optional_signal_list(path)
            if signals:
                return source_label, path, signals
        return "", None, []

    def _refresh_raw_signals_ledger_from_authoritative_plan(
        self, artifacts: Dict[str, Any]
    ) -> None:
        source_label, plan_path, plan_signals = self._authoritative_signal_plan(
            artifacts
        )
        if not plan_path or not plan_signals:
            return

        target_path = self.logs_dir / f"signals_{self.session_date}.json"
        existing_signals = self._load_optional_signal_list(target_path)
        plan_symbols = self._signal_symbol_set(plan_signals)
        existing_symbols = self._signal_symbol_set(existing_signals)
        if target_path.exists() and plan_symbols == existing_symbols:
            artifacts["signals_path"] = target_path
            artifacts["signals_today_path"] = target_path
            return

        payload = {
            "date": self.session_date,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "generator": "runtime_replay",
            "total_signals": len(plan_signals),
            "signals": plan_signals,
            "source": source_label,
            "signal_manifest": {
                "source": source_label,
                "input_total": len(plan_signals),
                "enforce_filters": False,
                "saved_total": len(plan_signals),
                "filtered_out_total": 0,
                "plan_file": plan_path.name,
                "replay_refresh": True,
            },
        }
        _write_json(target_path, payload)
        artifacts["signals_path"] = target_path
        artifacts["signals_today_path"] = target_path
        self._replay_notes.append(
            f"Refreshed raw signals ledger from {plan_path.name} for replay date {self.session_date}."
        )

    def _archive_enabled(self) -> bool:
        return bool(
            getattr(self._archive_cfg, "enabled", False)
            and getattr(self._archive_cfg, "prefer_local_archive", False)
            and self._archive_db_path is not None
        )

    def _archive_live_fallback_enabled(self) -> bool:
        return bool(
            getattr(self._archive_cfg, "allow_live_fallback_if_incomplete", True)
        )

    def _hydrate_archive_market_data(
        self,
        *,
        signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
        workflow_journal: Sequence[Dict[str, Any]] = (),
    ) -> None:
        requested_symbols = set(self.benchmark_symbols)
        requested_symbols.update(
            str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            for row in signals
        )
        requested_symbols.update(
            str(row.get("symbol") or "").upper().strip() for row in decisions
        )
        requested_symbols.update(
            str(row.get("symbol") or "").upper().strip() for row in trade_journal
        )
        requested_symbols.update(
            str(row.get("symbol") or (row.get("position") or {}).get("symbol") or "")
            .upper()
            .strip()
            for row in workflow_journal
            if isinstance(row, dict)
        )
        requested_symbols.discard("")
        if requested_symbols:
            self._load_archive_symbols(requested_symbols)

    def _load_archive_symbols(self, symbols: Iterable[str]) -> None:
        if not self._archive_enabled():
            return
        requested = sorted(
            {
                str(symbol or "").upper().strip()
                for symbol in symbols
                if str(symbol or "").strip()
                and str(symbol or "").upper().strip() not in self.market_bars
            }
        )
        if not requested:
            return
        archive_load = load_session_archive(
            db_path=self._archive_db_path,
            session_date=self.session_date,
            symbols=requested,
        )
        self._archive_diagnostics["archive_found"] = bool(archive_load.archive_exists)
        if not archive_load.archive_exists:
            if "Local minute-bar archive not found" not in " ".join(self._replay_notes):
                self._replay_notes.append(
                    f"Local minute-bar archive not found for {self.session_date}; replay may use live fetch."
                )
            return
        loaded_symbols = set()
        for symbol, frame in archive_load.bars_by_symbol.items():
            if frame is None or frame.empty:
                continue
            self.market_bars[symbol] = self._normalize_bars_frame(frame)
            loaded_symbols.add(symbol)
            if symbol not in self._archive_diagnostics["loaded_symbols"]:
                self._archive_diagnostics["loaded_symbols"].append(symbol)
        for symbol, manifest in archive_load.manifest_by_symbol.items():
            self._archive_manifest_by_symbol[symbol] = dict(manifest)
            if not bool(manifest.get("has_premarket_data")):
                if symbol not in self._archive_diagnostics["premarket_missing_symbols"]:
                    self._archive_diagnostics["premarket_missing_symbols"].append(
                        symbol
                    )
        for symbol in requested:
            if symbol in loaded_symbols:
                continue
            if symbol not in self._archive_missing_symbols:
                self._archive_missing_symbols.add(symbol)
                self._archive_diagnostics["missing_symbols"].append(symbol)

    def _record_live_fetch(self, symbol: str) -> None:
        ticker = str(symbol or "").upper().strip()
        if not ticker:
            return
        if ticker not in self._archive_diagnostics["live_fetch_symbols"]:
            self._archive_diagnostics["live_fetch_symbols"].append(ticker)

    def _market_data_coverage_mode(self) -> str:
        loaded = len(self._archive_diagnostics.get("loaded_symbols", []))
        live = len(self._archive_diagnostics.get("live_fetch_symbols", []))
        if loaded and live:
            return "mixed_archive_and_live_fetch"
        if loaded:
            return "fully_local_archive"
        if live:
            return "live_fetch_only"
        return "none"

    def _load_optional_signal_list(self, path: Optional[Path]) -> List[Dict[str, Any]]:
        if not path or not path.exists():
            return []
        return self._normalize_signals(_load_json(path))

    def _normalize_signals(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            items = (
                payload.get("signals")
                or payload.get("buy_signals")
                or payload.get("entry_candidates")
                or payload.get("ranked_watchlist")
                or payload.get("watchlist")
                or []
            )
        else:
            items = []
        normalized: List[Dict[str, Any]] = []
        for raw in items:
            ticker = str(raw.get("ticker") or raw.get("symbol") or "").upper().strip()
            if not ticker:
                continue
            entry = dict(raw)
            entry["ticker"] = ticker
            entry.setdefault("symbol", ticker)
            normalized.append(entry)
        return normalized

    def _load_decisions(self, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not artifacts.get("decision_path"):
            return []
        payload = _load_json(artifacts["decision_path"])
        if not isinstance(payload, list):
            raise RuntimeError(f"Expected a list in {artifacts['decision_path']}")
        return [dict(row) for row in payload]

    def _load_workflow_journal(self, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = artifacts.get("workflow_journal_path")
        if not path:
            return []
        return _load_jsonl(path)

    def _load_trade_journal(self, artifacts: Dict[str, Any]) -> List[Dict[str, Any]]:
        path = artifacts.get("trade_journal_path")
        if not path:
            return []
        payload = _load_json(path)
        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("trades") or payload.get("entries") or []
        else:
            rows = []
        normalized: List[Dict[str, Any]] = []
        for raw in rows:
            if not isinstance(raw, dict):
                continue
            timestamp = _trade_journal_event_timestamp(raw)
            if not timestamp:
                continue
            try:
                local_ts = _coerce_local_timestamp(timestamp)
            except Exception:
                continue
            if (
                local_ts.date().isoformat() != self.session_date
                and str(timestamp)[:10] != self.session_date
            ):
                continue
            entry = dict(raw)
            entry["_local_timestamp"] = local_ts.isoformat()
            entry["_event_timestamp_field"] = timestamp
            normalized.append(entry)
        return normalized

    def _load_production_claw_decisions_count(self) -> int:
        """Count rows in logs/decision_claw_decisions_YYYY-MM-DD.jsonl."""
        path = self.logs_dir / f"decision_claw_decisions_{self.session_date}.jsonl"
        if not path.exists():
            return 0
        try:
            return len(_load_jsonl(path))
        except Exception:
            return 0

    def _build_handoff_diagnostics(
        self,
        *,
        artifacts: Dict[str, Any],
        raw_signals: Sequence[Dict[str, Any]],
        replay_signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pm_signals = self._load_optional_signal_list(artifacts.get("pm_plan_path"))
        morning_signals = self._load_optional_signal_list(
            artifacts.get("morning_plan_path")
        )
        adjusted_signals = self._load_optional_signal_list(
            artifacts.get("adjusted_plan_path")
        )
        raw_signal_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            for row in raw_signals
        }
        raw_signal_symbols.discard("")
        replay_signal_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            for row in replay_signals
        }
        replay_signal_symbols.discard("")
        replay_rows_by_symbol: Dict[str, List[Dict[str, Any]]] = {}
        for row in replay_signals:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            replay_rows_by_symbol.setdefault(symbol, []).append(row)
        decision_symbols = {
            str(row.get("symbol") or "").upper().strip() for row in decisions
        }
        decision_symbols.discard("")
        pm_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper()
            for row in pm_signals
        }
        pm_symbols.discard("")
        morning_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper()
            for row in morning_signals
        }
        morning_symbols.discard("")
        adjusted_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper()
            for row in adjusted_signals
        }
        adjusted_symbols.discard("")
        day_plan_symbols = adjusted_symbols or morning_symbols
        optimized_symbols = (
            pm_symbols
            | morning_symbols
            | adjusted_symbols
            | raw_signal_symbols
            | replay_signal_symbols
        )
        overnight_missing_from_day_plan = (
            pm_symbols - day_plan_symbols if pm_symbols else set()
        )
        day_plan_missing_from_signals = (
            day_plan_symbols - raw_signal_symbols if day_plan_symbols else set()
        )
        decision_symbols_missing_from_raw_signals = (
            decision_symbols - raw_signal_symbols
        )
        decision_symbols_missing_from_replay_signals = (
            decision_symbols - replay_signal_symbols
        )
        decision_symbols_runtime_augmented = {
            symbol
            for symbol in decision_symbols_missing_from_raw_signals
            if any(
                self._is_runtime_augmented_signal_row(row)
                for row in replay_rows_by_symbol.get(symbol, [])
            )
        }
        decision_symbols_missing_from_raw_unexplained = (
            decision_symbols_missing_from_raw_signals
            - decision_symbols_runtime_augmented
        )
        recoverable_candidates = (
            pm_symbols | morning_symbols | adjusted_symbols
        ) - raw_signal_symbols

        expected_filtered_alignment = (
            bool(day_plan_missing_from_signals)
            and not decision_symbols_missing_from_replay_signals
            and not decision_symbols_missing_from_raw_unexplained
        )

        if expected_filtered_alignment:
            status = "filtered_expected"
            problem = ""
        elif pm_symbols and not day_plan_symbols:
            status = "collapsed"
            problem = "overnight_no_plan_generated"
        elif overnight_missing_from_day_plan:
            status = "degraded"
            problem = "overnight_candidates_dropped_before_day_plan"
        else:
            status = "healthy"
            problem = ""

        if day_plan_symbols and not raw_signal_symbols:
            signal_alignment_status = "missing"
            signal_alignment_problem = "no_signals_file"
        elif decision_symbols_missing_from_replay_signals:
            signal_alignment_status = "broken"
            signal_alignment_problem = "recorded_decisions_missing_from_replay_inputs"
        elif decision_symbols_missing_from_raw_unexplained:
            signal_alignment_status = "drifted"
            signal_alignment_problem = "recorded_decisions_missing_from_raw_signals"
        elif day_plan_missing_from_signals:
            signal_alignment_status = "filtered"
            signal_alignment_problem = "day_plan_candidates_filtered_before_signal_log"
        else:
            signal_alignment_status = "aligned"
            signal_alignment_problem = ""

        return {
            "status": status,
            "problem": problem,
            "signals_source": (
                artifacts["signals_path"].name
                if artifacts.get("signals_path")
                else "plan_fallback"
            ),
            "pm_candidates_total": len(pm_symbols),
            "morning_candidates_total": len(morning_symbols),
            "adjusted_candidates_total": len(adjusted_symbols),
            "loaded_signals_total": len(replay_signal_symbols),
            "raw_signals_total": len(raw_signal_symbols),
            "recorded_decision_symbols_total": len(decision_symbols),
            "optimized_candidates_total": len(optimized_symbols),
            "recoverable_candidates_count": len(recoverable_candidates),
            "overnight_candidates_missing_from_day_plan_count": len(
                overnight_missing_from_day_plan
            ),
            "day_plan_candidates_missing_from_signals_count": len(
                day_plan_missing_from_signals
            ),
            "decision_symbols_missing_from_signals_count": len(
                decision_symbols_missing_from_raw_unexplained
            ),
            "decision_symbols_missing_from_raw_signals_count": len(
                decision_symbols_missing_from_raw_signals
            ),
            "decision_symbols_runtime_augmented_count": len(
                decision_symbols_runtime_augmented
            ),
            "decision_symbols_missing_from_replay_signals_count": len(
                decision_symbols_missing_from_replay_signals
            ),
            "signal_alignment_status": signal_alignment_status,
            "signal_alignment_problem": signal_alignment_problem,
            "recoverable_candidates_examples": _sample_symbols(recoverable_candidates),
            "overnight_candidates_missing_from_day_plan_examples": _sample_symbols(
                overnight_missing_from_day_plan
            ),
            "day_plan_candidates_missing_from_signals_examples": _sample_symbols(
                day_plan_missing_from_signals
            ),
            "decision_symbols_missing_from_raw_signals_examples": _sample_symbols(
                decision_symbols_missing_from_raw_signals
            ),
            "decision_symbols_runtime_augmented_examples": _sample_symbols(
                decision_symbols_runtime_augmented
            ),
            "decision_symbols_missing_from_signals_examples": _sample_symbols(
                decision_symbols_missing_from_raw_unexplained
            ),
            "decision_symbols_missing_from_replay_signals_examples": _sample_symbols(
                decision_symbols_missing_from_replay_signals
            ),
            "source_paths": {
                "signals": str(artifacts.get("signals_path") or ""),
                "pm_plan": str(artifacts.get("pm_plan_path") or ""),
                "morning_plan": str(artifacts.get("morning_plan_path") or ""),
                "adjusted_plan": str(artifacts.get("adjusted_plan_path") or ""),
            },
            "interpretation": (
                "recorded_decisions_aligned_after_expected_filtering"
                if expected_filtered_alignment
                else problem or signal_alignment_problem or "healthy"
            ),
        }

    def _signal_pipeline_audit_path(self) -> Path:
        return self.logs_dir / f"signal_pipeline_audit_{self.session_date}.json"

    def _build_signal_pipeline_audit(
        self,
        *,
        artifacts: Dict[str, Any],
        raw_signals: Sequence[Dict[str, Any]],
        replay_signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
        watchlist_causality_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        pm_signals = self._load_optional_signal_list(artifacts.get("pm_plan_path"))
        morning_signals = self._load_optional_signal_list(
            artifacts.get("morning_plan_path")
        )
        adjusted_signals = self._load_optional_signal_list(
            artifacts.get("adjusted_plan_path")
        )

        raw_by_symbol = _pipeline_row_by_symbol(raw_signals)
        replay_by_symbol = _pipeline_row_by_symbol(replay_signals)
        pm_by_symbol = _pipeline_row_by_symbol(pm_signals)
        morning_by_symbol = _pipeline_row_by_symbol(morning_signals)
        adjusted_by_symbol = _pipeline_row_by_symbol(adjusted_signals)
        decision_by_symbol = _pipeline_row_by_symbol(decisions)
        trade_by_symbol = _pipeline_row_by_symbol(trade_journal)
        causality_rows = (
            (watchlist_causality_snapshot or {}).get("symbols")
            or (watchlist_causality_snapshot or {}).get("entries")
            or []
        )
        causality_by_symbol = _pipeline_row_by_symbol(causality_rows)

        all_symbols = sorted(
            set(raw_by_symbol)
            | set(replay_by_symbol)
            | set(pm_by_symbol)
            | set(morning_by_symbol)
            | set(adjusted_by_symbol)
            | set(decision_by_symbol)
            | set(trade_by_symbol)
            | set(causality_by_symbol)
        )

        rows: List[Dict[str, Any]] = []
        for symbol in all_symbols:
            causality = causality_by_symbol.get(symbol, {})
            decision = decision_by_symbol.get(symbol, {})
            trade = trade_by_symbol.get(symbol, {})
            preferred = (
                adjusted_by_symbol.get(symbol)
                or morning_by_symbol.get(symbol)
                or pm_by_symbol.get(symbol)
                or replay_by_symbol.get(symbol)
                or raw_by_symbol.get(symbol)
                or causality
                or {}
            )
            entry_source = _first_nonempty_value(
                adjusted_by_symbol.get(symbol, {}).get("entry_source"),
                morning_by_symbol.get(symbol, {}).get("entry_source"),
                pm_by_symbol.get(symbol, {}).get("entry_source"),
                replay_by_symbol.get(symbol, {}).get("entry_source"),
                raw_by_symbol.get(symbol, {}).get("entry_source"),
                causality.get("entry_source"),
            )
            source_bucket = _first_nonempty_value(
                adjusted_by_symbol.get(symbol, {}).get("source_bucket"),
                morning_by_symbol.get(symbol, {}).get("source_bucket"),
                pm_by_symbol.get(symbol, {}).get("source_bucket"),
                replay_by_symbol.get(symbol, {}).get("source_bucket"),
                raw_by_symbol.get(symbol, {}).get("source_bucket"),
                causality.get("source_bucket"),
            )
            plan_score_source = _first_nonempty_value(
                adjusted_by_symbol.get(symbol, {}).get("plan_score_source"),
                morning_by_symbol.get(symbol, {}).get("plan_score_source"),
                pm_by_symbol.get(symbol, {}).get("plan_score_source"),
                replay_by_symbol.get(symbol, {}).get("plan_score_source"),
                raw_by_symbol.get(symbol, {}).get("plan_score_source"),
                causality.get("plan_score_source"),
            )
            rows.append(
                {
                    "symbol": symbol,
                    "in_pm_plan": symbol in pm_by_symbol,
                    "in_morning_plan": symbol in morning_by_symbol,
                    "in_adjusted_plan": symbol in adjusted_by_symbol,
                    "in_raw_signals": symbol in raw_by_symbol,
                    "in_replay_signals": symbol in replay_by_symbol,
                    "in_recorded_decisions": symbol in decision_by_symbol,
                    "actually_executed": bool(decision.get("actually_executed")),
                    "in_trade_journal": symbol in trade_by_symbol,
                    "watchlist_member": bool(causality.get("watchlist_member")),
                    "current_status": str(
                        causality.get("current_status") or causality.get("status") or ""
                    ),
                    "blocking_reason": str(causality.get("blocking_reason") or ""),
                    "blocking_rule": str(causality.get("blocking_rule") or ""),
                    "entry_source": str(entry_source or ""),
                    "source_bucket": str(source_bucket or ""),
                    "plan_score_source": str(plan_score_source or ""),
                    "current_score": preferred.get(
                        "current_score", causality.get("current_score")
                    ),
                    "entry_threshold": preferred.get(
                        "entry_threshold", causality.get("entry_threshold")
                    ),
                    "score_gap_to_threshold": preferred.get(
                        "score_gap_to_threshold",
                        causality.get("score_gap_to_threshold"),
                    ),
                    "decision_score": decision.get("score"),
                    "decision_timestamp": decision.get("timestamp"),
                    "trade_timestamp": trade.get("_local_timestamp")
                    or trade.get("timestamp"),
                }
            )

        status_counts = Counter(
            str(row.get("current_status") or "").strip()
            for row in rows
            if str(row.get("current_status") or "").strip()
        )
        block_counts = Counter(
            str(row.get("blocking_reason") or "").strip()
            for row in rows
            if str(row.get("blocking_reason") or "").strip()
        )
        summary = {
            "symbols_total": len(rows),
            "executed_symbols": sum(1 for row in rows if row.get("actually_executed")),
            "blocked_symbols": sum(1 for row in rows if row.get("blocking_reason")),
            "stage_counts": {
                "pm_plan": len(pm_by_symbol),
                "morning_plan": len(morning_by_symbol),
                "adjusted_plan": len(adjusted_by_symbol),
                "raw_signals": len(raw_by_symbol),
                "replay_signals": len(replay_by_symbol),
                "recorded_decisions": len(decision_by_symbol),
                "trade_journal_symbols": len(trade_by_symbol),
                "watchlist_causality_symbols": len(causality_by_symbol),
            },
            "status_counts": dict(status_counts),
            "blocking_reason_counts": dict(block_counts),
        }
        return {
            "date": self.session_date,
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "summary": summary,
            "rows": rows,
            "pm_bridge": dict(
                (
                    (artifacts.get("pm_plan_payload") or {}).get(
                        "overnight_watchlist_bridge"
                    )
                    or {}
                )
            ),
            "pm_signal_pipeline_trace": dict(
                (
                    (artifacts.get("pm_plan_payload") or {}).get(
                        "signal_pipeline_trace"
                    )
                    or {}
                )
            ),
        }

    def _journal_metadata_candidates(
        self,
        *,
        symbol: str,
        trade_journal: List[Dict[str, Any]],
        decision_timestamp: str = "",
    ) -> List[Dict[str, Any]]:
        decision_ts: Optional[datetime] = None
        if decision_timestamp:
            try:
                decision_ts = _coerce_local_timestamp(decision_timestamp)
            except Exception:
                decision_ts = None
        candidates: List[Dict[str, Any]] = []
        for row in trade_journal:
            row_symbol = str(row.get("symbol") or "").upper().strip()
            if row_symbol != symbol:
                continue
            context = (
                row.get("signal_context")
                if isinstance(row.get("signal_context"), dict)
                else {}
            )
            entry_score = row.get("entry_score")
            if entry_score is None:
                entry_score = context.get("final_score")
            plan_score_source = str(
                row.get("plan_score_source") or context.get("plan_score_source") or ""
            )
            entry_source = str(
                row.get("entry_source") or context.get("entry_source") or ""
            )
            source_bucket = str(
                row.get("source_bucket") or context.get("source_bucket") or ""
            )
            metadata = {
                "ticker": symbol,
                "symbol": symbol,
                "score": entry_score,
                "final_score": entry_score,
                "realtime_score": entry_score,
                "plan_score_source": plan_score_source,
                "entry_source": entry_source,
                "source_bucket": source_bucket,
                "strategy_name": str(
                    row.get("strategy_name") or context.get("strategy_name") or ""
                ),
                "setup_type": str(
                    row.get("setup_type") or context.get("setup_type") or ""
                ),
                "strategy_id": str(
                    row.get("strategy_id") or context.get("strategy_id") or ""
                ),
                "origin_entry_source": str(
                    row.get("origin_entry_source")
                    or context.get("origin_entry_source")
                    or ""
                ),
                "runtime_entry_context": str(
                    row.get("runtime_entry_context")
                    or context.get("runtime_entry_context")
                    or ""
                ),
                "override_reason": str(
                    row.get("override_reason") or context.get("override_reason") or ""
                ),
            }
            completeness = int(entry_score is not None) + int(
                bool(plan_score_source or entry_source)
            )
            if metadata["strategy_name"]:
                completeness += 1
            row_ts = None
            try:
                row_ts = _coerce_local_timestamp(
                    str(row.get("timestamp") or row.get("entry_time") or "")
                )
            except Exception:
                row_ts = None
            time_delta = (
                abs((row_ts - decision_ts).total_seconds())
                if row_ts is not None and decision_ts is not None
                else 10**12
            )
            metadata["_completeness"] = completeness
            metadata["_time_delta_seconds"] = time_delta
            candidates.append(metadata)
        candidates.sort(
            key=lambda item: (
                -int(item.get("_completeness", 0)),
                float(item.get("_time_delta_seconds", 10**12)),
            )
        )
        return candidates

    def _best_journal_metadata(
        self,
        *,
        symbol: str,
        trade_journal: List[Dict[str, Any]],
        decision_timestamp: str = "",
    ) -> Dict[str, Any]:
        candidates = self._journal_metadata_candidates(
            symbol=symbol,
            trade_journal=trade_journal,
            decision_timestamp=decision_timestamp,
        )
        return dict(candidates[0]) if candidates else {}

    def _augment_signals_with_recorded_metadata(
        self,
        signals: List[Dict[str, Any]],
        *,
        decisions: List[Dict[str, Any]],
        trade_journal: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {
            str(signal.get("ticker") or signal.get("symbol") or "").upper(): dict(
                signal
            )
            for signal in signals
            if isinstance(signal, dict)
            and str(signal.get("ticker") or signal.get("symbol") or "").strip()
        }
        for decision in decisions:
            symbol = str(decision.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            signal_row = dict(
                merged.get(symbol) or {"ticker": symbol, "symbol": symbol}
            )
            score = decision.get("score")
            if score is not None:
                signal_row.setdefault("score", score)
                signal_row.setdefault("final_score", score)
                signal_row.setdefault("realtime_score", score)
            journal_metadata = self._best_journal_metadata(
                symbol=symbol,
                trade_journal=trade_journal,
                decision_timestamp=str(decision.get("timestamp") or ""),
            )
            if journal_metadata:
                for key in (
                    "score",
                    "final_score",
                    "realtime_score",
                    "plan_score_source",
                    "entry_source",
                    "source_bucket",
                    "strategy_name",
                    "setup_type",
                    "strategy_id",
                    "origin_entry_source",
                    "runtime_entry_context",
                    "override_reason",
                ):
                    value = journal_metadata.get(key)
                    if value not in (None, ""):
                        signal_row.setdefault(key, value)
            if journal_metadata:
                signal_row["_replay_metadata_backfilled"] = True
            merged[symbol] = signal_row
        return list(merged.values())

    def _augment_signals_with_plan_metadata(
        self,
        signals: List[Dict[str, Any]],
        *,
        artifacts: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {
            str(signal.get("ticker") or signal.get("symbol") or "").upper(): dict(
                signal
            )
            for signal in signals
            if isinstance(signal, dict)
            and str(signal.get("ticker") or signal.get("symbol") or "").strip()
        }
        preferred_plan_sources = (
            ("morning_plan", artifacts.get("morning_plan_path"), "overnight_plan"),
            ("pm_plan", artifacts.get("pm_plan_path"), "overnight_plan"),
            (
                "adjusted_plan",
                artifacts.get("adjusted_plan_path"),
                "legacy_adjusted_plan",
            ),
        )
        for source_name, plan_path, default_entry_source in preferred_plan_sources:
            plan_rows = self._load_optional_signal_list(plan_path)
            for plan_row in plan_rows:
                symbol = (
                    str(plan_row.get("ticker") or plan_row.get("symbol") or "")
                    .upper()
                    .strip()
                )
                if not symbol:
                    continue
                if symbol not in merged:
                    # Ensure plan candidates are represented in the signal list
                    # for hydration and replay discovery.
                    merged[symbol] = {
                        "ticker": symbol,
                        "symbol": symbol,
                        "entry_source": default_entry_source,
                        "source_bucket": "watchlist",
                        "status": "pending",
                        "reason": "plan_candidate",
                        "_candidate_source_plan": source_name,
                    }
                signal_row = merged[symbol]
                for key in (
                    "score",
                    "final_score",
                    "realtime_score",
                    "entry_price",
                    "price",
                    "qty",
                    "adjusted_qty",
                    "planned_qty",
                    "position_size",
                    "position_size_usd",
                    "risk_reward",
                    "rr_ratio",
                    "volume_ratio",
                    "relative_volume",
                    "rel_volume",
                    "strategy_name",
                    "setup_type",
                    "strategy_id",
                    "entry_source",
                    "source_bucket",
                    "plan_score_source",
                    "origin_entry_source",
                    "runtime_entry_context",
                    "override_reason",
                ):
                    value = plan_row.get(key)
                    if value not in (None, "", []):
                        signal_row.setdefault(key, value)
                signal_row.setdefault("entry_source", default_entry_source)
                signal_row.setdefault("source_bucket", "watchlist")
                if plan_path is not None:
                    signal_row.setdefault("plan_score_source", plan_path.name)
                signal_row.setdefault("_candidate_source_plan", source_name)
                if any(
                    key in plan_row and plan_row.get(key) not in (None, "", [])
                    for key in (
                        "entry_source",
                        "source_bucket",
                        "plan_score_source",
                        "strategy_name",
                        "setup_type",
                    )
                ):
                    signal_row["_replay_plan_metadata_backfilled"] = True
                merged[symbol] = signal_row
        return list(merged.values())

    def _build_replay_signal_data(
        self,
        *,
        dm: DayManager,
        symbol: str,
        decision: Dict[str, Any],
        trade_journal: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        signal_data = dict(
            dm._find_signal_data(symbol) or {"ticker": symbol, "symbol": symbol}
        )
        signal_data.setdefault("ticker", symbol)
        signal_data.setdefault("symbol", symbol)
        score = decision.get("score")
        if score is not None:
            signal_data.setdefault("score", score)
            signal_data.setdefault("final_score", score)
            signal_data.setdefault("realtime_score", score)
        journal_metadata = self._best_journal_metadata(
            symbol=symbol,
            trade_journal=trade_journal,
            decision_timestamp=str(decision.get("timestamp") or ""),
        )
        if journal_metadata:
            for key in (
                "score",
                "final_score",
                "realtime_score",
                "plan_score_source",
                "entry_source",
                "source_bucket",
                "strategy_name",
                "setup_type",
                "strategy_id",
                "origin_entry_source",
                "runtime_entry_context",
                "override_reason",
            ):
                value = journal_metadata.get(key)
                if value not in (None, ""):
                    signal_data.setdefault(key, value)
            signal_data["_replay_metadata_backfilled"] = True
        return signal_data

    def _dedupe_candidate_pool(
        self,
        *,
        raw_signals: Sequence[Dict[str, Any]],
        adjusted_signals: Sequence[Dict[str, Any]],
        morning_signals: Sequence[Dict[str, Any]],
        pm_signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        source_groups = (
            ("adjusted_plan", adjusted_signals),
            ("morning_plan", morning_signals),
            ("pm_plan", pm_signals),
            ("raw_signals", raw_signals),
        )
        merged: Dict[str, Dict[str, Any]] = {}
        for source_name, rows in source_groups:
            for raw in rows:
                if not isinstance(raw, dict):
                    continue
                symbol = (
                    str(raw.get("ticker") or raw.get("symbol") or "").upper().strip()
                )
                if not symbol or symbol in {"SPY", "QQQ"}:
                    continue
                if symbol not in merged:
                    entry = dict(raw)
                    entry["ticker"] = symbol
                    entry.setdefault("symbol", symbol)
                    entry["_candidate_source_plan"] = source_name
                    merged[symbol] = entry
        return merged

    def _discovery_candidate_contract(
        self, candidate: Dict[str, Any]
    ) -> Dict[str, Any]:
        symbol = (
            str(candidate.get("ticker") or candidate.get("symbol") or "")
            .upper()
            .strip()
        )
        if not symbol:
            return {"eligible": False, "reasons": ["missing_symbol"]}

        score = _safe_float(
            candidate.get("score")
            or candidate.get("final_score")
            or candidate.get("realtime_score")
            or candidate.get("confidence")
        )
        risk_reward = _safe_float(
            candidate.get("risk_reward") or candidate.get("rr_ratio")
        )
        rel_volume = _safe_float(
            candidate.get("relative_volume")
            or candidate.get("rel_volume")
            or candidate.get("volume_ratio")
        )
        day_return = _safe_float(
            candidate.get("day_return") or candidate.get("session_gain_pct")
        )
        short_momentum = _safe_float(candidate.get("short_momentum"))
        trend_strength = _safe_float(candidate.get("trend_strength"))
        pullback_pct = _safe_float(candidate.get("pullback_pct"))
        support_dist_atr = _safe_float(candidate.get("support_dist_atr"))
        resistance_dist_atr = _safe_float(candidate.get("resistance_dist_atr"))
        source_plan = str(candidate.get("_candidate_source_plan") or "")

        reasons: List[str] = []
        if score is None or score < 75.0:
            reasons.append(f"score<{75.0:.0f}")
        if risk_reward is not None and risk_reward < 1.5:
            reasons.append("rr<1.50")
        if rel_volume is not None and rel_volume < 1.15:
            reasons.append("rel_volume<1.15")
        if day_return is not None and day_return < 0.5:
            reasons.append("day_return<0.50")
        if short_momentum is not None and short_momentum < 0.40:
            reasons.append("short_momentum<0.40")
        if trend_strength is not None and trend_strength < 0.55:
            reasons.append("trend_strength<0.55")
        if pullback_pct is not None and pullback_pct > 3.0:
            reasons.append("pullback>3.0")
        if support_dist_atr is not None and support_dist_atr < 0.5:
            reasons.append("support_tight<0.50atr")
        if resistance_dist_atr is not None and resistance_dist_atr < 1.0:
            reasons.append("resistance_tight<1.00atr")
        if source_plan == "adjusted_plan":
            reasons.append("baseline_day_plan_symbol")

        return {
            "eligible": not reasons,
            "reasons": reasons,
            "metrics": {
                "score": score,
                "risk_reward": risk_reward,
                "relative_volume": rel_volume,
                "day_return": day_return,
                "short_momentum": short_momentum,
                "trend_strength": trend_strength,
                "pullback_pct": pullback_pct,
                "support_dist_atr": support_dist_atr,
                "resistance_dist_atr": resistance_dist_atr,
            },
        }

    def _bars_for_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        ticker = str(symbol or "").upper().strip()
        if not ticker:
            return None
        df = self.market_bars.get(ticker)
        if df is not None and not df.empty:
            return df
        self._load_archive_symbols([ticker])
        df = self.market_bars.get(ticker)
        if df is not None and not df.empty:
            return df
        if not self._archive_live_fallback_enabled():
            return None
        client = self.data_client or create_data_client(require_credentials=True)
        try:
            df = self._fetch_minute_bars(client, ticker)
        except Exception:
            return None
        self._record_live_fetch(ticker)
        self.market_bars[ticker] = df
        return df

    def _resolve_synthetic_levels(
        self,
        candidate: Dict[str, Any],
        *,
        entry_price: float,
    ) -> Dict[str, float]:
        stop_price = _safe_float(
            candidate.get("stop_price") or candidate.get("stop_loss")
        )
        target_price = _safe_float(
            candidate.get("target_price") or candidate.get("target")
        )
        risk_reward = (
            _safe_float(candidate.get("risk_reward") or candidate.get("rr_ratio"))
            or 1.8
        )
        if stop_price is None or stop_price >= entry_price:
            stop_price = round(entry_price * 0.96, 4)
        risk_per_share = max(entry_price - stop_price, entry_price * 0.02)
        if target_price is None or target_price <= entry_price:
            target_price = round(
                entry_price + risk_per_share * max(risk_reward, 1.5), 4
            )
        return {
            "stop_price": float(stop_price),
            "target_price": float(target_price),
            "risk_reward": float(max(risk_reward, 0.0)),
        }

    def _discovery_priority_score(
        self,
        candidate: Dict[str, Any],
        *,
        metrics: Dict[str, Any],
    ) -> float:
        score = _safe_float(metrics.get("score")) or 0.0
        volume_ratio = _safe_float(metrics.get("relative_volume")) or 0.0
        trend_strength = _safe_float(candidate.get("trend_strength")) or 0.0
        short_momentum = _safe_float(candidate.get("short_momentum")) or 0.0
        pullback_pct = _safe_float(candidate.get("pullback_pct")) or 0.0
        return (
            score
            + max(0.0, volume_ratio - 1.0) * 18.0
            + max(0.0, trend_strength - 0.5) * 20.0
            + max(0.0, short_momentum) * 6.0
            - max(0.0, pullback_pct - 1.0) * 4.0
        )

    def _simulate_discovery_trade(
        self,
        candidate: Dict[str, Any],
        *,
        default_notional: float = 1000.0,
        entry_start_time: Optional[datetime] = None,
        entry_price_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol = (
            str(candidate.get("ticker") or candidate.get("symbol") or "")
            .upper()
            .strip()
        )
        bars = self._bars_for_symbol(symbol)
        if bars is None or bars.empty:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["missing_bars"],
            }
        session_open = entry_start_time or datetime.combine(
            date.fromisoformat(self.session_date),
            time(8, 35),
            tzinfo=CT,
        )
        sliced = bars[bars.index >= pd.Timestamp(session_open)]
        if sliced.empty:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["no_valid_entry_window"],
            }
        entry_bar = sliced.iloc[0]
        entry_time = sliced.index[0]
        entry_price = (
            _safe_float(entry_price_override)
            or _safe_float(candidate.get("entry_price"))
            or _safe_float(entry_bar.get("close"))
        )
        if entry_price is None or entry_price <= 0:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["missing_entry_price"],
            }
        levels = self._resolve_synthetic_levels(candidate, entry_price=entry_price)
        stop_price = levels["stop_price"]
        target_price = levels["target_price"]
        notional = self._planned_signal_notional(candidate)
        if notional <= 0:
            notional = float(
                candidate.get("position_size")
                or candidate.get("position_size_usd")
                or default_notional
                or 1000.0
            )
        qty = max(1, int(notional / entry_price))
        exit_price = _safe_float(sliced.iloc[-1].get("close")) or entry_price
        exit_time = sliced.index[-1]
        exit_reason = "eod_close"
        max_favorable = 0.0
        max_adverse = 0.0
        trade_window = sliced[sliced.index > entry_time]
        if trade_window.empty:
            trade_window = sliced.tail(1)
        for ts, row in trade_window.iterrows():
            high = _safe_float(row.get("high")) or entry_price
            low = _safe_float(row.get("low")) or entry_price
            max_favorable = max(
                max_favorable, ((high - entry_price) / entry_price) * 100.0
            )
            max_adverse = min(max_adverse, ((low - entry_price) / entry_price) * 100.0)
            if low <= stop_price:
                exit_price = stop_price
                exit_time = ts
                exit_reason = "stop_loss"
                break
            if high >= target_price:
                exit_price = target_price
                exit_time = ts
                exit_reason = "profit_target"
                break
        pnl = (exit_price - entry_price) * qty
        return_pct = ((exit_price - entry_price) / entry_price) * 100.0
        return {
            "symbol": symbol,
            "status": "evaluated",
            "entry_time": entry_time.isoformat(),
            "entry_price": round(entry_price, 4),
            "stop_price": round(stop_price, 4),
            "target_price": round(target_price, 4),
            "exit_time": exit_time.isoformat(),
            "exit_price": round(float(exit_price), 4),
            "exit_reason": exit_reason,
            "qty": qty,
            "synthetic_notional": round(entry_price * qty, 2),
            "synthetic_pnl": round(float(pnl), 2),
            "return_pct": round(float(return_pct), 4),
            "max_favorable_excursion_pct": round(float(max_favorable), 4),
            "max_adverse_excursion_pct": round(float(max_adverse), 4),
            "candidate_source_plan": str(candidate.get("_candidate_source_plan") or ""),
        }

    def _simulate_counterfactual_management_trade(
        self,
        *,
        dm: DayManager,
        candidate: Dict[str, Any],
        entry_start_time: datetime,
        entry_price_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol = (
            str(candidate.get("ticker") or candidate.get("symbol") or "")
            .upper()
            .strip()
        )
        bars = self._bars_for_symbol(symbol)
        if bars is None or bars.empty:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["missing_bars"],
            }
        sliced = bars[bars.index >= pd.Timestamp(entry_start_time)]
        if sliced.empty:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["no_valid_entry_window"],
            }
        entry_bar = sliced.iloc[0]
        entry_time = sliced.index[0]
        entry_price = (
            _safe_float(entry_price_override)
            or _safe_float(candidate.get("entry_price"))
            or _safe_float(entry_bar.get("close"))
        )
        planned_notional = self._planned_signal_notional(candidate)
        qty = max(1, int(planned_notional / max(entry_price or 0.0, 0.01)))
        if entry_price is None or entry_price <= 0 or qty <= 0:
            return {
                "symbol": symbol,
                "status": "skipped",
                "reasons": ["missing_entry_price_or_size"],
            }

        remaining_qty = qty
        realized_pnl = 0.0
        trim_count = 0
        first_management_action = ""
        exit_price = _safe_float(sliced.iloc[-1].get("close")) or entry_price
        exit_time = sliced.index[-1]
        exit_reason = "eod_close"
        max_favorable = 0.0
        max_adverse = 0.0

        dm._entry_time_overrides[symbol] = entry_time
        dm.position_entries[symbol] = entry_time.astimezone(UTC)

        trade_window = sliced[sliced.index > entry_time]
        if trade_window.empty:
            trade_window = sliced.tail(1)

        for ts, row in trade_window.iterrows():
            current_price = (
                _safe_float(row.get("close"))
                or _safe_float(row.get("vwap"))
                or _safe_float(row.get("open"))
                or entry_price
            )
            high = _safe_float(row.get("high")) or current_price
            low = _safe_float(row.get("low")) or current_price
            max_favorable = max(
                max_favorable, ((high - entry_price) / entry_price) * 100.0
            )
            max_adverse = min(max_adverse, ((low - entry_price) / entry_price) * 100.0)

            synthetic_position = self._position_to_namespace(
                {
                    "symbol": symbol,
                    "qty": remaining_qty,
                    "avg_entry_price": entry_price,
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "market_value": current_price * remaining_qty,
                    "cost_basis": entry_price * remaining_qty,
                    "pnl_pct": ((current_price - entry_price) / entry_price) * 100.0,
                    "unrealized_pnl": (current_price - entry_price) * remaining_qty,
                    "side": "long",
                }
            )
            replay_action = self._evaluate_management_action(
                dm=dm,
                position=synthetic_position,
                replay_time=ts,
                entry_time=entry_time,
            )
            if replay_action and not first_management_action:
                first_management_action = replay_action

            if replay_action == "trim" and remaining_qty > 1:
                trim_qty = max(1, remaining_qty // 2)
                realized_pnl += (current_price - entry_price) * trim_qty
                remaining_qty -= trim_qty
                trim_count += 1
                exit_price = current_price
                exit_time = ts
                exit_reason = "management_trim"
                continue

            if replay_action == "exit":
                realized_pnl += (current_price - entry_price) * remaining_qty
                exit_price = current_price
                exit_time = ts
                exit_reason = "management_exit"
                remaining_qty = 0
                break

        if remaining_qty > 0:
            realized_pnl += (exit_price - entry_price) * remaining_qty
        total_return_pct = (
            ((realized_pnl / (entry_price * qty)) * 100.0) if qty > 0 else 0.0
        )
        return {
            "symbol": symbol,
            "status": "evaluated",
            "entry_time": entry_time.isoformat(),
            "entry_price": round(entry_price, 4),
            "exit_time": exit_time.isoformat(),
            "exit_price": round(float(exit_price), 4),
            "exit_reason": exit_reason,
            "qty": qty,
            "trim_count": int(trim_count),
            "synthetic_notional": round(entry_price * qty, 2),
            "synthetic_pnl": round(float(realized_pnl), 2),
            "return_pct": round(float(total_return_pct), 4),
            "max_favorable_excursion_pct": round(float(max_favorable), 4),
            "max_adverse_excursion_pct": round(float(max_adverse), 4),
            "candidate_source_plan": str(candidate.get("_candidate_source_plan") or ""),
            "management_style": "daymanager_position_health",
            "first_management_action": str(first_management_action or ""),
        }

    def _evaluate_discovery_strategy(
        self,
        *,
        artifacts: Dict[str, Any],
        raw_signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        adjusted_signals = self._load_optional_signal_list(
            artifacts.get("adjusted_plan_path")
        )
        morning_signals = self._load_optional_signal_list(
            artifacts.get("morning_plan_path")
        )
        pm_signals = self._load_optional_signal_list(artifacts.get("pm_plan_path"))
        pool = self._dedupe_candidate_pool(
            raw_signals=raw_signals,
            adjusted_signals=adjusted_signals,
            morning_signals=morning_signals,
            pm_signals=pm_signals,
        )
        baseline_symbols = {
            str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            for row in adjusted_signals
        }
        baseline_symbols.update(
            str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            for row in morning_signals
        )
        baseline_symbols.update(
            str(row.get("symbol") or "").upper().strip()
            for row in decisions
            if row.get("actually_executed")
        )
        baseline_symbols.discard("")

        candidate_rows: List[Dict[str, Any]] = []
        skipped_rows: List[Dict[str, Any]] = []
        data_quality_notes = Counter()
        discovery_rank_limit = 2
        for symbol, candidate in sorted(pool.items()):
            if symbol in baseline_symbols:
                continue
            contract = self._discovery_candidate_contract(candidate)
            if not contract["eligible"]:
                skipped_rows.append(
                    {
                        "symbol": symbol,
                        "candidate_source_plan": str(
                            candidate.get("_candidate_source_plan") or ""
                        ),
                        "status": "blocked",
                        "reasons": list(contract["reasons"]),
                    }
                )
                for reason in contract["reasons"]:
                    data_quality_notes[str(reason)] += 1
                continue
            simulation = self._simulate_discovery_trade(candidate)
            row = {
                "symbol": symbol,
                "candidate_source_plan": str(
                    candidate.get("_candidate_source_plan") or ""
                ),
                "contract": contract,
                "discovery_priority_score": round(
                    self._discovery_priority_score(
                        candidate, metrics=contract["metrics"]
                    ),
                    4,
                ),
                "candidate_snapshot": {
                    "score": contract["metrics"].get("score"),
                    "risk_reward": contract["metrics"].get("risk_reward"),
                    "relative_volume": contract["metrics"].get("relative_volume"),
                    "day_return": contract["metrics"].get("day_return"),
                    "short_momentum": contract["metrics"].get("short_momentum"),
                    "trend_strength": contract["metrics"].get("trend_strength"),
                    "pullback_pct": contract["metrics"].get("pullback_pct"),
                    "support_dist_atr": contract["metrics"].get("support_dist_atr"),
                    "resistance_dist_atr": contract["metrics"].get(
                        "resistance_dist_atr"
                    ),
                },
            }
            row.update(simulation)
            if simulation.get("status") != "evaluated":
                skipped_rows.append(
                    {
                        "symbol": symbol,
                        "candidate_source_plan": str(
                            candidate.get("_candidate_source_plan") or ""
                        ),
                        "status": str(simulation.get("status") or "skipped"),
                        "reasons": list(simulation.get("reasons") or []),
                    }
                )
                for reason in simulation.get("reasons") or ["skipped"]:
                    data_quality_notes[str(reason)] += 1
                continue
            candidate_rows.append(row)

        candidate_rows.sort(
            key=lambda row: float(row.get("discovery_priority_score", 0.0) or 0.0),
            reverse=True,
        )
        if discovery_rank_limit > 0 and len(candidate_rows) > discovery_rank_limit:
            for row in candidate_rows[discovery_rank_limit:]:
                skipped_rows.append(
                    {
                        "symbol": row.get("symbol") or "",
                        "candidate_source_plan": row.get("candidate_source_plan") or "",
                        "status": "trimmed",
                        "reasons": ["discovery_rank_limit"],
                    }
                )
            candidate_rows = candidate_rows[:discovery_rank_limit]
        evaluated = len(candidate_rows)
        winners = sum(
            1
            for row in candidate_rows
            if float(row.get("synthetic_pnl", 0.0) or 0.0) > 0
        )
        losers = sum(
            1
            for row in candidate_rows
            if float(row.get("synthetic_pnl", 0.0) or 0.0) < 0
        )
        net_pnl = round(
            sum(float(row.get("synthetic_pnl", 0.0) or 0.0) for row in candidate_rows),
            2,
        )
        avg_return_pct = round(
            (
                sum(float(row.get("return_pct", 0.0) or 0.0) for row in candidate_rows)
                / evaluated
            )
            if evaluated
            else 0.0,
            4,
        )
        return {
            "mode": "discovery",
            "variant": self.strategy_variant,
            "candidate_pool_size": len(pool),
            "baseline_symbols_total": len(baseline_symbols),
            "synthetic_candidates_evaluated": evaluated,
            "added_winners": winners,
            "added_losers": losers,
            "net_synthetic_pnl": net_pnl,
            "avg_return_pct": avg_return_pct,
            "slot_fill_delta": evaluated,
            "synthetic_notional_per_trade": 1000.0,
            "top_added_candidates": candidate_rows[:5],
            "blocked_or_skipped_candidates": skipped_rows[:10],
            "data_quality_notes": dict(data_quality_notes),
            "discovery_rank_limit": discovery_rank_limit,
        }

    def _evaluate_counterfactual_longs(
        self,
        *,
        dm: DayManager,
        signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
        actual_day_net_pnl: float = 0.0,
        capacity_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        signal_rows = [dict(row) for row in signals if isinstance(row, dict)]
        if not signal_rows:
            return {
                "mode": "long_counterfactual_v1",
                "candidate_pool_size": 0,
                "eligible_candidates_total": 0,
                "selected_candidates_total": 0,
                "selected_rank_limit": 0,
                "synthetic_default_notional": 0.0,
                "available_counterfactual_capital": 0.0,
                "available_counterfactual_slots": 0,
                "capacity_source": "",
                "net_synthetic_pnl_all": 0.0,
                "net_synthetic_pnl_selected": 0.0,
                "avg_return_pct_selected": 0.0,
                "estimated_day_net_pnl_selected": float(actual_day_net_pnl or 0.0),
                "added_winners_selected": 0,
                "added_losers_selected": 0,
                "top_added_candidates": [],
                "blocked_or_skipped_candidates": [],
            }

        executed_symbols = {
            str(row.get("symbol") or "").upper().strip()
            for row in decisions
            if isinstance(row, dict) and bool(row.get("actually_executed"))
        }
        executed_symbols.discard("")
        capacity = self._resolve_counterfactual_capacity(
            capacity_snapshot=capacity_snapshot or {},
            decisions=decisions,
            signals=signal_rows,
        )
        entry_start_time = datetime.combine(
            date.fromisoformat(self.session_date),
            time(8, 35),
            tzinfo=CT,
        )

        candidate_rows: List[Dict[str, Any]] = []
        blocked_rows: List[Dict[str, Any]] = []
        for raw in signal_rows:
            symbol = str(raw.get("ticker") or raw.get("symbol") or "").upper().strip()
            if not symbol or symbol in executed_symbols or symbol in {"SPY", "QQQ"}:
                continue

            signal_data = dict(raw)
            signal_data["ticker"] = symbol
            signal_data.setdefault("symbol", symbol)
            journal_metadata = self._best_journal_metadata(
                symbol=symbol,
                trade_journal=list(trade_journal),
            )
            if journal_metadata:
                for key in (
                    "score",
                    "final_score",
                    "realtime_score",
                    "plan_score_source",
                    "entry_source",
                    "source_bucket",
                    "strategy_name",
                    "setup_type",
                    "strategy_id",
                    "origin_entry_source",
                    "runtime_entry_context",
                    "override_reason",
                ):
                    value = journal_metadata.get(key)
                    if value not in (None, ""):
                        signal_data.setdefault(key, value)

            bars = self._bars_for_symbol(symbol)
            if bars is None or bars.empty:
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "status": "skipped",
                        "reasons": ["missing_bars"],
                    }
                )
                continue
            entry_slice = bars[bars.index >= pd.Timestamp(entry_start_time)]
            if entry_slice.empty:
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "status": "skipped",
                        "reasons": ["no_valid_entry_window"],
                    }
                )
                continue
            entry_bar = entry_slice.iloc[0]
            current_price = (
                _safe_float(entry_bar.get("open"))
                or _safe_float(entry_bar.get("close"))
                or 0.0
            )
            planned_entry = (
                _safe_float(signal_data.get("entry_price"))
                or _safe_float(signal_data.get("price"))
                or current_price
            )
            signal_data.setdefault(
                "prev_close",
                self.previous_closes.get(symbol),
            )
            entry_score = (
                _safe_float(signal_data.get("score"))
                or _safe_float(signal_data.get("final_score"))
                or _safe_float(signal_data.get("realtime_score"))
                or _safe_float(signal_data.get("confidence"))
                or 0.0
            )
            planned_notional = self._planned_signal_notional(signal_data)
            if planned_notional <= 0:
                inferred_qty = self._infer_counterfactual_qty_from_decisions(
                    signal_data,
                    decisions=decisions,
                )
                if inferred_qty > 0:
                    signal_data["qty"] = float(inferred_qty)
                    planned_notional = self._planned_signal_notional(
                        signal_data,
                        qty_override=inferred_qty,
                    )
            if planned_notional <= 0:
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "status": "blocked",
                        "reasons": ["missing_planned_size"],
                        "candidate_snapshot": {
                            "score": entry_score,
                            "entry_price": round(float(planned_entry or 0.0), 4),
                            "open_price": round(float(current_price or 0.0), 4),
                        },
                    }
                )
                continue

            self._current_replay_time = entry_start_time
            with self._patched_runtime_clock(), self._patched_inverse_screen():
                dm._refresh_entry_authority_state(self._positions)
                regime_blocked, regime_reason = dm._entries_blocked_by_regime(symbol)
                authority_contract = dm._resolve_entry_authority(signal_data)
                anchor_allowed, anchor_price, anchor_reason = (
                    dm._resolve_runtime_entry_anchor(
                        action=str(signal_data.get("action", "buy_open") or "buy_open"),
                        entry_price=float(planned_entry or current_price or 0.0),
                        current_price=float(current_price or 0.0),
                        prev_close=_safe_float(signal_data.get("prev_close")),
                        entry_score=float(entry_score or 0.0),
                        signal_data=signal_data,
                    )
                )

            replay_long_allowed = bool(
                (not regime_blocked)
                and authority_contract.get("eligible", False)
                and anchor_allowed
            )
            if not replay_long_allowed:
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "status": "blocked",
                        "reasons": [
                            reason
                            for reason in (
                                regime_reason,
                                str(authority_contract.get("reason", "") or ""),
                                anchor_reason,
                            )
                            if reason
                        ],
                        "candidate_snapshot": {
                            "score": entry_score,
                            "entry_price": round(float(planned_entry or 0.0), 4),
                            "open_price": round(float(current_price or 0.0), 4),
                        },
                    }
                )
                continue

            simulation = self._simulate_counterfactual_management_trade(
                dm=dm,
                candidate=signal_data,
                entry_start_time=entry_start_time,
                entry_price_override=anchor_price,
            )
            row = {
                "symbol": symbol,
                "candidate_source_plan": str(
                    signal_data.get("_candidate_source_plan") or ""
                ),
                "score": round(float(entry_score or 0.0), 4),
                "entry_source": str(signal_data.get("entry_source") or ""),
                "setup_type": str(signal_data.get("setup_type") or ""),
                "size_source": str(
                    signal_data.get("_replay_inferred_size_source") or "planned_signal"
                ),
                "inferred_notional": round(
                    float(
                        signal_data.get("_replay_inferred_notional", planned_notional)
                        or 0.0
                    ),
                    2,
                ),
                "replay_regime_reason": regime_reason,
                "replay_authority_reason": str(
                    authority_contract.get("reason", "") or ""
                ),
                "runtime_anchor_reason": anchor_reason,
                "runtime_anchor_price": round(float(anchor_price or 0.0), 4),
            }
            row.update(simulation)
            if simulation.get("status") != "evaluated":
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "status": str(simulation.get("status") or "skipped"),
                        "reasons": list(simulation.get("reasons") or []),
                    }
                )
                continue
            candidate_rows.append(row)

        candidate_rows.sort(
            key=lambda row: (
                -float(row.get("score", 0.0) or 0.0),
                -float(row.get("synthetic_pnl", 0.0) or 0.0),
                row.get("symbol", ""),
            )
        )
        remaining_capital = float(capacity.get("available_capital", 0.0) or 0.0)
        remaining_slots = int(capacity.get("available_slots", 0) or 0)
        selected_rows: List[Dict[str, Any]] = []
        if remaining_capital > 0 and remaining_slots > 0:
            for row in candidate_rows:
                trade_notional = float(row.get("synthetic_notional", 0.0) or 0.0)
                if trade_notional <= 0:
                    continue
                if trade_notional > remaining_capital + 0.01:
                    blocked_rows.append(
                        {
                            "symbol": str(row.get("symbol") or ""),
                            "status": "blocked",
                            "reasons": ["insufficient_verified_capital"],
                            "candidate_snapshot": {
                                "score": row.get("score"),
                                "synthetic_notional": round(trade_notional, 2),
                            },
                        }
                    )
                    continue
                selected_rows.append(row)
                remaining_capital = round(
                    max(0.0, remaining_capital - trade_notional), 2
                )
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break
        net_all = round(
            sum(float(row.get("synthetic_pnl", 0.0) or 0.0) for row in candidate_rows),
            2,
        )
        net_selected = round(
            sum(float(row.get("synthetic_pnl", 0.0) or 0.0) for row in selected_rows),
            2,
        )
        avg_return_selected = round(
            (
                sum(float(row.get("return_pct", 0.0) or 0.0) for row in selected_rows)
                / len(selected_rows)
            )
            if selected_rows
            else 0.0,
            4,
        )
        return {
            "mode": "long_counterfactual_v1",
            "candidate_pool_size": len(signal_rows),
            "eligible_candidates_total": len(candidate_rows),
            "selected_candidates_total": len(selected_rows),
            "selected_rank_limit": int(capacity.get("available_slots", 0) or 0),
            "synthetic_default_notional": 0.0,
            "available_counterfactual_capital": float(
                capacity.get("available_capital", 0.0) or 0.0
            ),
            "available_counterfactual_slots": int(
                capacity.get("available_slots", 0) or 0
            ),
            "capacity_source": str(capacity.get("source") or ""),
            "capacity_mode": str(capacity.get("mode") or "additive"),
            "capacity_snapshot_stale": bool(capacity.get("stale_snapshot", False)),
            "capacity_stale_reason": str(capacity.get("stale_reason") or ""),
            "reference_account_buying_power": float(
                capacity.get("reference_account_buying_power", 0.0) or 0.0
            ),
            "net_synthetic_pnl_all": float(net_all),
            "net_synthetic_pnl_selected": float(net_selected),
            "avg_return_pct_selected": float(avg_return_selected),
            "estimated_day_net_pnl_selected": round(
                float(actual_day_net_pnl or 0.0) + net_selected,
                2,
            )
            if str(capacity.get("mode") or "additive") == "additive"
            else round(float(actual_day_net_pnl or 0.0), 2),
            "replacement_budget_net_pnl_selected": round(float(net_selected), 2)
            if str(capacity.get("mode") or "") == "replacement_budget"
            else 0.0,
            "added_winners_selected": sum(
                1
                for row in selected_rows
                if float(row.get("synthetic_pnl", 0.0) or 0.0) > 0
            ),
            "added_losers_selected": sum(
                1
                for row in selected_rows
                if float(row.get("synthetic_pnl", 0.0) or 0.0) < 0
            ),
            "top_added_candidates": selected_rows[:10],
            "blocked_or_skipped_candidates": blocked_rows[:15],
        }

    def _ensure_market_data(self) -> None:
        missing_symbols = [
            symbol
            for symbol in self.benchmark_symbols
            if symbol not in self.market_bars
        ]
        if missing_symbols:
            self._load_archive_symbols(missing_symbols)
            missing_symbols = [
                symbol
                for symbol in self.benchmark_symbols
                if symbol not in self.market_bars
            ]
        if missing_symbols:
            if not self._archive_live_fallback_enabled():
                raise RuntimeError(
                    f"Missing benchmark minute bars in archive: {', '.join(missing_symbols)}"
                )
            client = self.data_client or create_data_client(require_credentials=True)
            for symbol in missing_symbols:
                self.market_bars[symbol] = self._fetch_minute_bars(client, symbol)
                self._record_live_fetch(symbol)
        missing_closes = [
            symbol
            for symbol in self.benchmark_symbols
            if symbol not in self.previous_closes
        ]
        if missing_closes:
            client = self.data_client or create_data_client(require_credentials=True)
            for symbol in missing_closes:
                close_value = self._fetch_previous_close(client, symbol)
                if close_value is not None:
                    self.previous_closes[symbol] = close_value
        unresolved = [
            symbol
            for symbol in self.benchmark_symbols
            if symbol not in self.previous_closes
        ]
        if unresolved:
            raise RuntimeError(
                f"Missing previous-close data for benchmarks: {', '.join(unresolved)}"
            )

    def _fetch_minute_bars(self, client: Any, symbol: str) -> pd.DataFrame:
        return self._fetch_bars_for_window(
            client,
            symbol,
            start_et=time(9, 30),
            end_et=time(16, 0),
        )

    def _fetch_extended_minute_bars(self, client: Any, symbol: str) -> pd.DataFrame:
        return self._fetch_bars_for_window(
            client,
            symbol,
            start_et=time(4, 0),
            end_et=time(16, 0),
        )

    def _fetch_bars_for_window(
        self,
        client: Any,
        symbol: str,
        *,
        start_et: time,
        end_et: time,
    ) -> pd.DataFrame:
        day_value = date.fromisoformat(self.session_date)
        start_utc = datetime.combine(day_value, start_et, tzinfo=ET).astimezone(UTC)
        end_utc = datetime.combine(day_value, end_et, tzinfo=ET).astimezone(UTC)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_utc,
            end=end_utc,
        )
        bars = client.get_stock_bars(request)
        if bars.df.empty:
            raise RuntimeError(
                f"No minute bars available for {symbol} on {self.session_date}"
            )
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            if symbol not in df.index.get_level_values(0):
                raise RuntimeError(f"Minute bars missing symbol slice for {symbol}")
            df = df.loc[symbol]
        return self._normalize_bars_frame(df)

    def _fetch_previous_close(self, client: Any, symbol: str) -> Optional[float]:
        day_value = date.fromisoformat(self.session_date)
        start_utc = datetime.combine(
            day_value - timedelta(days=10), time(0, 0), tzinfo=UTC
        )
        end_utc = datetime.combine(day_value, time(0, 0), tzinfo=UTC)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start_utc,
            end=end_utc,
        )
        bars = client.get_stock_bars(request)
        if bars.df.empty:
            return None
        df = bars.df
        if isinstance(df.index, pd.MultiIndex):
            if symbol not in df.index.get_level_values(0):
                return None
            df = df.loc[symbol]
        df = df.sort_index()
        prior = df[df.index < pd.Timestamp(day_value, tz=UTC)]
        if prior.empty:
            return None
        return float(prior.iloc[-1]["close"])

    def _normalize_bars_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        df = frame.copy()
        if df.empty:
            return df
        index = pd.DatetimeIndex(df.index)
        if index.tz is None:
            index = index.tz_localize(UTC)
        df.index = index.tz_convert(CT)
        return df.sort_index()

    def _current_price_for_symbol(self, symbol: str) -> Optional[float]:
        replay_time = self._current_replay_time
        if replay_time is None:
            return None
        df = self.market_bars.get(str(symbol or "").upper())
        if df is None or df.empty:
            return None
        sliced = df[df.index <= pd.Timestamp(replay_time)]
        if sliced.empty:
            return None
        row = sliced.iloc[-1]
        for key in ("close", "vwap", "open"):
            if key in row and pd.notna(row[key]):
                return float(row[key])
        return None

    def _load_inverse_universe_rows(self) -> List[Dict[str, Any]]:
        if self._inverse_universe_rows is None:
            from autotrade.utils.financial_db import FinancialDB

            db = FinancialDB()
            self._inverse_universe_rows = [
                dict(row) for row in db.get_all_inverse_etfs(active_only=True) or []
            ]
        return list(self._inverse_universe_rows)

    def _historical_inverse_bars(self, ticker: str) -> Optional[pd.DataFrame]:
        replay_time = self._current_replay_time
        if replay_time is None:
            return None
        symbol = str(ticker or "").upper()
        if symbol not in self.market_bars:
            self._load_archive_symbols([symbol])
        if symbol not in self.market_bars:
            if not self._archive_live_fallback_enabled():
                return None
            client = self.data_client or create_data_client(require_credentials=True)
            try:
                self.market_bars[symbol] = self._fetch_extended_minute_bars(
                    client, symbol
                )
                self._record_live_fetch(symbol)
            except Exception:
                return None
        df = self.market_bars.get(symbol)
        if df is None or df.empty:
            return None
        sliced = df[df.index <= pd.Timestamp(replay_time)]
        if sliced.empty:
            return None
        bars_5m = (
            sliced.resample("5min")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
        )
        return bars_5m.tail(80)

    @staticmethod
    def _coerce_replay_market_regime(value: Any) -> MarketRegime:
        text = str(value or "neutral").strip().lower().replace("-", "_")
        for regime in MarketRegime:
            if text in {regime.name.lower(), regime.value.lower()}:
                return regime
        aliases = {
            "sell_off": MarketRegime.SELLOFF,
            "risk_off": MarketRegime.SELLOFF,
            "bearish": MarketRegime.SELLOFF,
            "chop": MarketRegime.NEUTRAL,
        }
        return aliases.get(text, MarketRegime.NEUTRAL)

    @staticmethod
    def _sector_bias_from_replay_analysis(analysis: RegimeAnalysis) -> Dict[str, float]:
        momentum = dict(getattr(analysis, "sector_momentum", {}) or {})
        if not momentum:
            return {}
        ranked = sorted(momentum.items(), key=lambda item: float(item[1]), reverse=True)
        leaders = {name for name, _ in ranked[:3]}
        laggards = {name for name, _ in ranked[-3:]} if len(ranked) >= 3 else set()
        return {
            str(name): 15.0 if name in leaders else (-20.0 if name in laggards else 0.0)
            for name in momentum
        }

    def _build_replay_regime_analysis(
        self,
        *,
        resolved_regime: Dict[str, Any],
        artifacts: Dict[str, Any],
    ) -> RegimeAnalysis:
        plan_payload = dict((artifacts or {}).get("plan_payload") or {})
        regime_payload = dict(plan_payload.get("current_regime_analysis") or {})
        regime_payload.update(dict(plan_payload.get("regime_analysis") or {}))
        regime_payload.update(dict(resolved_regime or {}))
        sector_momentum = (
            regime_payload.get("sector_momentum")
            or plan_payload.get("sector_momentum")
            or {}
        )
        if not isinstance(sector_momentum, dict):
            sector_momentum = {}
        leading_sectors = regime_payload.get("leading_sectors") or []
        lagging_sectors = regime_payload.get("lagging_sectors") or []
        return RegimeAnalysis(
            regime=self._coerce_replay_market_regime(regime_payload.get("regime")),
            confidence=float(regime_payload.get("confidence") or 0.0),
            breadth_pct_positive=float(
                regime_payload.get("breadth_pct_positive")
                or regime_payload.get("breadth_pct")
                or 50.0
            ),
            breadth_5d_trend=str(regime_payload.get("breadth_5d_trend") or "stable"),
            avg_universe_return_1d=float(
                regime_payload.get("avg_universe_return_1d")
                or regime_payload.get("avg_return_1d")
                or 0.0
            ),
            avg_universe_return_5d=float(
                regime_payload.get("avg_universe_return_5d")
                or regime_payload.get("avg_return_5d")
                or 0.0
            ),
            top_decile_return=float(regime_payload.get("top_decile_return") or 0.0),
            bottom_decile_return=float(
                regime_payload.get("bottom_decile_return") or 0.0
            ),
            dispersion=float(regime_payload.get("dispersion") or 0.0),
            consecutive_down_days=int(regime_payload.get("consecutive_down_days") or 0),
            consecutive_up_days=int(regime_payload.get("consecutive_up_days") or 0),
            volume_trend=str(regime_payload.get("volume_trend") or "stable"),
            leading_sectors=[str(value) for value in leading_sectors],
            lagging_sectors=[str(value) for value in lagging_sectors],
            sector_momentum={
                str(key).strip().lower().replace(" ", "_"): float(value)
                for key, value in sector_momentum.items()
                if str(key).strip()
            },
            recommended_strategy=dict(regime_payload.get("recommended_strategy") or {}),
            pattern_detected=regime_payload.get("pattern_detected"),
            sources_degraded=bool(regime_payload.get("sources_degraded", False)),
            stale_session=bool(regime_payload.get("stale_session", False)),
        )

    def _build_replay_day_manager(
        self,
        *,
        signals: List[Dict[str, Any]],
        resolved_regime: Dict[str, Any],
        artifacts: Optional[Dict[str, Any]] = None,
        trading_client: Any = None,
    ) -> DayManager:
        # Patch the clock before initialization so __init__ and _load_signals
        # use the correct replay date.
        with self._patched_runtime_clock_for(
            _coerce_local_timestamp(self.session_date).replace(hour=8, minute=30)
        ):
            dm = DayManager(dry_run=True)

        # Force dry_run=False so DM walks its real submission path. The
        # execution adapter is patched to self._capture_inverse_order (a
        # no-op recorder) below, so "submitting" never reaches a broker —
        # but the orders still flow through DM's real gating logic, which
        # is what we need to capture for replay accounting.
        dm.dry_run = False
        dm.signals = list(signals)
        dm._artifact_session_date_override = self.session_date
        replay_artifacts = artifacts or {}
        active_plan_path = replay_artifacts.get("plan_path") or replay_artifacts.get(
            "pm_plan_path"
        )
        dm._active_plan_path = Path(active_plan_path) if active_plan_path else None
        dm._active_plan_payload = dict(replay_artifacts.get("plan_payload") or {})
        dm.signal_status = dm._init_signal_tracking(dm.signals)
        dm._initial_watchlist_tickers = {
            str(signal.get("ticker") or signal.get("symbol") or "").upper()
            for signal in signals
            if str(signal.get("ticker") or signal.get("symbol") or "").strip()
        }
        dm._watched_universe_tickers = set(dm._initial_watchlist_tickers)
        dm.youtube_context = {"resolved_regime": dict(resolved_regime)}
        dm.current_regime_analysis = self._build_replay_regime_analysis(
            resolved_regime=resolved_regime,
            artifacts=replay_artifacts,
        )
        dm.current_regime = getattr(dm.current_regime_analysis, "regime", None)
        dm.live_sector_bias = self._sector_bias_from_replay_analysis(
            dm.current_regime_analysis
        )
        dm.inverse_etf_manager = SimpleNamespace(
            get_instrument_profile=self.inverse_etf_manager.get_instrument_profile,
            evaluate_intraday_reversal=self.inverse_etf_manager.evaluate_intraday_reversal,
            check_hedge_conditions=self.inverse_etf_manager.check_hedge_conditions,
            calculate_current_allocation=self.inverse_etf_manager.calculate_current_allocation,
            rebalance_hedge=self.inverse_etf_manager.rebalance_hedge,
            check_hedge_exit=self.inverse_etf_manager.check_hedge_exit,
            manage_hedge_stop_loss=self.inverse_etf_manager.manage_hedge_stop_loss,
            select_hedge_instrument=self.inverse_etf_manager.select_hedge_instrument,
            calculate_hedge_size=self.inverse_etf_manager.calculate_hedge_size,
        )

        dm.position_health = {}
        dm._advisor_eval_cache = {}
        dm._advisor_eval_cache_lock = threading.RLock()
        dm.execution_accounting = ExecutionAccounting()
        dm.research_score_threshold_penalty = 0.0
        dm.research_position_size_multiplier = 1.0
        dm.research_freshness = {
            "status": "fresh",
            "age_hours": 0.0,
            "workflow_complete": True,
            "workflow_reason": "replay",
            "freshness_basis": "replay_session",
            "target_trade_date": self.session_date,
            "current_session_trade_date": self.session_date,
            "session_aligned": True,
        }
        dm.research_age_hours = 0.0
        replay_research_freshness = dict(dm.research_freshness)
        dm._check_research_freshness = lambda *args, **kwargs: dict(
            replay_research_freshness
        )
        if getattr(dm, "signal_pipeline", None):
            dm.signal_pipeline.run = lambda *args, **kwargs: list(dm.signals)
        dm._sequential_shadow_enabled = False
        dm._entry_time_overrides = {}
        dm._entry_time_for_advisor = lambda symbol: dm._entry_time_overrides.get(
            str(symbol or "").upper()
        )
        dm._apply_policy_risk_overlay = lambda position, health: dict(health)
        dm._core_data_readiness = {
            "is_fresh": True,
            "core_data_fresh": True,
            "pm_ready_for_execution": True,
            "primary_date": self.session_date,
            "blocking_reasons": [],
        }
        dm._benchmark_snapshot_cache = {}
        dm._entry_authority_state = {
            "session_date": self.session_date,
            "state": "open",
            "reason": "replay_start",
            "updated_at": None,
            "snapshot": {},
            "inverse_fast_entries_taken": 0,
            "safety_reentry_refresh": None,
        }
        dm._inverse_fast_entry_symbols = set()
        dm._last_defensive_screen = None
        dm._hedge_decision_state = {
            "session_date": self.session_date,
            "last_action": "",
            "last_symbol": "",
            "last_target_notional": 0.0,
            "last_decision_at": None,
            "cooldown_until": None,
            "open_order_ids": {},
            "decision_source": "",
        }
        dm.data_client = self.data_client
        dm._load_resolved_regime_context = lambda: dict(resolved_regime)
        dm._entry_authority_session_date_override = self.session_date
        dm._entry_authority_force_modern_session = self.force_modern_crash_protection
        if self.skip_position_advisor:
            dm.position_advisor = None
            dm.use_agentic = False
        dm.get_positions = lambda: self._mark_to_market_positions()
        dm.get_current_price = lambda ticker: self._current_price_for_symbol(ticker)
        dm._get_previous_close = lambda symbol: self.previous_closes.get(
            str(symbol or "").upper()
        )
        dm.get_current_phase = lambda now=None: self._phase_for_time(
            now or self._current_replay_time or _session_minutes(self.session_date)[0]
        )
        dm._has_open_buy_order = lambda symbol: (False, "")
        dm._record_hedge_decision = lambda **kwargs: None
        dm._capture_defensive_orders_in_dry_run = True
        dm._submit_order_via_execution_adapter = self._capture_inverse_order
        # Replay should not trigger live full-universe scanner paths. They are slow,
        # nondeterministic against shared local state, and can crash the baseline run.
        dm.watchlist_rotator = None
        dm.rotation_scheduler = None
        dm.universe_scanner = None
        dm.vwap_universe_scanner = None
        dm._intraday_reserved_scan_enabled = False
        dm._strength_reentry_scan_enabled = False
        dm.batch_rotate = lambda dropped_slots, held_tickers: 0
        dm.scan_watchlist_movers = lambda held_tickers: []
        dm.scan_power_hour_opportunities = lambda: []
        dm._scan_vwap_universe = lambda held_tickers: []
        dm._scan_strength_reentry_candidates = lambda held_tickers: []
        dm._scan_intraday_reserved_candidates = lambda held_tickers, **kwargs: []
        dm._enrich_signals_with_alpha_zoo = lambda signals, regime=None: signals
        dm._vl_confirm_entry = lambda **kwargs: (True, "replay_bypass")
        dm.client = trading_client or SimpleNamespace(
            get_account=lambda: SimpleNamespace(
                equity=100000.0,
                buying_power=100000.0,
                cash=100000.0,
                last_equity=100000.0,
                day_trade_count=0,
                pattern_day_trader=False,
            ),
            get_orders=lambda *args, **kwargs: [],
            cancel_order_by_id=lambda *args, **kwargs: True,
        )

        def _replay_should_run_advisor(
            self, symbol, pnl_pct, current_price, hold_minutes
        ):
            if self.position_advisor is None:
                return False
            cached = DayManager._get_cached_advisor_health(self, symbol)
            if cached and str(cached.get("action") or "").lower() in {"hold", "watch"}:
                return False
            return DayManager._should_run_advisor(
                self,
                symbol,
                pnl_pct,
                current_price,
                hold_minutes,
            )

        dm._should_run_advisor = MethodType(_replay_should_run_advisor, dm)

        # BUGFIX: Ensure DayManager uses a clean trade journal during replay.
        # Otherwise it loads today's trades from disk and blocks re-entry for those symbols.
        import autotrade.signals.trade_learner as trade_learner

        dm.trade_journal = trade_learner.TradeJournal()
        dm.trade_journal.trades = []

        return dm

    def _should_run_replay_daymanager_cycle(self, replay_minute: datetime) -> bool:
        if self.mode != REPLAY_MODE_DAYMANAGER_CORE:
            return False
        if self.profile != REPLAY_PROFILE_FAST:
            return False
        phase = self._phase_for_time(replay_minute)
        minute_of_day = int(replay_minute.hour * 60 + replay_minute.minute)
        if phase in (TradingPhase.OBSERVATION, TradingPhase.RESEARCH):
            return replay_minute.minute % 5 == 0
        if phase == TradingPhase.CORE_TRADING and minute_of_day <= (10 * 60):
            return replay_minute.minute % 5 == 0
        if phase == TradingPhase.CORE_TRADING and replay_minute.minute == 0:
            return True
        return False

    def _run_replay_daymanager_cycle(
        self,
        *,
        dm: DayManager,
        replay_minute: datetime,
    ) -> Dict[str, Any]:
        original_dry_run = bool(getattr(dm, "dry_run", True))
        # Force dry_run=False so DM calls its (patched) execution adapter
        # to record simulated trades into self._replay_orders.
        dm.dry_run = False
        try:
            with (
                self._patched_runtime_clock_for(replay_minute),
                self._patched_inverse_screen(),
                self._patched_intraday_provider(),
            ):
                return dict(dm.run_cycle() or {})
        finally:
            dm.dry_run = original_dry_run

    def _seed_workflow_positions(
        self,
        *,
        decisions: Sequence[Dict[str, Any]],
        workflow_journal: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        del trade_journal
        seeded: Dict[str, Dict[str, Any]] = {}
        executed_today = {
            str(row.get("symbol") or "").upper().strip()
            for row in decisions
            if isinstance(row, dict)
            and bool(row.get("actually_executed"))
            and str(row.get("symbol") or "").strip()
        }
        ordered_rows = self._initial_workflow_position_rows(workflow_journal)
        for row in ordered_rows:
            position = dict(row.get("position") or {})
            symbol = (
                str(row.get("symbol") or position.get("symbol") or "").upper().strip()
            )
            if not symbol or symbol in seeded or symbol in executed_today:
                continue
            qty = int(float(position.get("qty", 0) or 0))
            entry_price = float(
                position.get(
                    "entry_price",
                    position.get("avg_entry", position.get("avg_entry_price", 0.0)),
                )
                or 0.0
            )
            if qty <= 0 or entry_price <= 0:
                continue
            seeded[symbol] = {
                "symbol": symbol,
                "qty": qty,
                "avg_entry": entry_price,
                "avg_entry_price": entry_price,
                "entry_price": entry_price,
                "current_price": float(
                    position.get("current_price", entry_price) or entry_price
                ),
                "market_value": float(
                    position.get("market_value", qty * entry_price)
                    or (qty * entry_price)
                ),
                "unrealized_pnl": float(
                    position.get("unrealized_pnl", position.get("unrealized_pl", 0.0))
                    or 0.0
                ),
                "unrealized_pl": float(
                    position.get("unrealized_pl", position.get("unrealized_pnl", 0.0))
                    or 0.0
                ),
                "pnl_pct": float(
                    position.get(
                        "pnl_pct",
                        position.get(
                            "unrealized_pnl_pct",
                            position.get("unrealized_plpc", 0.0),
                        ),
                    )
                    or 0.0
                ),
                "unrealized_plpc": float(
                    position.get(
                        "unrealized_plpc",
                        position.get(
                            "unrealized_pnl_pct", position.get("pnl_pct", 0.0)
                        ),
                    )
                    or 0.0
                ),
                "cost_basis": float(
                    position.get("cost_basis", qty * entry_price) or (qty * entry_price)
                ),
                "side": "long",
            }
        return list(seeded.values())

    def _initial_workflow_position_rows(
        self,
        workflow_journal: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reconstruct the morning position roster from the workflow journal.

        Production journal events emit position snapshots in sweeps as the
        position scheduler ticks each holding. The previous "break on first
        repeat" approach only captured one sweep (e.g., 12 of 16 morning
        positions on 04-24) and undercounted the seed roster. With too few
        seeded positions, the replay starts with too much capacity and
        over-enters (180 phantom entries vs 16 real on 04-24).

        Use a time-window approach instead: take the FIRST row per symbol
        seen during the morning window (session open + 60 minutes). After
        that window, new symbols are intra-day entries, not initial holdings.
        """
        rows = sorted(
            [
                row
                for row in workflow_journal
                if isinstance(row, dict)
                and isinstance(row.get("position"), dict)
                and str(
                    row.get("symbol") or row.get("position", {}).get("symbol") or ""
                ).strip()
            ],
            key=lambda row: str(row.get("timestamp") or ""),
        )
        if not rows:
            return []

        # Filter for rows after 04:00 AM local time to avoid midnight ghosts
        # that often populate from stale caches during background syncs.
        filtered_rows = []
        for row in rows:
            try:
                row_ts = _coerce_local_timestamp(str(row.get("timestamp") or ""))
                if row_ts is not None and row_ts.hour >= 4:
                    filtered_rows.append(row)
            except Exception:
                continue

        if not filtered_rows:
            # Fallback to original rows if nothing found after 4 AM,
            # though this suggests a very sparse or broken journal.
            filtered_rows = rows

        # Anchor the morning window to the first valid journal timestamp
        try:
            first_ts = _coerce_local_timestamp(
                str(filtered_rows[0].get("timestamp") or "")
            )
        except Exception:
            first_ts = None
        morning_cutoff = (
            first_ts + timedelta(minutes=60) if first_ts is not None else None
        )
        initial_rows: List[Dict[str, Any]] = []
        seen_symbols: set[str] = set()
        for row in filtered_rows:
            symbol = (
                str(row.get("symbol") or row.get("position", {}).get("symbol") or "")
                .upper()
                .strip()
            )
            if not symbol or symbol in seen_symbols:
                continue
            if morning_cutoff is not None:
                try:
                    row_ts = _coerce_local_timestamp(str(row.get("timestamp") or ""))
                except Exception:
                    row_ts = None
                if row_ts is not None and row_ts > morning_cutoff:
                    break
            seen_symbols.add(symbol)
            initial_rows.append(row)
        return initial_rows

    def _prepare_workflow_workspace(self, artifacts: Dict[str, Any]) -> Path:
        workspace_root = Path(
            tempfile.mkdtemp(
                prefix=f"runtime_replay_{self.session_date.replace('-', '')}_"
            )
        )
        for dirname in ("logs", "plans", "data", "research", "charts", "tools"):
            (workspace_root / dirname).mkdir(parents=True, exist_ok=True)
        for key in (
            "signals_path",
            "signals_today_path",
            "signals_yesterday_path",
            "decision_path",
            "workflow_journal_path",
            "trade_journal_path",
            "adjusted_plan_path",
            "morning_plan_path",
            "pm_plan_path",
            "plan_path",
        ):
            path = artifacts.get(key)
            if path is None:
                continue
            source = Path(path)
            if not source.exists():
                continue
            target_dir = workspace_root / (
                "plans" if source.parent == self.plans_dir else "logs"
            )
            shutil.copy2(source, target_dir / source.name)
        compact = self.session_date.replace("-", "")
        exec_state_path = self.plans_dir / f".execution_state_{compact}.json"
        if exec_state_path.exists():
            exec_state_payload = _load_json(exec_state_path) or {}
            _write_json(
                workspace_root / "plans" / exec_state_path.name,
                self._sanitize_replay_execution_state(exec_state_payload),
            )
        overnight_state = self.project_dir / "research" / "overnight_state.json"
        if overnight_state.exists():
            shutil.copy2(
                overnight_state, workspace_root / "research" / overnight_state.name
            )
        return workspace_root

    @staticmethod
    def _planned_signal_notional(
        row: Dict[str, Any], *, qty_override: Optional[float] = None
    ) -> float:
        if not isinstance(row, dict):
            return 0.0
        price = float(row.get("entry_price") or row.get("planned_entry_price") or 0.0)
        qty = float(
            qty_override
            if qty_override is not None
            else (row.get("adjusted_qty") or row.get("qty") or 0.0)
        )
        if price <= 0 or qty <= 0:
            return 0.0
        return float(price * qty)

    @staticmethod
    def _infer_counterfactual_qty_from_decisions(
        row: Dict[str, Any],
        *,
        decisions: Sequence[Dict[str, Any]],
    ) -> float:
        if not isinstance(row, dict):
            return 0.0
        entry_price = float(
            row.get("entry_price") or row.get("planned_entry_price") or 0.0
        )
        if entry_price <= 0:
            return 0.0

        target_entry_source = str(row.get("entry_source") or "").strip().lower()
        target_source_bucket = str(row.get("source_bucket") or "").strip().lower()
        target_setup_type = str(row.get("setup_type") or "").strip().lower()

        buckets: Dict[str, List[float]] = {
            "exact": [],
            "entry_source": [],
            "all": [],
        }
        for decision in decisions:
            if not isinstance(decision, dict) or not bool(
                decision.get("actually_executed")
            ):
                continue
            decision_price = _safe_float(
                decision.get("planned_price")
                or decision.get("current_price")
                or decision.get("entry_price")
            )
            decision_qty = _safe_float(
                decision.get("adjusted_qty")
                or decision.get("planned_qty")
                or decision.get("qty")
            )
            decision_price = float(decision_price or 0.0)
            decision_qty = float(decision_qty or 0.0)
            if decision_price <= 0 or decision_qty <= 0:
                continue
            executed_notional = float(decision_price * decision_qty)
            buckets["all"].append(executed_notional)

            decision_entry_source = (
                str(decision.get("entry_source") or "").strip().lower()
            )
            decision_source_bucket = (
                str(decision.get("source_bucket") or "").strip().lower()
            )
            decision_setup_type = str(decision.get("setup_type") or "").strip().lower()
            if (
                target_entry_source
                and target_entry_source == decision_entry_source
                and (
                    not target_source_bucket
                    or target_source_bucket == decision_source_bucket
                )
                and (not target_setup_type or target_setup_type == decision_setup_type)
            ):
                buckets["exact"].append(executed_notional)
            if target_entry_source and target_entry_source == decision_entry_source:
                buckets["entry_source"].append(executed_notional)

        sample: List[float] = []
        inferred_from = ""
        for label in ("exact", "entry_source", "all"):
            if buckets[label]:
                sample = buckets[label]
                inferred_from = label
                break
        if not sample:
            return 0.0
        median_notional = float(statistics.median(sample))
        inferred_qty = max(1.0, math.floor(median_notional / entry_price))
        row["_replay_inferred_qty"] = float(inferred_qty)
        row["_replay_inferred_notional"] = round(float(inferred_qty * entry_price), 2)
        row["_replay_inferred_size_source"] = (
            f"executed_decision_{inferred_from}_median"
        )
        return float(inferred_qty)

    @staticmethod
    def _extract_counterfactual_capacity_snapshot(
        plan_payload: Dict[str, Any],
        pm_plan_payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        review = dict(
            ((plan_payload or {}).get("decision_claw") or {}).get("review") or {}
        )
        evidence = dict(review.get("evidence_summary") or {})
        deployment_request = dict(review.get("deployment_request") or {})
        pm_plan_payload = dict(pm_plan_payload or {})
        pm_account = dict(pm_plan_payload.get("account") or {})

        def _to_float(value: Any, default: float = 0.0) -> float:
            try:
                return float(value)
            except Exception:
                return float(default)

        def _to_int(value: Any, default: int = 0) -> int:
            try:
                return int(float(value))
            except Exception:
                return int(default)

        open_slots = _to_int(evidence.get("open_slots"), 0)
        redeploy_entry_cap = _to_int(
            evidence.get("redeployment_entry_cap", deployment_request.get("entry_cap")),
            0,
        )
        if redeploy_entry_cap > 0:
            open_slots = (
                min(open_slots, redeploy_entry_cap)
                if open_slots > 0
                else redeploy_entry_cap
            )
        return {
            "available_capital": round(_to_float(evidence.get("free_capital"), 0.0), 2),
            "available_slots": max(0, open_slots),
            "source": "decision_claw_review",
            "phase_agent": str(
                review.get("phase_agent") or evidence.get("phase_agent") or ""
            ),
            "positions_count": _to_int(evidence.get("positions_count"), 0),
            "current_positions_count": len(
                list(evidence.get("current_positions") or [])
            ),
            "reference_account_buying_power": round(
                _to_float(
                    pm_account.get("buying_power")
                    or pm_plan_payload.get("buying_power")
                    or pm_account.get("cash")
                    or pm_plan_payload.get("cash"),
                    0.0,
                ),
                2,
            ),
        }

    def _resolve_counterfactual_capacity(
        self,
        *,
        capacity_snapshot: Dict[str, Any],
        decisions: Sequence[Dict[str, Any]],
        signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        snapshot = dict(capacity_snapshot or {})
        try:
            available_capital = round(
                float(snapshot.get("available_capital", 0.0) or 0.0), 2
            )
        except Exception:
            available_capital = 0.0
        try:
            available_slots = max(
                0, int(float(snapshot.get("available_slots", 0) or 0))
            )
        except Exception:
            available_slots = 0
        phase_agent = str(snapshot.get("phase_agent") or "").strip().lower()
        positions_count = int(snapshot.get("positions_count", 0) or 0)
        current_positions_count = int(snapshot.get("current_positions_count", 0) or 0)
        stale_zero_snapshot = (
            available_capital <= 0.0
            and available_slots > 0
            and positions_count == 0
            and current_positions_count == 0
            and phase_agent in {"premarket", "overnight", "morning"}
        )
        reference_account_buying_power = round(
            float(snapshot.get("reference_account_buying_power", 0.0) or 0.0),
            2,
        )
        if stale_zero_snapshot:
            fallback = self._fallback_counterfactual_capacity_from_decisions(
                decisions=decisions,
                signals=signals,
            )
            if (
                fallback.get("available_capital", 0.0) > 0
                and fallback.get("available_slots", 0) > 0
            ):
                fallback["stale_snapshot"] = True
                fallback["stale_reason"] = "premarket_open_slots_with_zero_free_capital"
                fallback["reference_account_buying_power"] = (
                    reference_account_buying_power
                )
                return fallback
            if reference_account_buying_power > 0 and available_slots > 0:
                return {
                    "available_capital": float(reference_account_buying_power),
                    "available_slots": int(max(0, available_slots)),
                    "source": "pm_plan_account_reference",
                    "mode": "additive",
                    "stale_snapshot": True,
                    "stale_reason": "premarket_open_slots_with_zero_free_capital",
                    "reference_account_buying_power": reference_account_buying_power,
                }
        return {
            "available_capital": float(max(0.0, available_capital)),
            "available_slots": int(max(0, available_slots)),
            "source": str(snapshot.get("source") or "unverified_capacity"),
            "mode": "additive",
            "stale_snapshot": False,
            "stale_reason": "",
            "reference_account_buying_power": reference_account_buying_power,
        }

    def _fallback_counterfactual_capacity_from_decisions(
        self,
        *,
        decisions: Sequence[Dict[str, Any]],
        signals: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        signal_by_symbol = {
            str(row.get("symbol") or row.get("ticker") or "").upper().strip(): dict(row)
            for row in signals
            if isinstance(row, dict)
            and str(row.get("symbol") or row.get("ticker") or "").strip()
        }
        executed_notionals: List[float] = []
        executed_count = 0
        for decision in decisions:
            if not isinstance(decision, dict) or not bool(
                decision.get("actually_executed")
            ):
                continue
            executed_count += 1
            symbol = str(decision.get("symbol") or "").upper().strip()
            signal_row = signal_by_symbol.get(symbol, {})
            price = _safe_float(
                decision.get("planned_price")
                or decision.get("current_price")
                or signal_row.get("entry_price")
                or signal_row.get("planned_entry_price")
            )
            qty = _safe_float(
                decision.get("adjusted_qty")
                or decision.get("planned_qty")
                or decision.get("qty")
                or signal_row.get("adjusted_qty")
                or signal_row.get("qty")
            )
            if price and qty and price > 0 and qty > 0:
                executed_notionals.append(float(price * qty))
        return {
            "available_capital": round(sum(executed_notionals), 2),
            "available_slots": int(max(0, executed_count)),
            "source": "executed_decision_budget_fallback",
            "mode": "replacement_budget",
        }

    def _estimate_workflow_entry_budget(
        self,
        *,
        signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
    ) -> float:
        signal_rows = [dict(row) for row in signals if isinstance(row, dict)]
        signal_by_symbol = {
            str(row.get("symbol") or row.get("ticker") or "").upper().strip(): row
            for row in signal_rows
            if str(row.get("symbol") or row.get("ticker") or "").strip()
        }
        baseline_notionals = [
            self._planned_signal_notional(row)
            for row in signal_rows
            if self._planned_signal_notional(row) > 0
        ]
        baseline_notional = (
            float(statistics.median(baseline_notionals))
            if baseline_notionals
            else 2_000.0
        )
        executed_notionals: List[float] = []
        executed_count = 0
        for decision in decisions:
            if not isinstance(decision, dict) or not bool(
                decision.get("actually_executed")
            ):
                continue
            executed_count += 1
            symbol = str(decision.get("symbol") or "").upper().strip()
            signal_row = signal_by_symbol.get(symbol, {})
            qty_override = decision.get("adjusted_qty") or decision.get("qty")
            notional = self._planned_signal_notional(
                signal_row, qty_override=qty_override
            )
            if notional > 0:
                executed_notionals.append(notional)
        if executed_notionals:
            return round(sum(executed_notionals) * 1.1, 2)
        if executed_count > 0:
            return round(max(1, executed_count) * baseline_notional * 1.1, 2)
        fallback_entries = min(5, max(1, len(signal_rows)))
        return round(fallback_entries * baseline_notional, 2)

    @staticmethod
    def _sanitize_replay_execution_state(exec_state: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(exec_state or {})
        payload["executed_symbols"] = []
        payload["skipped_symbols"] = []
        payload["first_hour_recheck_done"] = False
        payload.pop("overnight_recheck_status", None)
        return payload

    def _build_workflow_replay_agent(
        self,
        *,
        workspace_root: Path,
        artifacts: Dict[str, Any],
        signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        workflow_journal: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
    ) -> Any:
        agent = autonomous_agent_mod.AutonomousAgent.__new__(
            autonomous_agent_mod.AutonomousAgent
        )
        agent.logger = logger
        agent.state = None
        agent.config = get_config()
        agent.runtime_cfg = agent.config
        agent.entry_quality_cfg = agent.config.entry_quality
        agent.research_freshness_cfg = getattr(agent.config, "research_freshness", None)
        agent.scheduler = _ReplayScheduler(self)
        agent.market_cycle_count = 0
        agent.cycle_count = 0
        agent.current_regime = "neutral"
        replay_regime_analysis = self._build_replay_regime_analysis(
            resolved_regime=dict(
                (artifacts.get("plan_payload") or {}).get("resolved_regime") or {}
            ),
            artifacts=artifacts,
        )
        agent.current_regime_analysis = replay_regime_analysis
        agent.current_regime = replay_regime_analysis.regime.value
        agent.resolved_regime_output = {}
        agent.regime_strategy_overrides = {}
        agent._youtube_context = {}
        agent.max_positions = int(
            getattr(agent.config.portfolio, "max_positions", 0) or 0
        )
        agent.decision_claw = None
        agent.task_router = None
        agent.error_diagnoser = SimpleNamespace(
            diagnose_error=lambda e, ctx: {"error": str(e), "context": ctx}
        )
        agent._day_manager_instance = None
        agent._day_manager_date = ""
        agent._day_manager_dry_run = None
        agent._minute_replay_archive_done_today = False
        agent._reflect_done_today = False
        agent._pm_plan_done_today = False
        agent._full_overnight_done_today = False
        agent._eod_review_done_today = False
        agent._broker_sync_done = False
        agent._last_reflect_result = {}
        agent._last_pm_plan_result = {}
        agent._premarket_state = {}
        agent._decision_claw_redeployment_state = {}
        agent._data_gateway = None
        agent._data_gateway_status = {}
        agent._signal_registry = None
        agent._signal_registry_status = {}
        agent.fast_loop_runtime = None
        agent.fast_loop_stream = None
        agent._fast_loop_status = {}
        agent.market_data_client = self.data_client
        agent.alpaca_client = None
        agent.ollama_url = ""
        agent.required_ollama_models = []
        agent.ollama_health_state = {}
        seed_positions = self._seed_workflow_positions(
            decisions=decisions,
            workflow_journal=workflow_journal,
            trade_journal=trade_journal,
        )
        agent.plan_generator = _ReplayPlanGenerator(
            self,
            plan_payload=artifacts.get("plan_payload") or {},
            logs_dir=workspace_root / "logs",
            positions=seed_positions,
            initial_deployable_capital=self._estimate_workflow_entry_budget(
                signals=signals,
                decisions=decisions,
            ),
        )
        agent.replay_trading_client = _ReplayTradingClient(agent.plan_generator)

        bind_names = [
            "_update_workflow_state_flag",
            "_decision_claw_market_rows",
            "_get_actionable_entry_rows",
            "_get_full_watchlist_rows",
            "_get_scalp_watchlist_symbols",
            "_build_early_runner_rows",
            "_get_current_position_symbols",
            "_tracked_early_runner_symbols",
            "_remaining_early_runner_capacity",
            "_recent_early_runner_empty_streak",
            "_build_early_runner_diagnostics",
            "_decision_claw_action_symbols",
            "_decision_claw_deployment_symbols",
            "_apply_decision_claw_symbols_to_rows",
            "_wave_entry_regime_gate",
            "_wave_hard_reject_gap_pct",
            "_is_wave_breakout_rescue_candidate",
            "_compute_wave_limit_price",
            "_execute_entry_waves",
        ]
        for name in bind_names:
            setattr(
                agent,
                name,
                MethodType(getattr(autonomous_agent_mod.AutonomousAgent, name), agent),
            )
        agent._merge_watchlist_rows = (
            autonomous_agent_mod.AutonomousAgent._merge_watchlist_rows
        )
        agent._wave_max_chase_pct = (
            autonomous_agent_mod.AutonomousAgent._wave_max_chase_pct
        )

        agent._refresh_strategy_failsafe = MethodType(
            lambda self, source="": None,
            agent,
        )
        agent._ensure_ollama_for_phase = MethodType(
            lambda self, phase: True,
            agent,
        )
        agent._monitor_ollama_health = MethodType(
            lambda self, cycle_count, phase: None,
            agent,
        )
        agent._record_ollama_result = MethodType(
            lambda self, result, context="": None,
            agent,
        )
        agent._check_ollama_health = MethodType(
            lambda self, require_models=True: True,
            agent,
        )
        agent._handle_error_with_healing = MethodType(
            lambda self, error, context="": None,
            agent,
        )
        agent.monitor_workflow_health = MethodType(
            lambda self, phase, result: None,
            agent,
        )
        agent._detect_and_fix_runtime_errors = MethodType(
            lambda self, error=None: 0,
            agent,
        )
        agent._run_decision_claw_review = MethodType(
            autonomous_agent_mod.AutonomousAgent._run_decision_claw_review,
            agent,
        )

        def _get_replay_day_manager(self, dry_run=True, harness=self):
            if harness._workflow_dm is None:
                harness._workflow_dm = harness._build_replay_day_manager(
                    signals=harness._workflow_runtime_signals or list(signals),
                    resolved_regime=dict(
                        artifacts.get("plan_payload", {}).get("resolved_regime") or {}
                    ),
                    artifacts=artifacts,
                    trading_client=getattr(self, "replay_trading_client", None),
                )
            return harness._workflow_dm

        agent._get_day_manager = MethodType(_get_replay_day_manager, agent)
        agent._replay_owner = self
        agent.run_day_manager_cycle = MethodType(
            lambda self, dry_run=True, **kwargs: (
                self._replay_owner._workflow_run_day_manager_cycle(
                    agent=self,
                    dry_run=dry_run,
                    **kwargs,
                )
            ),
            agent,
        )
        return agent

    def _set_workflow_plan_review(self, plan_payload: Dict[str, Any]) -> None:
        decision_claw = dict((plan_payload or {}).get("decision_claw") or {})
        selected = {
            str(symbol or "").upper().strip()
            for symbol in (decision_claw.get("selected_symbols") or [])
            if str(symbol or "").strip()
        }
        promoted: set[str] = set()
        trimmed: set[str] = set()
        review = dict(decision_claw.get("review") or {})
        for action in review.get("actions") or []:
            if not isinstance(action, dict):
                continue
            symbol = str(action.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            action_type = str(action.get("action_type") or "").strip().lower()
            if action_type == "promote_symbol":
                promoted.add(symbol)
            elif action_type == "trim_position":
                trimmed.add(symbol)
        self._workflow_plan_selected_symbols = selected
        self._workflow_plan_promoted_symbols = promoted
        self._workflow_plan_trimmed_symbols = trimmed

    def _workflow_signal_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(row or {})
        symbol = (
            str(payload.get("symbol") or payload.get("ticker") or "").upper().strip()
        )
        payload["symbol"] = symbol
        payload["ticker"] = symbol
        payload.setdefault("entry_source", "workflow_replay")
        payload.setdefault("source_bucket", "watchlist")
        payload.setdefault(
            "score",
            payload.get(
                "ranking_score",
                payload.get("final_score", payload.get("confidence", 0.0)),
            ),
        )
        return payload

    def _workflow_candidate_row(
        self, dm: DayManager, row: Dict[str, Any]
    ) -> Dict[str, Any]:
        payload = self._workflow_signal_row(row)
        symbol = str(payload.get("symbol") or "").upper().strip()
        if not symbol:
            return payload
        signal_data = dict(dm._find_signal_data(symbol) or {})
        if not signal_data:
            return payload
        for key in (
            "qty",
            "entry_price",
            "planned_entry_price",
            "score",
            "setup_type",
            "strategy_name",
            "stop_loss",
            "take_profit",
            "max_hold_days",
            "source_bucket",
            "origin_entry_source",
            "runtime_entry_context",
            "override_reason",
            "plan_score_source",
        ):
            if payload.get(key) in (None, "", 0, 0.0):
                value = signal_data.get(key)
                if value not in (None, ""):
                    payload[key] = value
        if not str(payload.get("entry_source") or "").strip():
            payload["entry_source"] = str(
                signal_data.get("entry_source")
                or payload.get("entry_source")
                or "workflow_replay"
            )
        return payload

    def _intraday_bars_until_replay_time(self, symbol: str) -> Optional[pd.DataFrame]:
        replay_time = self._current_replay_time
        if replay_time is None:
            return None
        df = self.market_bars.get(str(symbol or "").upper())
        if df is None or df.empty:
            return None
        sliced = df[df.index <= pd.Timestamp(replay_time)]
        if sliced.empty:
            return None
        return sliced.copy()

    @contextlib.contextmanager
    def _patched_intraday_provider(self) -> Iterable[None]:
        original_available = day_manager_mod.INTRADAY_PROVIDER_AVAILABLE
        original_batch = day_manager_mod.get_intraday_bars_batch
        original_direct = intraday_provider_mod.get_intraday_bars

        def replay_batch(symbols, _client=None, minutes_back=None, max_batch=None):
            del _client, minutes_back, max_batch
            payload: Dict[str, pd.DataFrame] = {}
            for symbol in symbols or []:
                symbol_key = str(symbol or "").upper().strip()
                bars = self._intraday_bars_until_replay_time(symbol_key)
                if bars is not None and not bars.empty:
                    payload[symbol_key] = bars
            return payload

        def replay_direct(
            ticker,
            data_client=None,
            minutes_back=240,
            interval="1m",
            **kwargs,
        ):
            del data_client, minutes_back, interval, kwargs
            symbol_key = str(ticker or "").upper().strip()
            bars = self._intraday_bars_until_replay_time(symbol_key)
            if bars is None or bars.empty:
                return None
            return bars

        day_manager_mod.INTRADAY_PROVIDER_AVAILABLE = True
        day_manager_mod.get_intraday_bars_batch = replay_batch
        intraday_provider_mod.get_intraday_bars = replay_direct
        try:
            yield
        finally:
            day_manager_mod.INTRADAY_PROVIDER_AVAILABLE = original_available
            day_manager_mod.get_intraday_bars_batch = original_batch
            intraday_provider_mod.get_intraday_bars = original_direct

    @staticmethod
    def _position_to_namespace(position: Dict[str, Any]) -> SimpleNamespace:
        qty = int(float(position.get("qty", 0) or 0))
        current_price = float(position.get("current_price", 0.0) or 0.0)
        avg_entry_price = float(
            position.get(
                "avg_entry_price",
                position.get("avg_entry", position.get("entry_price", 0.0)),
            )
            or 0.0
        )
        market_value = float(position.get("market_value", current_price * qty) or 0.0)
        pnl_pct = float(position.get("pnl_pct", 0.0) or 0.0)
        unrealized_pl = float(
            position.get(
                "unrealized_pl",
                position.get("unrealized_pnl", market_value - (avg_entry_price * qty)),
            )
            or 0.0
        )
        return SimpleNamespace(
            symbol=str(position.get("symbol") or "").upper(),
            avg_entry_price=avg_entry_price,
            avg_entry=avg_entry_price,
            entry_price=avg_entry_price,
            market_value=market_value,
            cost_basis=float(position.get("cost_basis", avg_entry_price * qty) or 0.0),
            unrealized_pl=unrealized_pl,
            unrealized_pnl=unrealized_pl,
            unrealized_plpc=pnl_pct / 100.0,
            pnl_pct=pnl_pct,
            current_price=current_price,
            qty=qty,
            side=str(position.get("side") or "long"),
            high_since_entry=position.get("high_since_entry"),
        )

    def _mark_to_market_positions(
        self, positions: Optional[Sequence[Any]] = None
    ) -> List[SimpleNamespace]:
        marked: List[SimpleNamespace] = []
        for raw in positions or self._positions:
            if raw is None:
                continue
            symbol = str(getattr(raw, "symbol", "") or "").upper().strip()
            if not symbol:
                continue
            qty = int(getattr(raw, "qty", 0) or 0)
            entry_price = float(
                getattr(raw, "avg_entry_price", None)
                or getattr(raw, "avg_entry", None)
                or getattr(raw, "entry_price", 0.0)
                or 0.0
            )
            if qty <= 0 or entry_price <= 0:
                continue
            current_price = float(self._current_price_for_symbol(symbol) or 0.0)
            if current_price <= 0:
                current_price = float(
                    getattr(raw, "current_price", entry_price) or entry_price
                )
            market_value = float(current_price * qty)
            cost_basis = float(getattr(raw, "cost_basis", 0.0) or (entry_price * qty))
            unrealized_pl = float(market_value - cost_basis)
            unrealized_plpc = float(
                (unrealized_pl / cost_basis) if cost_basis > 0 else 0.0
            )
            payload = {
                "symbol": symbol,
                "qty": qty,
                "avg_entry_price": entry_price,
                "avg_entry": entry_price,
                "entry_price": float(
                    getattr(raw, "entry_price", entry_price) or entry_price
                ),
                "current_price": current_price,
                "market_value": market_value,
                "cost_basis": cost_basis,
                "unrealized_pl": unrealized_pl,
                "unrealized_pnl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
                "pnl_pct": unrealized_plpc * 100.0,
                "side": str(getattr(raw, "side", "long") or "long"),
                "high_since_entry": getattr(raw, "high_since_entry", None),
            }
            for extra_name in ("entry_time", "profile", "order_id", "context"):
                if hasattr(raw, extra_name):
                    payload[extra_name] = getattr(raw, extra_name)
            marked.append(SimpleNamespace(**payload))
        return marked

    def _mark_to_close_pnl(self, symbol: str, entry_price: float, qty: int) -> float:
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key or entry_price <= 0 or qty <= 0:
            return 0.0
        df = self.market_bars.get(symbol_key)
        if df is None or df.empty:
            return 0.0
        row = df.iloc[-1]
        close_price = None
        for key in ("close", "vwap", "open"):
            if key in row and pd.notna(row[key]):
                close_price = float(row[key])
                break
        if not close_price or close_price <= 0:
            return 0.0
        return (close_price - float(entry_price)) * int(qty)

    def _record_workflow_bullish_action(
        self,
        *,
        symbol: str,
        action_type: str,
        qty: int,
        entry_price: float,
        entry_context: str,
        reason: str,
    ) -> Dict[str, Any]:
        payload = {
            "timestamp": (
                self._current_replay_time or _session_minutes(self.session_date)[0]
            ).isoformat(),
            "symbol": str(symbol or "").upper().strip(),
            "action_type": str(action_type or ""),
            "qty": int(qty or 0),
            "entry_price": round(float(entry_price or 0.0), 4),
            "entry_context": str(entry_context or ""),
            "reason": str(reason or ""),
        }
        payload["mark_to_close_pnl"] = round(
            self._mark_to_close_pnl(
                payload["symbol"], payload["entry_price"], payload["qty"]
            ),
            2,
        )
        self._workflow_bullish_actions.append(payload)
        return payload

    def _workflow_replay_add_guard(
        self,
        *,
        dm: DayManager,
        symbol: str,
        entry_price: float,
        entry_context: str,
    ) -> str:
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key or entry_price <= 0:
            return ""
        signal_data = dict(dm._find_signal_data(symbol_key) or {})
        planned_entry = float(
            signal_data.get("entry_price")
            or signal_data.get("planned_entry_price")
            or 0.0
        )
        normalized_entry_context = str(entry_context or "").strip().lower()
        if planned_entry <= 0:
            if normalized_entry_context == "scale_into_winner":
                return "replay_guard_missing_planned_entry_scale"
            if normalized_entry_context == "strength_reentry_add":
                return "replay_guard_missing_planned_entry_strength"
            return ""
        if entry_price < planned_entry * 0.98:
            return "replay_guard_below_planned_entry"
        if (
            normalized_entry_context == "scale_into_winner"
            and entry_price > planned_entry * 1.09
        ):
            return "replay_guard_overextended_scale_add"
        authority_state = str(
            (dm._get_entry_authority_state() or {}).get("state", "open") or "open"
        )
        if (
            normalized_entry_context == "strength_reentry_add"
            and authority_state in {"bull_lock", "inverse_fast"}
            and entry_price > planned_entry * 1.12
        ):
            return "replay_guard_overextended_strength_add"
        return ""

    def _workflow_replay_entry_context(
        self,
        *,
        agent: Any,
        row: Dict[str, Any],
        symbol: str,
        override_reason: str = "",
    ) -> Dict[str, float | str]:
        symbol_key = str(symbol or "").upper().strip()
        planned_entry = float(
            row.get("entry_price")
            or row.get("planned_entry_price")
            or row.get("current_price")
            or 0.0
        )
        current_price = float(self._current_price_for_symbol(symbol_key) or 0.0)
        entry_price = current_price if current_price > 0 else planned_entry
        score = float(row.get("score", row.get("ranking_score", 0.0)) or 0.0)
        if entry_price <= 0:
            return {
                "entry_price": 0.0,
                "planned_entry": planned_entry,
                "current_price": current_price,
                "gap_pct": 0.0,
                "score": score,
                "block_reason": "missing_replay_market_price",
            }
        if planned_entry <= 0:
            return {
                "entry_price": entry_price,
                "planned_entry": planned_entry,
                "current_price": current_price,
                "gap_pct": 0.0,
                "score": score,
                "block_reason": "",
            }
        gap_pct = ((entry_price - planned_entry) / planned_entry) * 100.0
        if gap_pct < -5.0:
            return {
                "entry_price": entry_price,
                "planned_entry": planned_entry,
                "current_price": current_price,
                "gap_pct": gap_pct,
                "score": score,
                "block_reason": "replay_guard_gap_down_reject",
            }
        replay_time = self._current_replay_time
        normalized_override_reason = str(override_reason or "").strip().lower()
        if normalized_override_reason == "overnight_first_hour_recheck":
            wave = 2
        else:
            wave = (
                1
                if replay_time is not None
                and ((replay_time.hour * 60) + replay_time.minute) < 615
                else 2
            )
        rescue_candidate = bool(
            agent._is_wave_breakout_rescue_candidate(
                gap_pct=gap_pct,
                score=score,
                wave=wave,
            )
        )
        hard_reject_gap = float(agent._wave_hard_reject_gap_pct() or 0.0)
        if gap_pct > hard_reject_gap:
            return {
                "entry_price": entry_price,
                "planned_entry": planned_entry,
                "current_price": current_price,
                "gap_pct": gap_pct,
                "score": score,
                "rescue_candidate": rescue_candidate,
                "qty_multiplier": 1.0,
                "block_reason": "replay_guard_hard_gap_reject",
            }
        max_chase_pct = float(agent._wave_max_chase_pct(score=score, wave=wave) or 0.0)
        max_allowed = planned_entry * (1.0 + max_chase_pct / 100.0)
        if entry_price > max_allowed and not rescue_candidate:
            return {
                "entry_price": entry_price,
                "planned_entry": planned_entry,
                "current_price": current_price,
                "gap_pct": gap_pct,
                "score": score,
                "rescue_candidate": False,
                "qty_multiplier": 1.0,
                "block_reason": "replay_guard_extended_above_plan",
            }
        qty_multiplier = 1.0
        if rescue_candidate:
            qty_multiplier = float(
                getattr(
                    agent.entry_quality_cfg, "wave_breakout_rescue_size_multiplier", 0.4
                )
                or 0.4
            )
        return {
            "entry_price": entry_price,
            "planned_entry": planned_entry,
            "current_price": current_price,
            "gap_pct": gap_pct,
            "score": score,
            "rescue_candidate": rescue_candidate,
            "qty_multiplier": qty_multiplier,
            "block_reason": "",
        }

    def _workflow_candidate_priority(
        self,
        *,
        row: Dict[str, Any],
        entry_ctx: Dict[str, Any],
        override_reason: str,
    ) -> float:
        symbol = str(row.get("symbol") or "").upper().strip()
        score = float(entry_ctx.get("score", row.get("score", 0.0)) or 0.0)
        gap_pct = float(entry_ctx.get("gap_pct", 0.0) or 0.0)
        entry_price = float(entry_ctx.get("entry_price", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        priority = score
        if symbol in self._workflow_plan_selected_symbols:
            priority += 8.0
        if symbol in self._workflow_plan_promoted_symbols:
            priority += 4.0
        if symbol in self._workflow_plan_trimmed_symbols:
            priority -= 6.0
        if str(override_reason or "").strip().lower() == "overnight_first_hour_recheck":
            if symbol in self._workflow_plan_selected_symbols:
                priority += 3.0
            if symbol in self._workflow_plan_trimmed_symbols:
                priority -= 2.0
        if -1.5 <= gap_pct <= 1.5:
            priority += 6.0
        elif -3.0 <= gap_pct < -1.5:
            priority += 1.5
        elif 1.5 < gap_pct <= 3.0:
            priority += 2.0
        elif gap_pct < -3.0:
            priority -= 6.0
        else:
            priority -= 3.0
        if bool(entry_ctx.get("rescue_candidate")):
            priority += 2.0
        if stop_loss > 0 and entry_price > 0:
            stop_buffer_pct = ((entry_price - stop_loss) / entry_price) * 100.0
            if stop_buffer_pct <= 0:
                priority -= 12.0
            elif stop_buffer_pct < 2.0:
                priority -= 5.0
            elif stop_buffer_pct >= 5.0:
                priority += 1.0
        return priority

    def _evaluate_workflow_held_strength(
        self,
        *,
        agent: Any,
        dm: DayManager,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        positions = [
            pos
            for pos in (agent.plan_generator.get_current_positions() or [])
            if isinstance(pos, dict) and str(pos.get("symbol") or "").strip()
        ]
        if not positions:
            return [], []

        scalable_rows = {
            str(row.get("symbol") or "").upper(): dict(row)
            for row in (
                dm.check_position_sizes(
                    [self._position_to_namespace(pos) for pos in positions]
                ).get("scalable", [])
                or []
            )
            if isinstance(row, dict)
        }
        traces: List[Dict[str, Any]] = []
        add_actions: List[Dict[str, Any]] = []
        status_priority = {"add_ready": 3, "blocked": 2, "watched_not_triggered": 1}

        def _evaluate_single_position(pos):
            symbol = str(pos.get("symbol") or "").upper().strip()
            if not symbol or dm._is_hedge_symbol(symbol):
                return None
            state = dm._strength_reentry_current_state(symbol, force_refresh=True)
            scalable = scalable_rows.get(symbol, {})
            add_ready = bool(scalable)
            block_reason = ""
            if add_ready:
                entry_price = float(pos.get("current_price", 0.0) or 0.0)
                block_reason = self._workflow_replay_add_guard(
                    dm=dm,
                    symbol=symbol,
                    entry_price=entry_price,
                    entry_context=str(scalable.get("entry_context") or ""),
                )
                if block_reason:
                    add_ready = False
                    scalable = {}
            if not add_ready and not block_reason:
                block_reason = dm._position_add_block_reason(symbol)
            trace = {
                "symbol": symbol,
                "qty": int(float(pos.get("qty", 0) or 0)),
                "market_value": round(float(pos.get("market_value", 0.0) or 0.0), 2),
                "pnl_pct": round(float(pos.get("pnl_pct", 0.0) or 0.0), 2),
                "phase": str(state.get("strength_reentry_phase") or "inactive"),
                "reason": str(state.get("strength_reentry_reason") or ""),
                "qualification": str(state.get("strength_reentry_qualification") or ""),
                "current_relative_strength_pct": state.get(
                    "strength_reentry_current_relative_strength_pct"
                ),
                "open_to_hour_gain_pct": state.get(
                    "strength_reentry_open_to_hour_gain_pct"
                ),
                "pullback_pct": state.get("pullback_pct"),
                "trigger": str(state.get("strength_reentry_trigger") or ""),
                "reclaim_detected_at": str(
                    state.get("strength_reentry_reclaim_detected_at") or ""
                ),
                "follow_through_ready_at": str(
                    state.get("strength_reentry_follow_through_ready_at") or ""
                ),
                "follow_through_failed_at": str(
                    state.get("strength_reentry_follow_through_failed_at") or ""
                ),
                "status": (
                    "add_ready"
                    if add_ready
                    else "blocked"
                    if block_reason
                    else "watched_not_triggered"
                ),
                "blocked_reason": str(block_reason or ""),
            }
            if scalable:
                trace["additional_qty"] = int(
                    float(scalable.get("additional_qty", 0) or 0)
                )
                trace["entry_context"] = str(scalable.get("entry_context") or "")
                trace["scale_reason"] = str(scalable.get("reason") or "")
            return trace, scalable, pos

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = list(executor.map(_evaluate_single_position, positions))

        for res in results:
            if res is None:
                continue
            trace, scalable, pos = res
            symbol = trace["symbol"]
            add_ready = trace["status"] == "add_ready"
            traces.append(trace)
            existing_trace = self._workflow_last_strength_trace_by_symbol.get(symbol)
            if existing_trace is None or status_priority.get(
                str(trace.get("status") or ""), 0
            ) >= status_priority.get(str(existing_trace.get("status") or ""), 0):
                self._workflow_last_strength_trace_by_symbol[symbol] = dict(trace)
            if add_ready and symbol not in self._workflow_add_selected_symbols:
                add_qty = int(float(scalable.get("additional_qty", 0) or 0))
                entry_price = float(pos.get("current_price", 0.0) or 0.0)
                if add_qty > 0 and entry_price > 0:
                    agent.plan_generator.apply_replay_entry(
                        symbol=symbol,
                        qty=add_qty,
                        price=entry_price,
                        entry_context=str(
                            scalable.get("entry_context") or "strength_reentry_add"
                        ),
                    )
                    add_actions.append(
                        self._record_workflow_bullish_action(
                            symbol=symbol,
                            action_type="add",
                            qty=add_qty,
                            entry_price=entry_price,
                            entry_context=str(
                                scalable.get("entry_context") or "strength_reentry_add"
                            ),
                            reason=str(scalable.get("reason") or ""),
                        )
                    )
                    self._workflow_add_selected_symbols.add(symbol)
        traces.sort(
            key=lambda row: (
                0 if row.get("status") == "add_ready" else 1,
                -float(row.get("pnl_pct", 0.0) or 0.0),
                row.get("symbol", ""),
            )
        )
        return traces, add_actions

    def _workflow_run_day_manager_cycle(
        self,
        *,
        agent: Any,
        dry_run: bool = True,
        candidate_universe_rows: Optional[List[Dict[str, Any]]] = None,
        override_reason: str = "",
        reset_signals: bool = False,
        max_new_entries_override: Optional[int] = None,
        entry_wave_override: Optional[Any] = None,
    ) -> Dict[str, Any]:
        with (
            self._patched_runtime_clock(),
            self._patched_inverse_screen(),
            self._patched_intraday_provider(),
        ):
            result = autonomous_agent_mod.AutonomousAgent.run_day_manager_cycle(
                agent,
                dry_run=dry_run,
                candidate_universe_rows=candidate_universe_rows,
                override_reason=override_reason,
                reset_signals=reset_signals,
                max_new_entries_override=max_new_entries_override,
                entry_wave_override=entry_wave_override,
            )
        self._capture_replay_cycle_result(
            agent=agent,
            result=result,
            candidate_universe_rows=candidate_universe_rows,
            override_reason=override_reason,
        )
        return result

    def _capture_replay_cycle_result(
        self,
        *,
        agent: Any,
        result: Dict[str, Any],
        candidate_universe_rows: Optional[List[Dict[str, Any]]],
        override_reason: str,
    ) -> None:
        dm = self._workflow_dm
        phase = self._phase_for_time(
            self._current_replay_time or _session_minutes(self.session_date)[0]
        )
        rows = (
            [
                self._workflow_candidate_row(dm, row)
                for row in (candidate_universe_rows or [])
                if isinstance(row, dict)
            ]
            if dm is not None
            else [
                dict(row)
                for row in (candidate_universe_rows or [])
                if isinstance(row, dict)
            ]
        )
        row_by_symbol = {
            str(row.get("symbol") or "").upper().strip(): row
            for row in rows
            if str(row.get("symbol") or "").strip()
        }
        selected_symbols = [
            str(symbol or "").upper().strip()
            for symbol in (result.get("selected_entry_symbols", []) or [])
            if str(symbol or "").strip()
        ]
        for symbol in selected_symbols:
            self._workflow_selected_symbols.add(symbol)
            row = row_by_symbol.get(symbol, {})
            entry_ctx = self._workflow_replay_entry_context(
                agent=agent,
                row=row,
                symbol=symbol,
                override_reason=override_reason,
            )
            qty = int(float(row.get("qty", 0) or 0))
            qty_multiplier = float(entry_ctx.get("qty_multiplier", 1.0) or 1.0)
            if qty > 0 and qty_multiplier > 0 and qty_multiplier != 1.0:
                qty = max(1, int(qty * qty_multiplier))
            entry_price = float(entry_ctx.get("entry_price", 0.0) or 0.0)
            if qty > 0 and entry_price > 0:
                agent.plan_generator.apply_replay_entry(
                    symbol=symbol,
                    qty=qty,
                    price=entry_price,
                    entry_context=str(row.get("entry_source") or "workflow_replay"),
                )
                self._record_workflow_bullish_action(
                    symbol=symbol,
                    action_type="new_entry",
                    qty=qty,
                    entry_price=entry_price,
                    entry_context=str(row.get("entry_source") or "workflow_replay"),
                    reason=str(result.get("block_reason") or "selected"),
                )
        blocked_symbols: List[Dict[str, Any]] = []
        top_ranked_unfilled = list(result.get("top_ranked_unfilled", []) or [])
        for row in top_ranked_unfilled[:20]:
            if not isinstance(row, dict):
                continue
            blocked_symbols.append(
                {
                    "symbol": str(row.get("ticker") or row.get("symbol") or "")
                    .upper()
                    .strip(),
                    "regime_reason": "",
                    "authority_reason": str(
                        row.get("reason")
                        or row.get("block_reason")
                        or result.get("block_reason")
                        or ""
                    ),
                    "candidate_priority": round(
                        float(
                            row.get(
                                "realtime_score",
                                row.get("score", row.get("ranking_score", 0.0)),
                            )
                            or 0.0
                        ),
                        2,
                    ),
                }
            )
        deployment_floor: Dict[str, Any] = {}
        if dm is not None:
            try:
                deployment_floor = dm._deployment_floor_status(
                    [
                        self._position_to_namespace(pos)
                        for pos in (agent.plan_generator.get_current_positions() or [])
                        if isinstance(pos, dict)
                    ]
                )
            except Exception:
                deployment_floor = {}
        deployment_floor["timestamp"] = (
            self._current_replay_time or _session_minutes(self.session_date)[0]
        ).isoformat()
        self._workflow_deployment_samples.append(dict(deployment_floor))
        action_row = {
            "timestamp": (
                self._current_replay_time or _session_minutes(self.session_date)[0]
            ).isoformat(),
            "phase": str(getattr(phase, "value", phase)),
            "override_reason": str(override_reason or ""),
            "candidate_count": len(rows),
            "open_slots": int(result.get("open_slots", 0) or 0),
            "max_new_entries": int(result.get("max_new_entries", 0) or 0),
            "orders_submitted": int(
                result.get("orders_submitted", result.get("entries_submitted", 0)) or 0
            ),
            "block_reason": str(result.get("block_reason") or ""),
            "blocked_by_reason": dict(result.get("blocked_by_reason", {}) or {}),
            "entry_audit": dict(result.get("entry_audit", {}) or {}),
            "selected_symbols": list(selected_symbols),
            "selected_candidates": [
                {
                    "symbol": symbol,
                    "entry_source": str(
                        row_by_symbol.get(symbol, {}).get("entry_source") or ""
                    ),
                    "origin_entry_source": str(
                        row_by_symbol.get(symbol, {}).get("origin_entry_source") or ""
                    ),
                    "runtime_entry_context": str(
                        row_by_symbol.get(symbol, {}).get("runtime_entry_context") or ""
                    ),
                    "override_reason": str(
                        row_by_symbol.get(symbol, {}).get("override_reason") or ""
                    ),
                    "plan_score_source": str(
                        row_by_symbol.get(symbol, {}).get("plan_score_source") or ""
                    ),
                    "source_bucket": str(
                        row_by_symbol.get(symbol, {}).get("source_bucket") or ""
                    ),
                }
                for symbol in selected_symbols
            ],
            "add_selected_symbols": [],
            "blocked_symbols": blocked_symbols,
            "held_strength_traces": [],
            "deployment_floor": dict(deployment_floor),
            "entry_authority_state": (
                dict(dm._get_entry_authority_state() or {}).get("state", "open")
                if dm is not None
                else "open"
            ),
        }
        if rows or selected_symbols or blocked_symbols:
            self._workflow_action_log.append(action_row)

    def _summarize_workflow_actions(
        self,
        generated_decisions: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        actions = list(self._workflow_action_log)
        deployment_samples = list(self._workflow_deployment_samples)
        floor_shortfall_cycles = sum(
            1
            for row in deployment_samples
            if float(row.get("shortfall_value", 0.0) or 0.0) > 0
        )
        latest_floor = deployment_samples[-1] if deployment_samples else {}
        selected_symbols_total = sum(
            len(row.get("selected_symbols", []) or []) for row in actions
        )
        add_selected_total = sum(
            len(row.get("add_selected_symbols", []) or []) for row in actions
        )
        bullish_mark_to_close_pnl = round(
            sum(
                float(row.get("mark_to_close_pnl", 0.0) or 0.0)
                for row in self._workflow_bullish_actions
            ),
            2,
        )
        generated_rows = [
            dict(row) for row in (generated_decisions or []) if isinstance(row, dict)
        ]
        executed_generated_rows = [
            row for row in generated_rows if bool(row.get("actually_executed"))
        ]
        if (
            selected_symbols_total == 0
            and add_selected_total == 0
            and executed_generated_rows
        ):
            selected_symbols_total = len(executed_generated_rows)
            bullish_mark_to_close_pnl = round(
                sum(
                    self._mark_to_close_pnl(
                        str(row.get("symbol") or ""),
                        float(
                            row.get("current_price") or row.get("planned_price") or 0.0
                        ),
                        int(
                            float(
                                row.get("adjusted_qty") or row.get("planned_qty") or 0
                            )
                            or 0
                        ),
                    )
                    for row in executed_generated_rows
                ),
                2,
            )
        return {
            "total_actions": len(actions),
            "selected_symbols_total": int(selected_symbols_total),
            "add_selected_total": int(add_selected_total),
            "blocked_symbols_total": sum(
                len(row.get("blocked_symbols", []) or []) for row in actions
            ),
            "market_open_cycles": int(
                self._workflow_phase_cycles.get("market_open", 0) or 0
            ),
            "market_hours_cycles": int(
                self._workflow_phase_cycles.get("market_hours", 0) or 0
            ),
            "bullish_mark_to_close_pnl": float(bullish_mark_to_close_pnl),
            "deployment_floor_shortfall_cycles": int(floor_shortfall_cycles),
            "deployment_floor_final_pct": float(
                latest_floor.get("deployed_pct", 0.0) or 0.0
            ),
            "deployment_floor_final_shortfall_value": float(
                latest_floor.get("shortfall_value", 0.0) or 0.0
            ),
        }

    def _load_workspace_decisions(self, workspace_root: Path) -> List[Dict[str, Any]]:
        compact = self.session_date.replace("-", "")
        path = workspace_root / "logs" / f"trade_decisions_{compact}.json"
        if not path.exists():
            return []
        payload = _load_json(path)
        return payload if isinstance(payload, list) else []

    @contextlib.contextmanager
    def _patched_agent_replay_environment(
        self,
        *,
        replay_time: datetime,
        workspace_root: Path,
    ) -> Iterable[None]:
        original_agent_datetime = autonomous_agent_mod.datetime
        original_agent_project_dir = autonomous_agent_mod.PROJECT_DIR
        original_agent_log_dir = autonomous_agent_mod.LOG_DIR
        original_agent_logs_dir = autonomous_agent_mod.LOGS_DIR
        original_agent_plans_dir = autonomous_agent_mod.PLANS_DIR
        original_agent_data_dir = autonomous_agent_mod.DATA_DIR
        original_agent_time_sleep = autonomous_agent_mod.time.sleep

        class ReplayDateTime(original_agent_datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return replay_time.astimezone(tz)
                return replay_time.replace(tzinfo=None)

        autonomous_agent_mod.datetime = ReplayDateTime
        autonomous_agent_mod.PROJECT_DIR = workspace_root
        autonomous_agent_mod.LOG_DIR = workspace_root / "logs"
        autonomous_agent_mod.LOGS_DIR = workspace_root / "logs"
        autonomous_agent_mod.PLANS_DIR = workspace_root / "plans"
        autonomous_agent_mod.DATA_DIR = workspace_root / "data"
        autonomous_agent_mod.time.sleep = lambda *_args, **_kwargs: None
        try:
            with self._patched_runtime_clock_for(replay_time):
                yield
        finally:
            autonomous_agent_mod.datetime = original_agent_datetime
            autonomous_agent_mod.PROJECT_DIR = original_agent_project_dir
            autonomous_agent_mod.LOG_DIR = original_agent_log_dir
            autonomous_agent_mod.LOGS_DIR = original_agent_logs_dir
            autonomous_agent_mod.PLANS_DIR = original_agent_plans_dir
            autonomous_agent_mod.DATA_DIR = original_agent_data_dir
            autonomous_agent_mod.time.sleep = original_agent_time_sleep

    def _run_agent_workflow(self, *, persist: bool = True) -> Dict[str, Any]:
        artifacts = self._resolve_artifacts()
        decisions = self._load_decisions(artifacts)
        workflow_journal = self._load_workflow_journal(artifacts)
        trade_journal = self._load_trade_journal(artifacts)
        raw_signals = self._load_signals(artifacts)
        enriched_signals = self._augment_signals_with_plan_metadata(
            raw_signals,
            artifacts=artifacts,
        )
        signals = self._augment_signals_with_recorded_metadata(
            enriched_signals,
            decisions=decisions,
            trade_journal=trade_journal,
        )
        self._workflow_runtime_signals = [
            dict(row) for row in signals if isinstance(row, dict)
        ]
        self._set_workflow_plan_review(artifacts.get("plan_payload") or {})
        self._hydrate_archive_market_data(
            signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            workflow_journal=workflow_journal,
        )
        self._ensure_market_data()

        workspace_root = self._prepare_workflow_workspace(artifacts)
        try:
            agent = self._build_workflow_replay_agent(
                workspace_root=workspace_root,
                artifacts=artifacts,
                signals=signals,
                decisions=decisions,
                workflow_journal=workflow_journal,
                trade_journal=trade_journal,
            )
            self._workflow_agent = agent
            session_minutes = _session_minutes(self.session_date)
            total_steps = len(session_minutes)
            # Honor the interval each cycle method returns. Production waits
            # interval_seconds before the next cycle; the replay was running
            # one cycle per minute regardless, which over-cycles vs production
            # and lets DM keep picking new entries every minute. Use the
            # returned value to skip ahead.
            next_cycle_step = 0
            for step_index, replay_minute in enumerate(session_minutes, start=1):
                self._current_replay_time = replay_minute
                phase_name = (
                    "market_open"
                    if replay_minute.hour == 8 and replay_minute.minute < 45
                    else "market_hours"
                )
                if step_index - 1 < next_cycle_step:
                    # Gated by the previous cycle's interval — skip this minute.
                    continue
                self._maybe_log_progress(
                    replay_minute=replay_minute,
                    step_index=step_index,
                    total_steps=total_steps,
                    phase=phase_name,
                    extra={
                        "inverse_orders": len(self._inverse_orders),
                        "actions": len(self._workflow_action_log),
                        "positions": len(self._positions),
                    },
                )
                self._workflow_phase_cycles[phase_name] = (
                    int(self._workflow_phase_cycles.get(phase_name, 0) or 0) + 1
                )
                with self._patched_agent_replay_environment(
                    replay_time=replay_minute,
                    workspace_root=workspace_root,
                ):
                    if phase_name == "market_open":
                        interval_seconds = agent._run_market_open_cycle(
                            cycle_count=self._workflow_phase_cycles[phase_name],
                            execute=False,
                        )
                    else:
                        interval_seconds = agent._run_market_hours_cycle(
                            cycle_count=self._workflow_phase_cycles[phase_name],
                            execute=False,
                        )
                # Convert returned interval (seconds) to minutes-to-skip.
                # Floor at 1 minute (always advance), cap at 30 minutes (avoid
                # silly waits if production returned a stale large value).
                try:
                    interval_int = int(interval_seconds or 0)
                except (TypeError, ValueError):
                    interval_int = 60
                interval_minutes = max(1, min(30, interval_int // 60))
                next_cycle_step = (step_index - 1) + interval_minutes
            self._finalize_inverse_positions()
            generated_decisions = self._load_workspace_decisions(workspace_root)
        finally:
            shutil.rmtree(workspace_root, ignore_errors=True)

        workflow_summary = self._summarize_workflow_actions(
            generated_decisions=generated_decisions,
        )
        inverse_performance = self._summarize_inverse_trade_results()
        order_breakdown = self._summarize_replay_orders()
        actual_day_context = _load_actual_day_context(
            self.project_dir, self.session_date
        )
        handoff_diagnostics = self._build_handoff_diagnostics(
            artifacts=artifacts,
            raw_signals=raw_signals,
            replay_signals=signals,
            decisions=decisions,
        )
        watchlist_causality_snapshot: Dict[str, Any] = {}
        watchlist_causality_path = ""
        try:
            replay_dm = self._workflow_dm
            if replay_dm is None and hasattr(agent, "_get_day_manager"):
                replay_dm = agent._get_day_manager(dry_run=False)
            if replay_dm is None:
                raise RuntimeError("workflow replay day manager unavailable")
            watchlist_causality_snapshot = (
                replay_dm._persist_watchlist_causality_snapshot(
                    phase=TradingPhase.CORE_TRADING
                )
            )
            watchlist_causality_path = str(replay_dm._watchlist_causality_path())
        except Exception as exc:
            self._replay_notes.append(
                f"Workflow watchlist causality snapshot persistence failed: {exc}"
            )
        signal_pipeline_audit = self._build_signal_pipeline_audit(
            artifacts=artifacts,
            raw_signals=raw_signals,
            replay_signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            watchlist_causality_snapshot=watchlist_causality_snapshot,
        )
        signal_pipeline_audit_path = str(self._signal_pipeline_audit_path())
        if persist:
            _write_json(Path(signal_pipeline_audit_path), signal_pipeline_audit)
        replay_dm = self._workflow_dm
        if replay_dm is None and hasattr(agent, "_get_day_manager"):
            replay_dm = agent._get_day_manager(dry_run=False)
        if replay_dm is None:
            replay_dm = self._build_replay_day_manager(
                signals=signals,
                resolved_regime=dict(
                    artifacts.get("plan_payload", {}).get("resolved_regime") or {}
                ),
                artifacts=artifacts,
            )
        trade_journal_management_events = self._trade_journal_management_events(
            trade_journal
        )
        management_replay = self._replay_position_management(
            dm=replay_dm,
            journal_entries=list(workflow_journal) + trade_journal_management_events,
        )
        management_coverage = self._build_management_coverage_diagnostics(
            management_replay=management_replay,
            actual_day_context=actual_day_context,
            workflow_journal=workflow_journal,
            trade_journal_management_events=trade_journal_management_events,
        )
        stop_loss_diagnostics = self._build_stop_loss_diagnostics(management_replay)
        intraday_bar_coverage = self._build_intraday_bar_coverage(
            signals=signals,
            decisions=decisions,
            trade_journal=trade_journal,
            workflow_actions=self._workflow_action_log,
        )
        replay_order_accounting = self._build_replay_order_accounting(
            order_breakdown=order_breakdown,
            workflow_summary=workflow_summary,
        )
        replay_environment = self._build_replay_environment_snapshot(artifacts)
        divergences: List[Dict[str, Any]] = []
        if management_coverage["status"] in {"incomplete", "degraded"}:
            divergences.append(
                {
                    "type": "management_replay_coverage_gap",
                    "timestamp": "",
                    "symbol": "",
                    "replay_regime_reason": str(
                        management_coverage.get("problem") or ""
                    ),
                    "replay_authority_reason": str(
                        management_coverage.get("status") or ""
                    ),
                }
            )
        divergence_summary = _summarize_divergences(divergences)
        report = {
            "summary": {
                "date": self.session_date,
                "replay_mode": self.mode,
                "replay_profile": self.profile,
                "phase_coverage": "market_open+market_hours",
                "force_modern_crash_protection": self.force_modern_crash_protection,
                "position_advisor_mode": (
                    "skipped" if self.skip_position_advisor else "enabled"
                ),
                "signals_total": len(signals),
                "recorded_decisions_total": len(decisions),
                # Decision-ledger executions only — counts rows in
                # trade_decisions_<date>.json with actually_executed=True. This
                # is NOT the production trade count: trims, exits, and entries
                # via alternative paths are not in this file. See
                # production_total_journaled / actual_total_trades for truth.
                "recorded_longs_executed": sum(
                    1 for row in decisions if row.get("actually_executed")
                ),
                "production_entries_journaled": int(
                    actual_day_context.get("entries_journaled", 0) or 0
                ),
                "production_exits_journaled": int(
                    actual_day_context.get("exits_journaled", 0) or 0
                ),
                "production_trims_journaled": int(
                    actual_day_context.get("trims_journaled", 0) or 0
                ),
                "production_total_journaled": int(
                    actual_day_context.get("total_journaled", 0) or 0
                ),
                "production_unique_symbols_journaled": int(
                    actual_day_context.get("unique_symbols_journaled", 0) or 0
                ),
                "workflow_generated_decisions_total": len(generated_decisions),
                "workflow_selected_symbols_total": int(
                    workflow_summary.get("selected_symbols_total", 0) or 0
                ),
                "workflow_add_selected_total": int(
                    workflow_summary.get("add_selected_total", 0) or 0
                ),
                # Unique-symbol counts. Use these instead of the legacy
                # "bullish_actions_total" which was just selections + adds
                # combined (double-counting).
                "replay_unique_symbols_entered": int(
                    order_breakdown["unique_symbol_count_by_bucket"].get(
                        "equity_entry", 0
                    )
                    + order_breakdown["unique_symbol_count_by_bucket"].get(
                        "inverse_etf", 0
                    )
                ),
                "replay_entries_executed": int(
                    order_breakdown["by_bucket"].get("equity_entry", 0)
                    + order_breakdown["by_bucket"].get("inverse_etf", 0)
                ),
                "replay_adds_executed": int(
                    workflow_summary.get("add_selected_total", 0) or 0
                ),
                "workflow_bullish_mark_to_close_pnl": float(
                    workflow_summary.get("bullish_mark_to_close_pnl", 0.0) or 0.0
                ),
                "deployment_floor_shortfall_cycles": int(
                    workflow_summary.get("deployment_floor_shortfall_cycles", 0) or 0
                ),
                "deployment_floor_final_pct": float(
                    workflow_summary.get("deployment_floor_final_pct", 0.0) or 0.0
                ),
                "deployment_floor_final_shortfall_value": float(
                    workflow_summary.get("deployment_floor_final_shortfall_value", 0.0)
                    or 0.0
                ),
                "workflow_blocked_symbols_total": int(
                    workflow_summary.get("blocked_symbols_total", 0) or 0
                ),
                "market_open_cycles": int(
                    workflow_summary.get("market_open_cycles", 0) or 0
                ),
                "market_hours_cycles": int(
                    workflow_summary.get("market_hours_cycles", 0) or 0
                ),
                # NOTE: legacy field. This used to be "all DM orders" mislabeled
                # as inverse. It is now the true inverse-ETF count. Sum of all
                # buckets is in replay_orders_total.
                "inverse_orders_captured": int(
                    order_breakdown["by_bucket"].get("inverse_etf", 0)
                ),
                "replay_orders_total": int(order_breakdown.get("total", 0) or 0),
                "replay_orders_by_bucket": dict(order_breakdown.get("by_bucket", {})),
                "replay_orders_unique_symbols_by_bucket": dict(
                    order_breakdown.get("unique_symbol_count_by_bucket", {})
                ),
                "replay_orders_top_repeated_exits": list(
                    order_breakdown.get("top_repeated_exit_symbols", [])
                ),
                "replay_data_gap_skipped": int(
                    order_breakdown.get("data_gap_skipped_total", 0) or 0
                ),
                "replay_data_gap_skipped_unique_symbols": int(
                    order_breakdown.get("data_gap_skipped_unique_symbols", 0) or 0
                ),
                "replay_data_gap_skipped_examples": list(
                    order_breakdown.get("data_gap_skipped_unique_symbols_examples", [])
                ),
                "inverse_trades_closed": int(
                    inverse_performance.get("closed_trades", 0) or 0
                ),
                "inverse_net_pnl": float(
                    inverse_performance.get("net_pnl", 0.0) or 0.0
                ),
                "inverse_avg_return_pct": float(
                    inverse_performance.get("avg_return_pct", 0.0) or 0.0
                ),
                "inverse_win_rate": float(
                    inverse_performance.get("win_rate", 0.0) or 0.0
                ),
                "divergences_count": len(divergences),
                "decision_artifact_missing": bool(
                    artifacts.get("decision_path") is None
                ),
                "actual_net_pnl": float(actual_day_context.get("net_pnl", 0.0) or 0.0),
                "actual_total_trades": int(
                    actual_day_context.get("total_trades", 0) or 0
                ),
                "handoff_status": str(handoff_diagnostics.get("status") or "healthy"),
                "handoff_problem": str(handoff_diagnostics.get("problem") or ""),
                "handoff_recoverable_candidates": int(
                    handoff_diagnostics.get("recoverable_candidates_count", 0) or 0
                ),
                "signal_alignment_status": str(
                    handoff_diagnostics.get("signal_alignment_status") or "aligned"
                ),
                "signal_alignment_problem": str(
                    handoff_diagnostics.get("signal_alignment_problem") or ""
                ),
                "market_data_coverage_mode": self._market_data_coverage_mode(),
                "archive_found": bool(self._archive_diagnostics.get("archive_found")),
                "archive_symbols_loaded": len(
                    self._archive_diagnostics.get("loaded_symbols", [])
                ),
                "archive_symbols_missing": len(
                    self._archive_diagnostics.get("missing_symbols", [])
                ),
                "archive_live_fetch_symbols": len(
                    self._archive_diagnostics.get("live_fetch_symbols", [])
                ),
                "data_quality": self._build_data_quality_status(),
                "management_replay_events": int(
                    management_replay.get("summary", {}).get("events_replayed", 0) or 0
                ),
                "management_replay_differences": int(
                    management_replay.get("summary", {}).get("action_differences", 0)
                    or 0
                ),
                "management_replay_exit_signals": int(
                    management_replay.get("summary", {}).get("exit_like_signals", 0)
                    or 0
                ),
                "management_replay_coverage_status": str(
                    management_coverage.get("status") or ""
                ),
                "unresolved_stop_loss_warnings": int(
                    stop_loss_diagnostics.get("unresolved_stop_loss_warnings", 0) or 0
                ),
                "intraday_zero_bar_symbols": int(
                    intraday_bar_coverage.get("symbols_with_zero_bars", 0) or 0
                ),
                # Side-by-side compare: how the recorded production session ran
                # vs. how the replay simulator ran. Three production sources
                # because each captures a different slice:
                #   - decisions ledger: only Wave-1 entry decisions DM scored
                #   - trade journal: actual journaled trades (entries + exits
                #     + trims) on the date — closer to truth
                #   - eod_review.total_trades: post-session round-trip count
                "recorded_vs_replay": {
                    "production_decisions_logged": len(decisions),
                    "production_claw_decisions_logged": self._load_production_claw_decisions_count(),
                    "production_decisions_marked_executed": sum(
                        1 for row in decisions if row.get("actually_executed")
                    ),
                    "production_journaled_entries": int(
                        actual_day_context.get("entries_journaled", 0) or 0
                    ),
                    "production_journaled_exits": int(
                        actual_day_context.get("exits_journaled", 0) or 0
                    ),
                    "production_journaled_trims": int(
                        actual_day_context.get("trims_journaled", 0) or 0
                    ),
                    "production_journaled_total": int(
                        actual_day_context.get("total_journaled", 0) or 0
                    ),
                    "production_eod_total_trades": int(
                        actual_day_context.get("total_trades", 0) or 0
                    ),
                    "production_net_pnl": float(
                        actual_day_context.get("net_pnl", 0.0) or 0.0
                    ),
                    "replay_decisions": len(generated_decisions),
                    "replay_unique_symbols_entered": int(
                        order_breakdown["unique_symbol_count_by_bucket"].get(
                            "equity_entry", 0
                        )
                        + order_breakdown["unique_symbol_count_by_bucket"].get(
                            "inverse_etf", 0
                        )
                    ),
                    "replay_orders_total": int(order_breakdown.get("total", 0) or 0),
                },
            },
            "artifacts": {
                "signals_file": artifacts["signals_path"].name
                if artifacts["signals_path"]
                else "",
                "decision_file": artifacts["decision_path"].name
                if artifacts["decision_path"]
                else "",
                "plan_file": artifacts["plan_path"].name
                if artifacts["plan_path"]
                else "",
                "workflow_journal_file": (
                    artifacts["workflow_journal_path"].name
                    if artifacts["workflow_journal_path"]
                    else ""
                ),
                "trade_journal_file": (
                    artifacts["trade_journal_path"].name
                    if artifacts["trade_journal_path"]
                    else ""
                ),
                "watchlist_causality_file": watchlist_causality_path,
                "signal_pipeline_audit_file": signal_pipeline_audit_path,
            },
            "workflow_replay": {
                "action_log": list(self._workflow_action_log),
                "generated_trade_decisions": generated_decisions,
                "phase_cycles": dict(self._workflow_phase_cycles),
                "bullish_actions": list(self._workflow_bullish_actions),
                "deployment_floor_samples": list(self._workflow_deployment_samples),
                "held_strength_latest": sorted(
                    self._workflow_last_strength_trace_by_symbol.values(),
                    key=lambda row: (
                        0 if row.get("status") == "add_ready" else 1,
                        -float(row.get("pnl_pct", 0.0) or 0.0),
                        row.get("symbol", ""),
                    ),
                ),
            },
            "market_data_archive": dict(self._archive_diagnostics),
            "replay_environment": replay_environment,
            "intraday_bar_coverage": intraday_bar_coverage,
            "replay_order_accounting": replay_order_accounting,
            "inverse_orders": list(self._inverse_orders),
            "inverse_trade_results": list(self._inverse_trade_results),
            "replay_notes": list(self._replay_notes),
            "actual_day_context": actual_day_context,
            "handoff_diagnostics": handoff_diagnostics,
            "watchlist_causality_snapshot": watchlist_causality_snapshot,
            "signal_pipeline_audit": signal_pipeline_audit,
            "divergence_summary": divergence_summary,
            "divergences": divergences,
            "long_decision_checks": [],
            "position_management_replay": management_replay,
            "management_replay_coverage": management_coverage,
            "stop_loss_diagnostics": stop_loss_diagnostics,
        }
        replay_entry_gate = build_replay_entry_diagnostic_gate(report)
        report["replay_entry_diagnostic_gate"] = replay_entry_gate
        report["summary"]["replay_entry_gate_status"] = replay_entry_gate["status"]
        report["summary"]["replay_entry_gate_reason"] = replay_entry_gate["reason"]
        report["summary"]["replay_entry_gate_passed"] = bool(
            replay_entry_gate["passed"]
        )
        if persist:
            _write_json(self.output_path, report)
        return report

    def _replay_position_management(
        self,
        *,
        dm: DayManager,
        journal_entries: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        relevant_rows = [
            row
            for row in journal_entries
            if isinstance(row.get("position"), dict)
            and str(
                row.get("symbol") or row.get("position", {}).get("symbol") or ""
            ).strip()
        ]
        unavailable_rows = [
            row
            for row in journal_entries
            if isinstance(row, dict)
            and row not in relevant_rows
            and str(row.get("final_action", {}).get("action") or "").lower().strip()
            == "unavailable_source"
        ]
        if not relevant_rows:
            return {
                "summary": {
                    "events_replayed": 0,
                    "action_differences": 0,
                    "exit_like_signals": 0,
                    "hard_stop_exit_signals": 0,
                    "recorded_trim_events": 0,
                    "recorded_exit_events": 0,
                    "trim_signals": 0,
                    "watch_signals": 0,
                    "unavailable_source_events": len(unavailable_rows),
                    "coverage_status": "incomplete" if unavailable_rows else "empty",
                },
                "events": [],
                "unavailable_events": unavailable_rows,
                "hard_stop_audit_rows": [],
                "first_exit_like_by_symbol": [],
                "first_hard_stop_by_symbol": [],
            }

        events: List[Dict[str, Any]] = []
        first_exit_like_by_symbol: Dict[str, Dict[str, Any]] = {}
        first_hard_stop_by_symbol: Dict[str, Dict[str, Any]] = {}
        hard_stop_audit_rows: List[Dict[str, Any]] = []
        action_differences = 0
        exit_like_signals = 0
        hard_stop_exit_signals = 0
        recorded_trim_events = 0
        recorded_exit_events = 0
        trim_signals = 0
        watch_signals = 0

        for row in relevant_rows:
            timestamp_raw = row.get("timestamp")
            if not timestamp_raw:
                continue
            timestamp = _coerce_local_timestamp(str(timestamp_raw))
            self._current_replay_time = timestamp

            # Refresh entry authority with simulated time
            dm._refresh_entry_authority_state(now_local=timestamp)

            position_payload = dict(row.get("position") or {})
            symbol = (
                str(row.get("symbol") or position_payload.get("symbol") or "")
                .upper()
                .strip()
            )
            entry_time_raw = (
                row.get("entry_time")
                or row.get("entered_at")
                or position_payload.get("entry_time")
                or position_payload.get("entered_at")
            )
            entry_time = (
                _coerce_local_timestamp(str(entry_time_raw))
                if entry_time_raw
                else timestamp
            )
            qty = int(position_payload.get("qty") or 0)
            if qty <= 0:
                continue
            current_price = float(position_payload.get("current_price") or 0.0)
            entry_price = float(
                position_payload.get("entry_price")
                or position_payload.get("avg_entry_price")
                or 0.0
            )
            if current_price <= 0 or entry_price <= 0:
                continue
            unrealized_plpc = float(
                position_payload.get("unrealized_plpc")
                if position_payload.get("unrealized_plpc") is not None
                else (current_price - entry_price) / entry_price
            )
            dm._entry_time_overrides[symbol] = entry_time
            dm.position_entries[symbol] = entry_time.astimezone(UTC)
            synthetic_position = SimpleNamespace(
                symbol=symbol,
                avg_entry_price=entry_price,
                avg_entry=entry_price,
                entry_price=entry_price,
                current_price=current_price,
                qty=qty,
                unrealized_pl=float(
                    position_payload.get("unrealized_pl")
                    or position_payload.get("unrealized_pnl")
                    or ((current_price - entry_price) * qty)
                ),
                unrealized_pnl=float(
                    position_payload.get("unrealized_pnl")
                    or position_payload.get("unrealized_pl")
                    or ((current_price - entry_price) * qty)
                ),
                unrealized_plpc=unrealized_plpc,
                pnl_pct=unrealized_plpc * 100.0,
                market_value=float(
                    position_payload.get("market_value") or current_price * qty
                ),
                cost_basis=float(
                    position_payload.get("cost_basis") or entry_price * qty
                ),
                high_since_entry=position_payload.get("high_since_entry"),
                side=str(position_payload.get("side") or "long"),
            )
            replay_health = self._evaluate_management_health(
                dm=dm,
                position=synthetic_position,
                replay_time=timestamp,
                entry_time=entry_time,
            )
            replay_action = str(replay_health.get("action") or "").lower().strip()
            recorded_action = (
                str(row.get("final_action", {}).get("action") or "").lower().strip()
            )
            source = str(row.get("_replay_source") or "workflow_journal")
            if recorded_action == "trim":
                recorded_trim_events += 1
            if recorded_action in {"exit", "stop", "sell"}:
                recorded_exit_events += 1
            differs = bool(
                replay_action and recorded_action and replay_action != recorded_action
            )
            if differs:
                action_differences += 1
            if replay_action == "watch":
                watch_signals += 1
            if replay_action == "trim":
                trim_signals += 1
            if replay_action in {"trim", "exit"}:
                exit_like_signals += 1
                first_exit_like_by_symbol.setdefault(
                    symbol,
                    {
                        "symbol": symbol,
                        "timestamp": timestamp.isoformat(),
                        "recorded_action": recorded_action,
                        "replay_action": replay_action,
                        "current_price": current_price,
                        "entry_price": entry_price,
                        "pnl_pct": round(unrealized_plpc * 100.0, 4),
                    },
                )
            if bool(replay_health.get("hard_stop_forced_exit")):
                hard_stop_exit_signals += 1
                hard_stop_row = {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "recorded_action": recorded_action,
                    "replay_action": replay_action,
                    "action": "hard_stop_exit",
                    "pnl_pct": round(unrealized_plpc * 100.0, 4),
                    "hard_stop_pct": replay_health.get("hard_stop_pct"),
                    "exit_reason": "; ".join(
                        str(reason)
                        for reason in (replay_health.get("signals") or [])
                        if str(reason).strip()
                    ),
                }
                hard_stop_audit_rows.append(hard_stop_row)
                first_hard_stop_by_symbol.setdefault(symbol, hard_stop_row)
            events.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol,
                    "source": source,
                    "recorded_action": recorded_action,
                    "replay_action": replay_action,
                    "differs": differs,
                    "hard_stop_forced_exit": bool(
                        replay_health.get("hard_stop_forced_exit")
                    ),
                    "hard_stop_pct": replay_health.get("hard_stop_pct"),
                    "entry_price": entry_price,
                    "current_price": current_price,
                    "pnl_pct": round(unrealized_plpc * 100.0, 4),
                    "hold_minutes": round(
                        max(0.0, (timestamp - entry_time).total_seconds() / 60.0), 2
                    ),
                }
            )

        coverage_status = "covered" if events else "empty"
        if unavailable_rows:
            coverage_status = "partial"
        return {
            "summary": {
                "events_replayed": len(events),
                "action_differences": action_differences,
                "exit_like_signals": exit_like_signals,
                "hard_stop_exit_signals": hard_stop_exit_signals,
                "recorded_trim_events": recorded_trim_events,
                "recorded_exit_events": recorded_exit_events,
                "trim_signals": trim_signals,
                "watch_signals": watch_signals,
                "unavailable_source_events": len(unavailable_rows),
                "coverage_status": coverage_status,
            },
            "events": events,
            "unavailable_events": unavailable_rows,
            "hard_stop_audit_rows": hard_stop_audit_rows,
            "first_exit_like_by_symbol": sorted(
                first_exit_like_by_symbol.values(),
                key=lambda row: (row.get("timestamp", ""), row.get("symbol", "")),
            ),
            "first_hard_stop_by_symbol": sorted(
                first_hard_stop_by_symbol.values(),
                key=lambda row: (row.get("timestamp", ""), row.get("symbol", "")),
            ),
        }

    def _trade_journal_management_events(
        self,
        trade_journal: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        management_types = {
            "trim": "trim",
            "trim_oversized_position": "trim",
            "exit": "exit",
            "stop": "exit",
            "sell": "exit",
            "scale_out": "trim",
        }
        for row in trade_journal or []:
            if not isinstance(row, dict):
                continue
            trade_type = str(row.get("trade_type") or "").lower().strip()
            recorded_action = management_types.get(trade_type)
            if not recorded_action:
                continue
            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            timestamp = str(
                row.get("_local_timestamp") or _trade_journal_event_timestamp(row)
            )
            if not timestamp:
                continue
            qty = int(
                float(row.get("quantity", row.get("qty", row.get("shares", 0))) or 0)
            )
            entry_price = _safe_float(row.get("entry_price") or row.get("price")) or 0.0
            current_price = (
                _safe_float(row.get("exit_price") or row.get("fill_price"))
                or entry_price
            )
            if qty <= 0 or entry_price <= 0 or current_price <= 0:
                events.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "final_action": {
                            "action": "unavailable_source",
                            "source_action": recorded_action,
                            "reason": "trade_journal_missing_position_fields",
                        },
                        "_replay_source": "trade_journal",
                        "_source_trade_type": trade_type,
                    }
                )
                continue
            entry_time = str(row.get("entry_time") or timestamp)
            unrealized_plpc = (
                float((current_price - entry_price) / entry_price)
                if entry_price > 0
                else 0.0
            )
            events.append(
                {
                    "timestamp": timestamp,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "position": {
                        "symbol": symbol,
                        "entry_price": entry_price,
                        "avg_entry_price": entry_price,
                        "current_price": current_price,
                        "qty": qty,
                        "unrealized_plpc": unrealized_plpc,
                        "pnl_pct": unrealized_plpc * 100.0,
                        "cost_basis": entry_price * qty,
                        "market_value": current_price * qty,
                    },
                    "final_action": {
                        "action": recorded_action,
                        "source_action": recorded_action,
                        "trade_type": trade_type,
                    },
                    "_replay_source": "trade_journal",
                    "_source_trade_type": trade_type,
                }
            )
        return events

    def _build_management_coverage_diagnostics(
        self,
        *,
        management_replay: Dict[str, Any],
        actual_day_context: Dict[str, Any],
        workflow_journal: Sequence[Dict[str, Any]],
        trade_journal_management_events: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        summary = dict(management_replay.get("summary") or {})
        production_management_total = int(
            actual_day_context.get("trims_journaled", 0) or 0
        ) + int(actual_day_context.get("exits_journaled", 0) or 0)
        events_replayed = int(summary.get("events_replayed", 0) or 0)
        status = "covered"
        problem = ""
        if production_management_total > 0 and events_replayed <= 0:
            status = "incomplete"
            problem = "production_management_events_not_replayed"
        elif (
            production_management_total > 0
            and int(summary.get("recorded_trim_events", 0) or 0) <= 0
        ):
            status = "degraded"
            problem = "production_trim_events_not_classified"
        elif int(summary.get("unavailable_source_events", 0) or 0) > 0:
            status = "partial"
            problem = "some_management_events_missing_position_fields"
        return {
            "status": status,
            "problem": problem,
            "production_management_events": production_management_total,
            "production_trim_events": int(
                actual_day_context.get("trims_journaled", 0) or 0
            ),
            "production_exit_events": int(
                actual_day_context.get("exits_journaled", 0) or 0
            ),
            "events_replayed": events_replayed,
            "workflow_journal_events_loaded": len(workflow_journal or []),
            "trade_journal_management_events_loaded": len(
                trade_journal_management_events or []
            ),
            "workflow_journal_path": str(
                self.logs_dir / f"workflow_journal_{self.session_date}.jsonl"
            ),
            "trade_journal_path": str(self.logs_dir / "trade_journal.json"),
        }

    def _build_stop_loss_diagnostics(
        self,
        management_replay: Dict[str, Any],
    ) -> Dict[str, Any]:
        unresolved_by_symbol: Counter = Counter()
        resolved_by_action: Counter = Counter()
        warning_like_events = 0
        for event in management_replay.get("events", []) or []:
            if not isinstance(event, dict):
                continue
            replay_action = str(event.get("replay_action") or "").lower().strip()
            recorded_action = str(event.get("recorded_action") or "").lower().strip()
            hard_stop = bool(event.get("hard_stop_forced_exit"))
            stop_like = hard_stop or "stop" in recorded_action
            if not stop_like:
                continue
            warning_like_events += 1
            symbol = str(event.get("symbol") or "").upper().strip()
            if replay_action in {"trim", "exit"}:
                resolved_by_action[replay_action] += 1
            elif recorded_action in {"trim", "exit", "sell"}:
                resolved_by_action[f"production_{recorded_action}"] += 1
            else:
                unresolved_by_symbol[symbol] += 1
        repeated_unresolved = [
            {"symbol": symbol, "count": int(count)}
            for symbol, count in unresolved_by_symbol.most_common()
            if count > 1
        ]
        return {
            "stop_loss_like_events": int(warning_like_events),
            "resolved_by_action": dict(resolved_by_action),
            "unresolved_stop_loss_warnings": int(sum(unresolved_by_symbol.values())),
            "repeated_unresolved_stop_loss_warnings": repeated_unresolved[:10],
            "status": "degraded" if repeated_unresolved else "covered",
        }

    def _build_intraday_bar_coverage(
        self,
        *,
        signals: Sequence[Dict[str, Any]],
        decisions: Sequence[Dict[str, Any]],
        trade_journal: Sequence[Dict[str, Any]],
        workflow_actions: Sequence[Dict[str, Any]] = (),
    ) -> Dict[str, Any]:
        source_by_symbol: Dict[str, str] = {}
        for row in signals or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("ticker") or row.get("symbol") or "").upper().strip()
            if symbol:
                source_by_symbol.setdefault(
                    symbol,
                    str(
                        row.get("source_bucket") or row.get("entry_source") or "signals"
                    ),
                )
        for row in decisions or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").upper().strip()
            if symbol:
                source_by_symbol.setdefault(symbol, "recorded_decision")
        for row in trade_journal or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").upper().strip()
            if symbol:
                source_by_symbol.setdefault(symbol, "trade_journal")
        for action in workflow_actions or []:
            if not isinstance(action, dict):
                continue
            for symbol in action.get("selected_symbols", []) or []:
                symbol_key = str(symbol or "").upper().strip()
                if symbol_key:
                    source_by_symbol.setdefault(symbol_key, "workflow_selected")
        rows = []
        for symbol in sorted(source_by_symbol):
            frame = self.market_bars.get(symbol)
            bars = int(len(frame)) if frame is not None else 0
            rows.append(
                {
                    "symbol": symbol,
                    "bars": bars,
                    "source_bucket": source_by_symbol.get(symbol, ""),
                    "archive_loaded": symbol
                    in set(self._archive_diagnostics.get("loaded_symbols", [])),
                    "archive_missing": symbol
                    in set(self._archive_diagnostics.get("missing_symbols", [])),
                    "live_fetch": symbol
                    in set(self._archive_diagnostics.get("live_fetch_symbols", [])),
                }
            )
        zero_rows = [row for row in rows if int(row.get("bars", 0) or 0) <= 0]
        actionable_rows = [
            row
            for row in rows
            if str(row.get("source_bucket") or "") != "workflow_selected"
        ]
        actionable_zero_rows = [
            row for row in actionable_rows if int(row.get("bars", 0) or 0) <= 0
        ]
        by_source = Counter(
            str(row.get("source_bucket") or "unknown") for row in zero_rows
        )
        return {
            "symbols_scored": len(rows),
            "symbols_with_bars": len(rows) - len(zero_rows),
            "symbols_with_zero_bars": len(zero_rows),
            "actionable_symbols_scored": len(actionable_rows),
            "actionable_symbols_with_bars": len(actionable_rows)
            - len(actionable_zero_rows),
            "actionable_symbols_with_zero_bars": len(actionable_zero_rows),
            "diagnostic_only_zero_bar_symbols": max(
                0, len(zero_rows) - len(actionable_zero_rows)
            ),
            "top_missing_symbols": zero_rows[:15],
            "top_actionable_missing_symbols": actionable_zero_rows[:15],
            "zero_bar_symbols_by_source_bucket": dict(by_source),
            "interpretation": (
                "zero_bar_symbols_are_diagnostic_only"
                if zero_rows and not actionable_zero_rows
                else "actionable_symbols_missing_intraday_bars"
                if actionable_zero_rows
                else "covered"
            ),
        }

    def _build_replay_order_accounting(
        self,
        *,
        order_breakdown: Dict[str, Any],
        workflow_summary: Optional[Dict[str, Any]] = None,
        counterfactual_candidates: int = 0,
    ) -> Dict[str, Any]:
        workflow_summary = dict(workflow_summary or {})
        selected = int(workflow_summary.get("selected_symbols_total", 0) or 0)
        adds = int(workflow_summary.get("add_selected_total", 0) or 0)
        actual_orders = int(order_breakdown.get("total", 0) or 0)
        return {
            "actual_replay_order_submissions": actual_orders,
            "actual_replay_orders_by_bucket": dict(
                order_breakdown.get("by_bucket", {})
            ),
            "selected_candidates": selected,
            "diagnostic_only_candidates": max(0, selected + adds - actual_orders),
            "counterfactual_candidates": int(counterfactual_candidates or 0),
            "note": (
                "selected_candidates are DayManager diagnostics unless they also appear "
                "in actual_replay_order_submissions"
            ),
        }

    def _evaluate_management_health(
        self,
        *,
        dm: DayManager,
        position: SimpleNamespace,
        replay_time: datetime,
        entry_time: datetime,
    ) -> Dict[str, Any]:
        hold_minutes = int(max(0.0, (replay_time - entry_time).total_seconds() / 60.0))
        dm._hold_minutes = lambda symbol: hold_minutes
        dm._entry_time_for_advisor = lambda symbol: entry_time.replace(tzinfo=None)
        with self._patched_runtime_clock_for(replay_time):
            health = dict(dm.calculate_position_health(position) or {})
        health.setdefault("action", "")
        health.setdefault("signals", [])
        return health

    def _evaluate_management_action(
        self,
        *,
        dm: DayManager,
        position: SimpleNamespace,
        replay_time: datetime,
        entry_time: datetime,
    ) -> str:
        health = self._evaluate_management_health(
            dm=dm,
            position=position,
            replay_time=replay_time,
            entry_time=entry_time,
        )
        return str(health.get("action") or "").lower().strip()

    def _classify_replay_order(self, *, symbol: str, side: str, context: str) -> str:
        """Bucket label for replay order accounting."""
        ctx = (context or "").lower()
        if "inverse" in ctx or "hedge" in ctx:
            return "inverse_etf"
        if is_inverse_etf(symbol):
            return "inverse_etf"
        if "trim" in ctx:
            return "trim"
        if side == "buy":
            return "equity_entry"
        if side == "sell":
            return "equity_exit"
        return "other"

    def _capture_inverse_order(self, **kwargs: Any) -> SimpleNamespace:
        symbol = str(kwargs.get("symbol") or "").upper()
        order_id = f"replay-order-{len(self._replay_orders) + 1}"
        replay_time = (
            self._current_replay_time or _session_minutes(self.session_date)[0]
        )
        fill_price = float(self._current_price_for_symbol(symbol) or 0.0)
        profile = self.inverse_etf_manager.get_instrument_profile(symbol)
        side = str(kwargs.get("side") or "").lower()
        context = str(kwargs.get("context") or "")
        qty = int(kwargs.get("qty") or 0)
        bucket = self._classify_replay_order(symbol=symbol, side=side, context=context)

        # Reject phantom orders for symbols with no minute-bar data. DM should
        # not have selected these (the replay's get_current_price returns 0),
        # but it does. Without this gate, the replay reports 100s of "entries"
        # against price=0 symbols that were never actually tradable in the
        # archived session. Track them in a separate skip bucket so the
        # divergence is visible.
        if side == "buy" and (qty <= 0 or fill_price <= 0):
            skip_event = {
                "id": order_id,
                "timestamp": replay_time.isoformat(),
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "context": context,
                "bucket": "data_gap_skipped",
                "skip_reason": ("no_price_data" if fill_price <= 0 else "zero_qty"),
                "fill_price": fill_price,
                "notional": 0.0,
            }
            self._replay_data_gap_skips.append(skip_event)
            return SimpleNamespace(id=order_id, **kwargs)

        order_event = {
            "id": order_id,
            "timestamp": replay_time.isoformat(),
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "context": context,
            "bucket": bucket,
            # Kept as `entry_price` for backwards-compat with downstream tooling
            # (it's actually the fill price for either side).
            "entry_price": fill_price,
            "fill_price": fill_price,
            "notional": float(fill_price * qty),
            "leverage": int(profile.get("leverage", 1) or 1),
        }
        self._replay_orders.append(order_event)

        # Inverse ETFs have their own simulation engine in
        # _evaluate_inverse_positions; we maintain self._positions for them.
        # Equity orders belong to the plan_generator's seed_positions (which is
        # what DM actually reads via plan_generator.get_current_positions()).
        if bucket == "inverse_etf":
            if side == "buy" and qty > 0 and fill_price > 0:
                self._positions.append(
                    SimpleNamespace(
                        symbol=symbol,
                        qty=qty,
                        avg_entry_price=fill_price,
                        avg_entry=fill_price,
                        entry_price=fill_price,
                        current_price=fill_price,
                        market_value=float(fill_price * qty),
                        cost_basis=float(fill_price * qty),
                        unrealized_pl=0.0,
                        unrealized_pnl=0.0,
                        unrealized_plpc=0.0,
                        pnl_pct=0.0,
                        side="long",
                        entry_time=replay_time,
                        profile=profile,
                        order_id=order_id,
                        context=context,
                    )
                )
            elif side == "sell" and qty > 0:
                self._apply_replay_sell_fill(
                    symbol=symbol,
                    qty=qty,
                    fill_price=fill_price,
                    timestamp=replay_time,
                    order_id=order_id,
                    context=context,
                )
        else:
            # Equity flow. apply_replay_entry for the buy side is invoked by the
            # workflow path (_capture_replay_cycle_result), so we only handle
            # the exit drain here. Without this, DM keeps re-issuing the same
            # exit every minute because seed_positions never decrements.
            agent = self._workflow_agent
            plan_gen = (
                getattr(agent, "plan_generator", None) if agent is not None else None
            )
            if (
                side == "sell"
                and qty > 0
                and fill_price > 0
                and plan_gen is not None
                and hasattr(plan_gen, "apply_replay_exit")
            ):
                closed = plan_gen.apply_replay_exit(
                    symbol=symbol,
                    qty=qty,
                    price=fill_price,
                    exit_context=context,
                )
                self._workflow_action_log.append(
                    {
                        "timestamp": replay_time.isoformat(),
                        "symbol": symbol,
                        "action": "equity_sell_fill",
                        "qty": int(closed),
                        "requested_qty": int(qty),
                        "fill_price": float(fill_price),
                        "order_id": order_id,
                        "context": context,
                    }
                )
        return SimpleNamespace(id=order_id, **kwargs)

    def _apply_replay_sell_fill(
        self,
        *,
        symbol: str,
        qty: int,
        fill_price: float,
        timestamp: datetime,
        order_id: str,
        context: str,
    ) -> None:
        """Mutate replay position state after a simulated sell fill."""
        symbol_key = str(symbol or "").upper().strip()
        remaining_qty = int(qty or 0)
        if not symbol_key or remaining_qty <= 0:
            return
        for position in list(self._positions):
            if remaining_qty <= 0:
                break
            if str(getattr(position, "symbol", "") or "").upper() != symbol_key:
                continue
            position_qty = int(float(getattr(position, "qty", 0) or 0))
            if position_qty <= 0:
                continue
            closed_qty = min(position_qty, remaining_qty)
            remaining_qty -= closed_qty
            new_qty = position_qty - closed_qty
            avg_entry = float(
                getattr(
                    position,
                    "avg_entry_price",
                    getattr(position, "entry_price", fill_price),
                )
                or fill_price
            )
            if new_qty <= 0:
                self._positions.remove(position)
            else:
                position.qty = new_qty
                position.market_value = float(new_qty * fill_price)
                position.cost_basis = float(new_qty * avg_entry)
                position.current_price = float(fill_price)
                position.unrealized_pl = float((fill_price - avg_entry) * new_qty)
                position.unrealized_pnl = position.unrealized_pl
                position.unrealized_plpc = (
                    float((fill_price - avg_entry) / avg_entry)
                    if avg_entry > 0
                    else 0.0
                )
                position.pnl_pct = float(position.unrealized_plpc * 100.0)
            self._workflow_action_log.append(
                {
                    "timestamp": timestamp.isoformat(),
                    "symbol": symbol_key,
                    "action": "sell_fill",
                    "qty": int(closed_qty),
                    "remaining_qty": int(new_qty),
                    "fill_price": float(fill_price),
                    "order_id": str(order_id),
                    "context": str(context or ""),
                }
            )

    def _bars_through_time(self, symbol: str, replay_time: datetime) -> pd.DataFrame:
        ticker = str(symbol or "").upper()
        if ticker not in self.market_bars:
            client = self.data_client or create_data_client(require_credentials=True)
            self.market_bars[ticker] = self._fetch_extended_minute_bars(client, ticker)
        frame = self.market_bars.get(ticker)
        if frame is None or frame.empty:
            return pd.DataFrame()
        return frame[frame.index <= pd.Timestamp(replay_time)]

    def _evaluate_inverse_positions(self, replay_minute: datetime) -> None:
        active_positions = list(self._positions)
        for position in active_positions:
            symbol = str(getattr(position, "symbol", "") or "").upper()
            entry_time = getattr(position, "entry_time", None)
            entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
            qty = int(getattr(position, "qty", 0) or 0)
            profile = dict(getattr(position, "profile", {}) or {})
            if not symbol or entry_time is None or qty <= 0 or entry_price <= 0:
                continue
            if replay_minute <= entry_time:
                continue
            bars = self._bars_through_time(symbol, replay_minute)
            if bars.empty:
                continue
            current_price = float(bars.iloc[-1]["close"])
            pnl_pct = ((current_price - entry_price) / entry_price) * 100.0
            exit_reason = ""
            if pnl_pct <= float(profile.get("stop_loss_pct", -4.0) or -4.0):
                exit_reason = "hedge_stop_loss"
            elif pnl_pct >= float(profile.get("profit_target_pct", 6.0) or 6.0):
                exit_reason = "profit_target_hit"
            else:
                reversal_signal = self.inverse_etf_manager.evaluate_intraday_reversal(
                    symbol,
                    bars=bars.tail(20),
                )
                if bool(reversal_signal.get("should_exit")):
                    reversal_reasons = ",".join(reversal_signal.get("reasons", []))
                    exit_reason = (
                        f"intraday_reversal:{reversal_reasons}"
                        if reversal_reasons
                        else "intraday_reversal"
                    )
                elif replay_minute.time() >= time(15, 0):
                    exit_reason = "eod_no_overnight"
            if not exit_reason:
                continue
            self._close_inverse_position(
                position,
                exit_time=replay_minute,
                exit_price=current_price,
                exit_reason=exit_reason,
            )

    def _close_inverse_position(
        self,
        position: SimpleNamespace,
        *,
        exit_time: datetime,
        exit_price: float,
        exit_reason: str,
    ) -> None:
        if position not in self._positions:
            return
        symbol = str(getattr(position, "symbol", "") or "").upper()
        qty = int(getattr(position, "qty", 0) or 0)
        entry_price = float(getattr(position, "entry_price", 0.0) or 0.0)
        entry_time = getattr(position, "entry_time", None) or exit_time
        pnl_dollars = (float(exit_price) - entry_price) * qty
        return_pct = ((float(exit_price) - entry_price) / entry_price) * 100.0
        hold_minutes = max(0.0, (exit_time - entry_time).total_seconds() / 60.0)
        self._inverse_trade_results.append(
            {
                "symbol": symbol,
                "qty": qty,
                "entry_time": entry_time.isoformat(),
                "entry_price": entry_price,
                "exit_time": exit_time.isoformat(),
                "exit_price": float(exit_price),
                "hold_minutes": round(hold_minutes, 2),
                "return_pct": round(return_pct, 4),
                "pnl_dollars": round(pnl_dollars, 2),
                "exit_reason": exit_reason,
                "order_id": str(getattr(position, "order_id", "") or ""),
                "context": str(getattr(position, "context", "") or ""),
                "leverage": int(
                    (getattr(position, "profile", {}) or {}).get("leverage", 1) or 1
                ),
            }
        )
        self._positions.remove(position)

    def _finalize_inverse_positions(self) -> None:
        session_close = datetime.combine(
            date.fromisoformat(self.session_date),
            time(15, 0),
            tzinfo=CT,
        )
        for position in list(self._positions):
            symbol = str(getattr(position, "symbol", "") or "").upper()
            price = float(self._current_price_for_symbol(symbol) or 0.0)
            if price <= 0:
                bars = self._bars_through_time(symbol, session_close)
                if not bars.empty:
                    price = float(bars.iloc[-1]["close"])
            if price <= 0:
                continue
            self._close_inverse_position(
                position,
                exit_time=session_close,
                exit_price=price,
                exit_reason="eod_no_overnight",
            )

    def _build_data_quality_status(self) -> Dict[str, Any]:
        """Surface archive-coverage degradation as a top-level warning."""
        loaded = list(self._archive_diagnostics.get("loaded_symbols", []) or [])
        missing = list(self._archive_diagnostics.get("missing_symbols", []) or [])
        live = list(self._archive_diagnostics.get("live_fetch_symbols", []) or [])
        total_requested = len(loaded) + len(missing)
        coverage_pct = (
            float(len(loaded)) / float(total_requested) if total_requested > 0 else 0.0
        )
        warnings: List[str] = []
        if missing and not live:
            warnings.append(
                "minute_bar_archive_incomplete: "
                f"{len(missing)} symbols missing with no live fetch fallback "
                "— decisions on those symbols silently no-op"
            )
        if total_requested > 0 and coverage_pct < 0.75:
            warnings.append(
                f"low_archive_coverage: {coverage_pct * 100.0:.1f}% of requested symbols loaded"
            )
        status = "ok"
        if warnings:
            status = "degraded" if coverage_pct >= 0.5 else "critical"
        return {
            "status": status,
            "coverage_mode": self._market_data_coverage_mode(),
            "coverage_pct": round(coverage_pct, 4),
            "symbols_requested": total_requested,
            "symbols_loaded": len(loaded),
            "symbols_missing": len(missing),
            "symbols_live_fetch": len(live),
            "missing_symbols_examples": sorted(missing)[:15],
            "warnings": warnings,
        }

    def _summarize_replay_orders(self) -> Dict[str, Any]:
        """Bucket the captured replay orders into typed counters."""
        buckets: Dict[str, int] = {
            "equity_entry": 0,
            "equity_exit": 0,
            "inverse_etf": 0,
            "trim": 0,
            "other": 0,
        }
        unique_by_bucket: Dict[str, set[str]] = {key: set() for key in buckets}
        per_symbol_exits: Counter = Counter()
        for order in self._replay_orders:
            bucket = str(order.get("bucket") or "other")
            if bucket not in buckets:
                bucket = "other"
            buckets[bucket] += 1
            symbol = str(order.get("symbol") or "").upper()
            if symbol:
                unique_by_bucket[bucket].add(symbol)
                if (
                    bucket in {"equity_exit", "inverse_etf"}
                    and order.get("side") == "sell"
                ):
                    per_symbol_exits[symbol] += 1
        most_repeated = per_symbol_exits.most_common(5)
        # Tally data-gap rejections by reason and unique symbol.
        skip_unique: set[str] = set()
        skip_reasons: Counter = Counter()
        for skip in self._replay_data_gap_skips:
            symbol = str(skip.get("symbol") or "").upper()
            if symbol:
                skip_unique.add(symbol)
            skip_reasons[str(skip.get("skip_reason") or "unknown")] += 1
        return {
            "total": len(self._replay_orders),
            "by_bucket": dict(buckets),
            "unique_symbols_by_bucket": {
                key: sorted(value) for key, value in unique_by_bucket.items()
            },
            "unique_symbol_count_by_bucket": {
                key: len(value) for key, value in unique_by_bucket.items()
            },
            "max_repeated_exits_per_symbol": (
                {"symbol": most_repeated[0][0], "count": int(most_repeated[0][1])}
                if most_repeated
                else None
            ),
            "top_repeated_exit_symbols": [
                {"symbol": sym, "count": int(count)} for sym, count in most_repeated
            ],
            # Phantom-order rejections: DM tried to buy/sell symbols where the
            # archive returned no minute bars. Production never traded these
            # because they had real data; the replay rejects them so they
            # don't inflate the entry count.
            "data_gap_skipped_total": len(self._replay_data_gap_skips),
            "data_gap_skipped_unique_symbols": len(skip_unique),
            "data_gap_skipped_unique_symbols_examples": sorted(skip_unique)[:15],
            "data_gap_skipped_by_reason": dict(skip_reasons),
        }

    def _summarize_inverse_trade_results(self) -> Dict[str, Any]:
        trades = list(self._inverse_trade_results)
        if not trades:
            return {
                "closed_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "avg_return_pct": 0.0,
                "best_trade": None,
                "worst_trade": None,
            }
        wins = sum(
            1 for trade in trades if float(trade.get("pnl_dollars", 0.0) or 0.0) > 0
        )
        losses = sum(
            1 for trade in trades if float(trade.get("pnl_dollars", 0.0) or 0.0) < 0
        )
        net_pnl = sum(float(trade.get("pnl_dollars", 0.0) or 0.0) for trade in trades)
        avg_return_pct = sum(
            float(trade.get("return_pct", 0.0) or 0.0) for trade in trades
        ) / len(trades)
        best_trade = max(
            trades, key=lambda trade: float(trade.get("pnl_dollars", 0.0) or 0.0)
        )
        worst_trade = min(
            trades, key=lambda trade: float(trade.get("pnl_dollars", 0.0) or 0.0)
        )
        return {
            "closed_trades": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate": round((wins / len(trades)) * 100.0, 2),
            "net_pnl": round(net_pnl, 2),
            "avg_return_pct": round(avg_return_pct, 4),
            "best_trade": dict(best_trade),
            "worst_trade": dict(worst_trade),
        }

    def _phase_for_time(self, replay_time: datetime) -> TradingPhase:
        minute_of_day = replay_time.hour * 60 + replay_time.minute
        import autotrade.core.day_manager as day_manager_mod

        if minute_of_day < day_manager_mod.PHASE_OBSERVATION_START:
            return TradingPhase.PREMARKET
        if minute_of_day < day_manager_mod.PHASE_RESEARCH_START:
            return TradingPhase.OBSERVATION
        if minute_of_day < day_manager_mod.PHASE_CORE_START:
            return TradingPhase.RESEARCH
        if minute_of_day < day_manager_mod.PHASE_WINDDOWN_START:
            return TradingPhase.CORE_TRADING
        if minute_of_day < day_manager_mod.PHASE_MARKET_CLOSE:
            return TradingPhase.WIND_DOWN
        return TradingPhase.AFTER_HOURS

    @contextlib.contextmanager
    def _patched_runtime_clock_for(self, replay_time: datetime) -> Iterable[None]:
        original_datetime = day_manager_mod.datetime

        class ReplayDateTime(original_datetime):
            @classmethod
            def now(cls, tz=None):
                if tz is not None:
                    return replay_time.astimezone(tz)
                return replay_time.replace(tzinfo=None)

        day_manager_mod.datetime = ReplayDateTime
        try:
            yield
        finally:
            day_manager_mod.datetime = original_datetime

    @contextlib.contextmanager
    def _patched_runtime_clock(self) -> Iterable[None]:
        replay_time = self._current_replay_time
        if replay_time is None:
            yield
            return
        with self._patched_runtime_clock_for(replay_time):
            yield

    @contextlib.contextmanager
    def _patched_inverse_screen(self) -> Iterable[None]:
        from autotrade.signals import inverse_etf_screener as screener_mod
        from autotrade.utils import financial_db as financial_db_mod

        replay_time = self._current_replay_time
        original_screener = screener_mod.InverseETFScreener
        original_db = financial_db_mod.FinancialDB

        if self.inverse_screen_provider is not None:
            provider = self.inverse_screen_provider

            class ReplayScreener:
                def __init__(self, *args, **kwargs):
                    pass

                def screen_universe(
                    self,
                    regime,
                    portfolio_holdings=None,
                    entry_mode="",
                    minutes_since_open=None,
                    sources_degraded=False,
                ):
                    return provider(replay_time, regime, portfolio_holdings)

            screener_mod.InverseETFScreener = ReplayScreener
            financial_db_mod.FinancialDB = lambda: object()
        else:
            harness = self
            rows = self._load_inverse_universe_rows()

            class ReplayFinancialDB:
                def get_all_inverse_etfs(self, active_only=True):
                    return list(rows)

                def upsert_screen_result(self, payload):
                    return None

            class ReplayScreener(original_screener):
                def _fetch_intraday_bars(self, ticker):
                    return harness._historical_inverse_bars(ticker)

            screener_mod.InverseETFScreener = ReplayScreener
            financial_db_mod.FinancialDB = ReplayFinancialDB
        try:
            yield
        finally:
            screener_mod.InverseETFScreener = original_screener
            financial_db_mod.FinancialDB = original_db


class _ReplayTradingClient:
    def __init__(self, plan_generator: _ReplayPlanGenerator) -> None:
        self._plan_generator = plan_generator
        self.cancelled_order_ids: List[str] = []

    def get_account(self) -> SimpleNamespace:
        return SimpleNamespace(**self._plan_generator.get_account_info())

    def get_orders(self, *args: Any, **kwargs: Any) -> List[Any]:
        return []

    def cancel_order_by_id(self, order_id: Any) -> None:
        self.cancelled_order_ids.append(str(order_id))
        return None


def replay_runtime_session(
    session_date: str,
    *,
    mode: str = REPLAY_MODE_AGENT_WORKFLOW,
    profile: str = REPLAY_PROFILE_FULL,
    project_dir: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    strategy_eval: Optional[str] = None,
    strategy_variant: str = DEFAULT_DISCOVERY_VARIANT,
    force_modern_crash_protection: bool = False,
    market_bars: Optional[Dict[str, pd.DataFrame]] = None,
    previous_closes: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    replay = RuntimeSessionReplay(
        session_date=session_date,
        mode=mode,
        profile=profile,
        project_dir=project_dir,
        output_path=output_path,
        strategy_eval=strategy_eval,
        strategy_variant=strategy_variant,
        force_modern_crash_protection=force_modern_crash_protection,
        market_bars=market_bars,
        previous_closes=previous_closes,
    )
    return replay.run(persist=True)


def _benchmark_confidence(summary: Dict[str, Any]) -> str:
    if bool(summary.get("decision_artifact_missing")):
        return "medium"
    return "high"


def run_runtime_replay_benchmark(
    *,
    dates: Sequence[str],
    benchmark_set: str = "custom",
    mode: str = REPLAY_MODE_DAYMANAGER_CORE,
    profile: str = REPLAY_PROFILE_FULL,
    project_dir: Optional[Path | str] = None,
    output_path: Optional[Path | str] = None,
    strategy_eval: Optional[str] = None,
    strategy_variant: str = DEFAULT_DISCOVERY_VARIANT,
    force_modern_crash_protection: bool = False,
    persist: bool = True,
    completion_status_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    selected_dates = [str(item) for item in dates if str(item).strip()]
    if not selected_dates:
        raise ValueError("At least one benchmark date is required")
    project_root = Path(project_dir) if project_dir else PROJECT_DIR
    if output_path:
        benchmark_output = Path(output_path)
    else:
        benchmark_output = (
            project_root
            / "logs"
            / "runtime_replay_benchmark_{benchmark_set}{suffix}_{stamp}.json".format(
                benchmark_set=benchmark_set,
                suffix="_modern_crash" if force_modern_crash_protection else "",
                stamp=datetime.now().strftime("%Y%m%d_%H%M%S"),
            )
        )
    status_path = (
        Path(completion_status_path)
        if completion_status_path is not None
        else _benchmark_completion_status_path(benchmark_output)
    )
    emit_status = bool(persist or completion_status_path is not None)
    started_at = datetime.now().isoformat()
    benchmark_status: Dict[str, Any] = {
        "benchmark_set": benchmark_set,
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "failed_at": None,
        "output_path": str(benchmark_output),
        "completion_status_path": str(status_path),
        "mode": mode,
        "profile": profile,
        "dates_total": len(selected_dates),
        "completed_dates": 0,
        "current_date": None,
        "last_completed_date": None,
    }
    if emit_status:
        _write_benchmark_status(status_path, benchmark_status)

    day_rows: List[Dict[str, Any]] = []
    current_date = None
    try:
        for session_date in selected_dates:
            current_date = session_date
            report = RuntimeSessionReplay(
                session_date=session_date,
                mode=mode,
                profile=profile,
                project_dir=project_root,
                strategy_eval=strategy_eval,
                strategy_variant=strategy_variant,
                force_modern_crash_protection=force_modern_crash_protection,
            ).run(persist=False)
            summary = report.get("summary", {})
            divergence_summary = report.get("divergence_summary", {}) or {}
            actual_day_context = report.get("actual_day_context", {}) or {}
            handoff_diagnostics = report.get("handoff_diagnostics", {}) or {}
            discovery_eval = report.get("discovery_strategy_eval", {}) or {}
            confidence = _benchmark_confidence(summary)
            gating_tier = (
                "secondary" if session_date in NON_GATING_BENCHMARK_DATES else "primary"
            )
            day_rows.append(
                {
                    "date": session_date,
                    "force_modern_crash_protection": bool(
                        summary.get(
                            "force_modern_crash_protection",
                            force_modern_crash_protection,
                        )
                    ),
                    "gating_tier": gating_tier,
                    "confidence": confidence,
                    "recorded_decisions_total": int(
                        summary.get("recorded_decisions_total", 0) or 0
                    ),
                    "recorded_longs_executed": int(
                        summary.get("recorded_longs_executed", 0) or 0
                    ),
                    "replay_longs_allowed": int(
                        summary.get("replay_longs_allowed", 0) or 0
                    ),
                    "divergences_count": int(summary.get("divergences_count", 0) or 0),
                    "inverse_orders_captured": int(
                        summary.get("inverse_orders_captured", 0) or 0
                    ),
                    "inverse_net_pnl": float(
                        summary.get("inverse_net_pnl", 0.0) or 0.0
                    ),
                    "actual_net_pnl": float(summary.get("actual_net_pnl", 0.0) or 0.0),
                    "actual_total_trades": int(
                        summary.get("actual_total_trades", 0) or 0
                    ),
                    "management_replay_differences": int(
                        summary.get("management_replay_differences", 0) or 0
                    ),
                    "decision_artifact_missing": bool(
                        summary.get("decision_artifact_missing", False)
                    ),
                    "metadata_backfilled_checks": int(
                        summary.get("metadata_backfilled_checks", 0) or 0
                    ),
                    "metadata_inferred_checks": int(
                        summary.get("metadata_inferred_checks", 0) or 0
                    ),
                    "first_inverse_fast_at": str(
                        summary.get("first_inverse_fast_at") or ""
                    ),
                    "inverse_absence_reason": str(
                        summary.get("inverse_absence_reason") or ""
                    ),
                    "divergence_examples": list(divergence_summary.get("examples", [])),
                    "divergence_top_reasons": list(
                        divergence_summary.get("top_reasons", [])
                    ),
                    "handoff_status": str(
                        handoff_diagnostics.get("status") or "healthy"
                    ),
                    "handoff_problem": str(handoff_diagnostics.get("problem") or ""),
                    "handoff_recoverable_candidates": int(
                        handoff_diagnostics.get("recoverable_candidates_count", 0) or 0
                    ),
                    "handoff_recoverable_examples": list(
                        handoff_diagnostics.get("recoverable_candidates_examples", [])
                    ),
                    "handoff_signals_source": str(
                        handoff_diagnostics.get("signals_source") or ""
                    ),
                    "signal_alignment_status": str(
                        handoff_diagnostics.get("signal_alignment_status") or "aligned"
                    ),
                    "signal_alignment_problem": str(
                        handoff_diagnostics.get("signal_alignment_problem") or ""
                    ),
                    "signal_alignment_missing_decisions": int(
                        handoff_diagnostics.get(
                            "decision_symbols_missing_from_signals_count", 0
                        )
                        or 0
                    ),
                    "signal_alignment_missing_decision_examples": list(
                        handoff_diagnostics.get(
                            "decision_symbols_missing_from_signals_examples", []
                        )
                    ),
                    "eod_review_file": str(
                        actual_day_context.get("eod_review_file") or ""
                    ),
                    "discovery_variant": str(discovery_eval.get("variant") or ""),
                    "discovery_candidate_pool_size": int(
                        discovery_eval.get("candidate_pool_size", 0) or 0
                    ),
                    "discovery_candidates_evaluated": int(
                        discovery_eval.get("synthetic_candidates_evaluated", 0) or 0
                    ),
                    "discovery_added_winners": int(
                        discovery_eval.get("added_winners", 0) or 0
                    ),
                    "discovery_added_losers": int(
                        discovery_eval.get("added_losers", 0) or 0
                    ),
                    "discovery_net_synthetic_pnl": float(
                        discovery_eval.get("net_synthetic_pnl", 0.0) or 0.0
                    ),
                    "discovery_avg_return_pct": float(
                        discovery_eval.get("avg_return_pct", 0.0) or 0.0
                    ),
                    "discovery_slot_fill_delta": int(
                        discovery_eval.get("slot_fill_delta", 0) or 0
                    ),
                    "discovery_top_added_candidates": list(
                        discovery_eval.get("top_added_candidates", [])
                    ),
                    "notes": list(report.get("replay_notes", [])),
                }
            )
            benchmark_status.update(
                {
                    "completed_dates": len(day_rows),
                    "current_date": session_date,
                    "last_completed_date": session_date,
                    "last_completed_at": datetime.now().isoformat(),
                }
            )
            if emit_status:
                _write_benchmark_status(status_path, benchmark_status)

        payload = {
            "summary": {
                "benchmark_set": benchmark_set,
                "force_modern_crash_protection": force_modern_crash_protection,
                "dates_total": len(day_rows),
                "primary_dates": sum(
                    1 for row in day_rows if row["gating_tier"] == "primary"
                ),
                "secondary_dates": sum(
                    1 for row in day_rows if row["gating_tier"] == "secondary"
                ),
                "high_confidence_dates": sum(
                    1 for row in day_rows if row["confidence"] == "high"
                ),
                "medium_confidence_dates": sum(
                    1 for row in day_rows if row["confidence"] == "medium"
                ),
                "dates_with_divergences": sum(
                    1 for row in day_rows if row["divergences_count"] > 0
                ),
                "total_divergences": sum(row["divergences_count"] for row in day_rows),
                "inverse_net_pnl_total": round(
                    sum(row["inverse_net_pnl"] for row in day_rows),
                    2,
                ),
                "actual_net_pnl_total": round(
                    sum(row["actual_net_pnl"] for row in day_rows), 2
                ),
                "discovery_strategy_eval_enabled": bool(strategy_eval),
                "discovery_variant": strategy_variant
                if strategy_eval == "discovery"
                else "",
                "discovery_candidates_evaluated_total": sum(
                    row["discovery_candidates_evaluated"] for row in day_rows
                ),
                "discovery_added_winners_total": sum(
                    row["discovery_added_winners"] for row in day_rows
                ),
                "discovery_added_losers_total": sum(
                    row["discovery_added_losers"] for row in day_rows
                ),
                "discovery_net_synthetic_pnl_total": round(
                    sum(row["discovery_net_synthetic_pnl"] for row in day_rows),
                    2,
                ),
                "discovery_slot_fill_delta_total": sum(
                    row["discovery_slot_fill_delta"] for row in day_rows
                ),
            },
            "dates": day_rows,
            "status": "completed",
            "started_at": started_at,
            "completed_at": datetime.now().isoformat(),
            "failed_at": None,
            "output_path": str(benchmark_output),
            "completion_status_path": str(status_path),
            "completed_dates": len(day_rows),
            "current_date": current_date,
        }
        if persist:
            _write_json(benchmark_output, payload)
        if emit_status:
            _write_benchmark_status(status_path, payload)
        return payload
    except Exception as exc:
        benchmark_status.update(
            {
                "status": "failed",
                "failed_at": datetime.now().isoformat(),
                "error": str(exc),
                "completed_dates": len(day_rows),
                "current_date": current_date,
            }
        )
        if emit_status:
            _write_benchmark_status(status_path, benchmark_status)
        raise


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay one session through the runtime workflow or DayManager core path",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--date", help="Session date YYYY-MM-DD (local CT)")
    group.add_argument(
        "--benchmark-set",
        choices=sorted(DEFAULT_BENCHMARK_DATE_SETS.keys()),
        help="Run a named multi-date replay benchmark scorecard",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional output path for the replay report JSON",
    )
    parser.add_argument(
        "--completion-status-json",
        default="",
        help="Optional path for the benchmark completion status marker",
    )
    parser.add_argument(
        "--strategy-eval",
        choices=["discovery"],
        default="",
        help="Optional strategy evaluation mode for discovery-entry backtesting",
    )
    parser.add_argument(
        "--variant",
        default=DEFAULT_DISCOVERY_VARIANT,
        help="Strategy-eval variant name",
    )
    parser.add_argument(
        "--force-modern-crash-protection",
        action="store_true",
        help=(
            "Force the modern crash-open entry-authority logic for older sessions "
            "during replay and benchmark runs"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=REPLAY_MODE_CHOICES,
        default="",
        help=(
            "Replay mode. Daily --date defaults to agent_workflow in full profile "
            "and daymanager_core in fast profile; benchmarks default to daymanager_core."
        ),
    )
    parser.add_argument(
        "--profile",
        choices=REPLAY_PROFILE_CHOICES,
        default="",
        help=(
            "Replay profile. Daily --date defaults to fast; benchmarks default to full."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    selected_mode = args.mode or (
        REPLAY_MODE_DAYMANAGER_CORE
        if args.benchmark_set
        else REPLAY_MODE_AGENT_WORKFLOW
    )
    selected_profile = args.profile or (
        REPLAY_PROFILE_FULL if args.benchmark_set else REPLAY_PROFILE_FAST
    )
    if not args.mode and selected_profile == REPLAY_PROFILE_FAST:
        selected_mode = REPLAY_MODE_DAYMANAGER_CORE
    if args.benchmark_set:
        benchmark = run_runtime_replay_benchmark(
            dates=DEFAULT_BENCHMARK_DATE_SETS[args.benchmark_set],
            benchmark_set=args.benchmark_set,
            mode=selected_mode,
            profile=selected_profile,
            output_path=args.output_json or None,
            completion_status_path=args.completion_status_json or None,
            strategy_eval=args.strategy_eval or None,
            strategy_variant=args.variant,
            force_modern_crash_protection=args.force_modern_crash_protection,
            persist=True,
        )
        summary = benchmark["summary"]
        print("=" * 100)
        benchmark_mode = (
            "counterfactual-modern-crash"
            if args.force_modern_crash_protection
            else "historical"
        )
        print(
            f"RUNTIME REPLAY BENCHMARK | {args.benchmark_set} | mode={benchmark_mode} | replay={selected_mode} | profile={selected_profile}"
        )
        print("=" * 100)
        print(
            "Dates | primary={primary} secondary={secondary} high_conf={high} medium_conf={medium}".format(
                primary=summary["primary_dates"],
                secondary=summary["secondary_dates"],
                high=summary["high_confidence_dates"],
                medium=summary["medium_confidence_dates"],
            )
        )
        print(
            "Divergences | days={days} total={total} | inverse_net_pnl_total=${pnl:.2f} | actual_net_pnl_total=${actual:.2f}".format(
                days=summary["dates_with_divergences"],
                total=summary["total_divergences"],
                pnl=summary["inverse_net_pnl_total"],
                actual=summary["actual_net_pnl_total"],
            )
        )
        if summary.get("discovery_strategy_eval_enabled"):
            print(
                "Discovery | variant={variant} evaluated={evaluated} winners={winners} losers={losers} net=${pnl:.2f} slots={slots}".format(
                    variant=summary["discovery_variant"] or DEFAULT_DISCOVERY_VARIANT,
                    evaluated=summary["discovery_candidates_evaluated_total"],
                    winners=summary["discovery_added_winners_total"],
                    losers=summary["discovery_added_losers_total"],
                    pnl=summary["discovery_net_synthetic_pnl_total"],
                    slots=summary["discovery_slot_fill_delta_total"],
                )
            )
        print("-" * 100)
        print(
            "Date         Tier       Conf   Decisions  Allowed  Divergences  InversePnL  ActualPnL"
        )
        for row in benchmark["dates"]:
            print(
                f"{row['date']}  "
                f"{row['gating_tier']:<10} "
                f"{row['confidence']:<6} "
                f"{row['recorded_decisions_total']:<10} "
                f"{row['replay_longs_allowed']:<8} "
                f"{row['divergences_count']:<11} "
                f"${row['inverse_net_pnl']:<10.2f} "
                f"${row['actual_net_pnl']:<10.2f}"
            )
            if row["divergence_top_reasons"]:
                print(f"  divergence: {', '.join(row['divergence_top_reasons'])}")
            if row["divergence_examples"]:
                example = row["divergence_examples"][0]
                print(
                    "  example: {symbol} @ {timestamp} | {authority} | {regime}".format(
                        symbol=example.get("symbol") or "?",
                        timestamp=example.get("timestamp") or "?",
                        authority=example.get("replay_authority_reason")
                        or "no authority reason",
                        regime=example.get("replay_regime_reason")
                        or "no regime reason",
                    )
                )
            if row["handoff_status"] != "healthy":
                recoverable = ", ".join(row["handoff_recoverable_examples"]) or "none"
                print(
                    "  handoff: {status} | {problem} | recoverable={count} [{examples}] | source={source}".format(
                        status=row["handoff_status"],
                        problem=row["handoff_problem"] or "n/a",
                        count=row["handoff_recoverable_candidates"],
                        examples=recoverable,
                        source=row["handoff_signals_source"] or "plan_fallback",
                    )
                )
            if row["signal_alignment_status"] != "aligned":
                missing_examples = (
                    ", ".join(row["signal_alignment_missing_decision_examples"])
                    or "none"
                )
                print(
                    "  signal-ledger: {status} | {problem} | missing_decisions={count} [{examples}]".format(
                        status=row["signal_alignment_status"],
                        problem=row["signal_alignment_problem"] or "n/a",
                        count=row["signal_alignment_missing_decisions"],
                        examples=missing_examples,
                    )
                )
            elif row["notes"]:
                print(f"  note: {row['notes'][0]}")
            if row["discovery_candidates_evaluated"] > 0:
                print(
                    "  discovery: evaluated={evaluated} winners={winners} losers={losers} net=${pnl:.2f}".format(
                        evaluated=row["discovery_candidates_evaluated"],
                        winners=row["discovery_added_winners"],
                        losers=row["discovery_added_losers"],
                        pnl=row["discovery_net_synthetic_pnl"],
                    )
                )
                if row["discovery_top_added_candidates"]:
                    top = row["discovery_top_added_candidates"][0]
                    print(
                        "  discovery top: {symbol} | {source} | pnl=${pnl:.2f} | exit={exit_reason}".format(
                            symbol=top.get("symbol") or "?",
                            source=top.get("candidate_source_plan") or "unknown",
                            pnl=float(top.get("synthetic_pnl", 0.0) or 0.0),
                            exit_reason=top.get("exit_reason") or "n/a",
                        )
                    )
        print(f"Saved benchmark:      {benchmark['output_path']}")
        print(f"Completion status:    {benchmark.get('completion_status_path')}")
        return 0
    replay = RuntimeSessionReplay(
        session_date=args.date,
        mode=selected_mode,
        profile=selected_profile,
        output_path=args.output_json or None,
        strategy_eval=args.strategy_eval or None,
        strategy_variant=args.variant,
        force_modern_crash_protection=args.force_modern_crash_protection,
    )
    report = replay.run(persist=True)
    if not report:
        print("Replay failed to generate a report.")
        return 1
    summary = report.get("summary", {})
    print("=" * 80)
    replay_mode = (
        "counterfactual-modern-crash"
        if args.force_modern_crash_protection
        else "historical"
    )
    print(
        f"RUNTIME SESSION REPLAY | {args.date} | mode={replay_mode} | replay={selected_mode} | profile={selected_profile}"
    )
    print("=" * 80)
    print(f"Signals loaded:        {summary.get('signals_total', 0)}")
    print(f"Replay profile:        {summary.get('replay_profile', selected_profile)}")
    print(f"Position advisor:      {summary.get('position_advisor_mode', 'enabled')}")
    print(f"Decisions logged:      {summary.get('recorded_decisions_total', 0)}")
    print(
        f"  marked executed:     {summary.get('recorded_longs_executed', 0)}  "
        "(decision-ledger only — not full production trade count)"
    )
    prod_total = int(summary.get("production_total_journaled", 0) or 0)
    if prod_total:
        print(
            f"Production journaled:  {prod_total} actions  "
            f"(entries={summary.get('production_entries_journaled', 0)}, "
            f"exits={summary.get('production_exits_journaled', 0)}, "
            f"trims={summary.get('production_trims_journaled', 0)})  "
            f"unique symbols={summary.get('production_unique_symbols_journaled', 0)}"
        )
    if "replay_longs_allowed" in summary:
        print(f"Replay longs allowed:  {summary.get('replay_longs_allowed', 0)}")
    if "counterfactual_long_selected" in summary:
        print(
            f"Counterfactual longs:  {summary.get('counterfactual_long_selected', 0)}"
        )
        print(
            f"Counterfactual pnl:    ${float(summary.get('counterfactual_long_net_pnl_selected', 0.0) or 0.0):.2f}"
        )
        print(
            f"Estimated day pnl:     ${float(summary.get('counterfactual_estimated_day_net_pnl_selected', 0.0) or 0.0):.2f}"
        )
    if "workflow_generated_decisions_total" in summary:
        print(
            f"Workflow decisions:    {summary.get('workflow_generated_decisions_total', 0)}"
        )
        print(
            f"Workflow selections:   {summary.get('workflow_selected_symbols_total', 0)}"
        )
        print(f"Workflow adds:         {summary.get('workflow_add_selected_total', 0)}")
        print(
            f"Unique entries:        {summary.get('replay_unique_symbols_entered', 0)} symbols "
            f"({summary.get('replay_entries_executed', 0)} buy fills)"
        )
        print(
            f"Workflow m2c pnl:      ${float(summary['workflow_bullish_mark_to_close_pnl'] or 0.0):.2f}"
        )
        print(
            f"Deploy floor final:    {float(summary.get('deployment_floor_final_pct', 0.0) or 0.0) * 100.0:.1f}%"
        )
        print(
            f"Deploy shortfall:      ${float(summary.get('deployment_floor_final_shortfall_value', 0.0) or 0.0):.2f}"
        )
        print(f"Workflow blocked:      {summary['workflow_blocked_symbols_total']}")
        print(
            f"Phase cycles:          open={summary['market_open_cycles']} hours={summary['market_hours_cycles']}"
        )
    buckets = summary.get("replay_orders_by_bucket", {}) or {}
    print(
        f"Replay orders:         total={summary.get('replay_orders_total', 0)} "
        f"entries={buckets.get('equity_entry', 0)} "
        f"exits={buckets.get('equity_exit', 0)} "
        f"inverse={buckets.get('inverse_etf', 0)} "
        f"trims={buckets.get('trim', 0)}"
    )
    data_gap_skipped = int(summary.get("replay_data_gap_skipped", 0) or 0)
    if data_gap_skipped:
        gap_examples = summary.get("replay_data_gap_skipped_examples", []) or []
        print(
            f"  Data-gap rejections: {data_gap_skipped} orders across "
            f"{summary.get('replay_data_gap_skipped_unique_symbols', 0)} symbols "
            "(symbols had no minute-bar data — DM should not have picked them)"
        )
        if gap_examples:
            print("    examples: " + ", ".join(gap_examples[:10]))
    top_repeated = summary.get("replay_orders_top_repeated_exits", []) or []
    if top_repeated and int(top_repeated[0].get("count", 0) or 0) >= 5:
        worst = top_repeated[0]
        print(
            f"  WARN repeated exits: {worst.get('symbol')} ({worst.get('count')}x) "
            "— investigate exit-fill simulation for this symbol"
        )
    print(f"Inverse trades closed: {summary['inverse_trades_closed']}")
    print(f"Inverse net PnL:       ${summary['inverse_net_pnl']:.2f}")
    print(f"Inverse avg return:    {summary['inverse_avg_return_pct']:.2f}%")
    print(f"Inverse win rate:      {summary['inverse_win_rate']:.2f}%")
    if "management_replay_events" in summary:
        print(f"Mgmt events replayed:  {summary['management_replay_events']}")
        print(f"Mgmt action diffs:     {summary['management_replay_differences']}")
        print(f"Mgmt exit-like flags:  {summary['management_replay_exit_signals']}")
        print(
            f"Mgmt hard-stop flags:  {summary.get('management_hard_stop_exit_signals', 0)}"
        )
    print(f"Divergences:           {summary['divergences_count']}")
    print(f"Market data mode:      {summary['market_data_coverage_mode']}")
    print(
        f"Archive coverage:      found={summary['archive_found']} "
        f"loaded={summary['archive_symbols_loaded']} "
        f"missing={summary['archive_symbols_missing']} "
        f"live={summary['archive_live_fetch_symbols']}"
    )
    data_quality = summary.get("data_quality", {}) or {}
    dq_status = str(data_quality.get("status") or "ok")
    if dq_status != "ok":
        print(
            f"  DATA QUALITY {dq_status.upper()}: "
            f"{data_quality.get('coverage_pct', 0.0) * 100.0:.1f}% coverage "
            f"({data_quality.get('symbols_loaded', 0)}/{data_quality.get('symbols_requested', 0)})"
        )
        for warning in data_quality.get("warnings", []) or []:
            print(f"    ! {warning}")
        missing_examples = data_quality.get("missing_symbols_examples", []) or []
        if missing_examples:
            print("    missing examples: " + ", ".join(missing_examples[:10]))
    rvr = summary.get("recorded_vs_replay", {}) or {}
    # DEBUG: print(f"DEBUG: rvr={rvr}")
    print(
        "Production counts:     "
        f"legacy_decisions:{rvr.get('production_decisions_logged', 0)}  "
        f"claw_decisions:{rvr.get('production_claw_decisions_logged', 0)}  "
        f"workflow_events:{rvr.get('production_journaled_total', 0)}  "
        f"eod_trades:{rvr.get('production_eod_total_trades', 0)}"
    )
    print(
        "Production detail:     "
        f"journal_E:{rvr.get('production_journaled_entries', 0)}/"
        f"X:{rvr.get('production_journaled_exits', 0)}/"
        f"T:{rvr.get('production_journaled_trims', 0)}"
    )
    print(
        "                       "
        f"replay=decisions:{rvr.get('replay_decisions', 0)}/"
        f"unique_entries:{rvr.get('replay_unique_symbols_entered', 0)}/"
        f"orders:{rvr.get('replay_orders_total', 0)}"
    )
    print(f"Actual day net PnL:    ${summary['actual_net_pnl']:.2f}")
    print(
        f"Handoff:               {summary['handoff_status']} {summary['handoff_problem'] or ''}".rstrip()
    )
    print(
        "Signal ledger:         "
        + f"{summary['signal_alignment_status']} {summary['signal_alignment_problem'] or ''}".rstrip()
    )
    if "first_inverse_fast_at" in summary:
        print(
            f"First inverse_fast:    {summary['first_inverse_fast_at'] or 'not reached'}"
        )
    if summary.get("inverse_absence_reason"):
        print(f"Inverse absence:      {summary['inverse_absence_reason']}")
    divergence_summary = report.get("divergence_summary", {}) or {}
    if divergence_summary.get("top_reasons"):
        print(
            f"Top divergence reasons:{' '}{', '.join(divergence_summary['top_reasons'])}"
        )
    handoff = report.get("handoff_diagnostics", {}) or {}
    if handoff.get("recoverable_candidates_examples"):
        print(
            "Recoverable symbols:   "
            + ", ".join(handoff.get("recoverable_candidates_examples", []))
        )
    if handoff.get("decision_symbols_missing_from_signals_examples"):
        print(
            "Missing raw decision symbols: "
            + ", ".join(
                handoff.get("decision_symbols_missing_from_signals_examples", [])
            )
        )
    replay_entry_gate = report.get("replay_entry_diagnostic_gate", {}) or {}
    if replay_entry_gate:
        print(
            "Entry replay gate:     "
            f"{replay_entry_gate.get('status')} | {replay_entry_gate.get('reason')} "
            f"(silent={replay_entry_gate.get('silent_no_submit_cycles', 0)}, "
            f"explained={replay_entry_gate.get('explained_no_submit_cycles', 0)})"
        )
        if replay_entry_gate.get("zero_divergence_zero_entry_reproduces_failure"):
            print(
                "  INCIDENT: zero divergences with zero replay entries reproduced the no-entry failure."
            )
    discovery_eval = report.get("discovery_strategy_eval", {}) or {}
    if discovery_eval:
        print(
            "Discovery eval:        "
            + "variant={variant} evaluated={evaluated} winners={winners} losers={losers} net=${pnl:.2f}".format(
                variant=discovery_eval.get("variant") or DEFAULT_DISCOVERY_VARIANT,
                evaluated=int(
                    discovery_eval.get("synthetic_candidates_evaluated", 0) or 0
                ),
                winners=int(discovery_eval.get("added_winners", 0) or 0),
                losers=int(discovery_eval.get("added_losers", 0) or 0),
                pnl=float(discovery_eval.get("net_synthetic_pnl", 0.0) or 0.0),
            )
        )
        if discovery_eval.get("top_added_candidates"):
            top = discovery_eval["top_added_candidates"][0]
            print(
                "Top discovery add:    "
                + "{symbol} | {source} | pnl=${pnl:.2f} | exit={exit_reason}".format(
                    symbol=top.get("symbol") or "?",
                    source=top.get("candidate_source_plan") or "unknown",
                    pnl=float(top.get("synthetic_pnl", 0.0) or 0.0),
                    exit_reason=top.get("exit_reason") or "n/a",
                )
            )
    print(f"Saved report:          {replay.output_path}")
    if replay_entry_gate and replay_entry_gate.get("status") == "fail":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
