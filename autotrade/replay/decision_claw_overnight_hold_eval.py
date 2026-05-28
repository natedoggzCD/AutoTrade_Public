from __future__ import annotations

import argparse
import copy
import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd

from autotrade.core.decision_claw import DecisionClaw
from autotrade.replay.minute_bar_archive import (
    REGULAR_END_ET,
    REGULAR_START_ET,
    fetch_symbol_minute_bars,
)
from config.config_loader import get_config

logger = logging.getLogger("decision_claw_overnight_hold_eval")

PROJECT_DIR = Path(__file__).resolve().parents[2]
FIRST_HOUR_END_ET = time(10, 30)
DEFAULT_LOOKBACK_DATES = (
    "2026-03-24",
    "2026-03-25",
    "2026-03-26",
    "2026-03-27",
    "2026-03-30",
    "2026-03-31",
    "2026-04-01",
)
DEFAULT_OVERNIGHT_HOLD_BENCHMARK_SET = "recent_two_weeks"


def _derive_risk_reward(
    *,
    current_price: float,
    target_price: float,
    stop_price: float,
    raw_value: float,
) -> float:
    risk_reward = float(raw_value or 0.0)
    if risk_reward > 0.0:
        return risk_reward
    if current_price <= 0.0 or target_price <= current_price or stop_price <= 0.0:
        return 0.0
    risk = max(current_price - stop_price, 0.0)
    reward = max(target_price - current_price, 0.0)
    if risk <= 0.0 or reward <= 0.0:
        return 0.0
    return round(reward / risk, 4)


def _overnight_hold_bias(
    *,
    current_price: float,
    target_price: float,
    position_size_multiple: float,
    pnl_pct: float,
    risk_reward: float,
    headroom_pct: float,
) -> str:
    breakout_extension = (
        target_price > 0.0
        and current_price > target_price
        and pnl_pct >= 3.0
        and position_size_multiple <= 1.1
        and risk_reward >= 1.6
    )
    if breakout_extension:
        return "favorable"
    if headroom_pct >= 7.0 and risk_reward >= 1.4 and pnl_pct > 0.0:
        return "favorable"
    if headroom_pct < 2.0:
        return "unfavorable"
    return "neutral"


def _load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def _dedupe_preserve(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _next_trading_day(session_date: str) -> str:
    current = date.fromisoformat(session_date)
    probe = current + timedelta(days=1)
    while probe.weekday() >= 5:
        probe += timedelta(days=1)
    return probe.isoformat()


def _score_hold_outcome(
    *,
    approved_hold: bool,
    next_close_return_pct: float,
    first_hour_min_drawdown_pct: float,
) -> Dict[str, Any]:
    beneficial_hold = (
        float(next_close_return_pct) >= 1.0
        and float(first_hour_min_drawdown_pct) > -2.0
    )
    harmful_hold = (
        float(next_close_return_pct) <= -1.0
        or float(first_hour_min_drawdown_pct) <= -3.0
    )
    if beneficial_hold:
        label = "beneficial_hold"
    elif harmful_hold:
        label = "harmful_hold"
    else:
        label = "mixed"

    decision_quality = "mixed"
    score = 0.0
    if approved_hold and beneficial_hold:
        decision_quality = "correct_hold"
        score = 1.0
    elif approved_hold and harmful_hold:
        decision_quality = "bad_hold"
        score = -1.0
    elif (not approved_hold) and harmful_hold:
        decision_quality = "correct_trim"
        score = 1.0
    elif (not approved_hold) and beneficial_hold:
        decision_quality = "missed_hold"
        score = -1.0

    return {
        "outcome_label": label,
        "decision_quality": decision_quality,
        "score": score,
    }


def _compute_next_day_metrics(
    *,
    symbol: str,
    close_price: float,
    next_session_date: str,
) -> Dict[str, Any]:
    close_value = float(close_price or 0.0)
    if close_value <= 0.0:
        return {"available": False, "reason": "missing_close_price"}

    frame = fetch_symbol_minute_bars(
        None,
        session_date=next_session_date,
        symbol=symbol,
        start_et=REGULAR_START_ET,
        end_et=REGULAR_END_ET,
    )
    if frame is None or frame.empty:
        return {"available": False, "reason": "missing_next_day_bars"}

    et_index = pd.DatetimeIndex(frame.index).tz_convert("America/New_York")
    first_hour_mask = et_index.time <= FIRST_HOUR_END_ET
    first_hour = frame.loc[first_hour_mask]
    if first_hour.empty:
        first_hour = frame.iloc[: min(len(frame), 60)]

    open_price = float(frame.iloc[0].get("open", frame.iloc[0].get("close", 0.0)) or 0.0)
    close_next = float(frame.iloc[-1].get("close", 0.0) or 0.0)
    first_hour_close = float(
        first_hour.iloc[-1].get("close", first_hour.iloc[-1].get("open", 0.0)) or 0.0
    )
    first_hour_high = float(first_hour["high"].max() or 0.0)
    first_hour_low = float(first_hour["low"].min() or 0.0)

    def _pct(value: float) -> float:
        if close_value <= 0.0 or value <= 0.0:
            return 0.0
        return round(((float(value) - close_value) / close_value) * 100.0, 4)

    return {
        "available": True,
        "next_session_date": next_session_date,
        "open_price": round(open_price, 4),
        "first_hour_close": round(first_hour_close, 4),
        "first_hour_high": round(first_hour_high, 4),
        "first_hour_low": round(first_hour_low, 4),
        "next_close": round(close_next, 4),
        "gap_pct": _pct(open_price),
        "first_hour_return_pct": _pct(first_hour_close),
        "first_hour_max_gain_pct": _pct(first_hour_high),
        "first_hour_min_drawdown_pct": _pct(first_hour_low),
        "next_close_return_pct": _pct(close_next),
    }


@dataclass
class OvernightHoldScenario:
    session_date: str
    next_session_date: str
    payload: Dict[str, Any]
    legacy_recommendation: Dict[str, Any]
    candidates: List[Dict[str, Any]]


class DecisionClawOvernightHoldEvaluator:
    def __init__(
        self,
        *,
        project_dir: Optional[Path | str] = None,
        output_path: Optional[Path | str] = None,
    ) -> None:
        self.project_dir = Path(project_dir) if project_dir else PROJECT_DIR
        default_output = (
            self.project_dir
            / "logs"
            / f"decision_claw_overnight_hold_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        self.output_path = Path(output_path) if output_path else default_output
        self.config = copy.deepcopy(get_config().decision_claw)
        eval_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config.state_path = f"logs/decision_claw_eval/overnight_hold_state_{eval_stamp}.json"
        self.config.decisions_log_path = (
            f"logs/decision_claw_eval/overnight_hold_decisions_{eval_stamp}_{{date}}.jsonl"
        )
        self.config.actions_log_path = (
            f"logs/decision_claw_eval/overnight_hold_actions_{eval_stamp}_{{date}}.jsonl"
        )
        self.config.cost_log_path = (
            f"logs/decision_claw_eval/overnight_hold_cost_{eval_stamp}_{{date}}.jsonl"
        )
        self.config.phase_snapshot_path = (
            f"logs/decision_claw_eval/overnight_hold_snapshot_{eval_stamp}_{{date}}.json"
        )
        self.config.budget_controls.daily_max_calls = 10_000
        self.config.budget_controls.daily_max_cost_usd = 10_000.0
        if getattr(self.config, "market_state", None) is not None:
            self.config.market_state.cooldown_seconds = 0
        self.portfolio_cfg = get_config().portfolio
        self.controller = DecisionClaw(self.config, project_root=self.project_dir)

    def _load_signal_context(self, session_date: str) -> Dict[str, Dict[str, Any]]:
        rows_by_symbol: Dict[str, Dict[str, Any]] = {}
        candidates = [
            self.project_dir / "logs" / f"signals_{session_date}.json",
            self.project_dir / "plans" / f"morning_game_plan_{session_date.replace('-', '')}.json",
        ]
        for path in candidates:
            if not path.exists():
                continue
            try:
                payload = _load_json(path)
            except Exception:
                continue
            rows: List[Dict[str, Any]] = []
            if isinstance(payload, list):
                rows = [dict(row) for row in payload if isinstance(row, dict)]
            elif isinstance(payload, dict):
                for key in ("signals", "buy_signals", "entry_orders", "full_watchlist"):
                    value = payload.get(key)
                    if isinstance(value, list):
                        rows.extend(dict(row) for row in value if isinstance(row, dict))
            for row in rows:
                symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
                if not symbol:
                    continue
                rows_by_symbol.setdefault(symbol, {}).update(dict(row))
        return rows_by_symbol

    def build_scenario(self, *, session_date: str) -> OvernightHoldScenario:
        eod_path = self.project_dir / "data" / f"eod_review_{session_date}.json"
        eod_payload = _load_json(eod_path) if eod_path.exists() else {}
        signal_context = self._load_signal_context(session_date)
        next_session_date = _next_trading_day(session_date)
        target_value = float(getattr(self.portfolio_cfg, "position_size_target", 0.0) or 0.0)
        max_value = float(getattr(self.portfolio_cfg, "position_size_max", 0.0) or 0.0)

        candidates: List[Dict[str, Any]] = []
        for row in eod_payload.get("trades") or []:
            if not isinstance(row, dict):
                continue
            symbol = str(row.get("symbol") or "").upper().strip()
            if not symbol:
                continue
            market_value = float(
                row.get("market_value")
                or (float(row.get("qty", 0.0) or 0.0) * float(row.get("current_price", 0.0) or 0.0))
                or 0.0
            )
            pnl_pct = float(row.get("unrealized_plpc", 0.0) or 0.0) * 100.0
            close_price = float(row.get("current_price", 0.0) or 0.0)
            if market_value <= target_value * 1.01 or pnl_pct <= 0.0 or close_price <= 0.0:
                continue
            signal_row = signal_context.get(symbol, {})
            target_price = float(
                signal_row.get(
                    "target",
                    signal_row.get("take_profit", signal_row.get("target_price", 0.0)),
                )
                or 0.0
            )
            stop_price = float(
                signal_row.get(
                    "stop_loss",
                    signal_row.get("stop", signal_row.get("stop_price", 0.0)),
                )
                or 0.0
            )
            headroom_pct = (
                ((target_price - close_price) / close_price) * 100.0
                if target_price > 0.0
                else 0.0
            )
            position_size_multiple = round(market_value / max(target_value, 1.0), 2)
            max_size_multiple = round(market_value / max(max_value, 1.0), 2)
            risk_reward = _derive_risk_reward(
                current_price=close_price,
                target_price=target_price,
                stop_price=stop_price,
                raw_value=float(
                    signal_row.get("risk_reward", signal_row.get("rr_ratio", 0.0)) or 0.0
                ),
            )
            breakout_extension = bool(
                target_price > 0.0
                and close_price > target_price
                and pnl_pct >= 3.0
                and position_size_multiple <= 1.1
                and risk_reward >= 1.6
            )
            candidates.append(
                {
                    "symbol": symbol,
                    "qty": int(float(row.get("qty", 0.0) or 0.0)),
                    "market_value": round(market_value, 2),
                    "position_size_multiple": position_size_multiple,
                    "max_size_multiple": max_size_multiple,
                    "pnl_pct": round(pnl_pct, 4),
                    "current_price": round(close_price, 4),
                    "target_price": round(target_price, 4),
                    "stop_price": round(stop_price, 4),
                    "risk_reward": risk_reward,
                    "relative_strength": round(
                        float(signal_row.get("relative_strength", 0.0) or 0.0), 4
                    ),
                    "atr_14": round(float(signal_row.get("atr_14", 0.0) or 0.0), 4),
                    "headroom_pct": round(headroom_pct, 4),
                    "breakout_extension_candidate": breakout_extension,
                    "overnight_hold_bias": _overnight_hold_bias(
                        current_price=close_price,
                        target_price=target_price,
                        position_size_multiple=position_size_multiple,
                        pnl_pct=pnl_pct,
                        risk_reward=risk_reward,
                        headroom_pct=headroom_pct,
                    ),
                    "entry_source": str(row.get("signal_entry_source") or signal_row.get("entry_source") or ""),
                    "plan_score_source": str(row.get("plan_score_source") or signal_row.get("plan_score_source") or ""),
                    "next_day_metrics": _compute_next_day_metrics(
                        symbol=symbol,
                        close_price=close_price,
                        next_session_date=next_session_date,
                    ),
                }
            )

        candidates.sort(
            key=lambda item: (
                float(item.get("position_size_multiple", 0.0) or 0.0),
                float(item.get("pnl_pct", 0.0) or 0.0),
            ),
            reverse=True,
        )
        payload = {
            "phase": "wind_down",
            "wave": None,
            "open_slots": 0,
            "free_capital": 0.0,
            "positions_count": len(eod_payload.get("trades") or []),
            "current_positions": list(eod_payload.get("trades") or []),
            "candidate_count": 0,
            "candidates": [],
            "weak_positions": [],
            "wind_down_oversized_winners": [
                {key: value for key, value in row.items() if key != "next_day_metrics"}
                for row in candidates
            ],
            "legacy_wave": "overnight_hold_historical_eval",
        }
        legacy_recommendation = {
            "legacy_choice": [],
            "legacy_reason": "default_trim_to_target",
        }
        return OvernightHoldScenario(
            session_date=session_date,
            next_session_date=next_session_date,
            payload=payload,
            legacy_recommendation=legacy_recommendation,
            candidates=candidates,
        )

    def evaluate_dates(
        self,
        *,
        dates: Sequence[str],
        persist: bool = True,
    ) -> Dict[str, Any]:
        rows: List[Dict[str, Any]] = []
        approved_total = 0
        decisive_total = 0
        decisive_correct = 0

        for session_date in dates:
            scenario = self.build_scenario(session_date=session_date)
            if not scenario.candidates:
                rows.append(
                    {
                        "date": session_date,
                        "next_session_date": scenario.next_session_date,
                        "candidates": [],
                        "summary": {
                            "candidates_total": 0,
                            "approved_holds": 0,
                            "decisive_cases": 0,
                            "decisive_accuracy": 0.0,
                        },
                    }
                )
                continue
            result = self.controller.review(
                phase_agent="market_state",
                trigger=f"historical_wind_down_overnight_hold_eval_{session_date}",
                payload=scenario.payload,
                legacy_recommendation=scenario.legacy_recommendation,
            )
            hold_symbols = {
                action.symbol
                for action in result.actions
                if action.action_type == "hold_position"
                and bool(action.metadata.get("approve_overnight_oversize", False))
                and action.symbol
            }

            candidate_rows: List[Dict[str, Any]] = []
            decisive_cases = 0
            decisive_correct_cases = 0
            for candidate in scenario.candidates:
                symbol = str(candidate.get("symbol") or "").upper().strip()
                metrics = dict(candidate.get("next_day_metrics") or {})
                approved_hold = symbol in hold_symbols
                outcome = {"outcome_label": "unavailable", "decision_quality": "unscored", "score": 0.0}
                if bool(metrics.get("available")):
                    outcome = _score_hold_outcome(
                        approved_hold=approved_hold,
                        next_close_return_pct=float(metrics.get("next_close_return_pct", 0.0) or 0.0),
                        first_hour_min_drawdown_pct=float(
                            metrics.get("first_hour_min_drawdown_pct", 0.0) or 0.0
                        ),
                    )
                    if outcome["decision_quality"] in {
                        "correct_hold",
                        "bad_hold",
                        "correct_trim",
                        "missed_hold",
                    }:
                        decisive_cases += 1
                        if outcome["score"] > 0:
                            decisive_correct_cases += 1
                candidate_rows.append(
                    {
                        **{key: value for key, value in candidate.items() if key != "next_day_metrics"},
                        "approved_hold": approved_hold,
                        "next_day_metrics": metrics,
                        "evaluation": outcome,
                    }
                )

            approved_holds = sum(1 for row in candidate_rows if bool(row.get("approved_hold")))
            approved_total += approved_holds
            decisive_total += decisive_cases
            decisive_correct += decisive_correct_cases
            rows.append(
                {
                    "date": session_date,
                    "next_session_date": scenario.next_session_date,
                    "decision": result.decision,
                    "confidence": round(float(result.confidence or 0.0), 4),
                    "reasoning_summary": str(result.reasoning_summary or ""),
                    "actions": [action.__dict__ for action in result.actions],
                    "candidates": candidate_rows,
                    "summary": {
                        "candidates_total": len(candidate_rows),
                        "approved_holds": approved_holds,
                        "decisive_cases": decisive_cases,
                        "decisive_accuracy": round(
                            decisive_correct_cases / decisive_cases, 4
                        )
                        if decisive_cases
                        else 0.0,
                    },
                }
            )

        output = {
            "summary": {
                "dates_total": len(rows),
                "approved_holds_total": approved_total,
                "decisive_cases_total": decisive_total,
                "decisive_accuracy": round(decisive_correct / decisive_total, 4)
                if decisive_total
                else 0.0,
            },
            "dates": rows,
        }
        if persist:
            _write_json(self.output_path, output)
            latest_path = self.output_path.with_name(
                "decision_claw_overnight_hold_eval_latest.json"
            )
            _write_json(latest_path, output)
        return output


def _parse_dates(raw_dates: Sequence[str]) -> List[str]:
    parsed = _dedupe_preserve(raw_dates)
    return parsed or list(DEFAULT_LOOKBACK_DATES)


def _available_session_dates(project_dir: Path) -> List[str]:
    dates: List[str] = []
    for path in sorted((project_dir / "data").glob("eod_review_*.json")):
        suffix = path.stem.replace("eod_review_", "", 1)
        try:
            datetime.strptime(suffix, "%Y-%m-%d")
        except ValueError:
            continue
        dates.append(suffix)
    return dates


def _resolve_benchmark_set(*, project_dir: Path, benchmark_set: str) -> List[str]:
    available = _available_session_dates(project_dir)
    if benchmark_set == "recent_two_weeks":
        return available[-12:]
    raise ValueError(f"Unknown overnight hold benchmark set: {benchmark_set}")


def _select_dates(
    *,
    project_dir: Path,
    raw_dates: Sequence[str],
    benchmark_set: str = DEFAULT_OVERNIGHT_HOLD_BENCHMARK_SET,
    start_date: str = "",
    end_date: str = "",
) -> List[str]:
    parsed = _dedupe_preserve(raw_dates)
    if parsed:
        return parsed
    if not start_date and not end_date:
        if benchmark_set:
            return _resolve_benchmark_set(project_dir=project_dir, benchmark_set=benchmark_set)
        return list(DEFAULT_LOOKBACK_DATES)
    available = _available_session_dates(project_dir)
    if not available:
        return []
    start = start_date or available[0]
    end = end_date or available[-1]
    return [value for value in available if start <= value <= end]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate DecisionClaw wind-down oversized overnight hold decisions."
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        default=[],
        help="Session dates to evaluate (YYYY-MM-DD). Default uses recent completed sessions.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON output path.",
    )
    parser.add_argument(
        "--benchmark-set",
        default=DEFAULT_OVERNIGHT_HOLD_BENCHMARK_SET,
        help="Named overnight-hold benchmark set to evaluate when --dates is omitted.",
    )
    parser.add_argument(
        "--start-date",
        default="",
        help="Optional start date for a contiguous evaluation window (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        default="",
        help="Optional end date for a contiguous evaluation window (YYYY-MM-DD).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    evaluator = DecisionClawOvernightHoldEvaluator(
        output_path=Path(args.output) if str(args.output or "").strip() else None
    )
    report = evaluator.evaluate_dates(
        dates=_select_dates(
            project_dir=evaluator.project_dir,
            raw_dates=args.dates,
            benchmark_set=str(args.benchmark_set or "").strip(),
            start_date=str(args.start_date or "").strip(),
            end_date=str(args.end_date or "").strip(),
        ),
        persist=True,
    )
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
