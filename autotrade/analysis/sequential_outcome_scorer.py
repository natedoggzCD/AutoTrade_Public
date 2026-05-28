"""
Outcome scoring for sequential shadow predictions.

Primary metric: outcome correctness on executed events.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import yfinance as yf

from autotrade.analysis.sequential_shadow_schema import SequentialShadowOutcome


PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
JOURNAL_PATH = PROJECT_DIR / "logs" / "trade_journal.json"


def _load_journal_trades() -> List[Dict[str, Any]]:
    if not JOURNAL_PATH.exists():
        return []
    try:
        with open(JOURNAL_PATH, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []
    trades = payload.get("trades", [])
    return trades if isinstance(trades, list) else []


def _find_trade_by_id(trade_id: str) -> Optional[Dict[str, Any]]:
    tid = str(trade_id or "").strip()
    if not tid:
        return None
    for trade in _load_journal_trades():
        if str(trade.get("id") or "").strip() == tid:
            return trade
    return None


def _future_return_pct(symbol: str, start_time_iso: str, start_price: float, horizon_minutes: int) -> float:
    if not symbol or start_price <= 0:
        return 0.0
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="5m")
    except Exception:
        return 0.0
    if hist is None or hist.empty:
        return 0.0
    try:
        st = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        st = datetime.now() - timedelta(minutes=horizon_minutes)
    end_t = st + timedelta(minutes=max(15, int(horizon_minutes)))
    clipped = hist[(hist.index.tz_localize(None) >= st) & (hist.index.tz_localize(None) <= end_t)]
    if clipped is None or clipped.empty:
        return 0.0
    end_price = float(clipped["Close"].iloc[-1])
    return ((end_price - start_price) / start_price) * 100.0


def _future_profile(
    symbol: str,
    start_time_iso: str,
    start_price: float,
    horizon_minutes: int,
) -> Dict[str, float]:
    ret_pct = _future_return_pct(
        symbol,
        start_time_iso,
        start_price,
        horizon_minutes=horizon_minutes,
    )
    profile = {
        "ret_pct": float(ret_pct),
        "vol_pct": float(abs(ret_pct) * 0.25),
        "max_favorable_pct": float(max(0.0, ret_pct)),
        "max_adverse_pct": float(min(0.0, ret_pct)),
    }
    if not symbol or start_price <= 0:
        return profile
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="5m")
    except Exception:
        return profile
    if hist is None or hist.empty:
        return profile
    try:
        st = datetime.fromisoformat(start_time_iso.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        st = datetime.now() - timedelta(minutes=horizon_minutes)
    end_t = st + timedelta(minutes=max(15, int(horizon_minutes)))
    try:
        clipped = hist[
            (hist.index.tz_localize(None) >= st) & (hist.index.tz_localize(None) <= end_t)
        ]
    except Exception:
        return profile
    if clipped is None or clipped.empty:
        return profile

    try:
        rel = ((clipped["Close"] - float(start_price)) / float(start_price)) * 100.0
        profile["ret_pct"] = float(rel.iloc[-1])
        profile["vol_pct"] = float(rel.std(ddof=0) or profile["vol_pct"])
        profile["max_favorable_pct"] = float(rel.max() or profile["max_favorable_pct"])
        profile["max_adverse_pct"] = float(rel.min() or profile["max_adverse_pct"])
    except Exception:
        return profile
    return profile


def _clip_score(value: float, limit: float = 1.0) -> float:
    lim = float(abs(limit) or 1.0)
    return float(max(-lim, min(lim, value)))


def _entry_metric_breakdown(profile: Dict[str, float]) -> Dict[str, float]:
    ret_pct = float(profile.get("ret_pct", 0.0))
    vol_pct = float(profile.get("vol_pct", 0.0))
    max_favorable_pct = float(profile.get("max_favorable_pct", 0.0))
    max_adverse_pct = float(profile.get("max_adverse_pct", 0.0))

    profitability = _clip_score(ret_pct / 1.0)
    risk_adjusted = _clip_score((ret_pct - (0.5 * vol_pct)) / 1.0)
    timing = _clip_score((max_favorable_pct + max_adverse_pct) / 1.0)
    return {
        "profitability_actual": profitability,
        "risk_adjusted_actual": risk_adjusted,
        "timing_actual": timing,
    }


def _exit_metric_breakdown(profile: Dict[str, float]) -> Dict[str, float]:
    ret_pct = float(profile.get("ret_pct", 0.0))
    vol_pct = float(profile.get("vol_pct", 0.0))
    max_adverse_pct = float(profile.get("max_adverse_pct", 0.0))

    profitability = _clip_score((-ret_pct) / 1.0)
    risk_adjusted = _clip_score(((-ret_pct) - (0.5 * vol_pct)) / 1.0)
    timing = _clip_score((-max_adverse_pct) / 1.0)
    return {
        "profitability_actual": profitability,
        "risk_adjusted_actual": risk_adjusted,
        "timing_actual": timing,
    }


def _weighted_total(metrics: Dict[str, float]) -> float:
    return float(
        (0.60 * float(metrics.get("profitability_actual", 0.0)))
        + (0.25 * float(metrics.get("risk_adjusted_actual", 0.0)))
        + (0.15 * float(metrics.get("timing_actual", 0.0)))
    )


def _counterfactual_total(event_type: str, seq_action: str, actual_total: float) -> float:
    et = str(event_type or "").lower()
    action = str(seq_action or "").lower()
    if et in {"buy", "add"} and action in {"avoid", "exit"}:
        return float(-actual_total)
    if et in {"exit", "sell"} and action == "hold":
        return float(-actual_total)
    if et == "trim" and action == "hold":
        return float(-actual_total)
    return float(actual_total)


def _score_entry_event(
    event: Dict[str, Any], prediction: Dict[str, Any], horizon_minutes: int
) -> SequentialShadowOutcome:
    symbol = str(event.get("symbol", "")).upper()
    trade_id = str(event.get("trade_id") or "")
    fill_price = float(event.get("fill_price", 0.0) or 0.0)
    fill_time = str(event.get("fill_time") or event.get("created_at") or "")
    seq_action = str(prediction.get("recommended_action", "hold")).lower()

    trade = _find_trade_by_id(trade_id)
    if trade and trade.get("outcome"):
        outcome = str(trade.get("outcome")).lower()
        metrics = {
            "profitability_actual": 1.0 if outcome == "win" else (-1.0 if outcome == "loss" else 0.0),
            "risk_adjusted_actual": 1.0 if outcome == "win" else (-0.5 if outcome == "loss" else 0.0),
            "timing_actual": 0.0,
        }
    else:
        profile = _future_profile(symbol, fill_time, fill_price, horizon_minutes=horizon_minutes)
        metrics = _entry_metric_breakdown(profile)

    actual_score = _weighted_total(metrics)
    counterfactual = _counterfactual_total("buy", seq_action, actual_score)
    delta = counterfactual - actual_score
    metric_breakdown = dict(metrics)
    metric_breakdown["total_actual"] = float(actual_score)
    metric_breakdown["total_counterfactual"] = float(counterfactual)
    return SequentialShadowOutcome(
        event_id=str(event.get("event_id", "")),
        symbol=symbol,
        event_type="buy",
        baseline_action="buy",
        sequential_action=seq_action,
        actual_outcome_score=float(actual_score),
        counterfactual_score=float(counterfactual),
        score_delta=float(delta),
        sequential_more_accurate=bool(delta > 0),
        notes="entry_outcome_composite",
        metric_breakdown=metric_breakdown,
    )


def _score_exit_like_event(
    event: Dict[str, Any], prediction: Dict[str, Any], horizon_minutes: int
) -> SequentialShadowOutcome:
    symbol = str(event.get("symbol", "")).upper()
    event_type = str(event.get("event_type", "exit")).lower()
    fill_price = float(event.get("fill_price", 0.0) or 0.0)
    fill_time = str(event.get("fill_time") or event.get("created_at") or "")
    seq_action = str(prediction.get("recommended_action", "hold")).lower()

    profile = _future_profile(symbol, fill_time, fill_price, horizon_minutes=horizon_minutes)
    metrics = _exit_metric_breakdown(profile)
    actual_score = _weighted_total(metrics)
    counterfactual = _counterfactual_total(event_type, seq_action, actual_score)
    delta = counterfactual - actual_score
    metric_breakdown = dict(metrics)
    metric_breakdown["total_actual"] = float(actual_score)
    metric_breakdown["total_counterfactual"] = float(counterfactual)
    return SequentialShadowOutcome(
        event_id=str(event.get("event_id", "")),
        symbol=symbol,
        event_type=event_type,
        baseline_action="exit" if event_type in {"sell", "exit"} else "trim",
        sequential_action=seq_action,
        actual_outcome_score=float(actual_score),
        counterfactual_score=float(counterfactual),
        score_delta=float(delta),
        sequential_more_accurate=bool(delta > 0),
        notes="exit_outcome_composite",
        metric_breakdown=metric_breakdown,
    )


def score_event_outcome(
    event: Dict[str, Any],
    prediction: Dict[str, Any],
    horizon_minutes: int = 120,
) -> SequentialShadowOutcome:
    event_type = str(event.get("event_type", "")).lower()
    if event_type in {"buy", "add"}:
        return _score_entry_event(event, prediction, horizon_minutes=horizon_minutes)
    return _score_exit_like_event(event, prediction, horizon_minutes=horizon_minutes)


def build_comparative_report(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows = [r for r in outcomes if isinstance(r, dict)]
    total = len(rows)
    if total <= 0:
        return {
            "total_events": 0,
            "sequential_more_accurate_rate": 0.0,
            "mean_score_delta": 0.0,
            "by_event_type": {},
            "action_pairs": {},
        }

    wins = 0
    delta_sum = 0.0
    by_event_type: Dict[str, Dict[str, Any]] = {}
    action_pairs: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        delta = float(row.get("score_delta", 0.0) or 0.0)
        win = bool(row.get("sequential_more_accurate"))
        event_type = str(row.get("event_type", "unknown"))
        baseline = str(row.get("baseline_action", "unknown"))
        seq = str(row.get("sequential_action", "unknown"))
        pair_key = f"{baseline}->{seq}"

        if win:
            wins += 1
        delta_sum += delta

        et_slot = by_event_type.setdefault(event_type, {"count": 0, "wins": 0, "delta_sum": 0.0})
        et_slot["count"] += 1
        et_slot["delta_sum"] += delta
        if win:
            et_slot["wins"] += 1

        pair_slot = action_pairs.setdefault(pair_key, {"count": 0, "delta_sum": 0.0})
        pair_slot["count"] += 1
        pair_slot["delta_sum"] += delta

    for slot in by_event_type.values():
        count = max(1, int(slot["count"]))
        slot["win_rate"] = float(slot["wins"]) / float(count)
        slot["mean_score_delta"] = float(slot["delta_sum"]) / float(count)
        slot.pop("delta_sum", None)

    for slot in action_pairs.values():
        count = max(1, int(slot["count"]))
        slot["mean_score_delta"] = float(slot["delta_sum"]) / float(count)
        slot.pop("delta_sum", None)

    return {
        "total_events": total,
        "sequential_more_accurate_rate": float(wins) / float(total),
        "mean_score_delta": float(delta_sum) / float(total),
        "by_event_type": by_event_type,
        "action_pairs": action_pairs,
    }
