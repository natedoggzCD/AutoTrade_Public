"""Audit-driven overnight ranking helper.

Score contract:
- final_score: canonical actionable/display score for downstream consumers.
- ranking_score: legacy alias for final_score; keep bounded to [0, 100].
- ranking_score_with_boost: optional priority-only variant when another layer needs
  a sort boost without changing the canonical score.
- conviction_priority_score/confidence: secondary display or allocation context.
- entry_score/technical_score/sr_score/ml_score/sentiment_score: component inputs,
  not canonical ranking outputs.
"""

from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from autotrade.signals.overnight_model_fit import (
    derive_resistance_distance_atr,
    derive_support_distance_atr,
    load_model_artifact,
    predict_candidate_outcomes,
    predict_candidate_probabilities,
)
from autotrade.signals.universe_filters import (
    DEFAULT_POST_SPIKE_RANGE_ATR_THRESHOLD,
    DEFAULT_POST_SPIKE_VOLUME_THRESHOLD,
    is_post_spike_long_candidate,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_band(
    value: float,
    *,
    sweet_low: float,
    sweet_high: float,
    hard_low: float,
    hard_high: float,
) -> float:
    """Return a 0-100 score with a preferred central band."""
    if value <= hard_low or value >= hard_high:
        return 0.0
    if sweet_low <= value <= sweet_high:
        return 100.0
    if value < sweet_low:
        span = max(1e-9, sweet_low - hard_low)
        return _clamp(((value - hard_low) / span) * 100.0, 0.0, 100.0)
    span = max(1e-9, hard_high - sweet_high)
    return _clamp(((hard_high - value) / span) * 100.0, 0.0, 100.0)


def _score_positive(value: float, *, center: float, slope: float = 1.0) -> float:
    return _clamp(50.0 + ((value - center) * slope), 0.0, 100.0)


MODEL_ARTIFACT_PATH = Path("data/overnight_signal_model.json")
EDGE_ARTIFACT_PATH = Path("data/overnight_edge_context.json")


def _score_probability_input(value: Optional[float]) -> float:
    if value is None:
        return 50.0
    value = _safe_float(value, 0.5)
    if value > 1.0:
        value = value / 100.0
    return _clamp(value * 100.0, 0.0, 100.0)


def _derive_1d_model_score(
    candidate: Dict[str, Any], actionability_score: float
) -> float:
    atr_percent = _safe_float(candidate.get("atr_percent"), 0.0)
    volume_ratio = _safe_float(candidate.get("volume_ratio"), 0.0)
    vol_trend_ratio = _safe_float(candidate.get("vol_trend_ratio"), 1.0)
    weekly_return = _safe_float(candidate.get("weekly_return"), 0.0)
    rsi_14 = _safe_float(candidate.get("rsi_14"), 50.0)
    technical_score = _safe_float(candidate.get("technical_score"), 50.0)
    ml_score = _safe_float(candidate.get("ml_score"), 50.0)
    sr_score = _safe_float(candidate.get("sr_score"), 50.0)
    entry_score = _safe_float(candidate.get("entry_score"), sr_score)

    atr_score = _score_band(
        atr_percent, sweet_low=4.0, sweet_high=9.0, hard_low=1.0, hard_high=14.0
    )
    volume_score = _score_positive(volume_ratio, center=1.1, slope=24.0)
    trend_volume_score = _score_positive(vol_trend_ratio, center=1.0, slope=40.0)
    weekly_score = _score_band(
        weekly_return, sweet_low=1.0, sweet_high=8.0, hard_low=-8.0, hard_high=18.0
    )
    rsi_score = _score_band(
        rsi_14, sweet_low=42.0, sweet_high=60.0, hard_low=25.0, hard_high=78.0
    )

    return round(
        _clamp(
            (
                actionability_score * 0.20
                + atr_score * 0.18
                + volume_score * 0.16
                + trend_volume_score * 0.10
                + weekly_score * 0.08
                + rsi_score * 0.08
                + technical_score * 0.10
                + ml_score * 0.05
                + entry_score * 0.03
                + sr_score * 0.02
            ),
            0.0,
            100.0,
        ),
        2,
    )


def _derive_1w_model_score(
    candidate: Dict[str, Any], actionability_score: float
) -> float:
    weekly_return = _safe_float(candidate.get("weekly_return"), 0.0)
    technical_score = _safe_float(candidate.get("technical_score"), 50.0)
    ml_score = _safe_float(candidate.get("ml_score"), 50.0)
    ensemble_prob_1w = _score_probability_input(candidate.get("ensemble_prob_1w"))
    rsi_14 = _safe_float(candidate.get("rsi_14"), 50.0)
    atr_percent = _safe_float(candidate.get("atr_percent"), 0.0)

    weekly_score = _score_band(
        weekly_return, sweet_low=1.0, sweet_high=10.0, hard_low=-10.0, hard_high=22.0
    )
    rsi_score = _score_band(
        rsi_14, sweet_low=40.0, sweet_high=62.0, hard_low=25.0, hard_high=80.0
    )
    atr_score = _score_band(
        atr_percent, sweet_low=3.0, sweet_high=9.0, hard_low=1.0, hard_high=16.0
    )

    return round(
        _clamp(
            (
                ensemble_prob_1w * 0.36
                + weekly_score * 0.24
                + technical_score * 0.18
                + ml_score * 0.12
                + rsi_score * 0.05
                + atr_score * 0.03
                + actionability_score * 0.02
            ),
            0.0,
            100.0,
        ),
        2,
    )


def _outcome_to_score(
    value: float, *, center: float, slope: float, low: float = 0.0, high: float = 100.0
) -> float:
    return round(_clamp(50.0 + ((value - center) * slope), low, high), 2)


def _derive_actionability_score(candidate: Dict[str, Any]) -> Dict[str, float]:
    atr_percent = _safe_float(candidate.get("atr_percent"), 0.0)
    volume_ratio = _safe_float(candidate.get("volume_ratio"), 0.0)
    risk_reward = _safe_float(candidate.get("risk_reward"), 0.0)
    rsi_14 = _safe_float(candidate.get("rsi_14"), 50.0)
    support_dist_atr = derive_support_distance_atr(candidate)
    resistance_dist_atr = derive_resistance_distance_atr(candidate)

    atr_score = _score_band(
        atr_percent, sweet_low=4.0, sweet_high=9.0, hard_low=1.0, hard_high=16.0
    )
    volume_score = _score_positive(volume_ratio, center=1.0, slope=26.0)
    rr_score = _score_band(
        risk_reward, sweet_low=1.8, sweet_high=3.5, hard_low=0.8, hard_high=6.0
    )
    rsi_score = _score_band(
        rsi_14, sweet_low=42.0, sweet_high=60.0, hard_low=25.0, hard_high=80.0
    )

    if support_dist_atr <= 0:
        support_score = 45.0
    else:
        support_score = _score_band(
            support_dist_atr, sweet_low=0.0, sweet_high=1.2, hard_low=0.0, hard_high=4.5
        )
    if resistance_dist_atr <= 0:
        resistance_score = 50.0
    else:
        resistance_score = _score_band(
            resistance_dist_atr,
            sweet_low=1.2,
            sweet_high=4.0,
            hard_low=0.2,
            hard_high=7.0,
        )

    score = round(
        _clamp(
            (
                atr_score * 0.20
                + volume_score * 0.18
                + rr_score * 0.24
                + support_score * 0.18
                + resistance_score * 0.10
                + rsi_score * 0.10
            ),
            0.0,
            100.0,
        ),
        2,
    )
    return {
        "score": score,
        "support_dist_atr": round(support_dist_atr, 3),
        "resistance_dist_atr": round(resistance_dist_atr, 3),
    }


def _derive_catalyst_gate(candidate: Dict[str, Any]) -> Dict[str, float]:
    fresh_news = bool(candidate.get("fresh_news"))
    catalyst_count = _safe_int(candidate.get("catalyst_count"), 0)
    catalyst_score = _safe_float(candidate.get("catalyst_score"), 0.0)
    sentiment_score = _safe_float(candidate.get("sentiment_score"), 50.0)

    score = 50.0
    if fresh_news:
        score += 10.0
    score += min(12.0, catalyst_count * 4.0)
    score += min(10.0, catalyst_score * 12.0)

    # Sentiment is confirm/contradict only when an actual catalyst exists.
    if fresh_news or catalyst_count > 0 or catalyst_score > 0:
        if sentiment_score >= 65:
            score += 4.0
        elif sentiment_score <= 35:
            score -= 8.0

    return {"score": round(_clamp(score, 0.0, 100.0), 2)}


@lru_cache(maxsize=4)
def load_historical_context(project_root: str) -> Dict[str, Any]:
    try:
        root = Path(project_root)
        artifact_path = root / EDGE_ARTIFACT_PATH
        if artifact_path.exists():
            import json

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {"strategy": {}, "setup": {}, "symbol": {}}


def _edge_adjustment(
    candidate: Dict[str, Any], historical_context: Dict[str, Any]
) -> Dict[str, float]:
    strategy_name = str(candidate.get("strategy_name") or "unknown")
    setup_type = str(candidate.get("setup_type") or "unknown")
    symbol = (
        str(candidate.get("symbol") or candidate.get("ticker") or "").upper().strip()
    )

    strategy_row = (historical_context.get("strategy") or {}).get(strategy_name, {})
    setup_row = (historical_context.get("setup") or {}).get(setup_type, {})
    symbol_row = (historical_context.get("symbol") or {}).get(symbol, {})

    strategy_profit = _safe_float(strategy_row.get("profit_proxy"), 0.0)
    strategy_close = _safe_float(strategy_row.get("open_to_close"), 0.0)
    strategy_bad = _safe_float(strategy_row.get("bad_close_rate"), 0.0)
    strategy_good_close = _safe_float(strategy_row.get("positive_close_rate"), 0.0)
    strategy_adj = _clamp(
        (strategy_profit * 1.8)
        + (strategy_close * 1.8)
        + ((strategy_good_close - 0.5) * 8.0)
        - (strategy_bad * 10.0),
        -12.0,
        12.0,
    )
    setup_profit = _safe_float(setup_row.get("profit_proxy"), 0.0)
    setup_close = _safe_float(setup_row.get("open_to_close"), 0.0)
    setup_bad = _safe_float(setup_row.get("bad_close_rate"), 0.0)
    setup_good_close = _safe_float(setup_row.get("positive_close_rate"), 0.0)
    setup_adj = _clamp(
        (setup_profit * 1.3)
        + (setup_close * 1.4)
        + ((setup_good_close - 0.5) * 5.0)
        - (setup_bad * 7.0),
        -8.0,
        8.0,
    )

    symbol_profit = _safe_float(symbol_row.get("profit_proxy"), 0.0)
    symbol_close = _safe_float(symbol_row.get("open_to_close"), 0.0)
    symbol_bad = _safe_float(symbol_row.get("bad_close_rate"), 0.0)
    symbol_positive = _safe_float(symbol_row.get("positive_close_rate"), 0.0)
    appearances = _safe_int(symbol_row.get("count"), 0)
    symbol_adj = 0.0
    loser_penalty = 0.0
    if appearances >= 3:
        symbol_adj = _clamp(
            (symbol_profit * 1.1)
            + (symbol_close * 1.2)
            + ((symbol_positive - 0.5) * 4.0)
            - (symbol_bad * 6.0),
            -6.0,
            6.0,
        )
        if symbol_profit < 1.0 and symbol_close < 0.0:
            loser_penalty = min(
                10.0,
                abs(symbol_close) * 3.5
                + max(0.0, 1.0 - symbol_profit) * 1.8
                + (symbol_bad * 5.0),
            )

    total = round(strategy_adj + setup_adj + symbol_adj - loser_penalty, 2)
    return {
        "score": total,
        "strategy_edge": round(strategy_adj, 2),
        "setup_edge": round(setup_adj, 2),
        "symbol_edge": round(symbol_adj, 2),
        "symbol_loser_penalty": round(loser_penalty, 2),
    }


def _regime_bucket(label: str) -> str:
    value = str(label or "NEUTRAL").strip().upper()
    if any(
        token in value
        for token in ("SELL", "BEAR", "RISK_OFF", "CRISIS", "BULL_LOCK", "INVERSE")
    ):
        return "weak"
    if any(token in value for token in ("RISK_ON", "BULL", "ACCUM", "STRONG")):
        return "strong"
    return "neutral"


def _derive_stickiness_score(
    expected_high: float,
    expected_close: float,
    bad_close_prob: float,
    raw_value: Any,
) -> float:
    learned = _safe_float(raw_value, -1.0)
    if learned >= 0.0:
        return round(_clamp(learned, 0.0, 1.0), 4)
    if expected_high <= 0.0:
        fallback = 0.65 if expected_close > 0.0 else 0.1
    else:
        fallback = (max(0.0, expected_close) + 0.2) / max(expected_high, 0.85)
        if expected_close <= 0.0:
            fallback *= 0.35
        fallback -= bad_close_prob * 0.18
    return round(_clamp(fallback, 0.0, 1.0), 4)


def _derive_trap_risk(
    expected_high: float,
    expected_close: float,
    bad_close_prob: float,
    stickiness_score: float,
    raw_value: Any,
) -> float:
    learned = _safe_float(raw_value, -1.0)
    if learned >= 0.0:
        return round(_clamp(learned, 0.0, 1.0), 6)
    fade_span = max(0.0, expected_high - expected_close)
    fallback = (
        bad_close_prob * 0.55
        + max(0.0, 0.25 - expected_close) * 0.22
        + max(0.0, fade_span - 1.6) * 0.08
        + max(0.0, 0.4 - stickiness_score) * 0.55
    )
    if expected_high >= 2.8 and expected_close < 0.2:
        fallback += 0.12
    return round(_clamp(fallback, 0.0, 1.0), 6)


def _derive_fade_risk(
    expected_high: float,
    expected_close: float,
    trap_risk: float,
    raw_value: Any,
) -> float:
    learned = _safe_float(raw_value, -1.0)
    if learned >= 0.0:
        return round(_clamp(learned, 0.0, 1.0), 6)
    if expected_high <= 0.0:
        return 0.0
    fade_span = max(0.0, expected_high - expected_close)
    fallback = ((fade_span / max(expected_high, 0.85)) * 0.7) + (trap_risk * 0.35)
    if expected_high >= 3.0 and expected_close < 0.4:
        fallback += 0.08
    return round(_clamp(fallback, 0.0, 1.0), 6)


def _derive_execution_intent(
    *,
    expected_high: float,
    expected_close: float,
    trap_risk: float,
    fade_risk: float,
    stickiness_score: float,
    regime_bucket: str,
) -> str:
    intraday_skew_ratio = (
        (expected_close / expected_high) if expected_high > 0.0 else 1.0
    )
    if expected_high >= 2.4 and (
        expected_close < 0.35
        or trap_risk >= 0.42
        or fade_risk >= 0.48
        or stickiness_score < 0.34
    ):
        return "quick_turnover"
    if (
        expected_high >= 3.2
        and expected_close >= 0.35
        and intraday_skew_ratio < 0.82
        and stickiness_score < 0.4
        and fade_risk < 0.25
    ):
        return "quick_turnover"
    if (
        expected_close >= 0.35
        and trap_risk < 0.45
        and fade_risk < 0.5
        and stickiness_score >= 0.32
    ):
        return "hold_candidate"
    if regime_bucket == "weak":
        return (
            "hold_candidate"
            if stickiness_score >= 0.42 and expected_close >= 0.2
            else "quick_turnover"
        )
    return "quick_turnover" if expected_high >= 2.7 else "hold_candidate"


def _derive_support_break_trim_profile(
    *,
    execution_intent: str,
    regime_bucket: str,
    trap_risk: float,
    stickiness_score: float,
) -> str:
    if execution_intent == "quick_turnover":
        if regime_bucket == "weak" or trap_risk >= 0.5 or stickiness_score < 0.25:
            return "aggressive"
        return "balanced"
    if regime_bucket == "strong" and trap_risk < 0.32 and stickiness_score >= 0.5:
        return "patient"
    return "balanced"


def score_overnight_candidate(
    candidate: Dict[str, Any],
    *,
    project_root: Optional[Path] = None,
    historical_context: Optional[Dict[str, Any]] = None,
    model_artifact: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    history = (
        historical_context
        if historical_context is not None
        else load_historical_context(str(root))
    )
    if model_artifact is None:
        model_artifact = load_model_artifact(root / MODEL_ARTIFACT_PATH)

    actionability = _derive_actionability_score(candidate)
    learned_probs = predict_candidate_probabilities(candidate, model_artifact)
    learned_outcomes = predict_candidate_outcomes(candidate, model_artifact)
    model_1d = (
        round(float(learned_probs["hit_2pct"]) * 100.0, 2)
        if "hit_2pct" in learned_probs
        else _derive_1d_model_score(candidate, actionability["score"])
    )
    model_1w = (
        round(float(learned_probs["positive_close"]) * 100.0, 2)
        if "positive_close" in learned_probs
        else _derive_1w_model_score(candidate, actionability["score"])
    )
    bad_close_prob = float(learned_probs.get("bad_close", 0.0))
    expected_oh = max(
        0.0,
        _safe_float(learned_outcomes.get("open_to_high_pct"), (model_1d - 50.0) / 14.0),
    )
    expected_oc = _safe_float(
        learned_outcomes.get("open_to_close_pct"),
        ((model_1w - 50.0) / 18.0) - (bad_close_prob * 0.8),
    )
    expected_profit = _safe_float(
        learned_outcomes.get("profit_proxy"),
        (expected_oh * 0.65) + (expected_oc * 0.35),
    )
    expected_close_loss = max(
        0.0,
        _safe_float(
            learned_outcomes.get("close_loss_pct"),
            max(0.0, -expected_oc),
        ),
    )
    regime_label = str(
        candidate.get("regime")
        or candidate.get("market_regime")
        or candidate.get("strategic_regime")
        or "NEUTRAL"
    )
    regime_bucket = _regime_bucket(regime_label)
    stickiness_score = _derive_stickiness_score(
        expected_oh,
        expected_oc,
        bad_close_prob,
        learned_outcomes.get("stickiness_score"),
    )
    trap_risk = _derive_trap_risk(
        expected_oh,
        expected_oc,
        bad_close_prob,
        stickiness_score,
        learned_probs.get("trap_risk"),
    )
    fade_risk = _derive_fade_risk(
        expected_oh,
        expected_oc,
        trap_risk,
        learned_probs.get("fade_risk"),
    )
    execution_intent = _derive_execution_intent(
        expected_high=expected_oh,
        expected_close=expected_oc,
        trap_risk=trap_risk,
        fade_risk=fade_risk,
        stickiness_score=stickiness_score,
        regime_bucket=regime_bucket,
    )
    support_break_trim_profile = _derive_support_break_trim_profile(
        execution_intent=execution_intent,
        regime_bucket=regime_bucket,
        trap_risk=trap_risk,
        stickiness_score=stickiness_score,
    )
    catalyst_gate = _derive_catalyst_gate(candidate)
    edge = _edge_adjustment(candidate, history)

    legacy_repeat_penalty = _safe_float(candidate.get("fallback_repeat_penalty"), 0.0)
    loss_penalty = (
        (expected_close_loss * 4.5)
        + (bad_close_prob * 1.7)
        + (trap_risk * 4.8)
        + (fade_risk * 2.8)
    )
    if expected_oc < -0.5:
        loss_penalty += min(6.0, abs(expected_oc) * 2.0)
    if expected_profit < 0.8:
        loss_penalty += min(5.0, (0.8 - expected_profit) * 2.5)
    if regime_bucket == "weak":
        loss_penalty += (trap_risk * 1.8) + max(0.0, 0.15 - expected_oc) * 1.8
    close_quality_bonus = (stickiness_score * 7.5) + max(0.0, expected_oc) * 1.6
    turnover_bonus = max(0.0, expected_oh - 2.0) * (
        0.45 if execution_intent == "quick_turnover" else 0.22
    )
    final_score = (
        (expected_oh * 16.0)
        + (expected_profit * 6.5)
        + (expected_oc * 4.0)
        + close_quality_bonus
        + turnover_bonus
        + (actionability["score"] * 0.05)
        + (edge["score"] * 1.25)
        + (catalyst_gate["score"] * 0.015)
        + (2.0 if execution_intent == "hold_candidate" and expected_oc >= 0.35 else 0.0)
        - loss_penalty
        - legacy_repeat_penalty
    )
    final_score = round(_clamp(final_score, 0.0, 100.0), 2)

    reason = (
        f"expPP={expected_profit:.2f} expOC={expected_oc:.2f} expOH={expected_oh:.2f} "
        f"loss={expected_close_loss:.2f} bad={bad_close_prob * 100.0:.1f} "
        f"trap={trap_risk * 100.0:.1f} sticky={stickiness_score:.2f} "
        f"intent={execution_intent} action={actionability['score']:.1f} edge={edge['score']:+.1f}"
    )

    return {
        "final_score": final_score,
        "ranking_score": final_score,
        "overnight_model_score_1d": model_1d,
        "overnight_model_score_1w": model_1w,
        "overnight_hit_2pct_prob": round(float(learned_probs.get("hit_2pct", 0.0)), 6),
        "overnight_positive_close_prob": round(
            float(learned_probs.get("positive_close", 0.0)), 6
        ),
        "overnight_bad_close_prob": round(bad_close_prob, 6),
        "overnight_trap_risk": trap_risk,
        "overnight_fade_risk": fade_risk,
        "overnight_expected_open_to_high_pct": round(expected_oh, 4),
        "overnight_expected_open_to_close_pct": round(expected_oc, 4),
        "overnight_expected_profit_proxy": round(expected_profit, 4),
        "overnight_expected_close_loss_pct": round(expected_close_loss, 4),
        "overnight_stickiness_score": stickiness_score,
        "overnight_actionability_score": actionability["score"],
        "overnight_execution_intent": execution_intent,
        "overnight_catalyst_gate": catalyst_gate["score"],
        "overnight_strategy_edge": edge["strategy_edge"],
        "overnight_setup_edge": edge["setup_edge"],
        "overnight_symbol_edge": edge["symbol_edge"],
        "overnight_symbol_loser_penalty": edge["symbol_loser_penalty"],
        "support_break_trim_profile": support_break_trim_profile,
        "overnight_regime_bucket": regime_bucket,
        "support_dist_atr": actionability["support_dist_atr"],
        "resistance_dist_atr": actionability["resistance_dist_atr"],
        "overnight_rank_reason": reason,
        "legacy_backtest_diagnostic": {
            "backtest_win_rate": _safe_float(candidate.get("backtest_win_rate"), 0.0),
            "profit_factor": _safe_float(candidate.get("profit_factor"), 0.0),
            "backtest_scope": str(candidate.get("backtest_scope") or ""),
        },
    }


def _selection_priority(row: Dict[str, Any]) -> Tuple[float, float]:
    ranking_score = _safe_float(row.get("final_score", row.get("ranking_score")), 0.0)
    priority_score = _safe_float(row.get("ranking_score_with_boost"), ranking_score)
    expected_high = _safe_float(row.get("overnight_expected_open_to_high_pct"), 0.0)
    expected_profit = _safe_float(row.get("overnight_expected_profit_proxy"), 0.0)
    expected_close = _safe_float(row.get("overnight_expected_open_to_close_pct"), 0.0)
    hit_prob = _safe_float(row.get("overnight_hit_2pct_prob"), 0.0)
    bad_prob = _safe_float(row.get("overnight_bad_close_prob"), 0.0)
    trap_risk = _safe_float(row.get("overnight_trap_risk"), 0.0)
    fade_risk = _safe_float(row.get("overnight_fade_risk"), 0.0)
    stickiness_score = _safe_float(row.get("overnight_stickiness_score"), 0.0)
    actionability = _safe_float(row.get("overnight_actionability_score"), 0.0)
    execution_intent = str(row.get("overnight_execution_intent") or "").lower()
    primary = (
        (priority_score * 0.02)
        + (expected_high * 1.4)
        + expected_profit * 0.9
        + expected_close * 0.85
        + stickiness_score * 0.95
        + hit_prob * 0.35
        + actionability * 0.015
        - bad_prob * 0.55
        - trap_risk * 0.75
        - fade_risk * 0.35
    )
    if execution_intent == "quick_turnover":
        primary += max(0.0, expected_high - 2.0) * 0.2
    elif execution_intent == "hold_candidate":
        primary += max(0.0, expected_close) * 0.15 + stickiness_score * 0.2
    tiebreak = priority_score + (_safe_float(row.get("confidence"), 0.0) * 0.05)
    return primary, tiebreak


def _selection_risk_flags(row: Dict[str, Any]) -> Dict[str, bool]:
    setup = str(row.get("setup_type") or "")
    bad_prob = _safe_float(row.get("overnight_bad_close_prob"), 0.0)
    trap_risk = _safe_float(row.get("overnight_trap_risk"), 0.0)
    fade_risk = _safe_float(row.get("overnight_fade_risk"), 0.0)
    stickiness_score = _safe_float(row.get("overnight_stickiness_score"), 0.0)
    exp_close = _safe_float(row.get("overnight_expected_open_to_close_pct"), 0.0)
    exp_high = _safe_float(row.get("overnight_expected_open_to_high_pct"), 0.0)
    setup_edge = _safe_float(row.get("overnight_setup_edge"), 0.0)
    strategy_edge = _safe_float(row.get("overnight_strategy_edge"), 0.0)
    execution_intent = str(row.get("overnight_execution_intent") or "").lower()
    return {
        "is_momentum_breakout": setup == "momentum_breakout",
        "is_negative_close": exp_close < 0.0,
        "is_high_bad_close": bad_prob >= 0.22,
        "is_bad_setup": setup_edge < -0.5 or strategy_edge < -1.5,
        "is_extreme_tail": exp_close < -0.25
        and (bad_prob >= 0.20 or trap_risk >= 0.45),
        "is_high_trap": trap_risk >= 0.48 or fade_risk >= 0.55,
        "is_weak_stickiness": stickiness_score < 0.24,
        "is_quick_turnover": execution_intent == "quick_turnover",
        "hard_skip": (
            (
                setup == "momentum_breakout"
                and bad_prob >= 0.18
                and trap_risk >= 0.45
                and exp_close < 0.25
            )
            or (exp_close < -0.35 and bad_prob >= 0.22 and exp_high < 3.4)
            or (
                (setup_edge < -0.5 or strategy_edge < -1.5)
                and exp_high < 2.8
                and exp_close < 0.1
            )
            or (trap_risk >= 0.62 and exp_close < 0.1 and stickiness_score < 0.24)
        ),
    }


def select_constrained_overnight_candidates(
    rows: Iterable[Dict[str, Any]],
    *,
    top_n: int = 50,
    regime_label: Optional[str] = None,
    post_spike_volume_threshold: float = DEFAULT_POST_SPIKE_VOLUME_THRESHOLD,
    post_spike_range_atr_threshold: float = DEFAULT_POST_SPIKE_RANGE_ATR_THRESHOLD,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    post_spike_skipped = 0
    ranked_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if is_post_spike_long_candidate(
            row,
            volume_threshold=post_spike_volume_threshold,
            range_atr_threshold=post_spike_range_atr_threshold,
        ):
            post_spike_skipped += 1
            continue
        ranked_rows.append(dict(row))
    ranked_rows.sort(key=_selection_priority, reverse=True)
    if top_n <= 0 or not ranked_rows:
        return [], {"requested": int(top_n), "selected": 0}

    inferred_regime = regime_label
    if not inferred_regime:
        for row in ranked_rows:
            inferred_regime = str(
                row.get("regime")
                or row.get("market_regime")
                or row.get("strategic_regime")
                or ""
            ).strip()
            if inferred_regime:
                break
    regime_bucket = _regime_bucket(str(inferred_regime or "NEUTRAL"))
    max_negative_close = max(
        1,
        int(
            round(
                top_n * {"weak": 0.02, "neutral": 0.04, "strong": 0.06}[regime_bucket]
            )
        ),
    )
    max_high_bad = max(
        1,
        int(
            round(
                top_n * {"weak": 0.02, "neutral": 0.04, "strong": 0.06}[regime_bucket]
            )
        ),
    )
    max_bad_setup = max(
        1,
        int(
            round(
                top_n * {"weak": 0.05, "neutral": 0.06, "strong": 0.08}[regime_bucket]
            )
        ),
    )
    max_extreme_tail = max(
        1,
        int(
            round(
                top_n * {"weak": 0.02, "neutral": 0.04, "strong": 0.05}[regime_bucket]
            )
        ),
    )
    max_high_trap = max(
        1,
        int(
            round(
                top_n * {"weak": 0.08, "neutral": 0.10, "strong": 0.14}[regime_bucket]
            )
        ),
    )
    max_quick_turnover = max(
        1,
        int(
            round(top_n * {"weak": 0.55, "neutral": 0.7, "strong": 0.82}[regime_bucket])
        ),
    )
    selected: List[Dict[str, Any]] = []
    counts = {
        "negative_close": 0,
        "high_bad_close": 0,
        "bad_setup": 0,
        "extreme_tail": 0,
        "high_trap": 0,
        "quick_turnover": 0,
        "hold_candidate": 0,
        "weak_stickiness": 0,
        "momentum_breakout": 0,
        "hard_skipped": 0,
        "cap_skipped": 0,
        "post_spike_skipped": post_spike_skipped,
    }
    selected_symbols: set[str] = set()

    for row in ranked_rows:
        symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
        if symbol and symbol in selected_symbols:
            continue
        flags = _selection_risk_flags(row)
        if flags["hard_skip"]:
            counts["hard_skipped"] += 1
            continue
        if (
            flags["is_negative_close"]
            and counts["negative_close"] >= max_negative_close
        ):
            counts["cap_skipped"] += 1
            continue
        if flags["is_high_bad_close"] and counts["high_bad_close"] >= max_high_bad:
            counts["cap_skipped"] += 1
            continue
        if flags["is_bad_setup"] and counts["bad_setup"] >= max_bad_setup:
            counts["cap_skipped"] += 1
            continue
        if flags["is_extreme_tail"] and counts["extreme_tail"] >= max_extreme_tail:
            counts["cap_skipped"] += 1
            continue
        if flags["is_high_trap"] and counts["high_trap"] >= max_high_trap:
            counts["cap_skipped"] += 1
            continue
        if (
            flags["is_quick_turnover"]
            and counts["quick_turnover"] >= max_quick_turnover
        ):
            counts["cap_skipped"] += 1
            continue
        selected.append(row)
        if symbol:
            selected_symbols.add(symbol)
        counts["negative_close"] += int(flags["is_negative_close"])
        counts["high_bad_close"] += int(flags["is_high_bad_close"])
        counts["bad_setup"] += int(flags["is_bad_setup"])
        counts["extreme_tail"] += int(flags["is_extreme_tail"])
        counts["high_trap"] += int(flags["is_high_trap"])
        counts["quick_turnover"] += int(flags["is_quick_turnover"])
        counts["hold_candidate"] += int(not flags["is_quick_turnover"])
        counts["weak_stickiness"] += int(flags["is_weak_stickiness"])
        counts["momentum_breakout"] += int(flags["is_momentum_breakout"])
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        for row in ranked_rows:
            symbol = str(row.get("symbol") or row.get("ticker") or "").upper().strip()
            if symbol and symbol in selected_symbols:
                continue
            selected.append(row)
            if symbol:
                selected_symbols.add(symbol)
            if len(selected) >= top_n:
                break

    for idx, row in enumerate(selected, start=1):
        row["overnight_actionable_priority"] = idx

    diagnostics = {
        "requested": int(top_n),
        "selected": int(len(selected)),
        "regime_label": str(inferred_regime or "NEUTRAL"),
        "regime_bucket": regime_bucket,
        "max_negative_close": max_negative_close,
        "max_high_bad_close": max_high_bad,
        "max_bad_setup": max_bad_setup,
        "max_extreme_tail": max_extreme_tail,
        "max_high_trap": max_high_trap,
        "max_quick_turnover": max_quick_turnover,
        **counts,
    }
    return selected, diagnostics
