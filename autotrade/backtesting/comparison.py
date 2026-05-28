"""
Strategy comparison helpers with lightweight significance testing.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Any, Dict, Tuple


@dataclass
class ComparisonResult:
    is_better: bool
    verdict: str
    confidence: float
    deltas: Dict[str, float]
    details: Dict[str, Any]


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _coerce_metrics(result: Any) -> Dict[str, float]:
    if isinstance(result, dict):
        raw = result
    elif hasattr(result, "__dict__"):
        raw = dict(result.__dict__)
    else:
        raw = {}

    win_rate = float(raw.get("win_rate", 0.0) or 0.0)
    if win_rate > 1.0:
        win_rate /= 100.0

    max_drawdown = float(raw.get("max_drawdown", raw.get("max_drawdown_pct", 0.0)) or 0.0)
    # Normalize to positive fraction scale.
    if max_drawdown > 1.5:
        max_drawdown /= 100.0
    if max_drawdown < 0:
        max_drawdown = abs(max_drawdown)

    sharpe = float(raw.get("sharpe_ratio", raw.get("sharpe", 0.0)) or 0.0)

    total_pnl = float(
        raw.get("total_pnl", raw.get("total_pnl_dollars", raw.get("net_pnl", 0.0)))
        or 0.0
    )
    profit_factor = float(raw.get("profit_factor", 0.0) or 0.0)
    total_trades = int(raw.get("total_trades", 0) or 0)

    return {
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
        "total_pnl": total_pnl,
        "total_trades": float(total_trades),
    }


def _two_proportion_p_value(win_a: float, n_a: int, win_b: float, n_b: int) -> float:
    if n_a <= 0 or n_b <= 0:
        return 1.0
    p1 = max(0.0, min(1.0, win_a))
    p2 = max(0.0, min(1.0, win_b))
    x1 = p1 * n_a
    x2 = p2 * n_b
    pooled = (x1 + x2) / (n_a + n_b)
    denom = pooled * (1.0 - pooled) * ((1.0 / n_a) + (1.0 / n_b))
    if denom <= 0:
        return 1.0
    z = (p2 - p1) / sqrt(denom)
    return max(0.0, min(1.0, 2.0 * (1.0 - _normal_cdf(abs(z)))))


class StrategyComparator:
    """
    Compare challenger strategy metrics against baseline metrics.

    Supports dataclass-like results and plain dict payloads.
    """

    def _metric_winners(self, base: Dict[str, float], chall: Dict[str, float]) -> Dict[str, str]:
        winners: Dict[str, str] = {}
        for metric in ("win_rate", "profit_factor", "sharpe_ratio", "total_pnl", "total_trades"):
            if chall[metric] > base[metric]:
                winners[metric] = "challenger"
            elif chall[metric] < base[metric]:
                winners[metric] = "baseline"
            else:
                winners[metric] = "tie"

        # Lower drawdown is better.
        if chall["max_drawdown"] < base["max_drawdown"]:
            winners["max_drawdown"] = "challenger"
        elif chall["max_drawdown"] > base["max_drawdown"]:
            winners["max_drawdown"] = "baseline"
        else:
            winners["max_drawdown"] = "tie"
        return winners

    def compare(self, baseline: Any, challenger: Any) -> ComparisonResult:
        base = _coerce_metrics(baseline)
        chall = _coerce_metrics(challenger)

        deltas = {
            "win_rate": chall["win_rate"] - base["win_rate"],
            "profit_factor": chall["profit_factor"] - base["profit_factor"],
            "sharpe_ratio": chall["sharpe_ratio"] - base["sharpe_ratio"],
            "max_drawdown": chall["max_drawdown"] - base["max_drawdown"],
            "total_pnl": chall["total_pnl"] - base["total_pnl"],
            "total_trades": chall["total_trades"] - base["total_trades"],
        }

        n_a = int(base["total_trades"])
        n_b = int(chall["total_trades"])
        p_value_win_rate = _two_proportion_p_value(base["win_rate"], n_a, chall["win_rate"], n_b)

        score = 0.0
        if deltas["profit_factor"] > 0.05:
            score += 1.8
        if deltas["sharpe_ratio"] > 0.08:
            score += 1.2
        if deltas["total_pnl"] > 0:
            score += 1.0
        if deltas["max_drawdown"] < -0.01:
            score += 0.8

        if deltas["win_rate"] > 0:
            score += 0.6 if p_value_win_rate < 0.10 else 0.3
        elif deltas["win_rate"] < -0.015:
            score -= 1.2
        if n_b < 10:
            score -= 1.0

        is_better = score >= 2.0 and deltas["profit_factor"] >= 0
        if n_b < 5:
            verdict = "TOO_FEW_TRADES"
            confidence = 0.95
            is_better = False
        elif is_better:
            verdict = "IMPROVED_SIGNIFICANT" if p_value_win_rate < 0.05 else "IMPROVED"
            confidence = min(0.95, 0.55 + (score * 0.08))
        elif score >= 1.2:
            verdict = "MARGINAL"
            confidence = 0.65
        else:
            verdict = "NO_IMPROVEMENT"
            confidence = 0.85

        details = {
            "baseline_metrics": base,
            "challenger_metrics": chall,
            "metric_winners": self._metric_winners(base, chall),
            "significance": {
                "win_rate_p_value": p_value_win_rate,
                "win_rate_significant_95": p_value_win_rate < 0.05,
                "win_rate_significant_90": p_value_win_rate < 0.10,
            },
            "score": score,
        }

        return ComparisonResult(
            is_better=is_better,
            verdict=verdict,
            confidence=float(max(0.0, min(1.0, confidence))),
            deltas=deltas,
            details=details,
        )
