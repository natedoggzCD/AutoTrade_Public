from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

import numpy as np


_CI_LEVELS: tuple[float, ...] = (0.80, 0.90, 0.95)


def _safe_returns(trade_returns: Sequence[float]) -> List[float]:
    cleaned: List[float] = []
    for value in trade_returns or []:
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed):
            cleaned.append(parsed)
    return cleaned


def _profit_factor(returns: Sequence[float], pf_cap: float = 10.0) -> float:
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    if gross_loss <= 1e-12:
        return float(pf_cap if gross_profit > 0 else 0.0)
    return float(gross_profit / gross_loss)


def _sharpe_ratio(returns: Sequence[float]) -> float:
    if len(returns) < 2:
        return 0.0
    arr = np.asarray(returns, dtype=float)
    std = float(arr.std(ddof=1))
    if std <= 1e-12:
        return 0.0
    return float(arr.mean() / std)


def _win_rate(returns: Sequence[float]) -> float:
    if not returns:
        return 0.0
    wins = sum(1 for r in returns if r > 0)
    return float(wins / len(returns))


def _metrics(returns: Sequence[float]) -> Dict[str, float]:
    return {
        "profit_factor": _profit_factor(returns),
        "sharpe": _sharpe_ratio(returns),
        "win_rate": _win_rate(returns),
    }


def _stratified_resample(returns: Sequence[float], rng: random.Random) -> List[float]:
    wins = [r for r in returns if r > 0]
    non_wins = [r for r in returns if r <= 0]

    sampled: List[float] = []
    if wins:
        sampled.extend(rng.choices(wins, k=len(wins)))
    if non_wins:
        sampled.extend(rng.choices(non_wins, k=len(non_wins)))
    if not sampled:
        return []
    rng.shuffle(sampled)
    return sampled


def bootstrap_trade_metrics(
    trade_returns: Sequence[float],
    *,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> Dict[str, object]:
    """
    Bootstrap confidence intervals for trade-level metrics.

    Returns confidence intervals for 80/90/95% on:
    - profit_factor
    - sharpe
    - win_rate
    """
    returns = _safe_returns(trade_returns)
    point = _metrics(returns)
    n_sims = max(1, int(n_simulations))

    if not returns:
        empty_ci = {
            f"{int(level * 100)}%": {
                "profit_factor": {"low": 0.0, "high": 0.0},
                "sharpe": {"low": 0.0, "high": 0.0},
                "win_rate": {"low": 0.0, "high": 0.0},
            }
            for level in _CI_LEVELS
        }
        return {
            "n_trades": 0,
            "n_simulations": n_sims,
            "random_seed": int(random_seed),
            "point_estimate": point,
            "confidence_intervals": empty_ci,
        }

    rng = random.Random(int(random_seed))
    pf_samples: List[float] = []
    sharpe_samples: List[float] = []
    wr_samples: List[float] = []

    for _ in range(n_sims):
        sampled = _stratified_resample(returns, rng)
        stats = _metrics(sampled)
        pf_samples.append(float(stats["profit_factor"]))
        sharpe_samples.append(float(stats["sharpe"]))
        wr_samples.append(float(stats["win_rate"]))

    ci: Dict[str, Dict[str, Dict[str, float]]] = {}
    for level in _CI_LEVELS:
        low_q = (1.0 - float(level)) / 2.0
        high_q = 1.0 - low_q
        key = f"{int(level * 100)}%"
        ci[key] = {
            "profit_factor": {
                "low": float(np.quantile(pf_samples, low_q)),
                "high": float(np.quantile(pf_samples, high_q)),
            },
            "sharpe": {
                "low": float(np.quantile(sharpe_samples, low_q)),
                "high": float(np.quantile(sharpe_samples, high_q)),
            },
            "win_rate": {
                "low": float(np.quantile(wr_samples, low_q)),
                "high": float(np.quantile(wr_samples, high_q)),
            },
        }

    return {
        "n_trades": len(returns),
        "n_simulations": n_sims,
        "random_seed": int(random_seed),
        "point_estimate": point,
        "confidence_intervals": ci,
    }

