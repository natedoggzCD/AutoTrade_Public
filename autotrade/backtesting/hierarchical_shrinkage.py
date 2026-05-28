"""Hierarchical Bayesian shrinkage for per-symbol alpha metrics.

For each (symbol, alpha) pair we observe a raw estimate (e.g. profit factor,
expectancy) from N trades. If N is small the estimate is noisy. We shrink
toward a cluster prior so thin-history symbols don't get promoted on luck.

The shrinkage weight is w = n / (n + k), where k controls the shrinkage
strength (k=30 by default — at n=30 the posterior is the average of symbol
and cluster; at n=300 the symbol dominates 10:1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np

DEFAULT_K = 30.0
MIN_CLUSTER_N = 50
MAX_POSTERIOR_PF = 5.0


@dataclass
class SymbolAlphaEstimate:
    symbol: str
    alpha_id: str
    cluster_key: str
    n_trades: int
    raw_mean_return: float
    raw_std_return: float
    raw_pf: float


@dataclass
class ShrinkageResult:
    symbol: str
    alpha_id: str
    cluster_key: str
    n_trades: int
    n_cluster_trades: int
    raw_mean_return: float
    cluster_mean_return: float
    posterior_mean_return: float
    posterior_se: float
    posterior_pf: float
    weight: float

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "alpha_id": self.alpha_id,
            "cluster_key": self.cluster_key,
            "n_trades": self.n_trades,
            "n_cluster_trades": self.n_cluster_trades,
            "raw_mean_return": self.raw_mean_return,
            "cluster_mean_return": self.cluster_mean_return,
            "posterior_mean_return": self.posterior_mean_return,
            "posterior_se": self.posterior_se,
            "posterior_pf": self.posterior_pf,
            "weight": self.weight,
        }


def shrink_estimates(
    estimates: Iterable[SymbolAlphaEstimate],
    k: float = DEFAULT_K,
) -> List[ShrinkageResult]:
    """Apply per-alpha cluster shrinkage to a list of symbol estimates.

    Clusters are grouped by (alpha_id, cluster_key). The cluster prior is the
    trade-weighted mean of all symbol estimates in the cluster.
    """
    items = list(estimates)
    if not items:
        return []

    cluster_summary: Dict[tuple, dict] = {}
    for it in items:
        key = (it.alpha_id, it.cluster_key)
        s = cluster_summary.setdefault(
            key, {"sum_weighted": 0.0, "sum_n": 0, "stds": []}
        )
        s["sum_weighted"] += it.raw_mean_return * it.n_trades
        s["sum_n"] += it.n_trades
        s["stds"].append(it.raw_std_return)

    cluster_mean: Dict[tuple, float] = {}
    cluster_n: Dict[tuple, int] = {}
    cluster_std: Dict[tuple, float] = {}
    for key, s in cluster_summary.items():
        if s["sum_n"] > 0:
            cluster_mean[key] = s["sum_weighted"] / s["sum_n"]
        else:
            cluster_mean[key] = 0.0
        cluster_n[key] = s["sum_n"]
        # average within-symbol std as cluster volatility proxy
        cluster_std[key] = float(np.nanmean(s["stds"])) if s["stds"] else 0.0

    results: List[ShrinkageResult] = []
    for it in items:
        key = (it.alpha_id, it.cluster_key)
        prior_mean = cluster_mean.get(key, 0.0)
        prior_n = cluster_n.get(key, 0)

        # If the cluster is too sparse, fall back to no-prior (posterior == raw).
        if prior_n < MIN_CLUSTER_N:
            w = 1.0
            posterior = it.raw_mean_return
        else:
            w = it.n_trades / (it.n_trades + k)
            posterior = w * it.raw_mean_return + (1.0 - w) * prior_mean

        # Posterior standard error: blend of symbol SE and cluster SE,
        # weighted by inverse-variance proxy.
        symbol_se = (
            (it.raw_std_return / np.sqrt(it.n_trades))
            if it.n_trades > 0
            else float("inf")
        )
        cluster_se = (
            (cluster_std.get(key, 0.0) / np.sqrt(max(prior_n, 1)))
            if prior_n > 0
            else float("inf")
        )
        if not np.isfinite(symbol_se) and not np.isfinite(cluster_se):
            posterior_se = float("inf")
        else:
            posterior_se = float(
                np.sqrt(
                    (w**2) * (symbol_se**2 if np.isfinite(symbol_se) else 0.0)
                    + ((1.0 - w) ** 2)
                    * (cluster_se**2 if np.isfinite(cluster_se) else 0.0)
                )
            )

        # Posterior PF is approximated by shrinking the raw PF toward 1.0
        # (no-edge prior) using the same weight, unless the cluster prior PF
        # is computed elsewhere. This keeps PF and mean-return consistent in
        # sign without re-running the trade ledger.
        posterior_pf = min(MAX_POSTERIOR_PF, w * it.raw_pf + (1.0 - w) * 1.0)

        results.append(
            ShrinkageResult(
                symbol=it.symbol,
                alpha_id=it.alpha_id,
                cluster_key=it.cluster_key,
                n_trades=it.n_trades,
                n_cluster_trades=prior_n,
                raw_mean_return=it.raw_mean_return,
                cluster_mean_return=prior_mean,
                posterior_mean_return=posterior,
                posterior_se=posterior_se,
                posterior_pf=posterior_pf,
                weight=w,
            )
        )

    return results


def assign_cluster_key(
    symbol: str,
    sector: Optional[str],
    atr_pct: Optional[float],
    market_cap_bucket: Optional[str],
) -> str:
    """Compose a cluster identifier from sector + ATR percentile band + cap bucket.

    Bands keep cluster count manageable while preserving regime homogeneity.
    """
    sector_part = (sector or "UNK").upper()[:8]
    if atr_pct is None or not np.isfinite(atr_pct):
        atr_band = "atrUNK"
    elif atr_pct < 2.0:
        atr_band = "atrLOW"
    elif atr_pct < 4.0:
        atr_band = "atrMID"
    else:
        atr_band = "atrHI"
    cap_part = (market_cap_bucket or "UNK").upper()[:6]
    return f"{sector_part}|{atr_band}|{cap_part}"
