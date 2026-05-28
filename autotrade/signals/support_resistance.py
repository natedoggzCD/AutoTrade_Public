"""
Internal support/resistance estimation from OHLCV history.

This module provides a lightweight, technical-data-only S/R estimator used by
the scanner and day execution paths. It avoids external SR databases and
derives levels directly from recent price structure + ATR.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def _resolve_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _compute_atr(df: pd.DataFrame) -> float:
    high_col = _resolve_col(df, ["high", "High"])
    low_col = _resolve_col(df, ["low", "Low"])
    close_col = _resolve_col(df, ["close", "Close"])
    if not high_col or not low_col or not close_col or len(df) < 3:
        return 0.0

    high = pd.to_numeric(df[high_col], errors="coerce")
    low = pd.to_numeric(df[low_col], errors="coerce")
    close = pd.to_numeric(df[close_col], errors="coerce")
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = float(tr.rolling(14, min_periods=5).mean().iloc[-1] or 0.0)
    return atr if np.isfinite(atr) else 0.0


def _cluster_pivots(
    prices: np.ndarray,
    bar_idx: np.ndarray,
    volumes: np.ndarray,
    tolerance: float,
) -> List[Dict[str, float]]:
    if prices.size == 0:
        return []
    order = np.argsort(prices)
    prices = prices[order]
    bar_idx = bar_idx[order]
    volumes = volumes[order]

    clusters: List[Dict[str, float]] = []
    cur_prices: List[float] = [float(prices[0])]
    cur_idx: List[int] = [int(bar_idx[0])]
    cur_vols: List[float] = [float(volumes[0])]

    for i in range(1, len(prices)):
        px = float(prices[i])
        center = float(np.mean(cur_prices))
        if abs(px - center) <= tolerance:
            cur_prices.append(px)
            cur_idx.append(int(bar_idx[i]))
            cur_vols.append(float(volumes[i]))
            continue

        clusters.append(
            {
                "price": float(np.mean(cur_prices)),
                "touches": float(len(cur_prices)),
                "last_idx": float(max(cur_idx)),
                "avg_volume": float(np.mean(cur_vols)),
            }
        )
        cur_prices = [px]
        cur_idx = [int(bar_idx[i])]
        cur_vols = [float(volumes[i])]

    clusters.append(
        {
            "price": float(np.mean(cur_prices)),
            "touches": float(len(cur_prices)),
            "last_idx": float(max(cur_idx)),
            "avg_volume": float(np.mean(cur_vols)),
        }
    )
    return clusters


def _score_cluster(
    cluster: Dict[str, float],
    *,
    current_price: float,
    atr: float,
    max_idx: int,
    median_volume: float,
    is_support: bool,
) -> float:
    price = float(cluster.get("price", 0.0) or 0.0)
    if price <= 0:
        return -1.0
    if is_support and price >= current_price:
        return -1.0
    if (not is_support) and price <= current_price:
        return -1.0

    atr_safe = max(atr, current_price * 0.01, 1e-6)
    dist_atr = abs(current_price - price) / atr_safe
    if dist_atr > 4.0:
        return -1.0

    touches = float(cluster.get("touches", 0.0) or 0.0)
    last_idx = float(cluster.get("last_idx", 0.0) or 0.0)
    avg_vol = float(cluster.get("avg_volume", 0.0) or 0.0)

    touches_score = min(62.0, touches * 16.0)
    recency_bars = max(0.0, float(max_idx) - last_idx)
    recency_score = max(0.0, 23.0 - recency_bars * 0.55)

    volume_score = 0.0
    if median_volume > 0:
        volume_score = min(15.0, max(0.0, (avg_vol / median_volume - 0.8) * 16.0))

    proximity_score = max(0.0, 20.0 - dist_atr * 8.0)
    return touches_score + recency_score + volume_score + proximity_score


def estimate_sr_levels(
    bars: pd.DataFrame,
    *,
    lookback_bars: int = 120,
    pivot_window: int = 3,
    min_touches: int = 2,
    cluster_atr_mult: float = 0.35,
) -> Optional[Dict[str, float]]:
    """
    Estimate S/R levels from OHLCV history.

    Returns scanner-compatible fields:
    s1_price/s1_strength/r1_price/r1_strength/support_dist_atr/resistance_dist_atr/
    distance_to_s1_pct/distance_to_r1_pct/sr_quality_score
    """
    if bars is None or bars.empty:
        return None

    high_col = _resolve_col(bars, ["high", "High"])
    low_col = _resolve_col(bars, ["low", "Low"])
    close_col = _resolve_col(bars, ["close", "Close"])
    vol_col = _resolve_col(bars, ["volume", "Volume"])
    atr_col = _resolve_col(bars, ["atr_14", "ATR_14"])
    if not high_col or not low_col or not close_col:
        return None

    df = bars.copy()
    if "Date" in df.columns:
        df = df.sort_values("Date")
    elif "date" in df.columns:
        df = df.sort_values("date")
    else:
        df = df.sort_index()

    df = df.tail(max(30, int(lookback_bars))).copy()
    highs = pd.to_numeric(df[high_col], errors="coerce")
    lows = pd.to_numeric(df[low_col], errors="coerce")
    closes = pd.to_numeric(df[close_col], errors="coerce")
    volumes = (
        pd.to_numeric(df[vol_col], errors="coerce").fillna(0.0)
        if vol_col
        else pd.Series(0.0, index=df.index)
    )

    if len(df) < max(10, pivot_window * 2 + 3):
        return None
    current_price = float(closes.iloc[-1] or 0.0)
    if current_price <= 0:
        return None

    atr = 0.0
    if atr_col:
        atr = float(pd.to_numeric(df[atr_col], errors="coerce").iloc[-1] or 0.0)
    if atr <= 0:
        atr = _compute_atr(df)
    if atr <= 0:
        atr = current_price * 0.02

    tolerance = max(atr * float(cluster_atr_mult), current_price * 0.0035)
    window_size = int(pivot_window) * 2 + 1
    piv_hi = highs.eq(
        highs.rolling(window_size, center=True, min_periods=window_size).max()
    )
    piv_lo = lows.eq(
        lows.rolling(window_size, center=True, min_periods=window_size).min()
    )
    pivot_idx = np.arange(len(df), dtype=np.int64)

    high_prices = highs[piv_hi].dropna().to_numpy(dtype=np.float64)
    high_indices = pivot_idx[piv_hi.to_numpy()]
    high_vols = volumes[piv_hi].fillna(0.0).to_numpy(dtype=np.float64)

    low_prices = lows[piv_lo].dropna().to_numpy(dtype=np.float64)
    low_indices = pivot_idx[piv_lo.to_numpy()]
    low_vols = volumes[piv_lo].fillna(0.0).to_numpy(dtype=np.float64)

    support_clusters = [
        c
        for c in _cluster_pivots(low_prices, low_indices, low_vols, tolerance)
        if c.get("touches", 0.0) >= float(min_touches)
    ]
    resistance_clusters = [
        c
        for c in _cluster_pivots(high_prices, high_indices, high_vols, tolerance)
        if c.get("touches", 0.0) >= float(min_touches)
    ]

    max_idx = len(df) - 1
    median_volume = float(volumes.tail(40).median() or 0.0)

    best_support = None
    best_support_score = -1.0
    for c in support_clusters:
        score = _score_cluster(
            c,
            current_price=current_price,
            atr=atr,
            max_idx=max_idx,
            median_volume=median_volume,
            is_support=True,
        )
        if score > best_support_score:
            best_support_score = score
            best_support = c

    best_resistance = None
    best_resistance_score = -1.0
    for c in resistance_clusters:
        score = _score_cluster(
            c,
            current_price=current_price,
            atr=atr,
            max_idx=max_idx,
            median_volume=median_volume,
            is_support=False,
        )
        if score > best_resistance_score:
            best_resistance_score = score
            best_resistance = c

    fallback_window = min(len(df), 60)
    if fallback_window < 10:
        fallback_window = len(df)

    support_price = (
        float(best_support["price"])
        if best_support is not None
        else float(lows.tail(fallback_window).quantile(0.25))
    )
    resistance_price = (
        float(best_resistance["price"])
        if best_resistance is not None
        else float(highs.tail(fallback_window).quantile(0.75))
    )

    if support_price <= 0 or support_price >= current_price:
        support_price = max(0.01, current_price - atr * 1.2)
    if resistance_price <= current_price:
        resistance_price = current_price + atr * 1.6

    if resistance_price <= support_price:
        resistance_price = current_price + atr * 1.6
        support_price = max(0.01, current_price - atr * 1.2)

    if best_support is not None:
        s1_strength = float(max(20.0, min(95.0, best_support_score)))
    else:
        s1_strength = 40.0
    if best_resistance is not None:
        r1_strength = float(max(20.0, min(95.0, best_resistance_score)))
    else:
        r1_strength = 40.0

    atr_safe = max(atr, current_price * 0.01, 1e-6)
    support_dist_atr = max(0.0, (current_price - support_price) / atr_safe)
    resistance_dist_atr = max(0.0, (resistance_price - current_price) / atr_safe)

    distance_to_s1_pct = max(
        0.0, ((current_price - support_price) / max(current_price, 1e-6)) * 100.0
    )
    distance_to_r1_pct = max(
        0.0, ((resistance_price - current_price) / max(current_price, 1e-6)) * 100.0
    )

    range_quality = min(10.0, (resistance_price - support_price) / atr_safe)
    sr_quality_score = float(
        max(0.0, min(100.0, (s1_strength * 0.5) + (r1_strength * 0.4) + range_quality))
    )

    return {
        "s1_price": float(support_price),
        "s1_strength": float(s1_strength),
        "r1_price": float(resistance_price),
        "r1_strength": float(r1_strength),
        "support_dist_atr": float(support_dist_atr),
        "resistance_dist_atr": float(resistance_dist_atr),
        "distance_to_s1_pct": float(distance_to_s1_pct),
        "distance_to_r1_pct": float(distance_to_r1_pct),
        "sr_quality_score": float(sr_quality_score),
    }

