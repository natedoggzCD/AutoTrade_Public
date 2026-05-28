"""
Inverse ETF Intraday Screener
==============================
Screens inverse ETFs intraday for selective entries on bad market days.

Three-gate scoring system:
  Gate 1: Liquidity (volume + AUM)
  Gate 2: Momentum burst (volume ratio + price acceleration)
  Gate 3: VWAP/RSI entry timing

Usage:
    from autotrade.signals.inverse_etf_screener import InverseETFScreener
    from autotrade.utils.financial_db import FinancialDB

    screener = InverseETFScreener(FinancialDB(), data_client=alpaca_data_client)
    candidates = screener.screen_universe(regime="crisis", portfolio_holdings=["AAPL"])
"""

from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger("AutoTrade.InverseETFScreener")

BEARISH_REGIMES = {
    "crisis",
    "risk_off",
    "selloff",
    "capitulation",
    "crash",
    "bear",
    "bearish",
}

INVERSE_ETF_TIERS = {
    "1x": ["SH", "PSQ", "RWM", "DOG"],
    "2x": ["SDS", "QID", "SDD", "MZZ", "TWM"],
    "3x": ["SQQQ", "SDOW", "TZA", "FAZ", "SOXS", "SPXU", "SRTY"],
}

INVERSE_ETF_LONG_PROXIES = {
    "SH": {"benchmark": "SPY", "beta": 1.0},
    "SDS": {"benchmark": "SPY", "beta": 2.0},
    "SPXU": {"benchmark": "SPY", "beta": 3.0},
    "PSQ": {"benchmark": "QQQ", "beta": 1.0},
    "QID": {"benchmark": "QQQ", "beta": 2.0},
    "SQQQ": {"benchmark": "QQQ", "beta": 3.0},
    "RWM": {"benchmark": "IWM", "beta": 1.0},
    "TWM": {"benchmark": "IWM", "beta": 2.0},
    "TZA": {"benchmark": "IWM", "beta": 3.0},
    "DOG": {"benchmark": "DIA", "beta": 1.0},
    "SDD": {"benchmark": "DIA", "beta": 2.0},
    "SDOW": {"benchmark": "DIA", "beta": 3.0},
    "MZZ": {"benchmark": "MDY", "beta": 2.0},
    "FAZ": {"benchmark": "XLF", "beta": 3.0},
    "SOXS": {"benchmark": "SOXX", "beta": 3.0},
}


def inverse_etf_tier(symbol: str, leverage: Optional[int] = None) -> str:
    """Return the leverage tier for a known inverse ETF symbol."""
    ticker = str(symbol or "").upper().strip()
    for tier, symbols in INVERSE_ETF_TIERS.items():
        if ticker in symbols:
            return tier
    lev = int(leverage or 1)
    if lev >= 3:
        return "3x"
    if lev == 2:
        return "2x"
    return "1x"


def allowed_inverse_etf_tiers(regime: str, breadth_pct_positive: float) -> set[str]:
    """Select permitted inverse ETF leverage tiers from breadth and regime."""
    regime_key = str(regime or "").lower().strip()
    breadth = float(breadth_pct_positive or 50.0)
    allowed = {"1x"}
    if breadth < 30.0 or regime_key in {"capitulation", "crash"}:
        allowed.add("2x")
    if breadth < 25.0 and regime_key == "crash":
        allowed.add("3x")
    return allowed


class InverseETFScreener:
    """Screens inverse ETFs intraday for selective entries on bad market days."""

    # Gate thresholds
    MIN_AVG_VOLUME = 500_000
    MIN_AUM = 50.0  # millions
    VOLUME_RATIO_THRESHOLD = 1.5
    MOMENTUM_MIN_RETURN = 0.005  # +0.5% in last 30min
    RSI_LOW = 40.0
    RSI_HIGH = 70.0
    RSI_AVOID = 75.0
    VWAP_MAX_DISTANCE = 0.01  # 1%
    VWAP_AVOID_DISTANCE = 0.02  # 2%
    SCORE_ENTRY_THRESHOLD = 70
    FAST_ENTRY_SCORE_THRESHOLD = 40
    FAST_ENTRY_MIN_VOLUME_RATIO = 0.8
    FAST_ENTRY_MIN_RETURN = 0.002  # +0.2% continuation is enough in crash-open mode
    FAST_RSI_EXTREME = 92.0
    FAST_VWAP_CHASE_LIMIT = 0.05  # up to 5% above VWAP allowed in inverse_fast

    def __init__(self, financial_db, data_client=None):
        self.db = financial_db
        self.data_client = data_client

    def screen_universe(
        self,
        regime: str,
        portfolio_holdings: Optional[List[str]] = None,
        entry_mode: str = "",
        minutes_since_open: Optional[int] = None,
        sources_degraded: bool = False,
        breadth_pct_positive: float = 50.0,
        vix_level: float = 0.0,
        planned_hold_days: Optional[float] = None,
    ) -> List[Dict]:
        """
        Full screening pipeline:
        1. Pull active inverse ETFs from DB
        2. Filter by liquidity gate
        3. Fetch intraday bars (Alpaca primary, yfinance fallback)
        4. Score momentum + VWAP/RSI
        5. Return ranked candidates with ENTRY/NEUTRAL/AVOID signals
        """
        regime_lower = str(regime or "neutral").lower()
        fast_mode = str(entry_mode or "").lower() == "inverse_fast"

        # Broadened gate: Allow if explicitly bearish regime, OR
        # if sources are degraded and price action breadth is bearish (< 45%)
        is_bearish_regime = regime_lower in BEARISH_REGIMES
        degraded_bearish_breadth = sources_degraded and breadth_pct_positive < 45.0

        if not fast_mode and not is_bearish_regime and not degraded_bearish_breadth:
            logger.info(
                "[SCREEN] Regime '%s' is not bearish (degraded=%s, breadth=%.1f%%), skipping inverse ETF screen",
                regime,
                sources_degraded,
                breadth_pct_positive,
            )
            return []

        # Gate 1: Pull liquid ETFs from DB, then route by leverage tier.
        candidates = self._gate1_liquidity()
        allowed_tiers = allowed_inverse_etf_tiers(regime_lower, breadth_pct_positive)
        if fast_mode:
            allowed_tiers.update({"2x", "3x"})
        if float(vix_level or 0.0) > 35.0 and (
            planned_hold_days is None or float(planned_hold_days) >= 1.0
        ):
            allowed_tiers.discard("2x")
            allowed_tiers.discard("3x")
            logger.warning(
                "[SCREEN] VIX %.1f blocks new leveraged inverse ETF entries without sub-1-day hold tag",
                float(vix_level or 0.0),
            )
        candidates = [
            etf
            for etf in candidates
            if inverse_etf_tier(etf.get("ticker", ""), etf.get("leverage"))
            in allowed_tiers
        ]
        if not candidates:
            logger.warning("[SCREEN] No inverse ETFs pass liquidity gate")
            return []

        logger.info("[SCREEN] %d inverse ETFs pass liquidity gate", len(candidates))

        # Gates 2+3: Score each candidate with intraday data
        results = []
        for etf in candidates:
            scored = self._score_candidate(
                etf,
                entry_mode=entry_mode,
                minutes_since_open=minutes_since_open,
            )
            if scored:
                results.append(scored)

        # Sort by composite score descending
        results.sort(key=lambda x: x.get("composite_score", 0), reverse=True)

        # Save screen results to DB
        now = datetime.now()
        screen_time = now.strftime("%H:%M")
        screen_date = date.today().isoformat()
        for r in results:
            try:
                self.db.upsert_screen_result(
                    {
                        "ticker": r["ticker"],
                        "screen_date": screen_date,
                        "screen_time": screen_time,
                        "volume_ratio": r.get("volume_ratio", 0.0),
                        "momentum_score": r.get("momentum_score", 0.0),
                        "vwap_distance_pct": r.get("vwap_distance_pct", 0.0),
                        "rsi_14": r.get("rsi_14", 0.0),
                        "spread_bps": r.get("spread_bps", 0.0),
                        "signal": r.get("signal", "NEUTRAL"),
                        "entry_price": r.get("entry_price", 0.0),
                        "notes": r.get("notes", ""),
                    }
                )
            except Exception as e:
                logger.debug(
                    "[SCREEN] Failed to save screen result for %s: %s", r["ticker"], e
                )

        entry_candidates = [r for r in results if r.get("signal") == "ENTRY"]
        logger.info(
            "[SCREEN] Screened %d ETFs: %d ENTRY, %d NEUTRAL, %d AVOID",
            len(results),
            len(entry_candidates),
            sum(1 for r in results if r.get("signal") == "NEUTRAL"),
            sum(1 for r in results if r.get("signal") == "AVOID"),
        )
        return results

    def _gate1_liquidity(self) -> List[Dict]:
        """Gate 1: Filter by average daily volume and AUM."""
        all_etfs = self.db.get_all_inverse_etfs(active_only=True)
        passed = []
        for etf in all_etfs:
            vol = etf.get("avg_daily_volume", 0) or 0
            aum = etf.get("aum_millions", 0.0) or 0.0
            # If volume/AUM not populated yet (fresh seed), still include index ETFs
            # since they're known to be liquid
            if vol >= self.MIN_AVG_VOLUME or aum >= self.MIN_AUM:
                passed.append(etf)
            elif etf.get("category") == "index" and etf.get("leverage", 1) <= 3:
                # Always include canonical index hedges, including leveraged products.
                passed.append(etf)
        return passed

    def _score_candidate(
        self,
        etf: Dict,
        *,
        entry_mode: str = "",
        minutes_since_open: Optional[int] = None,
    ) -> Optional[Dict]:
        """Score a single ETF through Gates 2 and 3. Returns scored dict or None."""
        ticker = etf["ticker"]
        fast_mode = (
            str(entry_mode or "").lower() == "inverse_fast"
            and minutes_since_open is not None
            and 1 <= int(minutes_since_open) <= 60
        )
        try:
            bars = self._fetch_intraday_bars(ticker)
        except Exception as e:
            logger.debug("[SCREEN] Failed to fetch bars for %s: %s", ticker, e)
            return None

        if bars is None or (len(bars) < 10 and not fast_mode):
            return None
        if fast_mode and len(bars) < 1:
            return None

        # Calculate indicators
        volume_ratio = self._calc_volume_ratio(bars)
        momentum_return = self._calc_momentum_return(bars, lookback_minutes=30)
        if fast_mode:
            volume_ratio = max(
                volume_ratio, self._calc_fast_open_volume_ratio(etf, bars)
            )
            if len(bars) < 2:
                first = bars.iloc[-1]
                open_price = float(first.get("open", 0.0) or 0.0)
                close_price = float(first.get("close", 0.0) or 0.0)
                momentum_return = (
                    (close_price - open_price) / open_price if open_price > 0 else 0.0
                )
        rsi = self._calc_rsi(bars, period=14)
        vwap = self._calc_vwap(bars)
        current_price = float(bars["close"].iloc[-1])
        vwap_distance = (current_price - vwap) / vwap if vwap > 0 else 0.0

        # Gate 2: Momentum burst
        liquidity_score = self._score_liquidity(etf, volume_ratio)
        momentum_score = self._score_momentum(volume_ratio, momentum_return)

        # Gate 3: VWAP/RSI timing
        timing_score = self._score_timing(rsi, vwap_distance)

        composite = liquidity_score + momentum_score + timing_score

        signal = self._classify_signal(
            composite=composite,
            rsi=rsi,
            vwap_distance=vwap_distance,
            momentum_return=momentum_return,
            volume_ratio=volume_ratio,
            entry_mode=entry_mode,
            minutes_since_open=minutes_since_open,
        )

        return {
            "ticker": ticker,
            "name": etf.get("name", ""),
            "leverage": etf.get("leverage", 1),
            "underlying": etf.get("underlying", ""),
            "category": etf.get("category", ""),
            "leverage_tier": inverse_etf_tier(ticker, etf.get("leverage")),
            "max_hold_sessions": 2
            if inverse_etf_tier(ticker, etf.get("leverage")) == "3x"
            else (3 if inverse_etf_tier(ticker, etf.get("leverage")) == "2x" else 5),
            "composite_score": round(composite, 1),
            "liquidity_score": round(liquidity_score, 1),
            "momentum_score": round(momentum_score, 1),
            "timing_score": round(timing_score, 1),
            "volume_ratio": round(volume_ratio, 2),
            "rsi_14": round(rsi, 1),
            "vwap_distance_pct": round(vwap_distance * 100, 2),
            "entry_price": round(current_price, 2),
            "signal": signal,
            "notes": (
                f"mom_ret={momentum_return:.3f};"
                f"entry_mode={entry_mode or 'normal'};"
                f"mins_since_open={minutes_since_open}"
            ),
        }

    def _classify_signal(
        self,
        *,
        composite: float,
        rsi: float,
        vwap_distance: float,
        momentum_return: float,
        volume_ratio: float,
        entry_mode: str = "",
        minutes_since_open: Optional[int] = None,
    ) -> str:
        """Classify candidate signal for ordinary vs crash-open inverse fast lane."""
        fast_mode = (
            str(entry_mode or "").lower() == "inverse_fast"
            and minutes_since_open is not None
            and 1 <= int(minutes_since_open) <= 60
        )
        if fast_mode:
            if (
                rsi >= self.FAST_RSI_EXTREME
                or vwap_distance > self.FAST_VWAP_CHASE_LIMIT
            ):
                return "AVOID"
            if (
                composite >= self.FAST_ENTRY_SCORE_THRESHOLD
                and volume_ratio >= self.FAST_ENTRY_MIN_VOLUME_RATIO
                and momentum_return >= self.FAST_ENTRY_MIN_RETURN
            ):
                return "ENTRY"
        if rsi > self.RSI_AVOID or vwap_distance > self.VWAP_AVOID_DISTANCE:
            return "AVOID"
        if composite >= self.SCORE_ENTRY_THRESHOLD:
            return "ENTRY"
        return "NEUTRAL"

    def _fetch_intraday_bars(self, ticker: str) -> Optional[pd.DataFrame]:
        """Fetch intraday bars via existing utility."""
        try:
            from autotrade.utils.intraday_data_provider import get_intraday_bars

            return get_intraday_bars(
                ticker,
                self.data_client,
                minutes_back=240,
                interval="5m",
                min_bars=5,
            )
        except ImportError:
            logger.debug("[SCREEN] intraday_data_provider not available")
            return None

    def _calc_volume_ratio(self, bars: pd.DataFrame) -> float:
        """Current period volume vs. average volume."""
        if "volume" not in bars.columns or len(bars) < 20:
            return 0.0
        avg_vol = bars["volume"].iloc[:-1].mean()
        if avg_vol <= 0:
            return 0.0
        return float(bars["volume"].iloc[-1] / avg_vol)

    def _calc_fast_open_volume_ratio(self, etf: Dict, bars: pd.DataFrame) -> float:
        """Estimate opening 5-minute volume pressure for inverse_fast entries."""
        if "volume" not in bars.columns or bars.empty:
            return 0.0
        avg_daily_volume = float(etf.get("avg_daily_volume", 0.0) or 0.0)
        latest_volume = float(bars["volume"].iloc[-1] or 0.0)
        if avg_daily_volume <= 0:
            return 1.0 if latest_volume > 0 else 0.0
        expected_five_minute_volume = avg_daily_volume / 78.0
        if expected_five_minute_volume <= 0:
            return 0.0
        return latest_volume / expected_five_minute_volume

    def _calc_momentum_return(
        self, bars: pd.DataFrame, lookback_minutes: int = 30
    ) -> float:
        """Price return over lookback period."""
        if "close" not in bars.columns or len(bars) < 2:
            return 0.0
        # Approximate bars for lookback (5m bars -> 6 bars per 30min)
        lookback_bars = max(1, lookback_minutes // 5)
        if len(bars) < lookback_bars + 1:
            lookback_bars = len(bars) - 1
        start_price = float(bars["close"].iloc[-lookback_bars - 1])
        end_price = float(bars["close"].iloc[-1])
        if start_price <= 0:
            return 0.0
        return (end_price - start_price) / start_price

    def _calc_rsi(self, bars: pd.DataFrame, period: int = 14) -> float:
        """Calculate RSI from close prices."""
        if "close" not in bars.columns or len(bars) < period + 1:
            return 50.0  # neutral default
        close = bars["close"].astype(float)
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        last_gain = gain.iloc[-1]
        last_loss = loss.iloc[-1]
        if last_loss == 0:
            return 100.0 if last_gain > 0 else 50.0
        rs = last_gain / last_loss
        return float(100.0 - (100.0 / (1.0 + rs)))

    def _calc_vwap(self, bars: pd.DataFrame) -> float:
        """Calculate VWAP from OHLCV bars."""
        required = {"high", "low", "close", "volume"}
        if not required.issubset(bars.columns) or len(bars) < 1:
            return 0.0
        typical_price = (bars["high"] + bars["low"] + bars["close"]) / 3.0
        cum_vol = bars["volume"].cumsum()
        cum_tp_vol = (typical_price * bars["volume"]).cumsum()
        total_vol = cum_vol.iloc[-1]
        if total_vol <= 0:
            return float(bars["close"].iloc[-1])
        return float(cum_tp_vol.iloc[-1] / total_vol)

    def _score_liquidity(self, etf: Dict, volume_ratio: float) -> float:
        """Score 0-30 based on volume, AUM, and current volume ratio."""
        score = 0.0
        vol = etf.get("avg_daily_volume", 0) or 0
        aum = etf.get("aum_millions", 0.0) or 0.0

        # Volume component (0-15)
        if vol >= 10_000_000:
            score += 15.0
        elif vol >= 5_000_000:
            score += 12.0
        elif vol >= 1_000_000:
            score += 9.0
        elif vol >= self.MIN_AVG_VOLUME:
            score += 6.0

        # AUM component (0-10)
        if aum >= 1000:
            score += 10.0
        elif aum >= 500:
            score += 8.0
        elif aum >= 100:
            score += 5.0
        elif aum >= self.MIN_AUM:
            score += 3.0

        # Current volume ratio bonus (0-5)
        if volume_ratio >= 3.0:
            score += 5.0
        elif volume_ratio >= 2.0:
            score += 3.0
        elif volume_ratio >= 1.5:
            score += 1.0

        return min(score, 30.0)

    def _score_momentum(self, volume_ratio: float, momentum_return: float) -> float:
        """Score 0-40 based on volume ratio and price momentum."""
        score = 0.0

        # Volume ratio component (0-20)
        if volume_ratio >= 3.0:
            score += 20.0
        elif volume_ratio >= 2.0:
            score += 15.0
        elif volume_ratio >= self.VOLUME_RATIO_THRESHOLD:
            score += 10.0
        elif volume_ratio >= 1.0:
            score += 3.0

        # Momentum return component (0-20)
        if momentum_return >= 0.02:  # +2%
            score += 20.0
        elif momentum_return >= 0.01:  # +1%
            score += 15.0
        elif momentum_return >= self.MOMENTUM_MIN_RETURN:  # +0.5%
            score += 10.0
        elif momentum_return >= 0.002:  # +0.2%
            score += 3.0

        return min(score, 40.0)

    def _score_timing(self, rsi: float, vwap_distance: float) -> float:
        """Score 0-30 based on RSI positioning and VWAP distance."""
        score = 0.0

        # RSI component (0-15): sweet spot is 40-70
        if self.RSI_LOW <= rsi <= self.RSI_HIGH:
            # Closer to middle is better for entry
            if 45 <= rsi <= 60:
                score += 15.0
            else:
                score += 10.0
        elif rsi < self.RSI_LOW:
            score += 5.0  # oversold, might bounce further
        else:
            score += 0.0  # overbought

        # VWAP distance component (0-15): near or below VWAP is better
        abs_dist = abs(vwap_distance)
        if abs_dist <= 0.003:  # within 0.3%
            score += 15.0
        elif abs_dist <= self.VWAP_MAX_DISTANCE:  # within 1%
            score += 10.0
        elif vwap_distance < 0:  # below VWAP (pullback)
            score += 8.0
        elif abs_dist <= self.VWAP_AVOID_DISTANCE:  # 1-2% above
            score += 3.0

        return min(score, 30.0)
