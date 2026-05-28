"""
VWAP Mean Reversion Universe Scanner
=====================================
Scans the FULL DownDay universe (~4450 tickers) to find stocks above SMA-100
at the start of the trading day, then monitors them for VWAP mean reversion
setups during market hours.

This expands the day manager's intraday opportunity set far beyond the
~200-stock overnight watchlist.  The scanner is designed to be called once
at the start of each trading day to build the eligible universe, then the
day manager evaluates those tickers for live VWAP setups throughout the day.

Strategy (from VWAP Mean Reversion Guide):
- 64% win rate, 2.06 profit factor
- Best time: 10:00 AM - 3:00 PM ET (especially lunch 11:30-1:30)
- Entry: price >0.5-2% below VWAP, RSI divergence, exhaustion volume
- Exit: VWAP line (target), stop beyond recent extreme
- Max hold: 30-90 minutes
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import duckdb
import pandas as pd
import numpy as np

from config.config_loader import get_config

logger = logging.getLogger("AutoTrade.VWAPUniverseScanner")


@dataclass
class VWAPUniverse:
    """Container for pre-screened VWAP mean reversion universe."""

    tickers: List[str]
    ticker_data: Dict[str, Dict[str, Any]]  # ticker -> {close, sma100, atr_14, volume, ...}
    scan_date: date
    elapsed_sec: float
    total_scanned: int
    passed_sma100: int

    @property
    def size(self) -> int:
        return len(self.tickers)


@dataclass
class VWAPSetup:
    """A detected VWAP mean reversion setup on a specific ticker."""

    ticker: str
    current_price: float
    vwap: float
    deviation_pct: float       # % distance from VWAP (negative = below)
    zscore: float              # standard-deviation normalized distance
    rsi: float
    volume_exhausting: bool
    recent_reversal: bool
    sd_band: int               # which SD band: 1, 2, or 3
    score: float               # composite setup quality score (0-100)
    atr_14: float
    avg_daily_volume: float
    sma100: float
    stop_price: float
    target_price: float        # VWAP line
    risk_reward: float
    reason: str
    extra: Dict[str, Any] = field(default_factory=dict)


class VWAPUniverseScanner:
    """
    Pre-screens full DownDay universe by SMA-100 filter, then provides
    a fast VWAP mean reversion evaluation method for the day manager.

    Lifecycle:
      1. Call ``build_universe()`` once early in the trading day (pre-market
         or first cycle) to compute the SMA-100 eligible set from parquet.
      2. During market hours, call ``evaluate_setups(bars_map)`` to score
         live VWAP setups on the eligible tickers.
    """

    def __init__(
        self,
        parquet_path: Optional[Path] = None,
        parent_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = get_config()
        self.logger = parent_logger or logger

        # Resolve parquet path
        if parquet_path:
            self.parquet_path = Path(parquet_path)
        else:
            data_cfg = self.config.data
            base = Path(data_cfg.downday_root)
            rel = Path(data_cfg.daily_features_parquet)
            self.parquet_path = rel if rel.is_absolute() else base / rel

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet not found: {self.parquet_path}")

        # Config
        eq_cfg = self.config.entry_quality
        self.sma_period: int = int(getattr(eq_cfg, "vwap_universe_sma_period", 100))
        self.min_price: float = float(getattr(eq_cfg, "vwap_universe_min_price", 2.0))
        self.max_price: float = float(getattr(eq_cfg, "vwap_universe_max_price", 200.0))
        self.min_avg_volume: int = int(getattr(eq_cfg, "vwap_universe_min_avg_volume", 500_000))
        self.min_atr_pct: float = float(getattr(eq_cfg, "vwap_universe_min_atr_pct", 1.0))
        self.std_threshold: float = float(getattr(eq_cfg, "vwap_mean_reversion_std_threshold", 1.6))
        self.score_bonus: float = float(getattr(eq_cfg, "vwap_mean_reversion_score_bonus", 16.0))

        self.exclude_symbols: Set[str] = {
            "AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "GOOGL", "META", "SPY", "QQQ",
        }

        # Cached universe (rebuilt once per trading day)
        self._universe: Optional[VWAPUniverse] = None
        self._universe_date: Optional[date] = None

    # ------------------------------------------------------------------
    #  Stage 1: Build universe (once per day, from local parquet)
    # ------------------------------------------------------------------

    def build_universe(self, force: bool = False) -> VWAPUniverse:
        """
        Scan full DownDay parquet for tickers above SMA-100 with adequate
        volume and volatility.  Results are cached for the trading day.

        Returns:
            VWAPUniverse with eligible tickers and their latest daily stats.
        """
        today = date.today()
        if not force and self._universe is not None and self._universe_date == today:
            self.logger.debug(
                "[VWAP UNIVERSE] Using cached universe (%d tickers)", self._universe.size
            )
            return self._universe

        t0 = time.time()
        exclude_list = ", ".join(f"'{s}'" for s in sorted(self.exclude_symbols))

        query = f"""
        WITH base AS (
            SELECT
                ticker,
                Date,
                Close,
                Volume,
                atr_14,
                volume_sma_20,
                RSI_14,
                AVG(Close) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN {self.sma_period - 1} PRECEDING AND CURRENT ROW
                ) AS SMA_100,
                COUNT(*) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN {self.sma_period - 1} PRECEDING AND CURRENT ROW
                ) AS sma_row_count,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY Date DESC) AS rn
            FROM parquet_scan('{self.parquet_path.as_posix()}')
            WHERE Close IS NOT NULL
              AND Volume IS NOT NULL
              AND ticker NOT IN ({exclude_list})
        ),
        latest AS (
            SELECT * FROM base
            WHERE rn = 1
              AND sma_row_count >= {self.sma_period}
        )
        SELECT
            ticker,
            Date       AS last_date,
            Close      AS close,
            SMA_100    AS sma100,
            atr_14,
            Volume     AS volume,
            COALESCE(volume_sma_20, Volume) AS avg_volume,
            RSI_14     AS rsi,
            (atr_14 / NULLIF(Close, 0)) * 100 AS atr_pct,
            ((Close - SMA_100) / NULLIF(SMA_100, 0)) * 100 AS pct_above_sma100
        FROM latest
        WHERE Close BETWEEN {self.min_price} AND {self.max_price}
          AND COALESCE(volume_sma_20, Volume) >= {self.min_avg_volume}
          AND Close > SMA_100                              -- ABOVE SMA-100
          AND (atr_14 / NULLIF(Close, 0)) * 100 >= {self.min_atr_pct}
        ORDER BY pct_above_sma100 ASC
        """

        try:
            con = duckdb.connect(":memory:")
            df = con.execute(query).fetchdf()
            con.close()
        except Exception as e:
            self.logger.error("[VWAP UNIVERSE] DuckDB scan failed: %s", e)
            self._universe = VWAPUniverse(
                tickers=[], ticker_data={}, scan_date=today,
                elapsed_sec=time.time() - t0, total_scanned=0, passed_sma100=0,
            )
            return self._universe

        elapsed = time.time() - t0
        ticker_data: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            t = str(row["ticker"])
            ticker_data[t] = {
                "close": float(row["close"]),
                "sma100": float(row["sma100"]),
                "atr_14": float(row.get("atr_14") or 0),
                "volume": int(row.get("volume") or 0),
                "avg_volume": int(row.get("avg_volume") or 0),
                "rsi": float(row.get("rsi") or 50),
                "atr_pct": float(row.get("atr_pct") or 0),
                "pct_above_sma100": float(row.get("pct_above_sma100") or 0),
            }

        tickers = list(ticker_data.keys())
        self._universe = VWAPUniverse(
            tickers=tickers,
            ticker_data=ticker_data,
            scan_date=today,
            elapsed_sec=elapsed,
            total_scanned=len(df),
            passed_sma100=len(tickers),
        )
        self._universe_date = today
        self.logger.info(
            "[VWAP UNIVERSE] Built universe: %d tickers above SMA-%d "
            "(%.1fs, price $%.0f-$%.0f, min vol %dk, min ATR%% %.1f%%)",
            len(tickers), self.sma_period, elapsed,
            self.min_price, self.max_price,
            self.min_avg_volume / 1000, self.min_atr_pct,
        )
        return self._universe

    def get_universe(self) -> Optional[VWAPUniverse]:
        """Return cached universe or None if not yet built."""
        return self._universe

    def get_eligible_tickers(self) -> List[str]:
        """Return list of ticker symbols in the current SMA-100 universe."""
        if self._universe is None:
            return []
        return list(self._universe.ticker_data.keys())

    def is_in_universe(self, ticker: str) -> bool:
        """Check whether a ticker is in the current VWAP universe."""
        if self._universe is None:
            return False
        return ticker.upper() in self._universe.ticker_data

    # ------------------------------------------------------------------
    #  Stage 2: Evaluate live VWAP setups from intraday bars
    # ------------------------------------------------------------------

    def evaluate_setup(
        self,
        ticker: str,
        bars_df: pd.DataFrame,
        now: Optional[datetime] = None,
    ) -> Optional[VWAPSetup]:
        """
        Evaluate a single ticker's intraday bars for VWAP mean reversion.

        Args:
            ticker: Stock symbol (must be in universe).
            bars_df: Intraday 1-min bars with columns: open, high, low, close, volume.
            now: Current time for window checks (default: datetime.now()).

        Returns:
            VWAPSetup if a valid setup is detected, else None.
        """
        if bars_df is None or len(bars_df) < 20:
            return None

        required_cols = {"close", "volume"}
        if not required_cols.issubset(set(c.lower() for c in bars_df.columns)):
            return None

        # Normalize column names to lowercase
        bars_df = bars_df.copy()
        bars_df.columns = [c.lower() for c in bars_df.columns]

        close = bars_df["close"]
        raw_volume = bars_df["volume"].fillna(0).clip(lower=0)
        volume = raw_volume.replace(0, 1)

        # Compute intraday VWAP
        vwap = (close * volume).cumsum() / volume.cumsum()
        if vwap.empty:
            return None

        last_close = float(close.iloc[-1])
        last_vwap = float(vwap.iloc[-1])
        if last_vwap <= 0 or last_close <= 0:
            return None

        # Distance from VWAP
        deviation_pct = ((last_close - last_vwap) / last_vwap) * 100.0

        # Standard deviation bands
        dist_series = ((close - vwap) / vwap.replace(0, 1)) * 100.0
        lookback = min(60, len(dist_series))
        lookback_std = float(dist_series.tail(lookback).std()) if lookback > 5 else 0.5
        std_floor = 0.05
        effective_std = max(std_floor, abs(lookback_std))
        zscore = deviation_pct / effective_std

        # Determine SD band
        abs_z = abs(zscore)
        if abs_z >= 3.0:
            sd_band = 3
        elif abs_z >= 2.0:
            sd_band = 2
        elif abs_z >= 1.0:
            sd_band = 1
        else:
            sd_band = 0

        # RSI (14-period on intraday bars)
        rsi = self._compute_rsi(close, window=14)

        # Volume exhaustion: recent 5 bars avg vs prior 5 bars
        recent_vol = float(raw_volume.tail(5).mean()) if len(raw_volume) >= 5 else 0
        prior_vol = float(raw_volume.iloc[-10:-5].mean()) if len(raw_volume) >= 10 else recent_vol
        volume_exhausting = bool(recent_vol <= prior_vol * 0.95) if prior_vol > 0 else False

        # Recent reversal: price turning back toward VWAP
        recent_reversal = False
        if len(close) >= 4:
            # For long (price below VWAP): price starting to rise
            if deviation_pct < 0:
                recent_reversal = bool(close.iloc[-1] > close.iloc[-3])
            else:
                recent_reversal = bool(close.iloc[-1] < close.iloc[-3])

        # --- LONG SETUP CRITERIA (price below VWAP) ---
        # We focus on long setups only (consistent with AutoTrade's long-only approach)
        if deviation_pct >= -0.3:
            # Price not sufficiently below VWAP
            return None

        # Must be at least 1 SD below VWAP
        if zscore > -abs(self.std_threshold):
            return None

        # Need either RSI oversold OR recent reversal + volume exhaustion
        rsi_ok = rsi <= 45.0
        reversal_ok = recent_reversal and volume_exhausting
        if not (rsi_ok or reversal_ok):
            return None

        # --- Score the setup (0-100) ---
        score = 30.0  # baseline for passing all filters

        # Z-score depth bonus (deeper = better mean reversion)
        score += min(20.0, abs(zscore) * 5.0)

        # SD band bonus
        score += sd_band * 5.0

        # RSI bonus (more oversold = better)
        if rsi <= 30:
            score += 15.0
        elif rsi <= 35:
            score += 10.0
        elif rsi <= 40:
            score += 5.0

        # Volume exhaustion bonus
        if volume_exhausting:
            score += 8.0

        # Recent reversal bonus
        if recent_reversal:
            score += 7.0

        # Clamp
        score = min(100.0, max(0.0, score))

        # --- Stop and target ---
        universe_data = {}
        if self._universe and ticker.upper() in self._universe.ticker_data:
            universe_data = self._universe.ticker_data[ticker.upper()]

        atr_14 = float(universe_data.get("atr_14", last_close * 0.02))
        if atr_14 <= 0:
            atr_14 = last_close * 0.02
        avg_daily_volume = float(universe_data.get("avg_volume", 0))
        sma100 = float(universe_data.get("sma100", 0))

        # Stop: beyond recent low or 0.3% below entry (guide recommendation)
        recent_low = float(close.tail(20).min()) if len(close) >= 20 else last_close * 0.995
        stop_price = min(recent_low * 0.998, last_close * 0.997)

        # Target: VWAP line (primary), or if VWAP is very close just take the reversion
        target_price = last_vwap

        # Risk/reward
        risk = max(0.001, last_close - stop_price)
        reward = max(0.001, target_price - last_close)
        risk_reward = reward / risk

        reason_parts = [
            f"VWAP MR z={zscore:.2f}",
            f"dev={deviation_pct:+.2f}%",
            f"rsi={rsi:.1f}",
            f"sd_band={sd_band}",
        ]
        if volume_exhausting:
            reason_parts.append("vol_exhaust")
        if recent_reversal:
            reason_parts.append("reversal")

        return VWAPSetup(
            ticker=ticker.upper(),
            current_price=last_close,
            vwap=last_vwap,
            deviation_pct=deviation_pct,
            zscore=zscore,
            rsi=rsi,
            volume_exhausting=volume_exhausting,
            recent_reversal=recent_reversal,
            sd_band=sd_band,
            score=score,
            atr_14=atr_14,
            avg_daily_volume=avg_daily_volume,
            sma100=sma100,
            stop_price=stop_price,
            target_price=target_price,
            risk_reward=risk_reward,
            reason=", ".join(reason_parts),
        )

    def evaluate_setups_batch(
        self,
        bars_map: Dict[str, pd.DataFrame],
        now: Optional[datetime] = None,
        min_score: float = 40.0,
    ) -> List[VWAPSetup]:
        """
        Evaluate multiple tickers' intraday bars for VWAP setups.

        Args:
            bars_map: Dict of ticker -> intraday bars DataFrame.
            now: Current time.
            min_score: Minimum score to include in results.

        Returns:
            List of VWAPSetup objects sorted by score descending.
        """
        setups: List[VWAPSetup] = []
        for ticker, bars_df in bars_map.items():
            if not self.is_in_universe(ticker):
                continue
            try:
                setup = self.evaluate_setup(ticker, bars_df, now=now)
                if setup is not None and setup.score >= min_score:
                    setups.append(setup)
            except Exception as e:
                self.logger.debug(
                    "[VWAP SCAN] Error evaluating %s: %s", ticker, e
                )
        setups.sort(key=lambda s: s.score, reverse=True)
        return setups

    def setup_to_signal(self, setup: VWAPSetup) -> Dict[str, Any]:
        """
        Convert a VWAPSetup to a signal dict compatible with the day manager's
        candidate pipeline (find_replacement_candidates / execute_entry).
        """
        # Phase 3: Adaptive VWAP Scanning - Dynamic Sizing
        # Base multiplier from config (default 1.1x)
        base_size_mult = float(
            getattr(self.config.entry_quality, "vwap_mean_reversion_size_multiplier", 1.10)
        )
        
        # Add bonus for deeper discounts (oversize at discount)
        # abs(zscore) > 1.6 is the baseline. Every 1.0 increase in z-score adds 0.15 to the multiplier.
        z_bonus = max(0.0, (abs(setup.zscore) - 1.6) * 0.15)
        # Cap dynamic bonus at 0.40 (total max ~1.5x)
        dynamic_size_mult = base_size_mult + min(0.40, z_bonus)

        return {
            "ticker": setup.ticker,
            "action": "buy_open",
            "recommendation": "buy_open",
            "score": setup.score,
            "confidence": setup.score,
            "realtime_score": setup.score + self.score_bonus,
            "entry_price": setup.current_price,
            "stop_loss": setup.stop_price,
            "target": setup.target_price,
            "atr_14": setup.atr_14,
            "atr": setup.atr_14,
            "volume": setup.avg_daily_volume,
            "rsi": setup.rsi,
            "strategy_profile": "vwap_mean_reversion",
            "strategy_active": True,
            "strategy_score_delta": self.score_bonus + min(10.0, abs(setup.zscore) * 3.0),
            "strategy_size_multiplier": float(dynamic_size_mult),
            "strategy_window_required": True,
            "strategy_window_active": True,
            "strategy_momentum_penalty_multiplier": float(
                getattr(
                    self.config.entry_quality,
                    "vwap_mean_reversion_momentum_penalty_multiplier", 0.35,
                )
            ),
            "strategy_risk_budget_pct": 25.0,
            "signal_family": "vwap_mean_reversion",
            "alpha_family": "vwap_mean_reversion",
            "entry_source": "vwap_universe_scanner",
            "vwap_zscore": setup.zscore,
            "vwap_deviation_pct": setup.deviation_pct,
            "vwap_sd_band": setup.sd_band,
            "vwap_rsi": setup.rsi,
            "vwap_volume_exhausting": setup.volume_exhausting,
            "vwap_recent_reversal": setup.recent_reversal,
            "vwap_risk_reward": setup.risk_reward,
            "vwap_target_price": setup.target_price,
            "vwap_stop_price": setup.stop_price,
            "reason": f"[VWAP Universe] {setup.reason}",
            "catalyst_note": f"VWAP mean reversion: {setup.deviation_pct:+.2f}% below VWAP, z={setup.zscore:.2f}",
        }

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(close_series: pd.Series, window: int = 14) -> float:
        """Compute RSI from a close price series."""
        if close_series is None or len(close_series) < max(5, window + 1):
            return 50.0
        delta = close_series.diff().dropna()
        if delta.empty:
            return 50.0
        gains = delta.clip(lower=0.0)
        losses = (-delta.clip(upper=0.0)).clip(lower=0.0)
        avg_gain = gains.rolling(window=window, min_periods=window).mean().iloc[-1]
        avg_loss = losses.rolling(window=window, min_periods=window).mean().iloc[-1]
        if not pd.notna(avg_gain) or not pd.notna(avg_loss):
            return 50.0
        if float(avg_loss) <= 1e-9:
            return 100.0
        rs = float(avg_gain) / float(avg_loss)
        return max(0.0, min(100.0, 100.0 - (100.0 / (1.0 + rs))))
