"""
Screener V2 - Multi-Factor Technical Screener
==============================================

Uses DownDay local data to build a nightly candidate list without external API calls.

Core goals:
- Emphasize momentum + pullback + volume quality
- Keep stops/targets ATR-first
- Return schema compatible with existing entry_candidates/buy_signals
"""

from __future__ import annotations

import copy
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config_loader import get_config, ScreenerV2Config
from autotrade.feature_engineering.adapters import get_screener_v2_adapter
from autotrade.signals.strategy_pool import load_validated_strategies
from autotrade.signals.alpha_volume_divergence import score_divergence
from autotrade.signals.support_resistance import estimate_sr_levels
from autotrade.signals.universe_filters import (
    DEFAULT_MAX_MARKET_CAP,
    DEFAULT_MIN_MARKET_CAP,
    HARD_BLOCK_MEGA_CAP,
)
from autotrade.utils.data_sync import get_core_market_data_readiness
from autotrade.utils.security_metadata import load_security_metadata

logger = logging.getLogger(__name__)
DEFAULT_SECTOR_ETF_MAP: Dict[str, str] = {
    "basic_materials": "XLB",
    "communication_services": "XLC",
    "communications": "XLC",
    "consumer_cyclical": "XLY",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "financial": "XLF",
    "financials": "XLF",
    "health_care": "XLV",
    "healthcare": "XLV",
    "industrials": "XLI",
    "real_estate": "XLRE",
    "technology": "XLK",
    "utilities": "XLU",
}
# H10 (2026-05-21): per-process dedup of stale-core-data WARNING. Keyed on
# (primary_date, expected_date) so a genuine date change re-arms the alarm.
_SEEN_STALE_KEYS: set = set()

_LAST_SCREEN_RUN: Dict[str, Any] = {
    "status": "idle",
    "reason": None,
    "blocking_reasons": [],
    "result_count": None,
}


def _set_last_screen_run(diagnostics: Dict[str, Any]) -> None:
    global _LAST_SCREEN_RUN
    _LAST_SCREEN_RUN = copy.deepcopy(diagnostics or {})


def get_last_screen_run_diagnostics() -> Dict[str, Any]:
    return copy.deepcopy(_LAST_SCREEN_RUN)


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """Merge override dict into base dict (recursive)."""
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ScreenerResult:
    """Structured result for a single screener row (for type clarity)."""

    data: Dict


@dataclass
class PreparedHistorySession:
    """Reusable enriched history frame for one screener session."""

    data: pd.DataFrame
    source_key: Tuple[Any, ...]


@dataclass
class PreparedTickerContext:
    """Reusable per-ticker context derived from prepared history."""

    latest: pd.DataFrame
    sr_df: pd.DataFrame
    divergence_df: pd.DataFrame
    source_key: Tuple[Any, ...]


class ScreenerV2:
    """Multi-factor screener using local price data.

    Supports two scoring modes:
    - 'complex': Multi-factor weighted scoring with technical indicators
    - 'momentum_pullback': Fast momentum + pullback score
    """

    _PRICE_DATA_CACHE: Dict[Tuple[str, int, int], pd.DataFrame] = {}
    _PRICE_DATA_CACHE_KEY: Optional[Tuple[str, int, int]] = None
    _PREPARED_HISTORY_CACHE: Dict[Tuple[Any, ...], PreparedHistorySession] = {}
    _PREPARED_HISTORY_CACHE_KEY: Optional[Tuple[Any, ...]] = None
    _PREPARED_TICKER_CONTEXT_CACHE: Dict[Tuple[Any, ...], PreparedTickerContext] = {}
    _PREPARED_TICKER_CONTEXT_CACHE_KEY: Optional[Tuple[Any, ...]] = None
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        config_override: Optional[Dict] = None,
        parent_logger: Optional[logging.Logger] = None,
        scoring_mode: Optional[str] = None,
    ):
        root_cfg = get_config()
        base_cfg = root_cfg.screener_v2
        if config_override:
            merged = _deep_merge(base_cfg.model_dump(), config_override)
            base_cfg = ScreenerV2Config(**merged)

        self.config = base_cfg
        self.logger = parent_logger or logging.getLogger("screener_v2")
        self.data_cfg = root_cfg.data
        self.strategy_cfg = root_cfg.strategy
        self.feature_cfg = root_cfg.feature_engineering
        mode = (
            str(scoring_mode or self.config.default_scoring_mode or "momentum_pullback")
            .strip()
            .lower()
        )
        if mode == "simple_sr":
            self.logger.warning(
                "[ScreenerV2] scoring_mode='simple_sr' is deprecated; using 'momentum_pullback'"
            )
            mode = "momentum_pullback"
        self.scoring_mode = mode

        self.daily_features_path = self._resolve_daily_features_path()
        self.core_data_readiness = None
        self.last_run_diagnostics: Dict[str, Any] = {
            "status": "idle",
            "reason": None,
            "blocking_reasons": [],
            "result_count": None,
        }
        self._feature_adapter = None

        if self.feature_cfg.enabled:
            try:
                self._feature_adapter = get_screener_v2_adapter(
                    config=self.feature_cfg.model_dump()
                )
            except Exception as e:
                self.logger.warning(
                    "[ScreenerV2] Shared feature adapter unavailable, using legacy path: %s",
                    e,
                )

    def _resolve_daily_features_path(self) -> Path:
        path = Path(self.data_cfg.daily_features_parquet)
        if not path.is_absolute():
            path = Path(self.data_cfg.downday_root) / path
        return path

    def _load_security_metadata(self) -> pd.DataFrame:
        return load_security_metadata()

    def _get_core_data_readiness(self, force_refresh: bool = False) -> Dict[str, Any]:
        if self.core_data_readiness is None or force_refresh:
            try:
                self.core_data_readiness = get_core_market_data_readiness(
                    force_refresh=force_refresh
                )
            except TypeError:
                self.core_data_readiness = get_core_market_data_readiness()
        return self.core_data_readiness

    def _prepared_history_signature(self) -> Tuple[Any, ...]:
        return (
            str(self.scoring_mode),
            int(getattr(self.config, "history_days", 0) or 0),
            int(getattr(self.config, "volume_lookback_days", 0) or 0),
            float(getattr(self.config, "sma5_curl_scale", 0.0) or 0.0),
            float(getattr(self.config, "sma20_trend_scale", 0.0) or 0.0),
            float(getattr(self.config, "momentum_scale", 0.0) or 0.0),
            float(getattr(self.config, "macd_hist_scale", 0.0) or 0.0),
            float(getattr(self.config, "min_adx_for_trend_bonus", 0.0) or 0.0),
            bool(self.feature_cfg.enabled),
        )

    @staticmethod
    def _symbols_cache_key(symbols: Optional[List[str]]) -> Optional[Tuple[str, ...]]:
        if not symbols:
            return None
        normalized = sorted(
            {str(symbol).strip().upper() for symbol in symbols if symbol}
        )
        return tuple(normalized)

    def _get_prepared_history_session(
        self, symbols: Optional[List[str]] = None
    ) -> PreparedHistorySession:
        symbol_key = self._symbols_cache_key(symbols)
        try:
            path_key = str(self.daily_features_path.resolve())
        except Exception:
            path_key = str(self.daily_features_path)
        try:
            mtime_ns = int(self.daily_features_path.stat().st_mtime_ns)
        except Exception:
            mtime_ns = 0
        cache_key = (
            path_key,
            mtime_ns,
            self._prepared_history_signature(),
            symbol_key,
        )
        with self._CACHE_LOCK:
            cached = self._PREPARED_HISTORY_CACHE.get(cache_key)
        if cached is not None:
            self.logger.debug(
                "[ScreenerV2] Reusing prepared history cache (%s rows, %s tickers)",
                f"{len(cached.data):,}",
                f"{cached.data['ticker'].nunique():,}"
                if not cached.data.empty and "ticker" in cached.data.columns
                else "0",
            )
            return cached

        self.logger.info("[ScreenerV2] Preparing local price data...")
        price_df = self._load_price_data(symbols=symbols)
        if price_df.empty:
            return PreparedHistorySession(data=price_df, source_key=cache_key)

        self.logger.info(
            "[ScreenerV2] Loaded %s rows for %s tickers",
            f"{len(price_df):,}",
            f"{price_df['ticker'].nunique():,}",
        )
        prepared_df = self._compute_indicators(price_df)
        prepared_df = self._enrich_with_shared_features(prepared_df)
        session = PreparedHistorySession(data=prepared_df, source_key=cache_key)

        with self._CACHE_LOCK:
            prev_key = self._PREPARED_HISTORY_CACHE_KEY
            self._PREPARED_HISTORY_CACHE[cache_key] = session
            self._PREPARED_HISTORY_CACHE_KEY = cache_key
            if prev_key is not None and prev_key != cache_key:
                self._PREPARED_HISTORY_CACHE.pop(prev_key, None)
        return session

    def _compute_divergence_latest(self, history_df: pd.DataFrame) -> pd.DataFrame:
        """Build latest divergence context once per ticker from prepared history."""
        required = {"ticker", "date", "close", "volume"}
        if history_df.empty or not required.issubset(history_df.columns):
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "divergence_score",
                    "divergence_zscore",
                    "divergence_signal",
                    "volume_divergence_score",
                ]
            )

        div_df = score_divergence(history_df[list(required)].copy())
        if div_df.empty:
            return pd.DataFrame(
                columns=[
                    "ticker",
                    "divergence_score",
                    "divergence_zscore",
                    "divergence_signal",
                    "volume_divergence_score",
                ]
            )

        div_latest_idx = div_df.groupby("ticker")["date"].idxmax()
        div_latest = div_df.loc[div_latest_idx][
            ["ticker", "divergence_score", "divergence_zscore", "divergence_signal"]
        ].copy()
        div_latest["divergence_score"] = div_latest["divergence_score"].fillna(50.0)
        div_latest["divergence_zscore"] = div_latest["divergence_zscore"].fillna(0.0)
        div_latest["divergence_signal"] = div_latest["divergence_signal"].fillna("none")
        z = pd.to_numeric(div_latest["divergence_zscore"], errors="coerce").fillna(0.0)
        bullish_score = (80.0 + (z - 1.0).clip(lower=0.0, upper=2.0) * 10.0).clip(
            80.0, 100.0
        )
        div_latest["volume_divergence_score"] = np.where(
            z > 1.0,
            bullish_score,
            np.where(z < -1.0, 20.0, 50.0),
        )

        bullish_count = (div_latest["divergence_signal"] == "bullish").sum()
        self.logger.info(
            "[ScreenerV2] Divergence enrichment: %d bullish, %d bearish out of %d",
            bullish_count,
            (div_latest["divergence_signal"] == "bearish").sum(),
            len(div_latest),
        )
        return div_latest

    def _get_prepared_ticker_context(
        self, history_session: PreparedHistorySession
    ) -> PreparedTickerContext:
        cache_key = history_session.source_key
        with self._CACHE_LOCK:
            cached = self._PREPARED_TICKER_CONTEXT_CACHE.get(cache_key)
        if cached is not None:
            self.logger.debug(
                "[ScreenerV2] Reusing prepared ticker context (%s symbols)",
                f"{len(cached.latest):,}",
            )
            return cached

        history_df = history_session.data
        latest = self._latest_snapshot(history_df)
        sr_df = self._load_sr_context(history_df=history_df)
        divergence_df = self._compute_divergence_latest(history_df)
        context = PreparedTickerContext(
            latest=latest,
            sr_df=sr_df,
            divergence_df=divergence_df,
            source_key=cache_key,
        )

        with self._CACHE_LOCK:
            prev_key = self._PREPARED_TICKER_CONTEXT_CACHE_KEY
            self._PREPARED_TICKER_CONTEXT_CACHE[cache_key] = context
            self._PREPARED_TICKER_CONTEXT_CACHE_KEY = cache_key
            if prev_key is not None and prev_key != cache_key:
                self._PREPARED_TICKER_CONTEXT_CACHE.pop(prev_key, None)
        return context

    def _load_market_intelligence(self) -> Optional[Dict]:
        """Load today's YouTube market intelligence report."""
        import json
        from datetime import datetime, timedelta

        # Use same logic as DayManager/Agent for consistency
        rag_dir = Path("data/youtube/rag/daily_reports")

        for offset in [0, 1]:
            date_str = (datetime.now() - timedelta(days=offset)).strftime("%Y-%m-%d")
            report_path = rag_dir / f"{date_str}_consolidated.json"
            if report_path.exists():
                try:
                    with open(report_path) as f:
                        return json.load(f)
                except Exception:
                    continue
        return None

    def _apply_market_intelligence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply sector and regime bonuses from YouTube intelligence."""
        if df.empty:
            return df

        report = self._load_market_intelligence()
        if not report:
            return df

        signals = report.get("trading_signals", {})
        sector_bias = signals.get("sector_bias", [])

        degraded = False
        if not sector_bias and "raw_report" in report:
            # P11C Fallback: Extract from text blob
            self.logger.warning(
                "[YOUTUBE-SCREENER] JSON sector_bias missing. Entering Degraded Intelligence Mode."
            )
            degraded = True
            raw_text = str(report.get("raw_report", ""))

            # Simple keyword-based bias extraction (Surgical Regex fallback)
            sector_bias = []
            common_sectors = [
                "Semiconductors",
                "Software",
                "Technology",
                "Financials",
                "Energy",
                "Healthcare",
                "Autos",
            ]
            for sector in common_sectors:
                # Basic overweight regex
                ovw_match = re.search(
                    rf"(Overweight|Strong|Bullish|Favor).{{0,100}}{sector}",
                    raw_text,
                    re.I,
                ) or re.search(
                    rf"{sector}.{{0,100}}(Strong|Bullish|Momentum|Overweight)",
                    raw_text,
                    re.I,
                )
                if ovw_match:
                    sector_bias.append({"sector": sector, "bias": "OVERWEIGHT"})
                else:
                    # Basic avoid regex
                    avd_match = re.search(
                        rf"(Avoid|Underweight|Bearish|Weak).{{0,100}}{sector}",
                        raw_text,
                        re.I,
                    ) or re.search(
                        rf"{sector}.{{0,100}}(Weak|Avoid|Underweight)", raw_text, re.I
                    )
                    if avd_match:
                        sector_bias.append({"sector": sector, "bias": "AVOID"})

        avoid_sectors = [
            s["sector"].lower() for s in sector_bias if s.get("bias") == "AVOID"
        ]
        favor_sectors = [
            s["sector"].lower() for s in sector_bias if s.get("bias") == "OVERWEIGHT"
        ]

        if degraded:
            self.logger.info(
                f"[YOUTUBE-SCREENER] Degraded bias extracted: favor={favor_sectors}, avoid={avoid_sectors}"
            )
        else:
            self.logger.info(
                f"[YOUTUBE-SCREENER] Applying bias: favor={favor_sectors}, avoid={avoid_sectors}"
            )

        # We need sector info in the DataFrame to apply this.
        if "sector" in df.columns:
            # Sector penalty/bonus
            df["composite_score"] = df.apply(
                lambda x: (
                    x["composite_score"] - 15.0
                    if str(x.get("sector", "")).lower() in avoid_sectors
                    else x["composite_score"] + 8.0
                    if str(x.get("sector", "")).lower() in favor_sectors
                    else x["composite_score"]
                ),
                axis=1,
            )
            if degraded:
                # Add indicator that ranking was informed by degraded mode fallback
                df["degraded_intelligence"] = True

        return df

    def _load_price_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        self.core_data_readiness = self._get_core_data_readiness()
        if not bool(self.core_data_readiness.get("is_fresh", False)):
            reasons = ", ".join(self.core_data_readiness.get("blocking_reasons", []))
            self.last_run_diagnostics = {
                "status": "blocked",
                "reason": "core_data_not_ready",
                "blocking_reasons": list(
                    self.core_data_readiness.get("blocking_reasons", []) or []
                ),
                "core_data_readiness": copy.deepcopy(self.core_data_readiness),
                "result_count": 0,
            }
            # H10 (2026-05-21): log once at WARNING per distinct
            # (primary_date, expected_date) tuple, then DEBUG for repeats.
            # The screener is called many times per cycle and each call
            # produced an identical WARNING (12+ per session today), which
            # drowns the genuinely-actionable first occurrence in noise.
            stale_key = (
                self.core_data_readiness.get("primary_date"),
                self.core_data_readiness.get("expected_date"),
            )
            if stale_key not in _SEEN_STALE_KEYS:
                _SEEN_STALE_KEYS.add(stale_key)
                self.logger.warning(
                    "[ScreenerV2] Blocking screening on stale core data "
                    "(asof=%s expected=%s reasons=%s)",
                    stale_key[0],
                    stale_key[1],
                    reasons or "unknown",
                )
            else:
                self.logger.debug(
                    "[ScreenerV2] Blocking screening on stale core data "
                    "(asof=%s expected=%s reasons=%s) [dedup]",
                    stale_key[0],
                    stale_key[1],
                    reasons or "unknown",
                )
            return pd.DataFrame()
        if not self.daily_features_path.exists():
            self.logger.warning(
                f"Daily features file not found: {self.daily_features_path}"
            )
            return pd.DataFrame()

        cols = [
            "ticker",
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Adj Close",
            "Volume",
            "SMA_20",
            "EMA_20",
            "RSI_14",
            "MACD",
            "MACD_signal",
            "MACD_hist",
            "BB_mid",
            "BB_upper",
            "BB_lower",
            "Stoch_%K",
            "Stoch_%D",
            "atr_14",
            "ROC_5",
            "ROC_10",
        ]

        cache_key = (
            str(self.daily_features_path.resolve()),
            int(self.config.history_days),
            int(self.daily_features_path.stat().st_mtime_ns),
        )
        with self._CACHE_LOCK:
            cached = self._PRICE_DATA_CACHE.get(cache_key)
        if cached is not None:
            df = cached
            self.logger.info(
                "[ScreenerV2] Reusing cached local price data (%s rows, %s tickers)",
                f"{len(df):,}",
                f"{df['ticker'].nunique():,}"
                if not df.empty and "ticker" in df.columns
                else "0",
            )
        else:
            try:
                import duckdb

                query = f"""
                WITH base AS (
                    SELECT
                        ticker,
                        CAST(Date AS DATE) AS date,
                        "Open" AS open,
                        "High" AS high,
                        "Low" AS low,
                        "Close" AS close,
                        "Adj Close" AS adj_close,
                        Volume AS volume,
                        "SMA_20" AS sma20,
                        "EMA_20" AS ema20,
                        "RSI_14" AS rsi_14,
                        "MACD" AS macd,
                        "MACD_signal" AS macd_signal,
                        "MACD_hist" AS macd_hist,
                        "BB_mid" AS bb_mid,
                        "BB_upper" AS bb_upper,
                        "BB_lower" AS bb_lower,
                        "Stoch_%K" AS stoch_k,
                        "Stoch_%D" AS stoch_d,
                        "atr_14" AS atr_14,
                        "ROC_5" AS roc_5,
                        "ROC_10" AS roc_10
                    FROM read_parquet('{self.daily_features_path.as_posix()}')
                ), max_date AS (
                    SELECT max(date) AS max_date FROM base
                )
                SELECT *
                FROM base
                WHERE date >= (SELECT max_date - INTERVAL '{self.config.history_days} days' FROM max_date)
                """

                conn = duckdb.connect(database=":memory:")
                df = conn.execute(query).df()
                conn.close()
            except Exception as e:
                self.logger.warning(f"DuckDB load failed ({e}); falling back to pandas")
                df = pd.read_parquet(self.daily_features_path, columns=cols)
                df = df.rename(
                    columns={
                        "Date": "date",
                        "Open": "open",
                        "High": "high",
                        "Low": "low",
                        "Close": "close",
                        "Adj Close": "adj_close",
                        "Volume": "volume",
                        "SMA_20": "sma20",
                        "EMA_20": "ema20",
                        "RSI_14": "rsi_14",
                        "MACD": "macd",
                        "MACD_signal": "macd_signal",
                        "MACD_hist": "macd_hist",
                        "BB_mid": "bb_mid",
                        "BB_upper": "bb_upper",
                        "BB_lower": "bb_lower",
                        "Stoch_%K": "stoch_k",
                        "Stoch_%D": "stoch_d",
                        "atr_14": "atr_14",
                        "ROC_5": "roc_5",
                        "ROC_10": "roc_10",
                    }
                )
                df["date"] = pd.to_datetime(df["date"])
                max_date = df["date"].max()
                cutoff = max_date - pd.Timedelta(days=self.config.history_days)
                df = df[df["date"] >= cutoff]

            with self._CACHE_LOCK:
                # Keep only the latest snapshot in-memory to avoid unbounded growth.
                prev_key = self._PRICE_DATA_CACHE_KEY
                self._PRICE_DATA_CACHE[cache_key] = df
                self._PRICE_DATA_CACHE_KEY = cache_key
                if prev_key is not None and prev_key != cache_key:
                    self._PRICE_DATA_CACHE.pop(prev_key, None)

        if df.empty:
            return df

        if "adj_close" in df.columns and df["adj_close"].notna().any():
            # Avoid pandas' broken Series.fillna(DataFrame-symbol) path in this env.
            adj_close = pd.to_numeric(df["adj_close"], errors="coerce")
            close = pd.to_numeric(df["close"], errors="coerce")
            df["close"] = close.where(adj_close.isna(), adj_close)

        df["ticker"] = df["ticker"].astype(str).str.upper()

        metadata_df = self._load_security_metadata()
        if not metadata_df.empty:
            df = df.merge(metadata_df, on="ticker", how="left")

        if symbols:
            symbol_set = {s.upper() for s in symbols}
            df = df[df["ticker"].isin(symbol_set)]

        return df

    def _load_sr_context(
        self,
        symbols: Optional[List[str]] = None,
        history_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build local S/R + regime context from price history.

        This replaces the retired external SRbatch provider and keeps the
        screener fully technical-data-only.
        """
        try:
            if history_df is not None and not history_df.empty:
                df = history_df.copy()
            else:
                df = self._load_price_data(symbols=symbols)
            if df.empty:
                return pd.DataFrame()

            if symbols:
                symset = {str(s).upper() for s in symbols if s}
                if symset:
                    df = df[df["ticker"].astype(str).str.upper().isin(symset)].copy()
            if df.empty:
                return pd.DataFrame()

            rows: List[Dict[str, float]] = []
            grouped = df.sort_values(["ticker", "date"]).groupby("ticker", sort=False)
            for ticker, g in grouped:
                bars = g.tail(max(90, int(self.config.history_days))).copy()
                if len(bars) < 25:
                    continue

                sr = estimate_sr_levels(
                    bars,
                    lookback_bars=min(120, len(bars)),
                    pivot_window=3,
                    min_touches=2,
                    cluster_atr_mult=0.35,
                )
                if not isinstance(sr, dict):
                    continue

                latest = bars.iloc[-1]
                close = float(latest.get("close", 0) or 0)
                if close <= 0:
                    continue
                atr = float(latest.get("atr_14", 0) or 0)
                if atr <= 0:
                    atr = max(close * 0.02, 0.01)

                s1 = float(
                    sr.get("s1_price", max(0.01, close - atr * 1.2))
                    or max(0.01, close - atr * 1.2)
                )
                r1 = float(sr.get("r1_price", close + atr * 1.6) or (close + atr * 1.6))
                risk = max(close - s1, atr * 0.3, 1e-6)
                reward = max(r1 - close, 0.0)
                rr = float(reward / risk if risk > 0 else 0.0)

                sma20_series = (
                    pd.to_numeric(bars.get("sma20"), errors="coerce")
                    if "sma20" in bars.columns
                    else pd.Series(dtype=float)
                )
                if sma20_series.empty or sma20_series.notna().sum() < 5:
                    sma20_series = (
                        pd.to_numeric(bars["close"], errors="coerce")
                        .rolling(20, min_periods=5)
                        .mean()
                    )
                sma20_now = float(sma20_series.iloc[-1] or close)
                sma20_prev = (
                    float(sma20_series.iloc[-5] or sma20_now)
                    if len(sma20_series) >= 5
                    else sma20_now
                )
                slope_up = sma20_now >= sma20_prev

                if close > sma20_now and slope_up:
                    regime = "BULLISH"
                elif close < sma20_now and not slope_up:
                    regime = "BEARISH"
                else:
                    regime = "NEUTRAL"

                if regime == "BULLISH" and rr >= float(self.config.sr_rr_effective_min):
                    action_plan = "bullish trend continuation setup"
                elif regime == "BEARISH":
                    action_plan = "defensive risk-off setup"
                else:
                    action_plan = "neutral wait-for-confirmation setup"

                defensive_flags: List[str] = []
                if float(sr.get("distance_to_s1_pct", 0) or 0) > 8.0:
                    defensive_flags.append("far_from_support")
                if rr < 1.0:
                    defensive_flags.append("low_rr")

                rows.append(
                    {
                        "ticker": str(ticker).upper(),
                        "sr_s1_price": s1,
                        "sr_s1_strength": float(sr.get("s1_strength", 40.0) or 40.0),
                        "sr_r1_price": r1,
                        "sr_r1_strength": float(sr.get("r1_strength", 40.0) or 40.0),
                        "sr_support_dist_atr": float(
                            sr.get("support_dist_atr", 0.0) or 0.0
                        ),
                        "sr_resistance_dist_atr": float(
                            sr.get("resistance_dist_atr", 0.0) or 0.0
                        ),
                        "sr_distance_to_s1_pct": float(
                            sr.get("distance_to_s1_pct", 0.0) or 0.0
                        ),
                        "sr_distance_to_r1_pct": float(
                            sr.get("distance_to_r1_pct", 0.0) or 0.0
                        ),
                        "sr_quality_score": float(
                            sr.get("sr_quality_score", 0.0) or 0.0
                        ),
                        "sr_rr_ratio": rr,
                        "sr_regime": regime,
                        "sr_action_plan": action_plan,
                        "sr_defensive_flags": ",".join(defensive_flags),
                        "sr_atr_14": atr,
                    }
                )

            if not rows:
                return pd.DataFrame()
            out = pd.DataFrame(rows).drop_duplicates(subset=["ticker"], keep="last")
            self.logger.info(
                "[ScreenerV2] Local SR context built for %d symbols",
                len(out),
            )
            return out
        except Exception as e:
            self.logger.warning("[ScreenerV2] Local SR context build failed: %s", e)
            return pd.DataFrame()

    def _score_gap(self, gap_pct: pd.Series) -> pd.Series:
        gap = gap_pct.fillna(0.0)
        score = 70.0 - gap.abs() * self.config.gap_penalty_per_pct
        score = np.where(gap < 0, score - self.config.gap_negative_penalty, score)
        return pd.Series(np.clip(score, 0, 100), index=gap_pct.index)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        safe_weights = dict(weights or {})
        sr_weight = float(safe_weights.get("sr_alignment", 0.0))
        if sr_weight > self.config.sr_max_weight:
            self.logger.warning(
                "[ScreenerV2] sr_alignment weight %.3f exceeds cap %.3f; clamping",
                sr_weight,
                self.config.sr_max_weight,
            )
            safe_weights["sr_alignment"] = float(self.config.sr_max_weight)

        total = sum(float(v) for v in safe_weights.values())
        if total <= 0:
            return safe_weights
        if abs(total - 1.0) > 0.01:
            self.logger.warning(f"Screener weights sum to {total:.2f}; normalizing")
        normalized = {k: float(v) / total for k, v in safe_weights.items()}

        # Enforce hard S/R max after normalization as well.
        sr_norm = float(normalized.get("sr_alignment", 0.0))
        if sr_norm > self.config.sr_max_weight:
            other_total = sum(v for k, v in normalized.items() if k != "sr_alignment")
            if other_total <= 0:
                normalized["sr_alignment"] = float(self.config.sr_max_weight)
            else:
                scale = (1.0 - float(self.config.sr_max_weight)) / other_total
                for k in list(normalized.keys()):
                    if k == "sr_alignment":
                        normalized[k] = float(self.config.sr_max_weight)
                    else:
                        normalized[k] = float(normalized[k] * scale)
        return normalized

    def _series_or_default(
        self, df: pd.DataFrame, col: str, default: float = 0.0
    ) -> pd.Series:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(default)
        return pd.Series(default, index=df.index)

    def _coalesce_series(
        self,
        primary: pd.Series,
        fallback: pd.Series,
        *,
        numeric: bool = True,
    ) -> pd.Series:
        """Prefer primary values without triggering pandas Series.fillna(Series)."""
        if numeric:
            primary = pd.to_numeric(primary, errors="coerce")
            fallback = pd.to_numeric(fallback, errors="coerce")
        return primary.where(primary.notna(), fallback)

    def _safe_float(self, value, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            if isinstance(value, float) and np.isnan(value):
                return default
            return float(value)
        except Exception:
            return default

    def _safe_text(self, value, default: str = "") -> str:
        try:
            if value is None or pd.isna(value):
                return default
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return text if text else default

    def _compute_rsi(self, close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = -delta.clip(upper=0.0)
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.clip(0.0, 100.0)

    def _compute_adx(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=high.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=high.index,
        )

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(period, min_periods=period).mean()
        plus_di = 100.0 * (
            plus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
        )
        minus_di = 100.0 * (
            minus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan)
        )
        dx = 100.0 * (
            (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        )
        return dx.rolling(period, min_periods=period).mean().clip(0.0, 100.0)

    @staticmethod
    def _normalize_sector_key(value: Any) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        cleaned = str(value).strip().lower().replace("&", "and")
        cleaned = cleaned.replace("/", " ").replace("-", " ")
        cleaned = "_".join(cleaned.split())
        return cleaned or None

    def _sector_relative_strength_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build 5d/20d sector-relative return deltas plus a 0-100 rank score."""
        lookbacks = getattr(self.config, "sector_relative_strength_lookbacks", [5, 20])
        if not lookbacks:
            return {}

        sector = (
            df["sector"].map(self._normalize_sector_key)
            if "sector" in df.columns
            else pd.Series(pd.NA, index=df.index)
        )
        sector_map = {
            self._normalize_sector_key(k): str(v).upper()
            for k, v in dict(
                getattr(self.config, "sector_etf_map", DEFAULT_SECTOR_ETF_MAP) or {}
            ).items()
            if self._normalize_sector_key(k) and v
        }
        sector_etf = sector.map(sector_map)
        outputs: Dict[str, Any] = {}
        score_columns: List[pd.Series] = []

        for raw_lookback in lookbacks:
            lookback = int(raw_lookback)
            if lookback <= 0:
                continue

            ret_col = f"_ret_{lookback}d"
            stock_return = df.groupby("ticker")["close"].pct_change(lookback) * 100.0
            etf_frame = (
                pd.DataFrame(
                    {
                        "date": df["date"],
                        "ticker": df["ticker"].astype(str).str.upper(),
                        ret_col: stock_return,
                    },
                    index=df.index,
                )
                .dropna(subset=[ret_col])
                .drop_duplicates(subset=["date", "ticker"], keep="last")
            )
            etf_lookup = etf_frame.set_index(["date", "ticker"])[ret_col]
            lookup_index = pd.MultiIndex.from_arrays(
                [df["date"], sector_etf.fillna("")]
            )
            etf_return = pd.Series(
                etf_lookup.reindex(lookup_index).to_numpy(), index=df.index
            )

            sector_median = stock_return.groupby([df["date"], sector]).transform(
                "median"
            )
            benchmark_return = etf_return.where(etf_return.notna(), sector_median)
            delta = stock_return - benchmark_return
            delta = delta.where(sector.notna())
            score = delta.groupby(df["date"]).rank(pct=True).fillna(0.5) * 100.0

            outputs[f"rs_vs_sector_{lookback}d"] = delta
            outputs[f"rs_vs_sector_{lookback}d_score"] = score.clip(0.0, 100.0)
            score_columns.append(outputs[f"rs_vs_sector_{lookback}d_score"])

        if score_columns:
            outputs["rs_vs_sector_score"] = pd.concat(score_columns, axis=1).mean(
                axis=1
            )
            outputs["rs_vs_sector_pct"] = outputs["rs_vs_sector_score"]
        else:
            outputs["rs_vs_sector_score"] = pd.Series(50.0, index=df.index)
            outputs["rs_vs_sector_pct"] = outputs["rs_vs_sector_score"]
        return outputs

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.sort_values(["ticker", "date"]).copy()
        group = df.groupby("ticker", group_keys=False)

        def _write_column(name: str, value: Any) -> None:
            """Write a computed column using insert-based semantics."""
            if name in df.columns:
                df.drop(columns=[name], inplace=True)
            df.insert(len(df.columns), name, value)

        # SMA5 curl
        _write_column(
            "sma5",
            group["close"]
            .rolling(5, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True),
        )
        _write_column(
            "sma5_slope_pct",
            (df["sma5"].groupby(df["ticker"]).diff() / df["close"]) * 100,
        )
        _write_column("sma5_accel", df["sma5_slope_pct"].groupby(df["ticker"]).diff())
        _write_column(
            "sma5_curl_raw", (df["sma5_slope_pct"] * 60) + (df["sma5_accel"] * 40)
        )
        _write_column(
            "sma5_curl_score",
            (50 + df["sma5_curl_raw"] * self.config.sma5_curl_scale).clip(0, 100),
        )

        # SMA20 trend (use existing SMA_20 or compute if missing)
        if "sma20" not in df.columns or df["sma20"].isna().all():
            _write_column(
                "sma20",
                group["close"]
                .rolling(20, min_periods=20)
                .mean()
                .reset_index(level=0, drop=True),
            )
        _write_column(
            "sma20_trend_pct",
            (group["sma20"].diff(3) / group["sma20"].shift(3)) * 100,
        )
        _write_column(
            "sma20_trend_score",
            (50 + df["sma20_trend_pct"] * self.config.sma20_trend_scale).clip(0, 100),
        )

        # EMA alignment (5/21/63)
        _write_column(
            "ema5",
            group["close"].transform(lambda s: s.ewm(span=5, adjust=False).mean()),
        )
        _write_column(
            "ema21",
            group["close"].transform(lambda s: s.ewm(span=21, adjust=False).mean()),
        )
        _write_column(
            "ema63",
            group["close"].transform(lambda s: s.ewm(span=63, adjust=False).mean()),
        )
        _write_column(
            "ema_alignment_score",
            (df["ema5"] > df["ema21"]).astype(int) * 40
            + (df["ema21"] > df["ema63"]).astype(int) * 40
            + (df["close"] > df["ema5"]).astype(int) * 20,
        )

        # Gap behavior
        _write_column("prev_close", group["close"].shift(1))
        _write_column(
            "gap_pct", ((df["open"] - df["prev_close"]) / df["prev_close"]) * 100
        )
        _write_column("gap_score", self._score_gap(df["gap_pct"]))

        # Recent momentum
        _write_column("weekly_return", group["close"].pct_change(5) * 100)
        # Avoid pandas Series.fillna(Series) bug in this environment that can raise
        # NameError("DataFrame is not defined") inside pandas internals.
        # Fix (P10): Use numpy.where to bypass pandas namespace issues.
        momentum_base = np.where(
            df["roc_10"].notna(), df["roc_10"], df["weekly_return"]
        )
        momentum_base = pd.Series(momentum_base, index=df.index)
        _write_column(
            "momentum_score",
            (50 + momentum_base.clip(-10, 10) * self.config.momentum_scale).clip(
                0, 100
            ),
        )
        _write_column("momentum_roc_score", df["momentum_score"])
        _write_column(
            "relative_strength_score",
            df["weekly_return"].groupby(df["date"]).rank(pct=True).fillna(0.5) * 100.0,
        )
        for name, value in self._sector_relative_strength_columns(df).items():
            _write_column(name, value)

        # Volatility / BB position
        bb_range = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        _write_column("bb_position", ((df["close"] - df["bb_lower"]) / bb_range) * 100)
        _write_column("bb_score", (100 - df["bb_position"]).clip(0, 100))

        # RSI / MACD / Stoch
        _write_column("rsi_score", (100 - df["rsi_14"]).clip(0, 100))
        macd_scale = self.config.macd_hist_scale
        _write_column(
            "macd_score",
            (50 + (df["macd_hist"] / df["close"].replace(0, np.nan)) * macd_scale).clip(
                0, 100
            ),
        )
        _write_column("stoch_score", df["stoch_k"].clip(0, 100))

        # ADX trend quality:
        # - prefer existing ADX column if available
        # - otherwise compute ADX from OHLC
        adx_source = None
        for col_name in ("adx_14", "adx", "ADX"):
            if (
                col_name in df.columns
                and pd.to_numeric(df[col_name], errors="coerce").notna().any()
            ):
                adx_source = pd.to_numeric(df[col_name], errors="coerce")
                break
        if adx_source is None:
            adx_chunks = []
            for _ticker, ticker_df in df.groupby("ticker", sort=False):
                adx_chunks.append(
                    self._compute_adx(
                        ticker_df["high"],
                        ticker_df["low"],
                        ticker_df["close"],
                        period=14,
                    )
                )
            adx_source = (
                pd.concat(adx_chunks).sort_index()
                if adx_chunks
                else pd.Series(dtype=float)
            )
        _write_column("adx_14", pd.to_numeric(adx_source, errors="coerce"))
        adx_threshold = float(getattr(self.config, "min_adx_for_trend_bonus", 20.0))
        adx_score = np.select(
            [
                df["adx_14"] > 30.0,
                (df["adx_14"] >= adx_threshold) & (df["adx_14"] <= 30.0),
                df["adx_14"] < adx_threshold,
            ],
            [100.0, 60.0, 20.0],
            default=50.0,
        )
        _write_column(
            "adx_trend_quality_score", pd.Series(adx_score, index=df.index).fillna(50.0)
        )

        # ATR%
        _write_column("atr_pct", (df["atr_14"] / df["close"].replace(0, np.nan)) * 100)

        # Liquidity + movement potential hard-filter metrics.
        lookback = max(5, int(self.config.volume_lookback_days))
        _write_column(
            "avg_volume",
            group["volume"]
            .rolling(lookback, min_periods=min(5, lookback))
            .mean()
            .reset_index(level=0, drop=True),
        )
        _write_column("avg_dollar_volume", df["avg_volume"] * df["close"])
        vol_ratio = df["volume"] / df["avg_volume"].replace(0, np.nan)
        _write_column("volume_ratio", vol_ratio)
        _write_column(
            "volume_surge_score",
            (40 + vol_ratio.fillna(1.0).clip(0.2, 4.0) * 20).clip(0, 100),
        )

        rsi = df["rsi_14"].fillna(50.0)
        _write_column(
            "rsi_pullback_score",
            np.select(
                [
                    (rsi >= 40) & (rsi <= 55),
                    ((rsi >= 35) & (rsi < 40)) | ((rsi > 55) & (rsi <= 60)),
                    ((rsi >= 30) & (rsi < 35)) | ((rsi > 60) & (rsi <= 65)),
                ],
                [100.0, 75.0, 55.0],
                default=35.0,
            ),
        )

        high_5d = (
            group["high"]
            .rolling(5, min_periods=5)
            .max()
            .reset_index(level=0, drop=True)
        )
        low_5d = (
            group["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)
        )
        _write_column(
            "five_day_range_pct",
            (high_5d - low_5d) / df["close"].replace(0, np.nan) * 100,
        )

        # Mean-reversion component: RSI(2) + Bollinger position.
        _write_column(
            "rsi_2", group["close"].transform(lambda s: self._compute_rsi(s, period=2))
        )
        bb_pos = df["bb_position"].fillna(50.0)
        rsi2 = df["rsi_2"].fillna(50.0)
        mean_rev_score = np.select(
            [
                (rsi2 < 10.0) & (bb_pos < 20.0),
                rsi2 < 20.0,
                rsi2 > 80.0,
            ],
            [90.0, 70.0, 20.0],
            default=40.0,
        )
        _write_column(
            "mean_reversion_score",
            pd.Series(mean_rev_score, index=df.index).clip(0, 100),
        )

        # Squeeze detection (v0.12)
        _write_column(
            "bb_width",
            (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan),
        )
        # BB width percentile over 20 periods
        _write_column(
            "bb_width_rank",
            df["bb_width"]
            .groupby(df["ticker"])
            .transform(lambda x: x.rolling(20, min_periods=5).rank(pct=True)),
        )
        # Low width = pending expansion = high score
        _write_column(
            "squeeze_score",
            (100.0 * (1.0 - df["bb_width_rank"].fillna(0.5))).clip(0, 100),
        )

        # Momentum persistence (v0.12)
        _write_column("is_pos_day", (df["close"] > df["prev_close"]).astype(int))
        # Count of positive-return days in last 5
        _write_column(
            "momentum_persistence_score",
            df["is_pos_day"]
            .groupby(df["ticker"])
            .transform(lambda x: x.rolling(5, min_periods=1).sum())
            .fillna(0.0)
            * 20.0,
        )
        _write_column(
            "momentum_persistence_score",
            df["momentum_persistence_score"].clip(0, 100),
        )

        # Volume-weighted momentum (v0.12)
        # momentum_score * sqrt(volume_ratio), normalized to 0-100
        _write_column(
            "vol_weighted_momentum_score",
            (
                df["momentum_score"]
                * np.sqrt(df["volume_ratio"].fillna(1.0).clip(0.1, 5.0))
            ).clip(0, 100),
        )

        return df

    def _enrich_with_shared_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply shared feature layer while preserving existing columns and behavior."""
        if df.empty or self._feature_adapter is None:
            return df

        try:
            return self._feature_adapter.enrich_candidates(df)
        except Exception as e:
            self.logger.warning(
                "[ScreenerV2] Shared feature enrichment failed, continuing with legacy features: %s",
                e,
            )
            return df

    def _get_factor_score_series(self, latest: pd.DataFrame) -> Dict[str, pd.Series]:
        """Map configurable factor names to score series in [0, 100]."""
        default = pd.Series(50.0, index=latest.index)
        return {
            "sma5_curl": self._series_or_default(latest, "sma5_curl_score", 50.0),
            "momentum_roc": self._series_or_default(latest, "momentum_roc_score", 50.0),
            "relative_strength": self._series_or_default(
                latest, "relative_strength_score", 50.0
            ),
            "rs_vs_sector": self._series_or_default(latest, "rs_vs_sector_score", 50.0),
            "volume_surge": self._series_or_default(latest, "volume_surge_score", 50.0),
            "ema_alignment": self._series_or_default(
                latest, "ema_alignment_score", 50.0
            ),
            "rsi_pullback": self._series_or_default(latest, "rsi_pullback_score", 50.0),
            "regime": self._series_or_default(latest, "regime_score", 50.0),
            "sr_alignment": self._series_or_default(latest, "sr_alignment_score", 0.0),
            "adx_trend_quality": self._series_or_default(
                latest, "adx_trend_quality_score", 50.0
            ),
            "volume_divergence": self._series_or_default(
                latest, "volume_divergence_score", 50.0
            ),
            "mean_reversion": self._series_or_default(
                latest, "mean_reversion_score", 50.0
            ),
            "vol_weighted_momentum": self._series_or_default(
                latest, "vol_weighted_momentum_score", 50.0
            ),
            "squeeze": self._series_or_default(latest, "squeeze_score", 50.0),
            "momentum_persistence": self._series_or_default(
                latest, "momentum_persistence_score", 50.0
            ),
            # Backwards-compatible aliases
            "rsi": self._series_or_default(latest, "rsi_score", 50.0),
            "macd": self._series_or_default(latest, "macd_score", 50.0),
            "stoch": self._series_or_default(latest, "stoch_score", 50.0),
            "bb_position": self._series_or_default(latest, "bb_score", 50.0),
            "sma20_trend": self._series_or_default(latest, "sma20_trend_score", 50.0),
            "gap": self._series_or_default(latest, "gap_score", 50.0),
            "momentum": self._series_or_default(latest, "momentum_score", 50.0),
            "_default": default,
        }

    def _compute_sr_alignment_score(self, latest: pd.DataFrame) -> pd.Series:
        """Compute S/R confirmation score (0-100) using continuous scaling (trade_learner v0.12)."""
        s1_strength = self._series_or_default(latest, "sr_s1_strength")
        r1_strength = self._series_or_default(latest, "sr_r1_strength")
        dist_s1_pct = self._series_or_default(latest, "sr_distance_to_s1_pct")
        dist_r1_pct = self._series_or_default(latest, "sr_distance_to_r1_pct")

        if "sr_rr_ratio_effective" in latest.columns:
            rr_ratio_effective = self._series_or_default(
                latest, "sr_rr_ratio_effective", 0.0
            )
        else:
            rr_ratio_effective = self._series_or_default(
                latest, "sr_rr_ratio", 0.0
            ) * float(self.config.sr_rr_discount_factor)

        score = pd.Series(0.0, index=latest.index)

        # 1. S1 strength: continuous 0-30 scaled from 30-70 range
        score += ((s1_strength - 30.0) / 40.0 * 30.0).clip(0, 30.0)

        # 2. R1 strength: continuous 0-20 scaled from 30-65 range
        score += ((r1_strength - 30.0) / 35.0 * 20.0).clip(0, 20.0)

        # 3. S/R spread: continuous 0-15 based on distance sum > 10.8%
        spread_pct = dist_s1_pct + dist_r1_pct
        score += (spread_pct / 10.8 * 15.0).clip(0, 15.0)

        # 4. Distance to S1: continuous 0-15 (trade_learner: > 1.4%)
        score += (dist_s1_pct / 1.4 * 15.0).clip(0, 15.0)

        # 5. R:R effective: continuous 0-20 (scaled 1.2 -> 5.0)
        score += ((rr_ratio_effective - 1.2) / 3.8 * 20.0).clip(0, 20.0)

        action_plan = latest.get("sr_action_plan")
        if action_plan is not None:
            lower_plan = action_plan.fillna("").astype(str).str.lower()
            keyword_hit = lower_plan.apply(
                lambda s: any(k in s for k in self.config.sr_action_plan_keywords)
            )
            # Legacy keyword bonus, capped within the 100 overall
            score += np.where(keyword_hit, 10.0, 0.0)

        return score.clip(0.0, 100.0)

    def _enrich_with_divergence(
        self,
        history_df: pd.DataFrame,
        latest: pd.DataFrame,
        divergence_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Add volume-price divergence scores to latest snapshot."""
        try:
            if divergence_df is None:
                divergence_df = self._compute_divergence_latest(history_df)

            if divergence_df.empty:
                latest["volume_divergence_score"] = 50.0
                latest["divergence_score"] = 50.0
                latest["divergence_zscore"] = 0.0
                latest["divergence_signal"] = "none"
                return latest

            latest = latest.merge(divergence_df, on="ticker", how="left")
            latest["divergence_score"] = latest["divergence_score"].fillna(50.0)
            latest["divergence_zscore"] = latest["divergence_zscore"].fillna(0.0)
            latest["divergence_signal"] = latest["divergence_signal"].fillna("none")
            latest["volume_divergence_score"] = latest[
                "volume_divergence_score"
            ].fillna(50.0)
        except Exception as e:
            self.logger.warning("[ScreenerV2] Divergence enrichment failed: %s", e)
            latest["divergence_score"] = 50.0
            latest["divergence_zscore"] = 0.0
            latest["divergence_signal"] = "none"
            latest["volume_divergence_score"] = 50.0
        return latest

    def _apply_high_score_trap_demotion(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate performance data shows an inverse correlation for scores > 82.
        We fold these scores back into the high-performance 65-75 zone and
        penalize exhaustion (RSI/Divergence).

        Writes diagnostic provenance fields so downstream audits can separate
        demoted candidates from untouched ones in signals_*.json:
            pre_demotion_score    - the composite_score the demotion saw
            demotion_applied      - True iff composite_score > 82.0
            demotion_climax_penalty - sum of climax penalties applied (0/5/10)
        """
        if df.empty:
            return df

        def _demote_row(row):
            score = float(row.get("composite_score", 0.0))
            if score <= 82.0:
                return score, 0.0, False

            # Base penalty scales with the distance above 82.
            # A 90 becomes: 82 - (8.0 * 1.5) = 70.0
            new_score = 82.0 - (score - 82.0) * 1.5

            # Climax Detection: If high-scoring stock is also extremely overbought (RSI14 > 70)
            # or showing bearish volume divergence, it's a 'Toxic Peak'.
            rsi_14 = float(row.get("rsi_14", 50.0))
            vol_div = float(row.get("volume_divergence_score", 50.0))

            climax_penalty = 0.0
            if rsi_14 > 70.0:
                climax_penalty += 5.0
            if vol_div < 40.0:
                climax_penalty += 5.0

            return max(0, new_score - climax_penalty), climax_penalty, True

        applied = df.apply(_demote_row, axis=1, result_type="expand")
        df["final_score"] = applied[0]
        df["demotion_climax_penalty"] = applied[1]
        df["demotion_applied"] = applied[2]
        df["pre_demotion_score"] = df["composite_score"]
        return df

    def _latest_snapshot(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        latest_idx = df.groupby("ticker")["date"].idxmax()
        latest = df.loc[latest_idx].copy()
        history_counts = df.groupby("ticker")["date"].size()
        latest["history_count"] = latest["ticker"].map(history_counts)
        return latest

    def _apply_hard_universe_filters(
        self, df: pd.DataFrame, context: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Apply non-negotiable tradability filters BEFORE scoring.

        Returns:
            filtered_history_df, filtered_latest_snapshot
        """
        if df.empty:
            return df, pd.DataFrame()

        cfg = self.config
        latest = self._latest_snapshot(df)
        if latest.empty:
            return pd.DataFrame(), latest

        # Staleness (Delisting) check:
        # If the ticker's most recent date is older than the dataset's global most recent date,
        # it is likely delisted or has stale data.
        global_max_date = latest["date"].max()
        # Allow up to 3 days of lag (e.g. for weekends or data feed gaps).
        max_lag_days = getattr(cfg, "max_data_lag_days", 3)
        cond = pd.Series(True, index=latest.index)
        cond &= latest["date"] >= global_max_date - pd.Timedelta(days=max_lag_days)

        cond &= latest["history_count"] >= cfg.min_history_days
        cond &= latest["close"].between(cfg.min_price, cfg.max_price)
        cond &= latest["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        cond &= latest["avg_volume"].fillna(0) >= cfg.min_avg_volume
        cond &= latest["avg_dollar_volume"].fillna(0) >= cfg.min_avg_dollar_volume
        cond &= latest["five_day_range_pct"].fillna(0) >= cfg.five_day_range_min_pct

        from autotrade.risk.inverse_etf_manager import is_inverse_etf

        latest["is_inverse_etf"] = latest["ticker"].apply(is_inverse_etf)
        hard_block = latest["ticker"].astype(str).str.upper().isin(HARD_BLOCK_MEGA_CAP)
        if hard_block.any():
            for symbol in latest.loc[hard_block, "ticker"].astype(str).str.upper():
                self.logger.info(
                    "[ScreenerV2] Rejecting %s: hard mega-cap blocklist", symbol
                )
        cond &= latest["is_inverse_etf"] | ~hard_block

        # Market Cap filtering
        if "market_cap" in latest.columns:
            min_mkt_cap = getattr(cfg, "min_market_cap", DEFAULT_MIN_MARKET_CAP)
            max_mkt_cap = getattr(cfg, "max_market_cap", DEFAULT_MAX_MARKET_CAP)
            # Bypass market cap filter for Inverse ETFs
            cond &= latest["is_inverse_etf"] | latest["market_cap"].fillna(
                min_mkt_cap
            ).between(min_mkt_cap, max_mkt_cap)

        if {"volume_ratio", "high", "low", "close", "atr_14"}.issubset(latest.columns):
            post_spike_volume = float(
                getattr(cfg, "post_spike_volume_threshold", 5.0) or 5.0
            )
            post_spike_range_atr = float(
                getattr(cfg, "post_spike_range_atr_threshold", 1.5) or 1.5
            )
            range_pct = (latest["high"] - latest["low"]) / latest["close"].replace(
                0, np.nan
            )
            atr_pct = latest["atr_14"] / latest["close"].replace(0, np.nan)
            post_spike = (latest["volume_ratio"].fillna(0) >= post_spike_volume) & (
                range_pct.fillna(0) >= post_spike_range_atr * atr_pct.fillna(0)
            )
            if post_spike.any():
                for symbol in latest.loc[post_spike, "ticker"].astype(str).str.upper():
                    self.logger.info(
                        "[ScreenerV2] Rejecting %s: post-spike long exclusion", symbol
                    )
            cond &= latest["is_inverse_etf"] | ~post_spike

        # If these optional columns are present, reject income/bond-like profiles.
        # Bypass for Inverse ETFs.
        if "dividend_yield" in latest.columns:
            cond &= latest["is_inverse_etf"] | ~(
                (latest["dividend_yield"].fillna(0) > 3.0)
                & (latest["atr_pct"].fillna(0) < 2.0)
            )
        if "beta" in latest.columns:
            cond &= latest["is_inverse_etf"] | ~(
                (latest["beta"].fillna(1.0) < 0.3) & (latest["atr_pct"].fillna(0) < 1.5)
            )

        passed_latest = latest[cond].copy()
        passed_tickers = set(passed_latest["ticker"])
        filtered_history = df[df["ticker"].isin(passed_tickers)].copy()

        self.logger.info(
            "[ScreenerV2] Hard filters (%s): %s/%s tickers passed",
            context,
            len(passed_latest),
            len(latest),
        )
        self.logger.debug(
            "[ScreenerV2] Hard thresholds: price %.2f-%.2f, ATR%% %.2f-%.2f, avg_vol >= %s, avg_$vol >= %.0f, 5d_range >= %.2f",
            cfg.min_price,
            cfg.max_price,
            cfg.min_atr_pct,
            cfg.max_atr_pct,
            cfg.min_avg_volume,
            cfg.min_avg_dollar_volume,
            cfg.five_day_range_min_pct,
        )

        return filtered_history, passed_latest

    def _build_scored_df(
        self,
        df: pd.DataFrame,
        sr_df: pd.DataFrame,
        divergence_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        if df.empty:
            return df

        # Latest row per ticker
        latest_idx = df.groupby("ticker")["date"].idxmax()
        latest = df.loc[latest_idx].copy()

        # Ensure enough history
        history_counts = df.groupby("ticker")["date"].size()
        latest["history_count"] = latest["ticker"].map(history_counts)
        latest = latest[latest["history_count"] >= self.config.min_history_days]

        if latest.empty:
            return latest

        # Merge SR context
        if not sr_df.empty:
            latest = latest.merge(sr_df, on="ticker", how="left")

        # Apply R:R realism discount when SR fields are present.
        rr_discount = float(self.config.sr_rr_discount_factor)
        rr_raw = self._series_or_default(latest, "sr_rr_ratio", 0.0)
        latest["sr_rr_ratio_effective"] = rr_raw * rr_discount

        # Use SR ATR if local ATR missing
        if "sr_atr_14" in latest.columns:
            latest["atr_14"] = self._coalesce_series(
                latest["atr_14"], latest["sr_atr_14"]
            )
        latest["atr_pct"] = self._coalesce_series(
            latest["atr_pct"],
            (latest["atr_14"] / latest["close"]) * 100,
        )

        # Regime score from SR
        sr_regime = latest.get("sr_regime")
        if sr_regime is None:
            latest["regime_score"] = 50.0
            latest["regime_num"] = 0
            latest["has_regime_signal"] = 0
        else:
            regime_str = sr_regime.fillna("").astype(str).str.upper()
            latest["has_regime_signal"] = np.where(regime_str.str.len() > 0, 1, 0)
            latest["regime_num"] = np.where(
                regime_str.str.startswith("BULL"),
                1,
                np.where(regime_str.str.startswith("BEAR"), -1, 0),
            )
            latest["regime_score"] = np.where(
                latest["regime_num"] == 1,
                100,
                np.where(latest["regime_num"] == -1, 0, 50),
            )

        latest["sr_alignment_score"] = self._compute_sr_alignment_score(latest)
        latest["sr_quality_score"] = latest["sr_alignment_score"]

        # Volume-price divergence alpha enrichment
        latest = self._enrich_with_divergence(df, latest, divergence_df=divergence_df)

        # Factor weights + composite
        weights = self._normalize_weights(self.config.weights)
        factor_series = self._get_factor_score_series(latest)
        composite = pd.Series(0.0, index=latest.index)
        for factor, weight in weights.items():
            series = factor_series.get(factor, factor_series["_default"])
            composite += series.fillna(50.0) * float(weight)
        latest["composite_score"] = composite

        # Report SR contribution explicitly but keep it bounded by configured weight.
        sr_weight = float(weights.get("sr_alignment", 0.0))
        latest["sr_bonus"] = (latest["sr_alignment_score"] * sr_weight).clip(
            0.0, self.config.sr_bonus_cap
        )
        latest["final_score"] = latest["composite_score"]

        return latest

    def _passes_filters(self, df: pd.DataFrame) -> pd.Series:
        cfg = self.config
        cond = pd.Series(True, index=df.index)
        cond &= df["close"].between(cfg.min_price, cfg.max_price)
        cond &= df["atr_pct"].between(cfg.min_atr_pct, cfg.max_atr_pct)
        cond &= df["avg_volume"].fillna(0) >= cfg.min_avg_volume
        cond &= df["avg_dollar_volume"].fillna(0) >= cfg.min_avg_dollar_volume
        cond &= df["five_day_range_pct"].fillna(0) >= cfg.five_day_range_min_pct
        cond &= df["rsi_14"].between(cfg.rsi_min, cfg.rsi_max)
        if self.scoring_mode == "momentum_pullback":
            # ATR-first momentum pullback gate (no external S/R dependency).
            cond &= self._series_or_default(df, "momentum_roc_score", 0.0) >= float(
                getattr(cfg, "momentum_roc_min_score", 40.0) or 40.0
            )
            cond &= self._series_or_default(df, "rsi_pullback_score", 0.0) >= float(
                getattr(cfg, "rsi_pullback_min_score", 35.0) or 35.0
            )
        else:
            cond &= df["sma5_slope_pct"] >= cfg.sma5_min_slope
            cond &= df["sma5_accel"] >= cfg.sma5_min_accel
            cond &= df["composite_score"] >= cfg.min_composite_score
            max_composite_score = float(getattr(cfg, "max_composite_score", 0.0) or 0.0)
            if max_composite_score > 0:
                exempt_regimes = {
                    str(item).strip().upper()
                    for item in getattr(cfg, "max_composite_score_exempt_regimes", [])
                    if str(item).strip()
                }
                exempt = pd.Series(False, index=df.index)
                for regime_column in (
                    "market_regime",
                    "resolved_regime",
                    "regime",
                    "sr_regime",
                ):
                    if regime_column in df.columns:
                        values = df[regime_column].fillna("").astype(str).str.upper()
                        exempt |= values.isin(exempt_regimes)
                cond &= exempt | (df["composite_score"] <= max_composite_score)

        if cfg.prefer_bullish_regime:
            # Regime gate: reject BEARISH (regime_num == -1) but allow NEUTRAL (0) and BULLISH (1).
            # If SR/regime context is unavailable for a symbol, do not auto-reject it.
            regime_col = self._series_or_default(df, "regime_num", 0.0)
            if "has_regime_signal" in df.columns:
                has_regime = self._series_or_default(df, "has_regime_signal", 0.0) > 0
                cond &= (~has_regime) | (regime_col >= 0)
            else:
                cond &= regime_col >= 0

        # S/R Spread Quality Gate (v0.12)
        # require spread (dist_s1 + dist_r1) >= 8%
        if (
            "sr_distance_to_s1_pct" in df.columns
            and "sr_distance_to_r1_pct" in df.columns
        ):
            dist_s1 = self._series_or_default(df, "sr_distance_to_s1_pct", 0.0)
            dist_r1 = self._series_or_default(df, "sr_distance_to_r1_pct", 0.0)
            has_sr = (dist_s1 > 0) | (dist_r1 > 0)
            # Apply 8% spread gate only when S/R data is present
            cond &= (~has_sr) | ((dist_s1 + dist_r1) >= 8.0)

        return cond

    def _compute_levels(
        self, row: pd.Series, strategy_params: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Compute microstructure-aware entry levels using RiskGate logic."""
        price = self._safe_float(row.get("close", 0.0))
        atr = self._safe_float(row.get("atr_14", 0.0))

        # Microstructure-aware level calculation (Phase 1 Risk track)
        from autotrade.risk.risk_gate import get_risk_gate

        risk_gate = get_risk_gate()

        risk_levels = risk_gate.compute_entry_levels(
            entry_price=price,
            atr_14=atr,
            s1_price=self._safe_float(row.get("sr_s1_price", row.get("s1_price", 0.0))),
            s1_strength=self._safe_float(
                row.get("sr_s1_strength", row.get("s1_strength", 0.0))
            ),
            r1_price=self._safe_float(row.get("sr_r1_price", row.get("r1_price", 0.0))),
            r1_strength=self._safe_float(
                row.get("sr_r1_strength", row.get("r1_strength", 0.0))
            ),
            strategy_params=strategy_params,
            bid=self._safe_float(row.get("bid", price)),
            ask=self._safe_float(row.get("ask", price)),
        )

        return {
            "stop_price": float(risk_levels["stop_price"]),
            "target_price": float(risk_levels["target_price"]),
            "risk_reward": float(risk_levels["risk_reward"]),
            "partial_target_price": float(risk_levels.get("partial_target_price", 0.0)),
        }

    def screen(
        self,
        symbols: Optional[List[str]] = None,
        max_candidates: Optional[int] = None,
        exclude_symbols: Optional[List[str]] = None,
        log_samples: bool = True,
    ) -> List[Dict]:
        self.last_run_diagnostics = {
            "status": "running",
            "reason": None,
            "blocking_reasons": [],
            "result_count": None,
        }
        max_candidates = max_candidates or self.config.max_candidates
        exclude_symbols = exclude_symbols or []

        # Use momentum-pullback scoring when configured.
        if self.scoring_mode == "momentum_pullback":
            results = self._screen_momentum_pullback(
                symbols, max_candidates, exclude_symbols, log_samples
            )
        else:
            # Otherwise use complex scoring (default)
            results = self._screen_complex(
                symbols, max_candidates, exclude_symbols, log_samples
            )

        if self.last_run_diagnostics.get("status") != "blocked":
            self.last_run_diagnostics = {
                "status": "ok",
                "reason": None,
                "blocking_reasons": [],
                "result_count": len(results),
                "core_data_readiness": copy.deepcopy(self.core_data_readiness),
            }
        return results

    def _screen_momentum_pullback(
        self,
        symbols: Optional[List[str]] = None,
        max_candidates: int = 50,
        exclude_symbols: Optional[List[str]] = None,
        log_samples: bool = True,
    ) -> List[Dict]:
        """Momentum + pullback screening with optional S/R confirmation."""
        self.logger.info("[ScreenerV2] Using Momentum Pullback screening...")

        # Load prepared price data and enforce hard tradability filters before SR scoring.
        history_session = self._get_prepared_history_session(symbols=symbols)
        price_df = history_session.data
        if price_df.empty:
            if self.last_run_diagnostics.get("status") != "blocked":
                self.logger.warning("[ScreenerV2] No price data loaded")
            return []
        ticker_context = self._get_prepared_ticker_context(history_session)
        hard_filtered_df, latest_snapshot = self._apply_hard_universe_filters(
            price_df, context="momentum_pullback"
        )
        if hard_filtered_df.empty or latest_snapshot.empty:
            self.logger.warning(
                "[ScreenerV2] No symbols passed hard pre-scoring filters"
            )
            return []

        sr_symbols = set(latest_snapshot["ticker"].astype(str).str.upper().tolist())
        sr_df = ticker_context.sr_df[
            ticker_context.sr_df["ticker"].astype(str).str.upper().isin(sr_symbols)
        ].copy()
        divergence_df = ticker_context.divergence_df[
            ticker_context.divergence_df["ticker"]
            .astype(str)
            .str.upper()
            .isin(sr_symbols)
        ].copy()
        scored_df = self._build_scored_df(
            hard_filtered_df, sr_df, divergence_df=divergence_df
        )
        if scored_df.empty:
            self.logger.warning("[ScreenerV2] No momentum-pullback scored candidates")
            return []

        # Use config-driven composite score (v0.12)
        scored_df["momentum_pullback_score"] = scored_df["composite_score"]

        # Apply market intelligence (NEW Phase 1A)
        scored_df = self._apply_market_intelligence(scored_df)

        # High Score Trap Demotion (NEW Phase 1B)
        scored_df = self._apply_high_score_trap_demotion(scored_df)

        passed_mask = self._passes_filters(scored_df)
        filtered = scored_df[passed_mask]
        if filtered.empty:
            self.logger.warning(
                "[ScreenerV2] No momentum-pullback candidates passed filters; relaxing to scored set"
            )
            filtered = scored_df.copy()

        if exclude_symbols:
            filtered = filtered[
                ~filtered["ticker"].isin({s.upper() for s in exclude_symbols})
            ]

        ranked = filtered.sort_values("final_score", ascending=False).head(
            max_candidates
        )
        self.logger.info(
            f"[ScreenerV2] Momentum Pullback screening returned {len(ranked)} candidates"
        )

        if log_samples and not ranked.empty:
            for _, row in ranked.head(3).iterrows():
                self.logger.info(
                    f"[ScreenerV2] {row['ticker']}: final_score={row['final_score']:.1f}, "
                    f"RS={self._safe_float(row.get('relative_strength_score', 50), 50):.1f}, "
                    f"VOL={self._safe_float(row.get('volume_surge_score', 50), 50):.1f}, "
                    f"ADX={self._safe_float(row.get('adx_trend_quality_score', 50), 50):.1f}, "
                    f"DIV={self._safe_float(row.get('volume_divergence_score', 50), 50):.1f}, "
                    f"MR={self._safe_float(row.get('mean_reversion_score', 50), 50):.1f}"
                )

        return self._format_output(ranked)

    def _screen_complex(
        self,
        symbols: Optional[List[str]] = None,
        max_candidates: Optional[int] = None,
        exclude_symbols: Optional[List[str]] = None,
        log_samples: bool = True,
    ) -> List[Dict]:
        """Complex multi-factor screening (original method)."""
        max_candidates = max_candidates or self.config.max_candidates
        exclude_symbols = exclude_symbols or []
        history_session = self._get_prepared_history_session(symbols=symbols)
        price_df = history_session.data
        if price_df.empty:
            if self.last_run_diagnostics.get("status") != "blocked":
                self.logger.warning("[ScreenerV2] No price data loaded")
            return []
        ticker_context = self._get_prepared_ticker_context(history_session)
        hard_filtered_df, hard_latest = self._apply_hard_universe_filters(
            price_df, context="complex"
        )
        if hard_filtered_df.empty:
            self.logger.warning(
                "[ScreenerV2] No candidates passed hard pre-scoring filters"
            )
            return []

        sr_symbols = set(hard_latest["ticker"].astype(str).str.upper().tolist())
        sr_df = ticker_context.sr_df[
            ticker_context.sr_df["ticker"].astype(str).str.upper().isin(sr_symbols)
        ].copy()
        divergence_df = ticker_context.divergence_df[
            ticker_context.divergence_df["ticker"]
            .astype(str)
            .str.upper()
            .isin(sr_symbols)
        ].copy()

        scored_df = self._build_scored_df(
            hard_filtered_df, sr_df, divergence_df=divergence_df
        )
        if scored_df.empty:
            self.logger.warning("[ScreenerV2] No scored data available")
            return []

        # Apply market intelligence (NEW Phase 1A)
        scored_df = self._apply_market_intelligence(scored_df)

        # High Score Trap Demotion (NEW Phase 1B)
        scored_df = self._apply_high_score_trap_demotion(scored_df)

        # Apply filters
        passed_mask = self._passes_filters(scored_df)
        filtered = scored_df[passed_mask]

        if exclude_symbols:
            filtered = filtered[
                ~filtered["ticker"].isin({s.upper() for s in exclude_symbols})
            ]

        # Overnight/full-universe safety: never collapse to near-zero just because
        # strict quality gates are temporarily tight. Fall back to scored set
        # (already passed hard tradability filters) so discovery can continue.
        if filtered.empty:
            self.logger.warning("[ScreenerV2] No candidates passed filters")

        # Enforce minimum candidate count if data available
        if (
            self.config.enforce_min_candidates
            and len(filtered) < max_candidates
            and len(scored_df) >= max_candidates
        ):
            self.logger.warning(
                f"[ScreenerV2] Only {len(filtered)} passed filters; relaxing to top {max_candidates} by score"
            )
            filtered = scored_df.copy()

        # Rank by final score
        ranked = filtered.sort_values("final_score", ascending=False).head(
            max_candidates
        )

        if log_samples and not ranked.empty:
            self._log_sample_breakdown(ranked, count=3)

        return self._format_output(ranked)

    def _log_sample_breakdown(self, ranked: pd.DataFrame, count: int = 3) -> None:
        sample = ranked.head(count)
        for _, row in sample.iterrows():
            self.logger.info(
                "[ScreenerV2] %s score=%.1f (comp=%.1f, SR=%.1f) "
                "SMA5=%.1f MOM_ROC=%.1f RS=%.1f VOL=%.1f RSIpb=%.1f REG=%.1f",
                row["ticker"],
                self._safe_float(row.get("final_score", 0)),
                self._safe_float(row.get("composite_score", 0)),
                self._safe_float(row.get("sr_bonus", 0)),
                self._safe_float(row.get("sma5_curl_score", 0)),
                self._safe_float(
                    row.get("momentum_roc_score", row.get("momentum_score", 0))
                ),
                self._safe_float(row.get("relative_strength_score", 0)),
                self._safe_float(row.get("volume_surge_score", 0)),
                self._safe_float(
                    row.get("rsi_pullback_score", row.get("rsi_score", 0))
                ),
                self._safe_float(row.get("regime_score", 0)),
            )

    @staticmethod
    def _resolve_divergence_family_tags(
        divergence_signal: str,
        divergence_score: float,
        min_score: float,
    ) -> Dict[str, str]:
        """
        Map high-conviction bullish divergence rows to mean-reversion family tags.

        These tags are consumed downstream by day_manager/pipeline family routing.
        """
        signal = str(divergence_signal or "none").strip().lower()
        if signal == "bullish" and float(divergence_score or 0.0) >= float(min_score):
            return {
                "alpha_source": "volume_divergence",
                "signal_family": "mean_reversion",
                "alpha_family": "mean_reversion",
            }
        return {
            "alpha_source": "",
            "signal_family": "",
            "alpha_family": "",
        }

    def _format_output(self, ranked: pd.DataFrame) -> List[Dict]:
        weights = self._normalize_weights(self.config.weights)
        factor_map = self._get_factor_score_series(ranked)
        results = []

        for _, row in ranked.iterrows():
            ticker = self._safe_text(row.get("ticker"), default="UNKNOWN").upper()
            price = self._safe_float(row.get("close", 0))
            atr = self._safe_float(row.get("atr_14", 0))
            rsi_raw = row.get("rsi_14", None)
            rsi_value = self._safe_float(rsi_raw, default=50.0)
            if atr <= 0.0 or (
                rsi_raw is not None and not pd.isna(rsi_raw) and rsi_value <= 0.0
            ):
                self.logger.warning(
                    "[SCREENER REJECT] %s dropped before PMValidator: invalid features "
                    "(rsi_14=%s atr_14=%s price=%s)",
                    ticker,
                    rsi_raw,
                    row.get("atr_14", None),
                    row.get("close", None),
                )
                continue
            levels = self._compute_levels(row)

            factor_scores = {
                "sma5_curl": self._safe_float(row.get("sma5_curl_score", 0)),
                "momentum_roc": self._safe_float(
                    row.get("momentum_roc_score", row.get("momentum_score", 0))
                ),
                "relative_strength": self._safe_float(
                    row.get("relative_strength_score", 0)
                ),
                "rs_vs_sector": self._safe_float(
                    row.get("rs_vs_sector_score", 50), default=50
                ),
                "volume_surge": self._safe_float(row.get("volume_surge_score", 0)),
                "ema_alignment": self._safe_float(row.get("ema_alignment_score", 0)),
                "regime": self._safe_float(row.get("regime_score", 0)),
                "rsi_pullback": self._safe_float(
                    row.get("rsi_pullback_score", row.get("rsi_score", 0))
                ),
                "sr_alignment": self._safe_float(row.get("sr_alignment_score", 0)),
                "adx_trend_quality": self._safe_float(
                    row.get("adx_trend_quality_score", 50), default=50
                ),
                "volume_divergence": self._safe_float(
                    row.get("volume_divergence_score", row.get("divergence_score", 50)),
                    default=50,
                ),
                "mean_reversion": self._safe_float(
                    row.get("mean_reversion_score", 50), default=50
                ),
                # Legacy keys retained for compatibility.
                "rsi": self._safe_float(row.get("rsi_score", 0)),
                "macd": self._safe_float(row.get("macd_score", 0)),
                "stoch": self._safe_float(row.get("stoch_score", 0)),
                "bb_position": self._safe_float(row.get("bb_score", 0)),
                "sma20_trend": self._safe_float(row.get("sma20_trend_score", 0)),
                "gap": self._safe_float(row.get("gap_score", 0)),
                "momentum": self._safe_float(row.get("momentum_score", 0)),
            }

            score_breakdown = {}
            for k, w in weights.items():
                if k in factor_scores:
                    val = factor_scores[k]
                else:
                    val = self._safe_float(
                        factor_map.get(k, factor_map["_default"]).loc[row.name],
                        default=50.0,
                    )
                score_breakdown[k] = round(val * w, 2)

            sr_regime = self._safe_text(row.get("sr_regime"))
            caution_flags = self._safe_text(row.get("sr_defensive_flags"))
            action_plan = self._safe_text(row.get("sr_action_plan"))
            divergence_signal = self._safe_text(
                row.get("divergence_signal", "none"), default="none"
            )
            divergence_score = self._safe_float(
                row.get("divergence_score", 50), default=50
            )
            divergence_tags = self._resolve_divergence_family_tags(
                divergence_signal=divergence_signal,
                divergence_score=divergence_score,
                min_score=float(
                    getattr(self.config, "min_score", self.config.min_composite_score)
                ),
            )

            from autotrade.risk.inverse_etf_manager import is_inverse_etf

            is_inv = is_inverse_etf(ticker)

            sr_bonus_val = self._safe_float(row.get("sr_bonus", 0))

            results.append(
                {
                    "ticker": ticker,
                    "price": price,
                    "score": self._safe_float(row.get("final_score", 0)),
                    "normalized_score": self._safe_float(row.get("final_score", 0)),
                    "composite_score": self._safe_float(row.get("composite_score", 0)),
                    "pre_demotion_score": self._safe_float(
                        row.get("pre_demotion_score", row.get("composite_score", 0))
                    ),
                    "demotion_applied": bool(row.get("demotion_applied", False)),
                    "demotion_climax_penalty": self._safe_float(
                        row.get("demotion_climax_penalty", 0)
                    ),
                    "sr_bonus": self._safe_float(row.get("sr_bonus", 0)),
                    "rsi": self._safe_float(row.get("rsi_14", 50), default=50),
                    "weekly_return": self._safe_float(row.get("weekly_return", 0)),
                    "s1_price": self._safe_float(row.get("sr_s1_price", 0)),
                    "s1_strength": self._safe_float(row.get("sr_s1_strength", 0)),
                    "r1_price": self._safe_float(row.get("sr_r1_price", 0)),
                    "r1_strength": self._safe_float(row.get("sr_r1_strength", 0)),
                    "atr_14": atr,
                    "support_dist_atr": self._safe_float(
                        row.get("sr_support_dist_atr", 0)
                    ),
                    "resistance_dist_atr": self._safe_float(
                        row.get("sr_resistance_dist_atr", 0)
                    ),
                    "rr_ratio": self._safe_float(row.get("sr_rr_ratio", 0)),
                    "rr_ratio_effective": self._safe_float(
                        row.get("sr_rr_ratio_effective", 0)
                    ),
                    "risk_reward": levels["risk_reward"],
                    "stop_price": levels["stop_price"],
                    "target_price": levels["target_price"],
                    "partial_target_price": levels.get("partial_target_price", 0.0),
                    "distance_to_s1_pct": self._safe_float(
                        row.get("sr_distance_to_s1_pct", 0)
                    ),
                    "distance_to_r1_pct": self._safe_float(
                        row.get("sr_distance_to_r1_pct", 0)
                    ),
                    "regime": sr_regime,
                    "action_plan": action_plan,
                    "caution_flags": caution_flags,
                    "opportunity_flags": "",
                    "sma5_slope_pct": self._safe_float(row.get("sma5_slope_pct", 0)),
                    "sma5_accel": self._safe_float(row.get("sma5_accel", 0)),
                    "sma20_trend_pct": self._safe_float(row.get("sma20_trend_pct", 0)),
                    "gap_pct": self._safe_float(row.get("gap_pct", 0)),
                    "momentum_score": self._safe_float(row.get("momentum_score", 0)),
                    "atr_pct": self._safe_float(row.get("atr_pct", 0)),
                    "avg_volume": self._safe_float(row.get("avg_volume", 0)),
                    "avg_dollar_volume": self._safe_float(
                        row.get("avg_dollar_volume", 0)
                    ),
                    "five_day_range_pct": self._safe_float(
                        row.get("five_day_range_pct", 0)
                    ),
                    "sector": self._safe_text(row.get("sector")),
                    "industry": self._safe_text(row.get("industry")),
                    "rs_vs_sector_5d": self._safe_float(row.get("rs_vs_sector_5d", 0)),
                    "rs_vs_sector_20d": self._safe_float(
                        row.get("rs_vs_sector_20d", 0)
                    ),
                    "rs_vs_sector_pct": self._safe_float(
                        row.get("rs_vs_sector_pct", row.get("rs_vs_sector_score", 50)),
                        default=50,
                    ),
                    "score_breakdown": score_breakdown,
                    "factor_scores": factor_scores,
                    "sr_quality_score": self._safe_float(
                        row.get("sr_quality_score", 0)
                    ),
                    "divergence_score": divergence_score,
                    "divergence_zscore": self._safe_float(
                        row.get("divergence_zscore", 0)
                    ),
                    "divergence_signal": divergence_signal,
                    "alpha_source": divergence_tags["alpha_source"],
                    "signal_family": divergence_tags["signal_family"],
                    "alpha_family": divergence_tags["alpha_family"],
                    "is_inverse_etf": is_inv,
                    "tags": ["inverse_etf"] if is_inv else [],
                    "reason": (
                        f"MOM={factor_scores['momentum_roc']:.1f}, "
                        f"RS={factor_scores['relative_strength']:.1f}, "
                        f"SRS={factor_scores['rs_vs_sector']:.1f}, "
                        f"VOL={factor_scores['volume_surge']:.1f}, SR={sr_bonus_val:.1f}, "
                        f"ADX={factor_scores.get('adx_trend_quality', 50):.0f}, "
                        f"DIV={factor_scores.get('volume_divergence', 50):.0f}, "
                        f"MR={factor_scores.get('mean_reversion', 50):.0f}"
                    ),
                }
            )

        # Attach strategy metadata from validated strategies
        results = self._attach_strategy_metadata(results)

        return results

    def _attach_strategy_metadata(self, candidates: List[Dict]) -> List[Dict]:
        """Match candidates to validated strategies and attach metadata."""
        if not candidates:
            return candidates

        try:
            strategies = load_validated_strategies()
        except Exception as e:
            self.logger.warning(
                "[ScreenerV2] Could not load validated strategies: %s", e
            )
            return candidates

        if not strategies:
            return candidates

        # Build a lookup: pick the best strategy by win_rate for each setup_type
        best_by_type: Dict[str, Dict] = {}
        for strat in strategies:
            defn = strat.get("strategy_definition", strat)
            setup = defn.get("entry", {}).get("setup_type", "")
            bt = defn.get("backtest_results", {})
            wr = float(bt.get("win_rate", 0))
            if setup not in best_by_type or wr > float(
                best_by_type[setup].get("backtest_results", {}).get("win_rate", 0)
            ):
                best_by_type[setup] = defn

        # Also keep one "best overall" strategy for unmatched candidates
        best_overall = max(
            strategies,
            key=lambda s: float(
                s.get("strategy_definition", s)
                .get("backtest_results", {})
                .get("profit_factor", 0)
            ),
            default=None,
        )
        best_overall_defn = (
            best_overall.get("strategy_definition", best_overall)
            if best_overall
            else None
        )

        matched = 0
        for cand in candidates:
            # Try to match by setup type heuristic based on candidate characteristics
            rsi = cand.get("rsi", 50)
            weekly_ret = cand.get("weekly_return", 0)

            # Simple heuristic: pullback (RSI < 50), momentum (RSI > 55), else best overall
            if rsi < 50 and "pullback_support" in best_by_type:
                defn = best_by_type["pullback_support"]
            elif weekly_ret > 3 and "trend_follow" in best_by_type:
                defn = best_by_type["trend_follow"]
            elif "ma_bounce" in best_by_type:
                defn = best_by_type["ma_bounce"]
            else:
                if best_overall_defn:
                    defn = best_overall_defn
                else:
                    # Heuristic fallback if no strategies loaded (Phase 1)
                    cand["setup_type"] = "pullback" if rsi < 55 else "continuation"
                    continue

            bt = defn.get("backtest_results", {})
            exit_rules = defn.get("exit", {})
            cand["strategy_id"] = defn.get("name", "")
            cand["setup_type"] = defn.get("entry", {}).get("setup_type", "unknown")
            cand["strategy_params"] = {
                "stop_atr_mult": float(exit_rules.get("stop_atr_mult", 2.0)),
                "target_atr_mult": float(exit_rules.get("target_atr_mult", 3.0)),
                "trailing_stop": bool(exit_rules.get("trailing_stop", True)),
                "trailing_atr_mult": float(exit_rules.get("trailing_atr_mult", 1.5)),
                "max_hold_days": int(exit_rules.get("max_hold_days", 5)),
                "time_stop_if_flat_days": int(
                    exit_rules.get("time_stop_if_flat_days", 3)
                ),
            }
            cand["backtest_win_rate"] = float(bt.get("win_rate", 0))
            cand["backtest_profit_factor"] = float(bt.get("profit_factor", 0))
            cand["walk_forward_validated"] = bool(
                bt.get("walk_forward_validated", False)
            )
            matched += 1

        self.logger.info(
            "[ScreenerV2] Attached strategy metadata to %d/%d candidates",
            matched,
            len(candidates),
        )
        return candidates


# Convenience function for pipeline compatibility


def get_entry_candidates(
    max_candidates: int = 200,
    exclude_symbols: Optional[List[str]] = None,
    symbols: Optional[List[str]] = None,
    config_override: Optional[Dict] = None,
    parent_logger: Optional[logging.Logger] = None,
    log_samples: bool = True,
    scoring_mode: Optional[str] = None,
) -> List[Dict]:
    """
    Get entry candidates using screener.

    Args:
        max_candidates: Maximum candidates to return
        exclude_symbols: Symbols to exclude
        symbols: Optional list to limit universe
        config_override: Config override dict
        parent_logger: Logger instance
        log_samples: Whether to log sample breakdowns
        scoring_mode: Optional override. When omitted, uses config default.
    """
    screener = ScreenerV2(
        config_override=config_override,
        parent_logger=parent_logger,
        scoring_mode=scoring_mode,
    )
    try:
        base_candidates = screener.screen(
            symbols=symbols,
            max_candidates=max_candidates,
            exclude_symbols=exclude_symbols,
            log_samples=log_samples,
        )
    except Exception as exc:
        screener.last_run_diagnostics = {
            "status": "error",
            "reason": "screen_failed",
            "blocking_reasons": [],
            "result_count": 0,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "core_data_readiness": copy.deepcopy(screener.core_data_readiness),
        }
        _set_last_screen_run(screener.last_run_diagnostics)
        raise
    _set_last_screen_run(screener.last_run_diagnostics)

    # Stage-3 modular pipeline hook: keep legacy screener output as fallback.
    try:
        root_cfg = get_config()
        if not bool(getattr(root_cfg.signal_generation, "enabled", True)):
            return base_candidates

        # OVERNIGHT FIX: Skip pipeline when requesting large candidate counts (overnight research)
        # Pipeline caps results at max_signals_per_batch (default 200), so bypass it for full-universe runs
        if max_candidates > 500:
            if parent_logger:
                parent_logger.debug(
                    f"[ScreenerV2] Pipeline bypassed intentionally "
                    f"(max_candidates={max_candidates} > 500, full-universe overnight mode)"
                )
            return base_candidates

        from autotrade.signals.pipeline import SignalGenerationPipeline

        pipeline = SignalGenerationPipeline(parent_logger=parent_logger)
        ticker_list = [c.get("ticker") for c in base_candidates if c.get("ticker")]
        price_df = screener._load_price_data(symbols=ticker_list or symbols)
        context = pipeline.build_context(
            tickers=ticker_list or symbols,
            price_data=price_df,
            exclude_tickers=exclude_symbols,
        )
        result = pipeline.run(
            context=context,
            legacy_candidates=base_candidates,
            include_baselines=False,
        )
        if result.legacy_candidates:
            screener.last_run_diagnostics["result_count"] = len(
                result.legacy_candidates
            )
            _set_last_screen_run(screener.last_run_diagnostics)
            return result.legacy_candidates[:max_candidates]
        screener.last_run_diagnostics["result_count"] = len(base_candidates)
        _set_last_screen_run(screener.last_run_diagnostics)
        return base_candidates
    except Exception as e:
        log = parent_logger or logger
        log.warning(
            "[ScreenerV2] signal pipeline unavailable; using legacy path: %s", e
        )
        _set_last_screen_run(screener.last_run_diagnostics)
        return base_candidates


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run Screener V2 (local data only)")
    parser.add_argument("--max", type=int, default=200, help="Max candidates to return")
    parser.add_argument(
        "--limit", type=int, default=0, help="Limit to first N tickers (debug)"
    )
    args = parser.parse_args()

    symbols_list = None
    if args.limit > 0:
        try:
            from autotrade.utils.local_data_provider import get_provider

            symbols_list = get_provider().get_available_tickers()[: args.limit]
        except Exception as e:
            logger.warning(f"Could not load ticker list for limit: {e}")

    candidates = get_entry_candidates(
        max_candidates=args.max, symbols=symbols_list, log_samples=True
    )
    logger.info(f"Screener V2 returned {len(candidates)} candidates")
