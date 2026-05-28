"""
Conviction Engine - Scoring System for Hold/Add/Exit Decisions
================================================================
Need reasons TO HOLD, not just reasons to exit.

Conviction is a composite score (0-100) based on:
- Technical strength (trend, momentum, volume behavior)
- Fundamental catalyst (earnings, news)
- Relative strength vs market/sector
- Risk-reward ratio going forward
- Time decay (conviction decreases with time if flat)

High conviction (>60) = ADD consideration
Medium conviction (40-60) = HOLD
Low conviction (<40) = EXIT when unlocked
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_FACTOR_BASELINES = {
    'trend': 50.0,
    'momentum': 50.0,
    'support': 50.0,
    'resistance': 50.0,
    'volume': 50.0,
    'catalyst': 50.0,
    'news_sentiment': 50.0,
    'sector': 50.0,
    'relative_strength': 50.0,
    'pnl': 50.0,
    'risk_reward': 50.0,
    'time_decay': 100.0,
}

SECTOR_ETF_MAP = {
    'technology': 'XLK',
    'financial services': 'XLF',
    'healthcare': 'XLV',
    'consumer cyclical': 'XLY',
    'communication services': 'XLC',
    'industrials': 'XLI',
    'consumer defensive': 'XLP',
    'energy': 'XLE',
    'utilities': 'XLU',
    'real estate': 'XLRE',
    'basic materials': 'XLB',
    'materials': 'XLB',
}


@dataclass
class ConvictionFactors:
    """
    All factors that contribute to conviction score.
    Each factor is 0-100, weighted and combined.
    """
    # Technical factors
    trend_score: float = 50.0           # Price trend (>50 = uptrend)
    momentum_score: float = 50.0        # RSI, MACD, etc (>50 = bullish)
    support_score: float = 50.0         # Near support = higher score
    resistance_score: float = 50.0      # Near resistance = lower score
    volume_score: float = 50.0          # Volume confirmation
    
    # Fundamental factors
    catalyst_score: float = 50.0        # Upcoming earnings, events
    news_sentiment: float = 50.0        # Recent news sentiment
    
    # Relative factors
    sector_strength: float = 50.0       # Sector momentum
    relative_strength: float = 50.0     # vs SPY
    
    # Position factors
    pnl_score: float = 50.0             # Current P&L (winners keep winning)
    risk_reward_score: float = 50.0     # Forward-looking R:R
    time_decay_score: float = 100.0     # Decays if position is flat
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'trend': self.trend_score,
            'momentum': self.momentum_score,
            'support': self.support_score,
            'resistance': self.resistance_score,
            'volume': self.volume_score,
            'catalyst': self.catalyst_score,
            'news_sentiment': self.news_sentiment,
            'sector': self.sector_strength,
            'relative_strength': self.relative_strength,
            'pnl': self.pnl_score,
            'risk_reward': self.risk_reward_score,
            'time_decay': self.time_decay_score
        }


@dataclass
class ConvictionConfig:
    """Configuration for conviction scoring."""
    # Weight groups (must sum to 1.0)
    technical_weight: float = 0.30      # Trend, momentum, S/R
    fundamental_weight: float = 0.15    # Catalyst, news
    relative_weight: float = 0.15       # Sector, relative strength
    position_weight: float = 0.40       # P&L, R:R, time decay
    
    # Technical sub-weights
    trend_weight: float = 0.40
    momentum_weight: float = 0.30
    support_weight: float = 0.05
    resistance_weight: float = 0.05
    volume_weight: float = 0.20
    use_sr_inputs: bool = False
    
    # ADD thresholds
    add_conviction_min: float = 65.0    # Min conviction to consider ADD
    add_pnl_min: float = 4.0            # Min P&L % to consider ADD
    add_max_pct: float = 0.50           # Max position increase (50%)
    
    # HOLD thresholds
    hold_conviction_min: float = 40.0   # Below this, consider EXIT
    
    # Aggressive Time Decay Settings (v0.12.1)
    # Rules:
    # - Winners (pnl >= +2%): NO time decay (let winners run)
    # - Small winners (+0.5% to +2%): Slow decay after 8 hours
    # - Flat (-0.5% to +0.5%): Fast decay after 4 hours
    # - Losers (< -0.5%): Immediate heavy decay
    
    flat_threshold_pct: float = 0.5      # "Flat" = P&L within +/- 0.5%
    flat_decay_rate: float = 4.0         # Points lost per hour if flat (aggressive)
    flat_decay_max: float = 55.0         # Max penalty for flat positions (-55 points)
    flat_decay_aggressive_hours: float = 16.0  # Hours until guaranteed EXIT
    
    winner_decay_start_hours: float = 8.0   # Winners start decaying after 8h
    winner_decay_rate: float = 1.5          # Slow decay for winners
    
    loser_decay_rate: float = 5.0        # Points lost per hour if losing
    loser_decay_max: float = 60.0        # Max penalty for losers (-60 points)
    
    # Stagnation detection
    stagnation_penalty: float = 15.0     # Additional penalty for no movement
    stagnation_hours_threshold: float = 3.0  # Hours without meaningful movement
    
    # Legacy aliases (for backward compatibility)
    time_decay_start_hours: float = 4.0   # Start decaying after 4 hours
    time_decay_rate_per_hour: float = 2.0  # Legacy, not used in v0.12.1


class ConvictionEngine:
    """
    Computes conviction scores for positions.
    
    High conviction = reasons to ADD
    Medium conviction = reasons to HOLD
    Low conviction = should EXIT when unlocked
    """
    
    def __init__(self, config: Optional[ConvictionConfig] = None):
        self.config = config or self._load_config_from_yaml()
        self.local_provider = None
        self.financial_db = None
        self.news_cache = None
        self.daily_features_path: Optional[Path] = None
        self._benchmark_return_cache: Dict[str, Dict[str, float]] = {}
        self._universe_ret5: Optional[List[float]] = None
        self._universe_cache_ts: Optional[datetime] = None
        self._init_data_sources()

    def _load_config_from_yaml(self) -> ConvictionConfig:
        cfg = ConvictionConfig()
        try:
            from config.config_loader import get_config
            loaded = get_config().conviction_engine
            cfg.technical_weight = float(loaded.technical_weight)
            cfg.fundamental_weight = float(loaded.fundamental_weight)
            cfg.relative_weight = float(loaded.relative_weight)
            cfg.position_weight = float(loaded.position_weight)
            cfg.trend_weight = float(getattr(loaded, "trend_weight", cfg.trend_weight))
            cfg.momentum_weight = float(getattr(loaded, "momentum_weight", cfg.momentum_weight))
            cfg.support_weight = float(getattr(loaded, "support_weight", cfg.support_weight))
            cfg.resistance_weight = float(getattr(loaded, "resistance_weight", cfg.resistance_weight))
            cfg.volume_weight = float(getattr(loaded, "volume_weight", cfg.volume_weight))
            cfg.use_sr_inputs = False # Forced False: SR provider retired
            cfg.add_conviction_min = float(loaded.add_conviction_min)
            cfg.hold_conviction_min = float(loaded.hold_conviction_min)
            cfg.flat_threshold_pct = float(loaded.flat_threshold_pct)
            cfg.flat_decay_rate = float(loaded.flat_decay_rate)
            cfg.flat_decay_max = float(loaded.flat_decay_max)
            cfg.flat_decay_aggressive_hours = float(loaded.flat_decay_aggressive_hours)
            cfg.winner_decay_start_hours = float(loaded.winner_decay_start_hours)
            cfg.winner_decay_rate = float(loaded.winner_decay_rate)
            cfg.loser_decay_rate = float(loaded.loser_decay_rate)
            cfg.loser_decay_max = float(loaded.loser_decay_max)
            cfg.stagnation_penalty = float(loaded.stagnation_penalty)
            cfg.stagnation_hours_threshold = float(loaded.stagnation_hours_threshold)
            cfg.time_decay_start_hours = float(loaded.time_decay_start_hours)
            cfg.time_decay_rate_per_hour = float(loaded.time_decay_rate_per_hour)
        except Exception as e:
            logger.debug(f"Conviction config fallback to defaults: {e}")
        return cfg

    def _init_data_sources(self) -> None:
        try:
            from autotrade.utils.local_data_provider import get_provider
            self.local_provider = get_provider()
        except Exception as e:
            logger.debug(f"LocalDataProvider unavailable: {e}")

        try:
            from autotrade.utils.financial_db import FinancialDB
            self.financial_db = FinancialDB()
        except Exception as e:
            logger.debug(f"FinancialDB unavailable: {e}")

        try:
            from autotrade.utils.news_cache import get_cache
            self.news_cache = get_cache()
        except Exception as e:
            logger.debug(f"NewsCache unavailable: {e}")

        try:
            from config.config_loader import get_config
            data_cfg = get_config().data
            parquet = Path(data_cfg.daily_features_parquet)
            if not parquet.is_absolute():
                parquet = Path(data_cfg.downday_root) / parquet
            self.daily_features_path = parquet
        except Exception:
            self.daily_features_path = None

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _normalize_sentiment(value: float) -> float:
        v = float(value)
        if abs(v) > 1.5:
            v = (v - 50.0) / 50.0
        return max(-1.0, min(1.0, v))

    @staticmethod
    def _resolve_column(df, *candidates: str):
        for col in candidates:
            if col in df.columns:
                return col
        return None

    def _get_ticker_df(self, symbol: str, days: int = 260):
        if not self.local_provider or not symbol:
            return None
        try:
            if hasattr(self.local_provider, "has_ticker") and not self.local_provider.has_ticker(symbol):
                return None
            return self.local_provider.get_ticker_data(symbol, days=days, include_features=True)
        except Exception:
            return None

    def _get_price_context(self, symbol: str) -> Dict[str, float]:
        out: Dict[str, float] = {}
        df = self._get_ticker_df(symbol, days=260)
        if df is None or len(df) < 25:
            return out

        close_col = self._resolve_column(df, "Adj Close", "Close", "close")
        high_col = self._resolve_column(df, "High", "high")
        low_col = self._resolve_column(df, "Low", "low")
        vol_col = self._resolve_column(df, "Volume", "volume")
        rsi_col = self._resolve_column(df, "RSI_14", "rsi_14")
        macd_col = self._resolve_column(df, "MACD", "macd")

        if not close_col or not vol_col or not high_col or not low_col:
            return out

        closes = df[close_col].astype(float)
        highs = df[high_col].astype(float)
        lows = df[low_col].astype(float)
        volumes = df[vol_col].astype(float)
        current = self._safe_float(closes.iloc[-1], default=0.0)
        out["current_price"] = current

        if len(closes) >= 6 and closes.iloc[-6] > 0:
            out["ret_5d"] = ((closes.iloc[-1] / closes.iloc[-6]) - 1.0) * 100.0
        if len(closes) >= 21 and closes.iloc[-21] > 0:
            out["ret_20d"] = ((closes.iloc[-1] / closes.iloc[-21]) - 1.0) * 100.0
        out["roc_5d"] = out.get("ret_5d", 0.0)

        sma5 = closes.rolling(5).mean()
        if len(sma5.dropna()) >= 3 and current > 0:
            slope_now = ((sma5.iloc[-1] - sma5.iloc[-2]) / current) * 100.0
            slope_prev = ((sma5.iloc[-2] - sma5.iloc[-3]) / current) * 100.0
            out["sma5_curl"] = (slope_now * 0.6) + ((slope_now - slope_prev) * 0.4)

        if len(volumes) >= 20 and volumes.iloc[-20:].mean() > 0:
            out["volume_ratio"] = volumes.iloc[-1] / volumes.iloc[-20:].mean()
        if len(volumes) >= 10 and volumes.iloc[-10:].mean() > 0:
            out["volume_trend"] = (volumes.iloc[-3:].mean() / volumes.iloc[-10:].mean()) - 1.0

        up_mask = closes.diff().fillna(0) > 0
        up_vol = volumes[up_mask].tail(8).sum()
        down_vol = volumes[~up_mask].tail(8).sum()
        if down_vol > 0:
            out["up_down_volume_ratio"] = up_vol / down_vol
        elif up_vol > 0:
            out["up_down_volume_ratio"] = 2.0

        if rsi_col:
            out["rsi"] = self._safe_float(df[rsi_col].iloc[-1], default=50.0)

        if macd_col and len(df[macd_col]) >= 2:
            m_now = self._safe_float(df[macd_col].iloc[-1])
            m_prev = self._safe_float(df[macd_col].iloc[-2])
            if m_now > 0 and m_now > m_prev:
                out["macd_direction"] = 1.0
            elif m_now > 0 and m_now <= m_prev:
                out["macd_direction"] = 0.2
            elif m_now <= 0 and m_now > m_prev:
                out["macd_direction"] = -0.1
            else:
                out["macd_direction"] = -1.0

        prev_close = closes.shift(1)
        # Avoid Series.combine(max) here: pandas can recurse badly on
        # datetime-indexed series during scalar alignment checks in live paths.
        tr_components = pd.concat(
            [
                (highs - lows).abs(),
                (highs - prev_close).abs(),
                (lows - prev_close).abs(),
            ],
            axis=1,
        )
        tr = tr_components.max(axis=1)
        atr_14 = tr.rolling(14, min_periods=14).mean()
        if len(atr_14.dropna()) > 0 and current > 0:
            out["atr_pct"] = (self._safe_float(atr_14.iloc[-1]) / current) * 100.0

        return out

    def _get_benchmark_returns(self, symbol: str) -> Dict[str, float]:
        symbol = (symbol or "").upper()
        if not symbol:
            return {}
        cached = self._benchmark_return_cache.get(symbol)
        if cached:
            return cached

        out: Dict[str, float] = {}
        df = self._get_ticker_df(symbol, days=260)
        if df is not None and len(df) >= 25:
            close_col = self._resolve_column(df, "Adj Close", "Close", "close")
            if close_col:
                closes = df[close_col].astype(float)
                if len(closes) >= 6 and closes.iloc[-6] > 0:
                    out["ret_5d"] = ((closes.iloc[-1] / closes.iloc[-6]) - 1.0) * 100.0
                if len(closes) >= 21 and closes.iloc[-21] > 0:
                    out["ret_20d"] = ((closes.iloc[-1] / closes.iloc[-21]) - 1.0) * 100.0

        self._benchmark_return_cache[symbol] = out
        return out

    def _get_universe_percentile(self, ret_5d: float) -> float:
        now = datetime.now()
        if self._universe_ret5 and self._universe_cache_ts and (now - self._universe_cache_ts) < timedelta(hours=6):
            arr = self._universe_ret5
            hits = sum(1 for v in arr if v <= ret_5d)
            return (hits / len(arr)) * 100.0 if arr else 50.0

        arr: List[float] = []
        try:
            if self.daily_features_path and self.daily_features_path.exists():
                import duckdb
                query = f"""
                WITH prices AS (
                    SELECT
                        ticker,
                        CAST(Date AS DATE) AS date,
                        COALESCE("Adj Close", "Close") AS close
                    FROM read_parquet('{self.daily_features_path.as_posix()}')
                ), ranked AS (
                    SELECT
                        ticker,
                        date,
                        close,
                        ROW_NUMBER() OVER(PARTITION BY ticker ORDER BY date DESC) AS rn
                    FROM prices
                ), pivoted AS (
                    SELECT
                        ticker,
                        MAX(CASE WHEN rn = 1 THEN close END) AS close_0,
                        MAX(CASE WHEN rn = 6 THEN close END) AS close_5
                    FROM ranked
                    WHERE rn IN (1, 6)
                    GROUP BY ticker
                )
                SELECT ((close_0 / close_5) - 1.0) * 100.0 AS ret_5d
                FROM pivoted
                WHERE close_5 > 0
                """
                conn = duckdb.connect(database=':memory:')
                rows = conn.execute(query).fetchall()
                conn.close()
                arr = [self._safe_float(r[0]) for r in rows if r and r[0] is not None]
        except Exception as e:
            logger.debug(f"Universe percentile fallback: {e}")

        if not arr:
            return 50.0

        self._universe_ret5 = arr
        self._universe_cache_ts = now
        hits = sum(1 for v in arr if v <= ret_5d)
        return (hits / len(arr)) * 100.0

    def _get_sr_context(self, symbol: str) -> Dict[str, float]:
        if not self.sr_provider or not symbol:
            return {}
        try:
            raw = self.sr_provider.get_sr_levels(symbol) or {}
            return {
                "s1_price": self._safe_float(raw.get("s1_price"), 0.0),
                "s1_strength": self._safe_float(raw.get("s1_strength"), 0.0),
                "r1_price": self._safe_float(raw.get("r1_price"), 0.0),
                "r1_strength": self._safe_float(raw.get("r1_strength"), 0.0),
                "rr_ratio": self._safe_float(raw.get("rr_ratio"), 0.0),
                "atr_pct": self._safe_float(raw.get("atr_pct"), 0.0),
            }
        except Exception:
            return {}

    def _get_catalyst_context(self, symbol: str) -> Dict[str, bool]:
        ctx = {"earnings_soon": False, "event_soon": False}
        if not self.financial_db or not symbol:
            return ctx
        try:
            earnings = self.financial_db.get_upcoming_earnings(days=5, tickers=[symbol])
            ctx["earnings_soon"] = bool(earnings)
        except Exception:
            pass
        try:
            events = self.financial_db.get_events(symbol, limit=10)
            today = datetime.now().date()
            for e in events:
                raw_date = str(e.get("event_date", "")).strip()
                if not raw_date:
                    continue
                try:
                    event_date = datetime.fromisoformat(raw_date).date()
                except Exception:
                    continue
                days_out = (event_date - today).days
                if 0 <= days_out <= 7:
                    ctx["event_soon"] = True
                    break
        except Exception:
            pass
        return ctx

    def _get_cached_news_sentiment(self, symbol: str) -> Optional[float]:
        if not self.news_cache or not symbol:
            return None
        try:
            articles = self.news_cache.get_cached_news(symbol, max_age_hours=72, limit=30)
            if not articles:
                return None
            vals = []
            for a in articles:
                s = a.get("sentiment_score")
                if s is None:
                    continue
                vals.append(self._normalize_sentiment(self._safe_float(s)))
            if not vals:
                return None
            return sum(vals) / len(vals)
        except Exception:
            return None

    def _compute_momentum_score(
        self,
        rsi: float,
        roc_5d: float,
        macd_direction: float,
        sma5_curl: float,
    ) -> float:
        if rsi < 30:
            rsi_score = 25.0
        elif rsi < 40:
            rsi_score = 35.0 + (rsi - 30.0) * 2.0
        elif rsi <= 65:
            rsi_score = 65.0 + (rsi - 40.0) * 0.8
        elif rsi <= 75:
            rsi_score = 70.0 - (rsi - 65.0) * 1.5
        else:
            rsi_score = 50.0 - (min(rsi, 90.0) - 75.0) * 1.5

        macd_score = 50.0 + self._clamp(macd_direction, -1.0, 1.0) * 30.0
        roc_score = 50.0 + self._clamp(roc_5d, -6.0, 6.0) * 6.5
        curl_score = 50.0 + self._clamp(sma5_curl, -0.6, 0.6) * 60.0

        return self._clamp(
            (rsi_score * 0.30) +
            (macd_score * 0.30) +
            (roc_score * 0.20) +
            (curl_score * 0.20)
        )

    def _compute_support_score(
        self,
        current_price: float,
        support_price: float,
        support_strength: float,
        atr_pct: float,
    ) -> float:
        if support_price <= 0 or current_price <= 0:
            return 50.0
        if current_price <= support_price:
            return 20.0

        atr_abs = max(current_price * max(atr_pct, 0.5) / 100.0, 0.01)
        distance_atr = (current_price - support_price) / atr_abs

        if distance_atr < 0.25:
            dist_score = 60.0
        elif distance_atr <= 1.5:
            dist_score = 85.0 - (distance_atr - 0.25) * 20.0
        elif distance_atr <= 3.0:
            dist_score = 60.0 - (distance_atr - 1.5) * 10.0
        else:
            dist_score = 35.0

        strength_score = 20.0 + self._clamp(support_strength, 0.0, 100.0) * 0.8
        return self._clamp((dist_score * 0.65) + (strength_score * 0.35))

    def _compute_resistance_score(
        self,
        current_price: float,
        resistance_price: float,
        resistance_strength: float,
        atr_pct: float,
    ) -> float:
        if resistance_price <= 0 or current_price <= 0:
            return 50.0
        if resistance_price <= current_price:
            return 75.0

        atr_abs = max(current_price * max(atr_pct, 0.5) / 100.0, 0.01)
        headroom_atr = (resistance_price - current_price) / atr_abs

        if headroom_atr < 0.5:
            room_score = 35.0
        elif headroom_atr < 1.5:
            room_score = 45.0 + (headroom_atr - 0.5) * 15.0
        elif headroom_atr < 3.0:
            room_score = 60.0 + (headroom_atr - 1.5) * 10.0
        else:
            room_score = 80.0

        weak_resistance_score = 100.0 - self._clamp(resistance_strength, 0.0, 100.0)
        return self._clamp((room_score * 0.60) + (weak_resistance_score * 0.40))

    def _compute_volume_score(
        self,
        volume_ratio: float,
        volume_trend: float,
        up_down_volume_ratio: float,
    ) -> float:
        if volume_ratio <= 0:
            volume_ratio = 1.0
        ratio_score = 50.0 + self._clamp((volume_ratio - 1.0) * 45.0, -30.0, 35.0)
        trend_score = 50.0 + self._clamp(volume_trend, -0.6, 0.6) * 40.0
        flow_score = 50.0 + self._clamp(up_down_volume_ratio - 1.0, -1.0, 1.0) * 35.0
        return self._clamp((ratio_score * 0.50) + (trend_score * 0.25) + (flow_score * 0.25))
    
    def compute_conviction(
        self,
        symbol: str,
        pnl_pct: float,
        hold_minutes: float,
        atr_pct: float = 2.0,
        rsi: float = 50.0,
        current_price: float = 0.0,
        support_price: float = 0.0,
        resistance_price: float = 0.0,
        support_strength: float = 0.0,
        resistance_strength: float = 0.0,
        high_since_entry: float = 0.0,
        entry_price: float = 0.0,
        volume_ratio: float = 1.0,
        sector_momentum: float = 0.0,
        relative_strength: float = 0.0,
        news_sentiment: float = 0.0,
        has_catalyst: bool = False,
        sector: str = "",
        market_regime: str = "neutral"
    ) -> Tuple[float, ConvictionFactors, List[str]]:
        """
        Compute conviction score for a position.
        
        Returns:
            (score, factors, reasons) where:
            - score: 0-100 conviction score
            - factors: individual factor scores
            - reasons: list of human-readable reasons
        """
        symbol = (symbol or "").upper()
        factors = ConvictionFactors()
        reasons: List[str] = []

        # --------------------------------------------------------------------
        # Hydrate missing inputs from local providers.
        # --------------------------------------------------------------------
        price_ctx = self._get_price_context(symbol)
        catalyst_ctx = self._get_catalyst_context(symbol)
        cached_news = self._get_cached_news_sentiment(symbol)
        use_sr_inputs = bool(getattr(self.config, "use_sr_inputs", False))
        if use_sr_inputs:
            self._get_sr_context(symbol)

        current_price_eff = current_price if current_price > 0 else self._safe_float(price_ctx.get("current_price"), 0.0)
        entry_price_eff = entry_price if entry_price > 0 else current_price_eff

        atr_pct_eff = atr_pct if atr_pct > 0 else 0.0
        if atr_pct_eff <= 0:
            atr_pct_eff = self._safe_float(price_ctx.get("atr_pct"), 0.0)
        if atr_pct_eff <= 0:
            atr_pct_eff = 2.0

        # S/R inputs are optional and disabled by default.
        support_price_eff = support_price if use_sr_inputs and support_price > 0 else 0.0
        resistance_price_eff = resistance_price if use_sr_inputs and resistance_price > 0 else 0.0
        support_strength_eff = support_strength if use_sr_inputs and support_strength > 0 else 0.0
        resistance_strength_eff = resistance_strength if use_sr_inputs and resistance_strength > 0 else 0.0

        rsi_eff = self._safe_float(rsi, 50.0)
        if abs(rsi_eff - 50.0) < 0.01 and "rsi" in price_ctx:
            rsi_eff = self._safe_float(price_ctx.get("rsi"), 50.0)

        roc_5d = self._safe_float(price_ctx.get("roc_5d"), 0.0)
        macd_direction = self._safe_float(price_ctx.get("macd_direction"), 0.0)
        sma5_curl = self._safe_float(price_ctx.get("sma5_curl"), 0.0)
        stock_ret5 = self._safe_float(price_ctx.get("ret_5d"), 0.0)
        stock_ret20 = self._safe_float(price_ctx.get("ret_20d"), 0.0)

        volume_ratio_eff = volume_ratio if volume_ratio > 0 else 1.0
        if volume_ratio_eff <= 1.0 and "volume_ratio" in price_ctx:
            volume_ratio_eff = self._safe_float(price_ctx.get("volume_ratio"), volume_ratio_eff)
        volume_trend = self._safe_float(price_ctx.get("volume_trend"), 0.0)
        up_down_volume_ratio = self._safe_float(price_ctx.get("up_down_volume_ratio"), 1.0)

        # --------------------------------------------------------------------
        # TECHNICAL FACTORS (30%)
        # --------------------------------------------------------------------
        trend_from_pnl = 50.0 + self._clamp(pnl_pct, -8.0, 10.0) * 2.8
        trend_from_ret = 50.0 + self._clamp(stock_ret5, -6.0, 6.0) * 4.0
        trend_from_curl = 50.0 + self._clamp(sma5_curl, -0.6, 0.6) * 45.0

        retention_bonus = 0.0
        if high_since_entry > 0 and entry_price_eff > 0 and high_since_entry > entry_price_eff:
            max_gain_pct = ((high_since_entry - entry_price_eff) / entry_price_eff) * 100.0
            if max_gain_pct > 0.5:
                retention = pnl_pct / max_gain_pct
                retention = self._clamp(retention, -0.5, 1.2)
                retention_bonus = (retention - 0.2) * 20.0
                if retention > 0.75:
                    reasons.append(f"Trend retention strong ({retention * 100:.0f}% of max gain)")
                elif retention < 0.3:
                    reasons.append(f"Trend deterioration ({(1 - retention) * 100:.0f}% giveback)")

        factors.trend_score = self._clamp((trend_from_pnl * 0.45) + (trend_from_ret * 0.30) + (trend_from_curl * 0.25) + retention_bonus)

        factors.momentum_score = self._compute_momentum_score(
            rsi=rsi_eff,
            roc_5d=roc_5d,
            macd_direction=macd_direction,
            sma5_curl=sma5_curl,
        )
        if factors.momentum_score >= 65:
            reasons.append(f"Momentum strong (RSI {rsi_eff:.0f}, 5D ROC {roc_5d:+.1f}%)")
        elif factors.momentum_score <= 40:
            reasons.append(f"Momentum weak (RSI {rsi_eff:.0f}, 5D ROC {roc_5d:+.1f}%)")

        factors.support_score = self._compute_support_score(
            current_price=current_price_eff,
            support_price=support_price_eff,
            support_strength=support_strength_eff,
            atr_pct=atr_pct_eff,
        )
        if support_price_eff > 0 and factors.support_score >= 65:
            reasons.append(f"Support favorable (S1 ${support_price_eff:.2f})")
        elif support_price_eff > 0 and factors.support_score <= 35:
            reasons.append(f"Support compromised (near/below S1 ${support_price_eff:.2f})")

        factors.resistance_score = self._compute_resistance_score(
            current_price=current_price_eff,
            resistance_price=resistance_price_eff,
            resistance_strength=resistance_strength_eff,
            atr_pct=atr_pct_eff,
        )
        if resistance_price_eff > 0 and factors.resistance_score >= 65:
            reasons.append(f"Upside headroom to R1 (${resistance_price_eff:.2f})")
        elif resistance_price_eff > 0 and factors.resistance_score <= 40:
            reasons.append(f"R1 pressure near ${resistance_price_eff:.2f}")

        factors.volume_score = self._compute_volume_score(
            volume_ratio=volume_ratio_eff,
            volume_trend=volume_trend,
            up_down_volume_ratio=up_down_volume_ratio,
        )
        if volume_ratio_eff >= 1.4:
            reasons.append(f"Volume confirmation ({volume_ratio_eff:.2f}x vs 20D)")
        elif volume_ratio_eff <= 0.8:
            reasons.append(f"Weak volume ({volume_ratio_eff:.2f}x vs 20D)")

        # --------------------------------------------------------------------
        # FUNDAMENTAL FACTORS (15%)
        # --------------------------------------------------------------------
        earnings_soon = bool(catalyst_ctx.get("earnings_soon"))
        event_soon = bool(catalyst_ctx.get("event_soon"))
        has_catalyst_eff = bool(has_catalyst) or event_soon

        catalyst_score = 52.0
        if has_catalyst_eff:
            catalyst_score += 10.0
            reasons.append("Catalyst present")
        if event_soon:
            catalyst_score += 4.0
        if earnings_soon:
            catalyst_score -= 18.0
            reasons.append("Earnings within 5 days (event risk)")
        factors.catalyst_score = self._clamp(catalyst_score)

        news_input = self._normalize_sentiment(self._safe_float(news_sentiment))
        if abs(news_input) < 0.02 and cached_news is not None:
            news_norm = self._normalize_sentiment(cached_news)
        else:
            news_norm = news_input
        factors.news_sentiment = self._clamp(50.0 + (news_norm * 35.0))
        if news_norm > 0.20:
            reasons.append(f"Positive news sentiment ({news_norm:+.2f})")
        elif news_norm < -0.20:
            reasons.append(f"Negative news sentiment ({news_norm:+.2f})")

        # --------------------------------------------------------------------
        # RELATIVE FACTORS (15%)
        # --------------------------------------------------------------------
        sector_name = (sector or "").strip().lower()

        sector_etf = ""
        if sector_name:
            if sector_name in SECTOR_ETF_MAP:
                sector_etf = SECTOR_ETF_MAP[sector_name]
            else:
                for key, val in SECTOR_ETF_MAP.items():
                    if key in sector_name:
                        sector_etf = val
                        break

        spy_returns = self._get_benchmark_returns("SPY")
        spy_5d = self._safe_float(spy_returns.get("ret_5d"), 0.0)
        relative_5d = stock_ret5 - spy_5d

        sector_returns = self._get_benchmark_returns(sector_etf) if sector_etf else {}
        sector_5d = self._safe_float(sector_returns.get("ret_5d"), 0.0)
        sector_20d = self._safe_float(sector_returns.get("ret_20d"), 0.0)

        rel_input = self._safe_float(relative_strength, 0.0)
        if abs(rel_input) <= 1.5:
            rel_input_pct = rel_input * 5.0
        else:
            rel_input_pct = rel_input

        rel_candidates = []
        if stock_ret5 != 0.0 or spy_5d != 0.0:
            rel_candidates.append(50.0 + self._clamp(relative_5d, -8.0, 8.0) * 4.5)
        if stock_ret20 != 0.0 and sector_20d != 0.0:
            rel_candidates.append(50.0 + self._clamp(stock_ret20 - sector_20d, -10.0, 10.0) * 3.0)
        if abs(rel_input_pct) > 0.01:
            rel_candidates.append(50.0 + self._clamp(rel_input_pct, -10.0, 10.0) * 3.0)

        if stock_ret5 != 0.0:
            rel_candidates.append(self._get_universe_percentile(stock_ret5))

        if rel_candidates:
            factors.relative_strength = self._clamp(sum(rel_candidates) / len(rel_candidates))
        else:
            factors.relative_strength = DEFAULT_FACTOR_BASELINES["relative_strength"]

        if relative_5d >= 2.0:
            reasons.append(f"Outperforming SPY 5D by {relative_5d:.1f}%")
        elif relative_5d <= -2.0:
            reasons.append(f"Underperforming SPY 5D by {abs(relative_5d):.1f}%")

        sector_score_candidates = []
        if sector_5d != 0.0:
            sector_score_candidates.append(50.0 + self._clamp(sector_5d, -6.0, 6.0) * 5.0)
        if sector_20d != 0.0:
            sector_score_candidates.append(50.0 + self._clamp(sector_20d, -10.0, 10.0) * 2.5)
        if abs(sector_momentum) > 0.01:
            sector_score_candidates.append(50.0 + self._clamp(sector_momentum, -1.0, 1.0) * 25.0)

        if sector_score_candidates:
            factors.sector_strength = self._clamp(sum(sector_score_candidates) / len(sector_score_candidates))
        else:
            factors.sector_strength = DEFAULT_FACTOR_BASELINES["sector"]

        if factors.sector_strength >= 62:
            reasons.append("Sector tailwind")
        elif factors.sector_strength <= 40:
            reasons.append("Sector headwind")

        # --------------------------------------------------------------------
        # POSITION FACTORS (40%)
        # --------------------------------------------------------------------
        if pnl_pct >= 8.0:
            factors.pnl_score = 90.0
        elif pnl_pct >= 5.0:
            factors.pnl_score = 80.0 + ((pnl_pct - 5.0) * (10.0 / 3.0))
        elif pnl_pct >= 3.0:
            factors.pnl_score = 70.0 + ((pnl_pct - 3.0) * 5.0)
        elif pnl_pct >= 1.0:
            factors.pnl_score = 58.0 + ((pnl_pct - 1.0) * 6.0)
        elif pnl_pct >= -1.0:
            factors.pnl_score = 45.0 + (pnl_pct * 8.0)
        elif pnl_pct >= -2.0:
            factors.pnl_score = 35.0 + ((pnl_pct + 2.0) * 10.0)
        elif pnl_pct >= -4.0:
            factors.pnl_score = 20.0 + ((pnl_pct + 4.0) * 7.5)
        else:
            factors.pnl_score = 10.0
        factors.pnl_score = self._clamp(factors.pnl_score)

        if pnl_pct >= 3.0:
            reasons.append(f"Position winner (+{pnl_pct:.1f}%)")
        elif pnl_pct <= -2.0:
            reasons.append(f"Position loser ({pnl_pct:.1f}%)")

        # ATR-first forward R:R estimate (independent from S/R DB).
        atr_stop_pct = max(atr_pct_eff * 2.0, 0.5)
        trend_boost = (factors.trend_score - 50.0) / 100.0
        momentum_boost = (factors.momentum_score - 50.0) / 100.0
        target_mult = 2.2 + max(0.0, trend_boost * 0.8 + momentum_boost * 0.8)
        target_mult = self._clamp(target_mult, 1.6, 3.6)
        projected_reward_pct = max(atr_pct_eff * target_mult, 0.5)
        rr_ratio = projected_reward_pct / atr_stop_pct if atr_stop_pct > 0 else 0.0

        if rr_ratio >= 3.0:
            factors.risk_reward_score = 82.0
            reasons.append(f"Excellent R:R ({rr_ratio:.1f}:1)")
        elif rr_ratio >= 2.0:
            factors.risk_reward_score = 72.0
            reasons.append(f"Good R:R ({rr_ratio:.1f}:1)")
        elif rr_ratio >= 1.2:
            factors.risk_reward_score = 60.0
        elif rr_ratio >= 0.8:
            factors.risk_reward_score = 45.0
        elif rr_ratio > 0:
            factors.risk_reward_score = 30.0
            reasons.append(f"Poor R:R ({rr_ratio:.1f}:1)")
        else:
            factors.risk_reward_score = DEFAULT_FACTOR_BASELINES["risk_reward"]

        if hold_minutes > 0:
            hold_hours = hold_minutes / 60.0
            if pnl_pct >= 2.0:
                factors.time_decay_score = 100.0
            elif pnl_pct >= 0.5:
                if hold_hours <= self.config.winner_decay_start_hours:
                    factors.time_decay_score = 100.0
                else:
                    hours_over = hold_hours - self.config.winner_decay_start_hours
                    decay = hours_over * self.config.winner_decay_rate
                    factors.time_decay_score = max(50.0, 100.0 - decay)
            elif pnl_pct >= -self.config.flat_threshold_pct:
                if hold_hours <= self.config.time_decay_start_hours:
                    factors.time_decay_score = 100.0
                else:
                    hours_over = hold_hours - self.config.time_decay_start_hours
                    decay = hours_over * self.config.flat_decay_rate
                    factors.time_decay_score = max(100.0 - self.config.flat_decay_max, 100.0 - decay)
                if hold_hours > self.config.stagnation_hours_threshold and abs(roc_5d) < 0.8:
                    factors.time_decay_score -= self.config.stagnation_penalty
                    reasons.append("Stagnating position")
            else:
                decay = hold_hours * self.config.loser_decay_rate
                factors.time_decay_score = max(100.0 - self.config.loser_decay_max, 100.0 - decay)

            if factors.time_decay_score < 50:
                reasons.append(f"Time decay penalty ({hold_hours:.1f}h held)")

        factors.time_decay_score = self._clamp(factors.time_decay_score)

        # --------------------------------------------------------------------
        # FINAL SCORE
        # --------------------------------------------------------------------
        technical_score = (
            factors.trend_score * self.config.trend_weight +
            factors.momentum_score * self.config.momentum_weight +
            factors.support_score * self.config.support_weight +
            factors.resistance_score * self.config.resistance_weight +
            factors.volume_score * self.config.volume_weight
        )
        fundamental_score = (factors.catalyst_score + factors.news_sentiment) / 2.0
        relative_score = (factors.sector_strength + factors.relative_strength) / 2.0
        position_score = (
            factors.pnl_score * 0.40 +
            factors.risk_reward_score * 0.30 +
            factors.time_decay_score * 0.30
        )

        final_score = (
            technical_score * self.config.technical_weight +
            fundamental_score * self.config.fundamental_weight +
            relative_score * self.config.relative_weight +
            position_score * self.config.position_weight
        )

        regime = (market_regime or "").lower().strip()
        if regime == "risk_off":
            final_score *= 0.88
            reasons.append("Risk-off market regime")
        elif regime == "risk_on":
            final_score *= 1.04

        # Keep loss/winner calibration deterministic for failsafe triage.
        if pnl_pct <= -2.0:
            loss_cap = 44.0 - min(8.0, max(0.0, abs(pnl_pct) - 2.0) * 2.0)
            final_score = min(final_score, loss_cap)
        elif pnl_pct >= 3.0:
            winner_floor = 60.0 + min(12.0, (pnl_pct - 3.0) * 1.2)
            final_score = max(final_score, winner_floor)

        final_score = self._clamp(final_score)

        default_hits = []
        for key, value in factors.to_dict().items():
            baseline = DEFAULT_FACTOR_BASELINES.get(key)
            if baseline is not None and abs(value - baseline) < 0.01:
                default_hits.append(key)

        if len(default_hits) > 3:
            logger.debug(f"[{symbol}] Conviction defaults still active for factors={default_hits}")

        logger.debug(
            f"[{symbol}] Conviction {final_score:.1f} | "
            f"T={technical_score:.1f} F={fundamental_score:.1f} "
            f"R={relative_score:.1f} P={position_score:.1f}"
        )
        return final_score, factors, reasons
    
    def get_action_recommendation(
        self,
        conviction_score: float,
        pnl_pct: float,
        hold_phase: str,
        is_critical: bool = False,
        can_burn_day_trade: bool = True
    ) -> Tuple[str, float, List[str]]:
        """
        Get action recommendation based on conviction and constraints.
        
        Returns:
            (action, confidence, reasons)
        """
        reasons = []
        
        # Critical always exits
        if is_critical:
            return "exit", 0.95, ["CRITICAL: Must exit regardless of PDT"]
        
        # High conviction + profit = ADD consideration
        if conviction_score >= self.config.add_conviction_min and pnl_pct >= self.config.add_pnl_min:
            reasons.append(f"High conviction ({conviction_score:.0f}) + profit ({pnl_pct:.1f}%)")
            return "add", min(0.85, conviction_score / 100), reasons
        
        # High conviction = HOLD
        if conviction_score >= self.config.hold_conviction_min:
            reasons.append(f"Conviction ({conviction_score:.0f}) supports holding")
            return "hold", conviction_score / 100, reasons
        
        # Low conviction but locked = HOLD (can't sell without day trade)
        if hold_phase == "locked":
            if can_burn_day_trade and conviction_score < 25:
                reasons.append(f"Very low conviction ({conviction_score:.0f}) - consider emergency exit")
                return "hold", 0.4, reasons  # Still hold but flag it
            else:
                reasons.append("Low conviction but PDT-locked")
                return "hold", 0.5, reasons
        
        # Low conviction + unlocked = EXIT
        if conviction_score < self.config.hold_conviction_min:
            reasons.append(f"Low conviction ({conviction_score:.0f}) - rotate capital")
            return "exit", (100 - conviction_score) / 100, reasons
        
        return "hold", 0.5, ["Default: hold position"]


# ============================================================
# UNIT TESTS
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info("ConvictionEngine module loaded; use pytest for tests.")
