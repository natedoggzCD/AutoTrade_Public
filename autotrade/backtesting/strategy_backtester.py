"""
Strategy Backtester - Multi-strategy backtesting engine with yfinance data + SQLite cache.

Phase 1 of Backtest Engine Rebuild. Tests 4 entry × 3 exit = 12 strategy combinations
per ticker using 12 months of cached OHLCV data. No TA-Lib dependency.

Usage:
    python -m autotrade.backtesting.strategy_backtester AAPL
    python -m autotrade.backtesting.strategy_backtester CRWD PLTR SNOW --batch
"""

from __future__ import annotations

import logging
import math
import os
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from config.config_loader import get_config
except Exception:
    get_config = None
try:
    from autotrade.feature_engineering.adapters import get_backtest_adapter
except Exception:
    get_backtest_adapter = None
try:
    from autotrade.backtesting.data import resolve_backtest_paths, load_daily_bars
except Exception:
    resolve_backtest_paths = None
    load_daily_bars = None

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.parent  # autotrade/backtesting/ -> root
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DB_PATH = DATA_DIR / "backtest_cache.db"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_STALENESS_HOURS = 24
MIN_HISTORY_DAYS = 120          # ~6 months minimum trading days
MIN_TRADES_FOR_VALIDITY = 12
YFINANCE_RATE_LIMIT = 5         # max tickers/sec
RISK_FREE_RATE = 0.05           # for Sharpe ratio
TRADING_DAYS_PER_YEAR = 252
LOCAL_LOOKBACK_DAYS = 400       # calendar days (~250 trading days)


def _load_local_ohlcv(
    tickers: List[str],
    lookback_days: int = LOCAL_LOOKBACK_DAYS,
) -> Dict[str, pd.DataFrame]:
    if not tickers or resolve_backtest_paths is None or load_daily_bars is None:
        return {}

    try:
        paths = resolve_backtest_paths()
        daily_path = paths.daily_features_path
        if not daily_path.exists():
            return {}

        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=lookback_days)
        df = load_daily_bars(
            daily_features_path=daily_path,
            start_date=start_date,
            end_date=end_date,
            symbols=tickers,
        )
        if df.empty:
            return {}

        df["ticker"] = df["ticker"].astype(str).str.upper()
        df["date"] = pd.to_datetime(df["date"])

        out: Dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker"):
            group = group.sort_values("date")
            group = group.set_index("date")
            group = group[["open", "high", "low", "close", "volume"]]
            out[ticker] = group

        return out
    except Exception as e:
        logger.warning(f"Local OHLCV load failed: {e}")
        return {}


# ============================================================================
# BacktestResult dataclass
# ============================================================================
@dataclass
class BacktestResult:
    """Result of backtesting all strategy combinations for a single ticker."""
    ticker: str
    best_strategy: str            # e.g. "mean_reversion + atr_trail"
    win_rate: float               # 0-100
    avg_return: float             # per-trade %
    profit_factor: float          # gross_profit / gross_loss
    max_drawdown: float           # worst single-trade drawdown %
    sharpe_ratio: float           # annualized
    total_trades: int
    strategy_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cached: bool = False
    data_end_date: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"{self.ticker}: {self.best_strategy} | "
            f"WR={self.win_rate:.1f}% PF={self.profit_factor:.2f} "
            f"Sharpe={self.sharpe_ratio:.2f} Trades={self.total_trades}"
        )


# ============================================================================
# SQLite Cache Layer
# ============================================================================
class OHLCVCache:
    """SQLite-backed OHLCV cache for yfinance data."""

    def __init__(self, db_path: Path = CACHE_DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    ticker TEXT,
                    date DATE,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_meta (
                    ticker TEXT PRIMARY KEY,
                    last_updated TIMESTAMP,
                    rows_cached INTEGER
                )
            """)
            conn.commit()

    def is_stale(self, ticker: str, staleness_hours: float = CACHE_STALENESS_HOURS) -> bool:
        """Return True if cache for ticker is missing or older than staleness_hours."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT last_updated FROM cache_meta WHERE ticker = ?", (ticker,)
            ).fetchone()
            if row is None:
                return True
            last_updated = datetime.fromisoformat(row[0])
            return (datetime.now() - last_updated).total_seconds() > staleness_hours * 3600

    def get(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load cached OHLCV for a ticker. Returns None if no data."""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM ohlcv WHERE ticker = ? ORDER BY date",
                conn,
                params=(ticker,),
                parse_dates=["date"],
            )
        if df.empty:
            return None
        df.set_index("date", inplace=True)
        return df

    def put(self, ticker: str, df: pd.DataFrame):
        """Store OHLCV data for a ticker (replaces existing)."""
        if df is None or df.empty:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM ohlcv WHERE ticker = ?", (ticker,))
            records = []
            for idx, row in df.iterrows():
                dt = idx if isinstance(idx, str) else idx.strftime("%Y-%m-%d")
                records.append((
                    ticker, dt,
                    float(row.get("Open", row.get("open", 0))),
                    float(row.get("High", row.get("high", 0))),
                    float(row.get("Low", row.get("low", 0))),
                    float(row.get("Close", row.get("close", 0))),
                    int(row.get("Volume", row.get("volume", 0))),
                ))
            conn.executemany(
                "INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                records,
            )
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta (ticker, last_updated, rows_cached) "
                "VALUES (?, ?, ?)",
                (ticker, datetime.now().isoformat(), len(records)),
            )
            conn.commit()

    def put_batch(self, data: Dict[str, pd.DataFrame]):
        """Store OHLCV for multiple tickers efficiently."""
        with sqlite3.connect(self.db_path) as conn:
            for ticker, df in data.items():
                if df is None or df.empty:
                    continue
                conn.execute("DELETE FROM ohlcv WHERE ticker = ?", (ticker,))
                records = []
                for idx, row in df.iterrows():
                    dt = idx if isinstance(idx, str) else idx.strftime("%Y-%m-%d")
                    records.append((
                        ticker, dt,
                        float(row.get("Open", row.get("open", 0))),
                        float(row.get("High", row.get("high", 0))),
                        float(row.get("Low", row.get("low", 0))),
                        float(row.get("Close", row.get("close", 0))),
                        int(row.get("Volume", row.get("volume", 0))),
                    ))
                conn.executemany(
                    "INSERT OR REPLACE INTO ohlcv (ticker, date, open, high, low, close, volume) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    records,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO cache_meta (ticker, last_updated, rows_cached) "
                    "VALUES (?, ?, ?)",
                    (ticker, datetime.now().isoformat(), len(records)),
                )
            conn.commit()


# ============================================================================
# Data Fetching (yfinance)
# ============================================================================
def fetch_ohlcv(
    tickers: List[str],
    cache: OHLCVCache,
    force_refresh: bool = False,
    staleness_hours: float = CACHE_STALENESS_HOURS,
    rate_limit: int = YFINANCE_RATE_LIMIT,
    allow_fetch: bool = True,
    use_local_data: bool = False,
    local_lookback_days: int = LOCAL_LOOKBACK_DAYS,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for tickers, using cache when fresh.
    Returns {ticker: DataFrame} with columns [open, high, low, close, volume].
    """
    result: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []
    local_fetch: List[str] = []

    for t in tickers:
        cached = None
        if not force_refresh:
            cached = cache.get(t)
            if cached is not None and len(cached) >= MIN_HISTORY_DAYS and not cache.is_stale(t, staleness_hours=staleness_hours):
                result[t] = cached
                continue

        if use_local_data:
            local_fetch.append(t)
        elif allow_fetch:
            to_fetch.append(t)

    if use_local_data and local_fetch:
        local_data = _load_local_ohlcv(local_fetch, lookback_days=local_lookback_days)
        if local_data:
            cache.put_batch(local_data)
            result.update(local_data)
        missing = [t for t in local_fetch if t not in local_data]
        if missing and allow_fetch:
            to_fetch.extend(missing)

    if to_fetch and allow_fetch:
        import yfinance as yf
        logger.info(f"Fetching {len(to_fetch)} tickers from yfinance...")
        # Batch download is more efficient
        try:
            if len(to_fetch) == 1:
                raw = yf.download(to_fetch[0], period="1y", progress=False, threads=False)
                if raw is not None and not raw.empty:
                    # Flatten MultiIndex columns if present
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    # Normalize column names to lowercase
                    raw.columns = [c.lower() for c in raw.columns]
                    cache.put(to_fetch[0], raw)
                    result[to_fetch[0]] = raw
            else:
                # Batch download - rate limit chunks
                chunk_size = max(1, int(rate_limit))
                for i in range(0, len(to_fetch), chunk_size):
                    chunk = to_fetch[i : i + chunk_size]
                    raw = yf.download(chunk, period="1y", progress=False, threads=True, group_by="ticker")
                    if raw is None or raw.empty:
                        continue

                    batch_data: Dict[str, pd.DataFrame] = {}
                    if len(chunk) == 1:
                        t = chunk[0]
                        df = raw.copy()
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                        df.columns = [c.lower() for c in df.columns]
                        if len(df) >= MIN_HISTORY_DAYS:
                            batch_data[t] = df
                    else:
                        for t in chunk:
                            try:
                                if isinstance(raw.columns, pd.MultiIndex):
                                    df = raw[t].dropna(how="all")
                                else:
                                    df = raw.copy()
                                df.columns = [c.lower() for c in df.columns]
                                if len(df) >= MIN_HISTORY_DAYS:
                                    batch_data[t] = df
                            except (KeyError, Exception) as e:
                                logger.debug(f"Skip {t}: {e}")

                    cache.put_batch(batch_data)
                    result.update(batch_data)

                    # Rate limit between chunks
                    if i + chunk_size < len(to_fetch):
                        time.sleep(1.0)

        except Exception as e:
            logger.error(f"yfinance download error: {e}")

    return _enrich_with_shared_features(result)


def _enrich_with_shared_features(
    ohlcv_by_ticker: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Optionally enrich OHLCV datasets through the shared feature adapter."""
    if not ohlcv_by_ticker or get_config is None or get_backtest_adapter is None:
        return ohlcv_by_ticker

    try:
        feature_cfg = get_config().feature_engineering
    except Exception:
        return ohlcv_by_ticker

    if not bool(getattr(feature_cfg, "enabled", False)):
        return ohlcv_by_ticker

    try:
        adapter = get_backtest_adapter(config=feature_cfg.model_dump())
    except Exception as e:
        logger.warning("Backtest feature adapter unavailable, using legacy OHLCV: %s", e)
        return ohlcv_by_ticker

    enriched: Dict[str, pd.DataFrame] = {}
    for ticker, df in ohlcv_by_ticker.items():
        if df is None or df.empty:
            enriched[ticker] = df
            continue

        working = df.copy()
        rename_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        working = working.rename(columns=rename_map)
        working["ticker"] = str(ticker).upper()
        if isinstance(working.index, pd.DatetimeIndex):
            working["date"] = working.index

        try:
            enriched[ticker] = adapter.enrich_backtest_data(working)
        except Exception as e:
            logger.warning("Shared feature enrichment failed for %s, using legacy OHLCV: %s", ticker, e)
            enriched[ticker] = df

    return enriched


# ============================================================================
# Technical Indicators (no TA-Lib)
# ============================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to an OHLCV DataFrame.
    Columns expected: close, high, low, volume. Index = datetime.
    """
    df = df.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- SMA ---
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["sma_200"] = close.rolling(200).mean()

    # --- EMA ---
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()

    # --- RSI(14) - Wilder's smoothing ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # --- ATR(14) - True Range with Wilder's smoothing ---
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    # --- MACD ---
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # --- Bollinger Bands ---
    df["bb_mid"] = df["sma_20"]
    bb_std = close.rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # --- Rolling highs/lows for strategies ---
    df["high_20"] = high.rolling(20).max()
    df["low_20"] = low.rolling(20).min()
    df["avg_volume_20"] = volume.rolling(20).mean()

    return df


# ============================================================================
# Entry Strategies
# ============================================================================
def entry_mean_reversion(df: pd.DataFrame) -> pd.Series:
    """RSI(14) < 35 AND close > SMA(200). Oversold bounce in uptrend."""
    return (df["rsi_14"] < 35) & (df["close"] > df["sma_200"])


def entry_breakout(df: pd.DataFrame) -> pd.Series:
    """Close breaks above 20-day high on 1.5x avg volume. Momentum breakout."""
    prev_high_20 = df["high_20"].shift(1)
    return (df["close"] > prev_high_20) & (df["volume"] > 1.5 * df["avg_volume_20"])


def entry_support_bounce(df: pd.DataFrame) -> pd.Series:
    """Close within 2% of 20-day low AND RSI(14) < 45. Support test."""
    near_low = df["close"] <= df["low_20"] * 1.02
    return near_low & (df["rsi_14"] < 45)


def entry_pullback_buy(df: pd.DataFrame) -> pd.Series:
    """Close < EMA(10) AND close > SMA(50) AND MACD histogram turning positive."""
    macd_turning = (df["macd_histogram"] > 0) & (df["macd_histogram"].shift(1) <= 0)
    return (df["close"] < df["ema_10"]) & (df["close"] > df["sma_50"]) & macd_turning


ENTRY_STRATEGIES = {
    "mean_reversion": entry_mean_reversion,
    "breakout": entry_breakout,
    "support_bounce": entry_support_bounce,
    "pullback_buy": entry_pullback_buy,
}


# ============================================================================
# Exit Strategies
# ============================================================================
def exit_fixed_hold(
    df: pd.DataFrame, entry_idx: int, hold_days: int = 5
) -> Tuple[int, float]:
    """Sell after N trading days. Returns (exit_index, exit_price)."""
    exit_idx = min(entry_idx + hold_days, len(df) - 1)
    return exit_idx, float(df.iloc[exit_idx]["close"])


def exit_atr_trail(
    df: pd.DataFrame, entry_idx: int, multiplier: float = 2.0, max_hold: int = 30
) -> Tuple[int, float]:
    """Trailing stop at multiplier × ATR(14) below highest close since entry."""
    entry_price = float(df.iloc[entry_idx]["close"])
    highest = entry_price

    for i in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        current_close = float(df.iloc[i]["close"])
        atr = float(df.iloc[i]["atr_14"]) if not np.isnan(df.iloc[i]["atr_14"]) else 0
        highest = max(highest, current_close)
        stop = highest - multiplier * atr
        if current_close <= stop:
            return i, current_close

    # Hit max hold
    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return exit_idx, float(df.iloc[exit_idx]["close"])


def exit_target_stop(
    df: pd.DataFrame,
    entry_idx: int,
    target_pct: float = 0.05,
    stop_pct: float = 0.03,
    max_hold: int = 30,
) -> Tuple[int, float]:
    """Take profit at +target_pct%, stop loss at -stop_pct%."""
    entry_price = float(df.iloc[entry_idx]["close"])
    target = entry_price * (1 + target_pct)
    stop = entry_price * (1 - stop_pct)

    for i in range(entry_idx + 1, min(entry_idx + max_hold + 1, len(df))):
        high_price = float(df.iloc[i]["high"])
        low_price = float(df.iloc[i]["low"])
        close_price = float(df.iloc[i]["close"])

        # Check stop first (worst case)
        if low_price <= stop:
            return i, stop
        # Check target
        if high_price >= target:
            return i, target

    exit_idx = min(entry_idx + max_hold, len(df) - 1)
    return exit_idx, float(df.iloc[exit_idx]["close"])


EXIT_STRATEGIES = {
    "fixed_hold": exit_fixed_hold,
    "atr_trail": exit_atr_trail,
    "target_stop": exit_target_stop,
}


# ============================================================================
# Metrics Computation
# ============================================================================
@dataclass
class TradeRecord:
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    max_drawdown_pct: float = 0.0


def compute_trades(
    df: pd.DataFrame, entry_signal: pd.Series, exit_fn, exit_name: str
) -> List[TradeRecord]:
    """Generate trade list from entry signals and an exit strategy function."""
    trades: List[TradeRecord] = []
    entry_indices = df.index[entry_signal].tolist()

    i = 0
    cooldown_until = -1  # prevent overlapping trades

    for pos in range(len(df)):
        if df.index[pos] not in entry_indices:
            continue
        entry_idx = pos
        if entry_idx <= cooldown_until:
            continue

        entry_price = float(df.iloc[entry_idx]["close"])
        if entry_price <= 0:
            continue

        exit_idx, exit_price = exit_fn(df, entry_idx)
        if exit_idx <= entry_idx:
            continue

        ret = (exit_price - entry_price) / entry_price * 100

        # Max drawdown during trade
        trade_slice = df.iloc[entry_idx : exit_idx + 1]
        if len(trade_slice) > 1:
            running_max = trade_slice["close"].cummax()
            drawdowns = (trade_slice["close"] - running_max) / running_max * 100
            max_dd = float(drawdowns.min())
        else:
            max_dd = 0.0

        trades.append(TradeRecord(
            entry_date=str(df.index[entry_idx].date() if hasattr(df.index[entry_idx], "date") else df.index[entry_idx]),
            exit_date=str(df.index[exit_idx].date() if hasattr(df.index[exit_idx], "date") else df.index[exit_idx]),
            entry_price=round(entry_price, 4),
            exit_price=round(exit_price, 4),
            return_pct=round(ret, 4),
            max_drawdown_pct=round(max_dd, 4),
        ))
        cooldown_until = exit_idx

    return trades


def compute_metrics(
    trades: List[TradeRecord],
    min_trades_for_validity: int = MIN_TRADES_FOR_VALIDITY,
) -> Dict[str, Any]:
    """Compute strategy metrics from a list of trades."""
    if len(trades) < min_trades_for_validity:
        return {
            "valid": False,
            "total_trades": len(trades),
            "reason": f"Not enough trades ({len(trades)} < {min_trades_for_validity})",
        }

    returns = [t.return_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]

    win_rate = len(wins) / len(returns) * 100
    avg_return = np.mean(returns)

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001  # avoid div by zero
    profit_factor = gross_profit / gross_loss

    max_drawdown = min((t.max_drawdown_pct for t in trades), default=0.0)

    # Sharpe ratio (annualized)
    if len(returns) > 1:
        ret_std = np.std(returns, ddof=1)
        if ret_std > 0:
            avg_trade_days = np.mean([
                max((pd.Timestamp(t.exit_date) - pd.Timestamp(t.entry_date)).days, 1)
                for t in trades
            ])
            trades_per_year = TRADING_DAYS_PER_YEAR / max(avg_trade_days, 1)
            excess_return = avg_return - (RISK_FREE_RATE * 100 / trades_per_year)
            sharpe = (excess_return / ret_std) * math.sqrt(trades_per_year)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    return {
        "valid": True,
        "total_trades": len(trades),
        "win_rate": round(win_rate, 2),
        "avg_return": round(float(avg_return), 4),
        "profit_factor": round(profit_factor, 4),
        "max_drawdown": round(max_drawdown, 4),
        "sharpe_ratio": round(sharpe, 4),
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
    }


# ============================================================================
# Strategy Backtester
# ============================================================================
class StrategyBacktester:
    """
    Multi-strategy backtester: 4 entry × 3 exit = 12 combinations.
    Uses yfinance OHLCV cached in SQLite.
    """

    def __init__(
        self,
        cache: Optional[OHLCVCache] = None,
        config: Optional[Any] = None,
        allow_fetch: bool = True,
        use_local_data: Optional[bool] = None,
        local_lookback_days: int = LOCAL_LOOKBACK_DAYS,
    ):
        self.cache = cache or OHLCVCache()
        cfg = config
        if cfg is None and get_config is not None:
            try:
                cfg = get_config().backtest.strategy_backtester
            except Exception:
                cfg = None
        self.config = cfg
        self.cache_staleness_hours = getattr(cfg, "cache_staleness_hours", CACHE_STALENESS_HOURS)
        self.min_trades_for_validity = getattr(cfg, "min_trades_for_validity", MIN_TRADES_FOR_VALIDITY)
        self.yfinance_rate_limit = getattr(cfg, "yfinance_rate_limit", YFINANCE_RATE_LIMIT)
        self.allow_fetch = allow_fetch
        if use_local_data is None:
            try:
                data_cfg = get_config().data if get_config is not None else None
                use_local_data = bool(getattr(data_cfg, "use_local_data_primary", True))
            except Exception:
                use_local_data = True
        self.use_local_data = use_local_data
        self.local_lookback_days = local_lookback_days

    def backtest_ticker(
        self,
        ticker: str,
        force_refresh: bool = False,
        as_of_date: Optional[Any] = None,
    ) -> Optional[BacktestResult]:
        """
        Run all 12 strategy combinations on a single ticker.
        Returns BacktestResult or None if insufficient data.
        """
        start = time.time()

        # Fetch data
        data = fetch_ohlcv(
            [ticker],
            self.cache,
            force_refresh=force_refresh,
            staleness_hours=self.cache_staleness_hours,
            rate_limit=self.yfinance_rate_limit,
            allow_fetch=self.allow_fetch,
            use_local_data=self.use_local_data,
            local_lookback_days=self.local_lookback_days,
        )
        df = data.get(ticker)
        if df is None or len(df) < MIN_HISTORY_DAYS:
            logger.warning(f"{ticker}: insufficient data ({0 if df is None else len(df)} days)")
            return None

        if as_of_date is not None:
            try:
                cutoff = pd.Timestamp(as_of_date)
                if cutoff.tzinfo is not None:
                    cutoff = cutoff.tz_convert(None)
                cutoff = cutoff.normalize()
            except Exception as e:
                logger.warning(f"{ticker}: invalid as_of_date={as_of_date!r}: {e}")
                return None

            df = df[df.index <= cutoff]
            if df is None or len(df) < MIN_HISTORY_DAYS:
                logger.warning(
                    f"{ticker}: insufficient data before as_of_date {cutoff.date()} "
                    f"({0 if df is None else len(df)} days)"
                )
                return None

        cached = not self.cache.is_stale(ticker)
        data_end_date = ""
        try:
            if len(df.index) > 0:
                data_end_date = str(pd.Timestamp(df.index.max()).date())
        except Exception:
            data_end_date = ""

        # Compute indicators
        df = compute_indicators(df)

        # Drop rows where indicators aren't ready (need ~200 days for SMA200)
        # Use 50 as minimum warmup to not discard too much data
        warmup = 50
        df_ready = df.iloc[warmup:].copy()
        if len(df_ready) < MIN_HISTORY_DAYS - warmup:
            logger.warning(f"{ticker}: not enough data after indicator warmup")
            return None

        # Test all 12 combinations
        strategy_results: Dict[str, Dict[str, Any]] = {}
        best_score = -999
        best_name = "none"
        best_metrics: Dict[str, Any] = {}

        for entry_name, entry_fn in ENTRY_STRATEGIES.items():
            try:
                entry_signal = entry_fn(df_ready)
            except Exception as e:
                logger.debug(f"{ticker}/{entry_name}: entry signal error: {e}")
                continue

            if entry_signal.sum() == 0:
                # No entry signals for this strategy
                for exit_name in EXIT_STRATEGIES:
                    combo = f"{entry_name} + {exit_name}"
                    strategy_results[combo] = {
                        "valid": False,
                        "total_trades": 0,
                        "reason": "No entry signals",
                    }
                continue

            for exit_name, exit_fn in EXIT_STRATEGIES.items():
                combo = f"{entry_name} + {exit_name}"
                try:
                    trades = compute_trades(df_ready, entry_signal, exit_fn, exit_name)
                    metrics = compute_metrics(trades, min_trades_for_validity=self.min_trades_for_validity)
                    strategy_results[combo] = metrics

                    # Score: weighted combination for ranking
                    if metrics.get("valid"):
                        score = (
                            metrics["win_rate"] * 0.3
                            + metrics["profit_factor"] * 20
                            + metrics["sharpe_ratio"] * 10
                        )
                        if score > best_score:
                            best_score = score
                            best_name = combo
                            best_metrics = metrics
                except Exception as e:
                    logger.debug(f"{ticker}/{combo}: error: {e}")
                    strategy_results[combo] = {"valid": False, "error": str(e)}

        if not best_metrics or not best_metrics.get("valid"):
            logger.warning(f"{ticker}: no valid strategy combination found")
            return None

        elapsed = time.time() - start
        logger.info(f"{ticker}: best={best_name} WR={best_metrics['win_rate']:.1f}% "
                     f"PF={best_metrics['profit_factor']:.2f} ({elapsed:.2f}s)")

        return BacktestResult(
            ticker=ticker,
            best_strategy=best_name,
            win_rate=best_metrics["win_rate"],
            avg_return=best_metrics["avg_return"],
            profit_factor=best_metrics["profit_factor"],
            max_drawdown=best_metrics["max_drawdown"],
            sharpe_ratio=best_metrics["sharpe_ratio"],
            total_trades=best_metrics["total_trades"],
            strategy_results=strategy_results,
            cached=cached,
            data_end_date=data_end_date,
        )

    def backtest_batch(
        self,
        tickers: List[str],
        max_workers: int = 4,
        force_refresh: bool = False,
        as_of_dates: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Optional[BacktestResult]]:
        """
        Backtest multiple tickers in parallel using ThreadPoolExecutor.
        Pre-fetches all data in batch, then runs strategies in parallel.
        """
        start = time.time()

        # Pre-fetch all data in batch (much faster than individual)
        logger.info(f"Pre-fetching data for {len(tickers)} tickers...")
        fetch_ohlcv(
            tickers,
            self.cache,
            force_refresh=force_refresh,
            staleness_hours=self.cache_staleness_hours,
            rate_limit=self.yfinance_rate_limit,
            allow_fetch=self.allow_fetch,
            use_local_data=self.use_local_data,
            local_lookback_days=self.local_lookback_days,
        )

        results: Dict[str, Optional[BacktestResult]] = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.backtest_ticker,
                    t,
                    force_refresh,
                    (as_of_dates or {}).get(t),
                ): t
                for t in tickers
            }
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    results[ticker] = future.result()
                except Exception as e:
                    logger.error(f"{ticker}: backtest failed: {e}")
                    results[ticker] = None

        elapsed = time.time() - start
        valid = sum(1 for r in results.values() if r is not None)
        logger.info(f"Batch complete: {valid}/{len(tickers)} valid results in {elapsed:.1f}s")

        return results


# ============================================================================
# CLI
# ============================================================================
def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Backtester - multi-strategy testing")
    parser.add_argument("tickers", nargs="+", help="Ticker symbol(s) to backtest")
    parser.add_argument("--batch", action="store_true", help="Use batch mode with thread pool")
    parser.add_argument("--refresh", action="store_true", help="Force refresh cache")
    parser.add_argument("--workers", type=int, default=4, help="Max workers for batch mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    bt = StrategyBacktester()
    tickers = [t.upper() for t in args.tickers]

    if args.batch or len(tickers) > 1:
        print(f"\n=== BATCH BACKTEST: {len(tickers)} tickers ===\n")
        results = bt.backtest_batch(tickers, max_workers=args.workers, force_refresh=args.refresh)

        valid_results = {t: r for t, r in results.items() if r is not None}
        print(f"\nResults: {len(valid_results)}/{len(tickers)} valid\n")

        # Sort by profit factor descending
        sorted_results = sorted(
            valid_results.items(),
            key=lambda x: x[1].profit_factor,
            reverse=True,
        )

        print(f"{'Ticker':<8} {'Strategy':<30} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'AvgRet':>7} {'Trades':>6}")
        print("-" * 78)
        for ticker, r in sorted_results:
            print(
                f"{ticker:<8} {r.best_strategy:<30} "
                f"{r.win_rate:>5.1f}% {r.profit_factor:>6.2f} "
                f"{r.sharpe_ratio:>7.2f} {r.avg_return:>6.2f}% {r.total_trades:>6d}"
            )

        failed = [t for t, r in results.items() if r is None]
        if failed:
            print(f"\nFailed/insufficient data: {', '.join(failed)}")
    else:
        ticker = tickers[0]
        print(f"\n=== BACKTEST: {ticker} ===\n")
        start = time.time()
        result = bt.backtest_ticker(ticker, force_refresh=args.refresh)
        elapsed = time.time() - start

        if result is None:
            print(f"  No result for {ticker} (insufficient data or no valid strategies)")
            sys.exit(1)

        print(f"  Best Strategy:  {result.best_strategy}")
        print(f"  Win Rate:       {result.win_rate:.1f}%")
        print(f"  Avg Return:     {result.avg_return:.2f}%")
        print(f"  Profit Factor:  {result.profit_factor:.2f}")
        print(f"  Sharpe Ratio:   {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:   {result.max_drawdown:.2f}%")
        print(f"  Total Trades:   {result.total_trades}")
        print(f"  Data Source:    {'cache' if result.cached else 'yfinance (fresh)'}")
        print(f"  Time:           {elapsed:.2f}s")

        print(f"\n  All Strategy Results:")
        print(f"  {'Combo':<35} {'Valid':>5} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'Trades':>6}")
        print(f"  {'-'*72}")
        for combo, m in sorted(result.strategy_results.items()):
            if m.get("valid"):
                print(
                    f"  {combo:<35} {'YES':>5} "
                    f"{m['win_rate']:>5.1f}% {m['profit_factor']:>6.2f} "
                    f"{m['sharpe_ratio']:>7.2f} {m['total_trades']:>6d}"
                )
            else:
                reason = m.get("reason", m.get("error", "unknown"))
                print(f"  {combo:<35} {'NO':>5}  -- {reason}")


if __name__ == "__main__":
    main()
