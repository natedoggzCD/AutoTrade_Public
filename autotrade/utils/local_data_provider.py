"""
Local Data Provider - Primary data source for speed and efficiency.

Uses local data files from DownDay directory as PRIMARY source:
- daily_features.parquet (7M+ rows, 4453 tickers, 88 features including SMA/EMA/RSI/MACD)
- prices_hourly.parquet (intraday data)
- market_data.duckdb (structured views)

DATA SOURCE STRATEGY:
- OVERNIGHT/BULK WORKFLOWS: Use local data ONLY (speed priority)
- PRE-MARKET (4am-9:30am): Use live API for current prices
- MARKET HOURS (9:30am-4pm): Use live API for real-time data
- AFTER HOURS (4pm-8pm): Use live API for extended hours data

Falls back to yfinance/Alpaca for:
1. Ticker not found in local data
2. Real-time price needed during trading hours
3. Pre-market/after-hours price checks

Performance: Local parquet lookup < 50ms vs yfinance API ~500-2000ms
"""

import json
import os
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

import pandas as pd

from autotrade.data_ingestion.bootstrap import ensure_core_market_data_ready
from autotrade.data_ingestion.paths import get_ingestion_paths
from autotrade.utils.market_time import get_market_now, is_weekend_daytime

logger = logging.getLogger("AutoTrade.LocalDataProvider")

# ============================================================================
# Market Hours Detection
# ============================================================================


class DataMode(Enum):
    """Data source mode based on market hours."""

    LOCAL_ONLY = "local_only"  # Overnight/bulk - use local data
    LIVE_PRICES = "live_prices"  # Trading hours - use live API for prices
    HYBRID = "hybrid"  # Use local for history, live for current


def is_market_hours() -> bool:
    """Check if currently in market trading hours (9:30 AM - 4:00 PM ET)."""
    now = get_market_now()

    # Weekend check
    if now.weekday() >= 5:
        return False

    hour = now.hour
    minute = now.minute
    time_decimal = hour + minute / 60

    # Regular market hours: 9:30 AM - 4:00 PM ET
    return 9.5 <= time_decimal < 16.0


def is_extended_hours() -> bool:
    """Check if in pre-market (4am-9:30am) or after-hours (4pm-8pm)."""
    now = get_market_now()

    # Weekend check
    if now.weekday() >= 5:
        # Treat weekend daytime like after-hours for PM-mode testing
        return is_weekend_daytime(now)

    hour = now.hour
    minute = now.minute
    time_decimal = hour + minute / 60

    # Pre-market: 4 AM - 9:30 AM or After-hours: 4 PM - 8 PM
    return (4 <= time_decimal < 9.5) or (16 <= time_decimal < 20)


def get_data_mode() -> DataMode:
    """
    Determine appropriate data mode based on current time.

    Returns:
        DataMode.LOCAL_ONLY for overnight/bulk operations
        DataMode.LIVE_PRICES for trading hours
        DataMode.HYBRID for extended hours
    """
    if is_market_hours():
        return DataMode.LIVE_PRICES
    elif is_extended_hours():
        return DataMode.HYBRID
    else:
        return DataMode.LOCAL_ONLY


# ============================================================================
# Path Configuration
# ============================================================================

# Primary data directory (DownDay)
INGESTION_PATHS = get_ingestion_paths()
DOWNDAY_ROOT = INGESTION_PATHS.downday_root

# Data file paths
DAILY_FEATURES_PARQUET = INGESTION_PATHS.daily_features_parquet
HOURLY_PRICES_PARQUET = INGESTION_PATHS.hourly_prices_parquet
DAILY_FEATURES_H5 = INGESTION_PATHS.daily_features_h5
DUCKDB_PATH = INGESTION_PATHS.market_data_duckdb
PRICES_DAILY_CSV = INGESTION_PATHS.prices_daily_csv
PRICES_HOURLY_CSV = INGESTION_PATHS.prices_hourly_csv
DELISTED_SYMBOLS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "delisted_symbols.json"
)


class DelistedTombstones:
    """Persist yfinance delisting tombstones with a bounded recheck window."""

    def __init__(self, path: Path = DELISTED_SYMBOLS_PATH, recheck_days: int = 30):
        self._path = Path(path)
        self._recheck = timedelta(days=max(1, int(recheck_days or 30)))
        self._lock = threading.RLock()
        self._cache = self._load()

    def _load(self) -> Dict[str, Dict[str, str]]:
        if not self._path.exists():
            return {}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def is_delisted(self, symbol: str) -> bool:
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key:
            return False
        with self._lock:
            entry = self._cache.get(symbol_key)
            if not isinstance(entry, dict):
                return False
            try:
                ts = datetime.fromisoformat(str(entry.get("ts")))
            except Exception:
                self._cache.pop(symbol_key, None)
                self._flush_locked()
                return False
            if datetime.now() - ts > self._recheck:
                self._cache.pop(symbol_key, None)
                self._flush_locked()
                return False
            return True

    def mark_delisted(self, symbol: str, reason: str = "") -> None:
        symbol_key = str(symbol or "").upper().strip()
        if not symbol_key:
            return
        with self._lock:
            self._cache[symbol_key] = {
                "ts": datetime.now().isoformat(),
                "reason": str(reason or "possibly_delisted")[:300],
            }
            self._flush_locked()

    def _flush_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._cache, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)


DELISTED_TOMBSTONES = DelistedTombstones()


class _YFinanceDelistedSpamFilter(logging.Filter):
    """
    yfinance's internal logger emits an ERROR per call for every delisted
    symbol ("$XYZ: possibly delisted; no price data found"). When a
    watchlist contains 6-8 dead tickers, every cycle floods the log.
    These are not actionable -- our DelistedTombstones layer already
    skips repeat calls within the recheck window. Demote yfinance's own
    delisted/no-data spam to DEBUG so it stays out of the operational log.
    """

    _NOISY_TOKENS = ("possibly delisted", "no price data found", "no data found")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        msg_l = msg.lower()
        if any(tok in msg_l for tok in self._NOISY_TOKENS):
            return False
        return True


_yf_logger = logging.getLogger("yfinance")
if not any(isinstance(f, _YFinanceDelistedSpamFilter) for f in _yf_logger.filters):
    _yf_logger.addFilter(_YFinanceDelistedSpamFilter())
    if _yf_logger.level == logging.NOTSET or _yf_logger.level < logging.WARNING:
        _yf_logger.setLevel(logging.WARNING)


class LocalDataProvider:
    """
    Primary data source for price/feature data.

    Data priority:
    1. Parquet files (fastest, pre-computed features)
    2. HDF5 files (fast, less features)
    3. CSV files (fallback)
    4. yfinance API (last resort, for missing/real-time data)

    Features available in local data:
    - OHLCV (Open, High, Low, Close, Volume)
    - SMA_10, SMA_20 (can compute SMA_200 from history)
    - EMA_10, EMA_20
    - RSI_14
    - MACD, Signal, Histogram
    - And 70+ other features
    """

    def __init__(self, cache_size: int = 256):
        """
        Initialize LocalDataProvider.

        Args:
            cache_size: LRU cache size for ticker data
        """
        self._daily_df: Optional[pd.DataFrame] = None
        self._hourly_df: Optional[pd.DataFrame] = None
        self._ticker_cache: Dict[str, pd.DataFrame] = {}
        self._metadata: Dict[str, Any] = {}
        self._yf_available = False
        self._last_load: Optional[datetime] = None
        self._cache_size = cache_size

        # Check yfinance availability
        try:
            import yfinance as yf

            self._yf = yf
            self._yf_available = True
        except ImportError:
            logger.warning("yfinance not available - local data only")

        # Shared startup invariant: attempt core-data readiness once.
        try:
            ensure_core_market_data_ready(fail_fast=False)
        except Exception as e:
            logger.warning(f"Core data readiness check warning: {e}")

        # Load available tickers on init
        self._load_ticker_list()

    def _load_ticker_list(self):
        """Load list of available tickers from local data."""
        if DAILY_FEATURES_PARQUET.exists():
            try:
                # Just read ticker column for fast init
                df = pd.read_parquet(DAILY_FEATURES_PARQUET, columns=["ticker"])
                self._metadata["tickers"] = set(df["ticker"].unique())
                self._metadata["ticker_count"] = len(self._metadata["tickers"])
                self._metadata["source"] = "parquet"
                logger.info(
                    f"LocalDataProvider initialized: {self._metadata['ticker_count']} tickers "
                    f"from {DAILY_FEATURES_PARQUET.name}"
                )
            except Exception as e:
                logger.warning(f"Failed to load ticker list: {e}")
                self._metadata["tickers"] = set()
        else:
            self._metadata["tickers"] = set()
            logger.warning("No local data files found - using yfinance only")

    def has_ticker(self, ticker: str) -> bool:
        """Check if ticker exists in local data."""
        ticker = ticker.upper()
        if DELISTED_TOMBSTONES.is_delisted(ticker):
            return False
        return ticker in self._metadata.get("tickers", set())

    def get_available_tickers(self) -> List[str]:
        """Get list of all available tickers in local data."""
        return sorted(
            t
            for t in self._metadata.get("tickers", [])
            if not DELISTED_TOMBSTONES.is_delisted(t)
        )

    def get_ticker_data(
        self, ticker: str, days: int = 250, include_features: bool = True
    ) -> Optional[pd.DataFrame]:
        """
        Get historical data for a ticker.

        Args:
            ticker: Stock ticker symbol
            days: Number of trading days to fetch (default 250 = ~1 year)
            include_features: Include computed features (SMA, RSI, etc.)

        Returns:
            DataFrame with OHLCV + features, or None if not found
        """
        ticker = ticker.upper()
        if DELISTED_TOMBSTONES.is_delisted(ticker):
            return None
        cache_key = f"{ticker}_{days}_{include_features}"

        # Check cache first
        if cache_key in self._ticker_cache:
            cached_data, cached_time = self._ticker_cache[cache_key]
            if datetime.now() - cached_time < timedelta(minutes=30):
                return cached_data.copy()

        # Try local data first
        df = self._get_from_local(ticker, days, include_features)

        if df is not None and len(df) >= days * 0.8:  # At least 80% of requested data
            self._ticker_cache[cache_key] = (df, datetime.now())
            return df.copy()

        # Fallback to yfinance if local data insufficient
        if self._yf_available:
            yf_df = self._get_from_yfinance(ticker, days)
            if yf_df is not None:
                # Merge with local features if available
                if df is not None and include_features:
                    yf_df = self._merge_features(yf_df, df)
                self._ticker_cache[cache_key] = (yf_df, datetime.now())
                return yf_df.copy()

        return df  # Return whatever we have

    def _get_from_local(
        self, ticker: str, days: int, include_features: bool
    ) -> Optional[pd.DataFrame]:
        """Load ticker data from local parquet files."""
        if not DAILY_FEATURES_PARQUET.exists():
            return None

        try:
            # Columns to load
            cols = ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
            if include_features:
                cols.extend(
                    [
                        "SMA_10",
                        "SMA_20",
                        "EMA_10",
                        "EMA_20",
                        "RSI_14",
                        "MACD",
                        "Adj Close",
                    ]
                )

            # Load only required columns for this ticker
            df = pd.read_parquet(
                DAILY_FEATURES_PARQUET,
                columns=[c for c in cols if c != "Adj Close"] + ["Adj Close"]
                if include_features
                else cols,
            )

            # Filter to ticker
            ticker_df = df[df["ticker"] == ticker].copy()

            if ticker_df.empty:
                return None

            # Sort by date and get last N days
            ticker_df["Date"] = pd.to_datetime(ticker_df["Date"])
            ticker_df = ticker_df.sort_values("Date")
            ticker_df = ticker_df.tail(days)

            # Set date as index
            ticker_df.set_index("Date", inplace=True)
            ticker_df.drop(columns=["ticker"], inplace=True, errors="ignore")

            logger.debug(
                f"Loaded {len(ticker_df)} rows for {ticker} from local parquet"
            )
            return ticker_df

        except Exception as e:
            logger.warning(f"Failed to load {ticker} from local: {e}")
            return None

    def _get_from_yfinance(self, ticker: str, days: int) -> Optional[pd.DataFrame]:
        """Fallback to yfinance API."""
        if not self._yf_available:
            return None
        if DELISTED_TOMBSTONES.is_delisted(ticker):
            return None

        try:
            # Convert days to period string
            if days <= 5:
                period = "5d"
            elif days <= 30:
                period = "1mo"
            elif days <= 90:
                period = "3mo"
            elif days <= 180:
                period = "6mo"
            elif days <= 365:
                period = "1y"
            else:
                period = "2y"

            stock = self._yf.Ticker(ticker)
            df = stock.history(period=period)

            if df.empty:
                logger.debug(f"yfinance returned no data for {ticker}")
                DELISTED_TOMBSTONES.mark_delisted(ticker, "yfinance_empty_history")
                return None

            logger.info(f"Fetched {len(df)} rows for {ticker} from yfinance (fallback)")
            return df

        except Exception as e:
            message = str(e)
            if (
                "possibly delisted" in message.lower()
                or "no price data found" in message.lower()
            ):
                DELISTED_TOMBSTONES.mark_delisted(ticker, message)
            logger.warning(f"yfinance failed for {ticker}: {e}")
            return None

    def _merge_features(
        self, primary_df: pd.DataFrame, feature_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge features from local data into yfinance data."""
        try:
            # Align indexes
            feature_cols = [
                c
                for c in feature_df.columns
                if c not in ["Open", "High", "Low", "Close", "Volume"]
            ]

            for col in feature_cols:
                if col in feature_df.columns:
                    # Map by date
                    primary_df[col] = primary_df.index.map(feature_df[col].to_dict())

            return primary_df
        except Exception:
            return primary_df

    def get_sma(
        self, ticker: str, period: int = 200, use_cache: bool = True
    ) -> Tuple[Optional[float], Optional[float], str]:
        """
        Get SMA value for a ticker.

        Args:
            ticker: Stock ticker symbol
            period: SMA period (e.g., 200 for SMA-200)
            use_cache: Whether to use cached data

        Returns:
            Tuple of (sma_value, current_price, data_source)
        """
        # Need at least 'period' days of data
        df = self.get_ticker_data(ticker, days=max(period + 10, 250))

        if df is None or len(df) < period:
            return None, None, "insufficient_data"

        # Calculate SMA
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        sma = df[close_col].tail(period).mean()
        current_price = df[close_col].iloc[-1]

        source = "local" if self.has_ticker(ticker) else "yfinance"
        return float(sma), float(current_price), source

    def get_current_price(
        self, ticker: str, max_age_minutes: int = 15, force_live: bool = False
    ) -> Tuple[Optional[float], str]:
        """
        Get current/latest price for a ticker.

        Logic:
        1. If ticker NOT in local data → always use API
        2. If ticker in local data:
           - Market/extended hours → use live API
           - Overnight/bulk → use local for speed

        Args:
            ticker: Stock ticker symbol
            max_age_minutes: Maximum age of cached price (for live data)
            force_live: Force use of live API regardless of market hours

        Returns:
            Tuple of (price, source) where source is 'live', 'local', or 'none'
        """
        ticker = ticker.upper()
        if DELISTED_TOMBSTONES.is_delisted(ticker):
            return None, "delisted_tombstone"
        has_local = self.has_ticker(ticker)
        data_mode = get_data_mode()

        # If ticker not in local data, must use API
        if not has_local:
            price = self._get_live_price(ticker, mark_empty_as_delisted=True)
            if price is not None:
                return price, "live"
            return None, "none"

        # Ticker is in local data - use market-hours logic
        # Use live API during market hours or if forced
        if force_live or data_mode in (DataMode.LIVE_PRICES, DataMode.HYBRID):
            price = self._get_live_price(ticker)
            if price is not None:
                return price, "live"
            # Fall through to local if live fails

        # Use local data for overnight/bulk or as fallback
        df = self.get_ticker_data(ticker, days=5)

        if df is None or df.empty:
            return None, "none"

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        return float(df[close_col].iloc[-1]), "local"

    def _get_live_price(
        self, ticker: str, *, mark_empty_as_delisted: bool = False
    ) -> Optional[float]:
        """
        Get real-time price from live API (yfinance or Alpaca).

        Tries yfinance first, falls back to Alpaca if available.
        """
        pending_delisted_reason = ""

        # Try yfinance (works during extended hours too)
        if self._yf_available:
            try:
                if DELISTED_TOMBSTONES.is_delisted(ticker):
                    return None
                stock = self._yf.Ticker(ticker)
                # Try fast info first
                info = stock.fast_info
                if hasattr(info, "last_price") and info.last_price:
                    logger.debug(
                        f"Live price for {ticker}: ${info.last_price:.2f} (yfinance)"
                    )
                    return float(info.last_price)
                # Fall back to regular price
                if hasattr(info, "previous_close") and info.previous_close:
                    return float(info.previous_close)
                if mark_empty_as_delisted:
                    pending_delisted_reason = "yfinance_empty_live_price"
            except Exception as e:
                message = str(e)
                if (
                    "possibly delisted" in message.lower()
                    or "no price data found" in message.lower()
                ):
                    pending_delisted_reason = message
                logger.debug(f"yfinance live price failed for {ticker}: {e}")

        # Try Alpaca if available
        try:
            from autotrade.core.broker import get_broker

            broker = get_broker()
            if broker:
                latest = broker.api.get_latest_trade(ticker)
                if latest and latest.price:
                    logger.debug(
                        f"Live price for {ticker}: ${latest.price:.2f} (alpaca)"
                    )
                    return float(latest.price)
        except Exception as e:
            logger.debug(f"Alpaca live price failed for {ticker}: {e}")

        if mark_empty_as_delisted and pending_delisted_reason:
            DELISTED_TOMBSTONES.mark_delisted(ticker, pending_delisted_reason)

        return None

    def bulk_get_ohlcv(
        self, tickers: List[str], days: int = 120
    ) -> Dict[str, pd.DataFrame]:
        """Read OHLCV for many tickers in a single parquet pass.

        H5 (handoff 2026-05-22): per-ticker get_ticker_data() pays the full
        2.1 GB parquet read N times — 400 tickers = ~840 GB of disk I/O and
        was responsible for the 14.8-min Day Manager boot. Doing the read
        once and filtering with isin() collapses that to a single read.

        Returns: dict ticker -> DataFrame indexed by Date with columns
        Open/High/Low/Close/Volume. Tickers missing from local parquet
        are absent from the result.
        """
        if not tickers:
            return {}
        if not DAILY_FEATURES_PARQUET.exists():
            return {}
        upper = list({str(t).upper() for t in tickers if t})
        try:
            df = pd.read_parquet(
                DAILY_FEATURES_PARQUET,
                columns=["ticker", "Date", "Open", "High", "Low", "Close", "Volume"],
            )
        except Exception as e:
            logger.warning(f"bulk_get_ohlcv parquet read failed: {e}")
            return {}
        df = df[df["ticker"].isin(upper)]
        if df.empty:
            return {}
        df["Date"] = pd.to_datetime(df["Date"])
        out: Dict[str, pd.DataFrame] = {}
        for ticker, group in df.groupby("ticker", sort=False):
            g = group.sort_values("Date").tail(days)
            g = g.drop(columns=["ticker"])
            g.set_index("Date", inplace=True)
            out[str(ticker).upper()] = g
        return out

    def bulk_get_sma(
        self, tickers: List[str], period: int = 200, batch_size: int = 50
    ) -> Dict[str, Dict[str, Any]]:
        """
        Efficiently get SMA for multiple tickers.

        Optimized for bulk operations - loads parquet once.

        Args:
            tickers: List of ticker symbols
            period: SMA period
            batch_size: Process in batches

        Returns:
            Dict mapping ticker -> {sma, price, above_sma, source}
        """
        results = {}

        # Check which tickers are in local data
        local_tickers = [t for t in tickers if self.has_ticker(t)]
        api_tickers = [t for t in tickers if not self.has_ticker(t)]

        logger.info(
            f"SMA-{period} bulk lookup: {len(local_tickers)} local, "
            f"{len(api_tickers)} API"
        )

        # Load local data in one shot for efficiency
        if local_tickers and DAILY_FEATURES_PARQUET.exists():
            try:
                df = pd.read_parquet(
                    DAILY_FEATURES_PARQUET,
                    columns=["ticker", "Date", "Close", "Adj Close"],
                )
                df = df[df["ticker"].isin(local_tickers)]
                df["Date"] = pd.to_datetime(df["Date"])

                for ticker in local_tickers:
                    ticker_df = df[df["ticker"] == ticker].sort_values("Date")

                    if len(ticker_df) >= period:
                        close_col = (
                            "Adj Close" if "Adj Close" in ticker_df.columns else "Close"
                        )
                        closes = ticker_df[close_col].tail(period)
                        sma = closes.mean()
                        price = ticker_df[close_col].iloc[-1]

                        results[ticker] = {
                            "sma": float(sma),
                            "price": float(price),
                            "above_sma": price > sma,
                            "source": "local",
                            "data_points": len(closes),
                        }
                    else:
                        results[ticker] = {
                            "sma": None,
                            "price": None,
                            "above_sma": None,
                            "source": "insufficient_local_data",
                            "data_points": len(ticker_df),
                        }

            except Exception as e:
                logger.warning(f"Bulk local load failed: {e}")

        # Fallback to yfinance for API tickers
        if api_tickers and self._yf_available:
            for ticker in api_tickers:
                try:
                    sma, price, source = self.get_sma(ticker, period)
                    if sma is not None and price is not None:
                        results[ticker] = {
                            "sma": sma,
                            "price": price,
                            "above_sma": price > sma,
                            "source": "yfinance",
                        }
                except Exception as e:
                    logger.debug(f"Failed SMA for {ticker}: {e}")

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get provider statistics including current data mode."""
        current_mode = get_data_mode()
        return {
            "ticker_count": self._metadata.get("ticker_count", 0),
            "source": self._metadata.get("source", "none"),
            "cache_entries": len(self._ticker_cache),
            "yfinance_available": self._yf_available,
            "parquet_exists": DAILY_FEATURES_PARQUET.exists(),
            "h5_exists": DAILY_FEATURES_H5.exists(),
            "data_mode": current_mode.value,
            "is_market_hours": is_market_hours(),
            "is_extended_hours": is_extended_hours(),
        }


# ============================================================================
# Module-level convenience functions
# ============================================================================

_provider: Optional[LocalDataProvider] = None


def get_provider() -> LocalDataProvider:
    """Get or create singleton LocalDataProvider instance."""
    global _provider
    if _provider is None:
        _provider = LocalDataProvider()
    return _provider


def get_sma(
    ticker: str, period: int = 200
) -> Tuple[Optional[float], Optional[float], str]:
    """Convenience function to get SMA for a ticker."""
    return get_provider().get_sma(ticker, period)


def get_ticker_data(ticker: str, days: int = 250) -> Optional[pd.DataFrame]:
    """Convenience function to get ticker data."""
    return get_provider().get_ticker_data(ticker, days)


def has_local_data(ticker: str) -> bool:
    """Check if ticker has local data."""
    return get_provider().has_ticker(ticker)


def bulk_sma_check(tickers: List[str], period: int = 200) -> Dict[str, Dict[str, Any]]:
    """Bulk SMA check for multiple tickers - always uses local data for speed."""
    return get_provider().bulk_get_sma(tickers, period)


def get_current_price(
    ticker: str, force_live: bool = False
) -> Tuple[Optional[float], str]:
    """
    Get current price - auto-selects local vs live based on market hours.

    Args:
        ticker: Stock symbol
        force_live: Force live API even during overnight

    Returns:
        Tuple of (price, source) where source is 'live', 'local', or 'none'
    """
    return get_provider().get_current_price(ticker, force_live=force_live)


def get_data_source_mode() -> str:
    """
    Get current data source mode based on market hours.

    Returns:
        'local_only' - Overnight bulk operations, use local parquet
        'live_prices' - Market hours, use live API for prices
        'hybrid' - Extended hours, prefer live but fallback to local
    """
    return get_data_mode().value
