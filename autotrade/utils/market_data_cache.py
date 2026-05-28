"""
Market Data Cache - Fetches and caches SPY, QQQ, VIX, IWM data from Alpaca.

Updates once per trading day (or on demand).
Stores in local SQLite for fast access.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from autotrade.utils.alpaca_client_factory import resolve_alpaca_credentials
from autotrade.utils.safe_logging import get_safe_logger

logger = get_safe_logger(__name__)


class MarketDataCache:
    """
    Fetches and caches market-level data (SPY, QQQ, VIX, IWM) from Alpaca API.
    
    Provides:
    - SPY trend analysis (consecutive days, distance from SMAs)
    - VIX level and regime (low/normal/elevated/high)
    - IWM as small-cap breadth proxy (vs SPY)
    - Market breadth indicators
    """
    
    # Symbols to track
    SYMBOLS = ["SPY", "QQQ", "VIXY", "IWM"]  # VIXY is VIX ETF proxy since VIX not tradable
    
    def __init__(self, cache_path: Optional[str] = None, 
                 api_key: Optional[str] = None, 
                 secret_key: Optional[str] = None):
        """
        Initialize with local SQLite cache.
        
        Args:
            cache_path: Path to SQLite cache file
            api_key: Alpaca API key (optional, loads from config if not provided)
            secret_key: Alpaca secret key (optional, loads from config if not provided)
        """
        self.cache_path = cache_path or "data/market_context.db"
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        self.client: Optional[StockHistoricalDataClient] = None
        self._warned_no_client_update = False

        # Initialize Alpaca client with resilient credential lookup:
        # explicit args -> env vars -> config loader.
        try:
            creds = resolve_alpaca_credentials(
                api_key=api_key,
                secret_key=secret_key,
                require=False,
            )
            if creds:
                self.client = StockHistoricalDataClient(
                    api_key=creds.api_key,
                    secret_key=creds.secret_key,
                )
            else:
                logger.warning(
                    "Alpaca API keys not configured; market data cache update is disabled"
                )
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self.client = None

        self._init_cache()
    
    def _init_cache(self):
        """Initialize SQLite cache schema."""
        try:
            conn = sqlite3.connect(self.cache_path)
            cursor = conn.cursor()
            
            # Market data table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    updated_at TEXT,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Indicators table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS market_indicators (
                    date TEXT PRIMARY KEY,
                    spy_sma20 REAL,
                    spy_sma50 REAL,
                    vix_level TEXT,
                    vix_value REAL,
                    iwm_vs_spy_spread REAL,
                    breadth_proxy TEXT,
                    updated_at TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.debug(f"Market data cache initialized: {self.cache_path}")
            
        except Exception as e:
            logger.error(f"Error initializing cache: {e}")
    
    def update(self, days_back: int = 30) -> bool:
        """
        Fetch latest SPY, QQQ, VIXY, IWM bars from Alpaca and cache.
        
        Args:
            days_back: Number of days of history to fetch
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            if not self._warned_no_client_update:
                logger.warning("No Alpaca client available for market data update")
                self._warned_no_client_update = True
            return False
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            
            request = StockBarsRequest(
                symbol_or_symbols=self.SYMBOLS,
                timeframe=TimeFrame.Day,
                start=start_date,
                end=end_date
            )
            
            logger.info(f"Fetching market data for {', '.join(self.SYMBOLS)}...")
            bars = self.client.get_stock_bars(request)
            
            if not bars or not bars.data:
                logger.warning("No market data returned from Alpaca")
                return False
            
            conn = sqlite3.connect(self.cache_path)
            
            for symbol, bar_list in bars.data.items():
                for bar in bar_list:
                    date_str = bar.timestamp.strftime('%Y-%m-%d')
                    conn.execute("""
                        INSERT OR REPLACE INTO market_data 
                        (symbol, date, open, high, low, close, volume, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol, date_str, bar.open, bar.high, bar.low, 
                        bar.close, bar.volume, datetime.now().isoformat()
                    ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Market data cache updated with {len(bars.data)} symbols")
            
            # Compute indicators
            self._compute_indicators()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating market data cache: {e}")
            return False
    
    def _compute_indicators(self):
        """Compute technical indicators from cached data."""
        try:
            conn = sqlite3.connect(self.cache_path)
            
            # Load SPY data
            spy_df = pd.read_sql_query(
                "SELECT date, close, volume FROM market_data WHERE symbol = 'SPY' ORDER BY date",
                conn
            )
            
            if not spy_df.empty:
                spy_df['sma20'] = spy_df['close'].rolling(window=20).mean()
                spy_df['sma50'] = spy_df['close'].rolling(window=50).mean()
                spy_df['close'] = pd.to_numeric(spy_df['close'], errors='coerce')
            
            # Load VIXY data (VIX proxy)
            vix_df = pd.read_sql_query(
                "SELECT date, close FROM market_data WHERE symbol = 'VIXY' ORDER BY date",
                conn
            )
            
            # Load IWM data
            iwm_df = pd.read_sql_query(
                "SELECT date, close FROM market_data WHERE symbol = 'IWM' ORDER BY date",
                conn
            )
            
            # Get latest data
            if not spy_df.empty:
                latest = spy_df.iloc[-1]
                latest_date = latest['date']
                
                # Determine VIX regime
                vix_level = "normal"
                vix_value = None
                if not vix_df.empty:
                    vix_value = float(vix_df.iloc[-1]['close'])
                    if vix_value < 15:
                        vix_level = "low"
                    elif vix_value < 25:
                        vix_level = "normal"
                    elif vix_value < 35:
                        vix_level = "elevated"
                    else:
                        vix_level = "high"
                
                # Compute IWM vs SPY spread
                iwm_vs_spy = None
                if not iwm_df.empty and len(iwm_df) >= 2 and len(spy_df) >= 2:
                    iwm_return = (iwm_df.iloc[-1]['close'] / iwm_df.iloc[-2]['close'] - 1) * 100
                    spy_return = (spy_df.iloc[-1]['close'] / spy_df.iloc[-2]['close'] - 1) * 100
                    iwm_vs_spy = iwm_return - spy_return
                
                # Determine breadth proxy
                breadth_proxy = "neutral"
                if iwm_vs_spy is not None:
                    if iwm_vs_spy > 0.5:
                        breadth_proxy = "risk_on"
                    elif iwm_vs_spy < -0.5:
                        breadth_proxy = "risk_off"
                
                # Store indicators
                conn.execute("""
                    INSERT OR REPLACE INTO market_indicators 
                    (date, spy_sma20, spy_sma50, vix_level, vix_value, 
                     iwm_vs_spy_spread, breadth_proxy, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    latest_date,
                    float(latest['sma20']) if pd.notna(latest['sma20']) else None,
                    float(latest['sma50']) if pd.notna(latest['sma50']) else None,
                    vix_level,
                    vix_value,
                    iwm_vs_spy,
                    breadth_proxy,
                    datetime.now().isoformat()
                ))
                
                conn.commit()
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error computing indicators: {e}")
    
    def get_spy_trend(self, lookback_days: int = 5) -> Dict[str, Any]:
        """
        Get SPY trend analysis.
        
        Args:
            lookback_days: Days to look back for trend
            
        Returns:
            Dict with trend info:
            - trend: "up", "down", or "flat"
            - consecutive_days: Number of consecutive days in trend direction
            - distance_from_sma20: % distance from 20-day SMA
            - distance_from_sma50: % distance from 50-day SMA
        """
        try:
            conn = sqlite3.connect(self.cache_path)
            
            df = pd.read_sql_query(
                f"""SELECT date, close FROM market_data 
                    WHERE symbol = 'SPY' 
                    ORDER BY date DESC 
                    LIMIT {lookback_days + 10}""",
                conn
            )
            
            indicators = pd.read_sql_query(
                "SELECT * FROM market_indicators ORDER BY date DESC LIMIT 1",
                conn
            )
            
            conn.close()
            
            if df.empty:
                return {"trend": "unknown", "consecutive_days": 0}
            
            df = df.sort_values('date')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            df['prev_close'] = df['close'].shift(1)
            df['change'] = df['close'] - df['prev_close']
            
            # Count consecutive days
            consecutive = 0
            trend = "flat"
            
            for _, row in df.iloc[::-1].iterrows():
                if pd.isna(row['change']):
                    continue
                if row['change'] > 0:
                    if trend in ("flat", "up"):
                        trend = "up"
                        consecutive += 1
                    else:
                        break
                elif row['change'] < 0:
                    if trend in ("flat", "down"):
                        trend = "down"
                        consecutive += 1
                    else:
                        break
                else:
                    break
            
            latest_close = float(df.iloc[-1]['close'])
            
            result = {
                "trend": trend,
                "consecutive_days": consecutive,
                "latest_close": latest_close,
                "distance_from_sma20": None,
                "distance_from_sma50": None
            }
            
            if not indicators.empty:
                sma20 = indicators.iloc[0]['spy_sma20']
                sma50 = indicators.iloc[0]['spy_sma50']
                
                if sma20:
                    result['distance_from_sma20'] = ((latest_close / sma20) - 1) * 100
                if sma50:
                    result['distance_from_sma50'] = ((latest_close / sma50) - 1) * 100
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting SPY trend: {e}")
            return {"trend": "unknown", "consecutive_days": 0}
    
    def get_vix_level(self) -> Dict[str, Any]:
        """
        Get VIX (via VIXY) current level and regime.
        
        Returns:
            Dict with VIX info:
            - current: Current VIXY value
            - regime: "low", "normal", "elevated", "high"
            - change_5d: 5-day change %
        """
        try:
            conn = sqlite3.connect(self.cache_path)
            
            df = pd.read_sql_query(
                """SELECT date, close FROM market_data 
                   WHERE symbol = 'VIXY' 
                   ORDER BY date DESC 
                   LIMIT 6""",
                conn
            )
            
            indicators = pd.read_sql_query(
                "SELECT * FROM market_indicators ORDER BY date DESC LIMIT 1",
                conn
            )
            
            conn.close()
            
            if df.empty:
                return {"current": None, "regime": "unknown", "change_5d": None}
            
            df['close'] = pd.to_numeric(df['close'], errors='coerce')
            current = float(df.iloc[0]['close'])
            
            change_5d = None
            if len(df) >= 6:
                prev = float(df.iloc[5]['close'])
                change_5d = ((current / prev) - 1) * 100
            
            result = {
                "current": current,
                "regime": "unknown",
                "change_5d": change_5d
            }
            
            if not indicators.empty:
                result['regime'] = indicators.iloc[0]['vix_level']
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting VIX level: {e}")
            return {"current": None, "regime": "unknown", "change_5d": None}
    
    def get_market_breadth_proxy(self) -> Dict[str, Any]:
        """
        Get IWM (Russell 2000) as small-cap breadth proxy.
        
        Compare IWM vs SPY for risk-on/risk-off signal.
        - IWM outperforming = risk-on (bullish for our universe)
        - IWM underperforming = risk-off (defensive)
        
        Returns:
            Dict with breadth info:
            - iwm_return_1d: IWM 1-day return %
            - spy_return_1d: SPY 1-day return %
            - spread: IWM - SPY spread %
            - regime: "risk_on", "neutral", "risk_off"
        """
        try:
            conn = sqlite3.connect(self.cache_path)
            
            iwm_df = pd.read_sql_query(
                """SELECT date, close FROM market_data 
                   WHERE symbol = 'IWM' 
                   ORDER BY date DESC 
                   LIMIT 2""",
                conn
            )
            
            spy_df = pd.read_sql_query(
                """SELECT date, close FROM market_data 
                   WHERE symbol = 'SPY' 
                   ORDER BY date DESC 
                   LIMIT 2""",
                conn
            )
            
            indicators = pd.read_sql_query(
                "SELECT * FROM market_indicators ORDER BY date DESC LIMIT 1",
                conn
            )
            
            conn.close()
            
            if iwm_df.empty or spy_df.empty or len(iwm_df) < 2 or len(spy_df) < 2:
                return {
                    "iwm_return_1d": None,
                    "spy_return_1d": None,
                    "spread": None,
                    "regime": "unknown"
                }
            
            iwm_df['close'] = pd.to_numeric(iwm_df['close'], errors='coerce')
            spy_df['close'] = pd.to_numeric(spy_df['close'], errors='coerce')
            
            iwm_return = ((iwm_df.iloc[0]['close'] / iwm_df.iloc[1]['close']) - 1) * 100
            spy_return = ((spy_df.iloc[0]['close'] / spy_df.iloc[1]['close']) - 1) * 100
            spread = iwm_return - spy_return
            
            regime = "neutral"
            if spread > 0.5:
                regime = "risk_on"
            elif spread < -0.5:
                regime = "risk_off"
            
            return {
                "iwm_return_1d": round(iwm_return, 2),
                "spy_return_1d": round(spy_return, 2),
                "spread": round(spread, 2),
                "regime": regime
            }
            
        except Exception as e:
            logger.error(f"Error getting market breadth proxy: {e}")
            return {
                "iwm_return_1d": None,
                "spy_return_1d": None,
                "spread": None,
                "regime": "unknown"
            }
    
    def get_full_market_context(self) -> Dict[str, Any]:
        """
        Get complete market context from cache.
        
        Returns:
            Dict with all market indicators
        """
        spy_trend = self.get_spy_trend()
        vix = self.get_vix_level()
        breadth = self.get_market_breadth_proxy()
        
        # Determine overall market regime
        overall_regime = "neutral"
        
        if vix.get('regime') == 'high' or spy_trend.get('trend') == 'down' and spy_trend.get('consecutive_days', 0) >= 3:
            overall_regime = "risk_off"
        elif breadth.get('regime') == 'risk_on' and spy_trend.get('trend') == 'up':
            overall_regime = "risk_on"
        elif vix.get('regime') == 'elevated':
            overall_regime = "caution"
        
        return {
            "timestamp": datetime.now().isoformat(),
            "spy": spy_trend,
            "vix": vix,
            "breadth": breadth,
            "overall_regime": overall_regime,
            "cache_age_hours": self._get_cache_age_hours()
        }
    
    def _get_cache_age_hours(self) -> float:
        """Get age of cache in hours."""
        try:
            conn = sqlite3.connect(self.cache_path)
            result = conn.execute(
                "SELECT MAX(updated_at) FROM market_data"
            ).fetchone()
            conn.close()
            
            if result and result[0]:
                last_update = datetime.fromisoformat(result[0])
                age = (datetime.now() - last_update).total_seconds() / 3600
                return round(age, 1)
            
            return float('inf')
            
        except:
            return float('inf')
    
    def needs_update(self, max_age_hours: int = 6) -> bool:
        """
        Check if cache needs updating.
        
        Args:
            max_age_hours: Maximum acceptable cache age
            
        Returns:
            True if update needed
        """
        if not self.client:
            return False
        age = self._get_cache_age_hours()
        return age > max_age_hours


# Singleton instance
_cache: Optional[MarketDataCache] = None


def get_market_data_cache(**kwargs) -> MarketDataCache:
    """
    Get or create singleton MarketDataCache instance.
    
    Returns:
        MarketDataCache instance
    """
    global _cache
    if _cache is None:
        _cache = MarketDataCache(**kwargs)
    return _cache


def get_market_context() -> Dict[str, Any]:
    """
    Convenience function to get full market context.
    
    Returns:
        Dict with market indicators
    """
    cache = get_market_data_cache()
    
    if cache.needs_update():
        cache.update()
    
    return cache.get_full_market_context()
