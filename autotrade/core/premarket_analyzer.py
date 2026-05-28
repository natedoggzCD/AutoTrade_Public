"""
Pre-Market Analyzer
====================
Fetches and analyzes pre-market data for trading decisions.
Uses Alpaca's data feed for real pre-market prices.

Key features:
- Pre-market price & volume
- Gap analysis (vs previous close)
- Pre-market trend detection
- Liquidity assessment

Run at 6:30 AM or integrate into morning workflow.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd
from autotrade.utils.alpaca_client_factory import (
    create_data_client,
    create_trading_client,
    resolve_alpaca_credentials,
)
from autotrade.monitoring.liquidity_gate import LiquidityGate

# Load environment
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))
env_path = PROJECT_DIR / '.env'
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if '=' in line and not line.startswith('#'):
            key, val = line.split('=', 1)
            os.environ[key.strip()] = val.strip()

logger = logging.getLogger('premarket')


@dataclass
class PreMarketData:
    """Pre-market data for a single ticker."""
    ticker: str
    
    # Prices
    prev_close: float = 0.0
    premarket_price: float = 0.0
    premarket_high: float = 0.0
    premarket_low: float = 0.0
    premarket_open: float = 0.0
    
    # Gap analysis
    gap_pct: float = 0.0
    gap_direction: str = "flat"  # up, down, flat
    
    # Volume
    premarket_volume: int = 0
    avg_volume: int = 0
    volume_ratio: float = 0.0  # premarket vs avg daily
    
    # Trend
    premarket_trend: str = "flat"  # bullish, bearish, flat
    premarket_range_pct: float = 0.0
    
    # Quality
    has_data: bool = False
    liquidity_score: float = 0.0  # 0-100
    spread_pct: float = 0.0
    is_tradable: bool = True
    liquidity_block_reason: str = ""
    
    # Timestamps
    last_update: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PreMarketAnalyzer:
    """
    Analyzes pre-market data for watchlist stocks.
    
    Pre-market hours: 4:00 AM - 9:30 AM ET
    """
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        """Initialize with Alpaca credentials."""
        self.base_url = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
        default_paper = 'paper' in self.base_url.lower()
        creds = resolve_alpaca_credentials(
            api_key=api_key,
            secret_key=api_secret,
            paper=default_paper,
            require=False,
        )
        self.api_key = creds.api_key if creds else None
        self.api_secret = creds.secret_key if creds else None
        self.is_paper = bool(creds.paper) if creds else bool(default_paper)
        
        # Alpaca clients
        self.api = None
        self.data_api = None
        self.liquidity_gate = LiquidityGate()
        self._init_clients()
    
    def _init_clients(self):
        """Initialize Alpaca API clients."""
        try:
            self.api = create_trading_client(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.is_paper,
                validate_connection=True,
                retries=3,
                retry_delay_seconds=2.0,
                logger=logger,
                require_credentials=True,
            )

            self.data_api = create_data_client(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.is_paper,
                require_credentials=True,
            )
            
            logger.debug("Alpaca clients initialized for pre-market data")
            
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca clients: {e}")
            raise
    
    def get_premarket_quote(self, ticker: str) -> Optional[Dict]:
        """Get latest pre-market quote for a ticker."""
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
            quote = self.data_api.get_stock_latest_quote(request)
            
            if ticker in quote:
                q = quote[ticker]
                return {
                    'bid': float(q.bid_price) if q.bid_price else 0,
                    'ask': float(q.ask_price) if q.ask_price else 0,
                    'mid': (float(q.bid_price) + float(q.ask_price)) / 2 if q.bid_price and q.ask_price else 0,
                    'bid_size': int(q.bid_size) if q.bid_size else 0,
                    'ask_size': int(q.ask_size) if q.ask_size else 0,
                    'timestamp': str(q.timestamp)
                }
            return None
        except Exception as e:
            logger.debug(f"Quote error for {ticker}: {e}")
            return None
    
    def get_premarket_bars(self, ticker: str) -> pd.DataFrame:
        """Get pre-market bars for today."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Pre-market starts 4 AM ET
            now = datetime.now()
            today_start = now.replace(hour=4, minute=0, second=0, microsecond=0)
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Minute,
                start=today_start
            )
            
            bars = self.data_api.get_stock_bars(request)
            
            if ticker in bars and len(bars[ticker]) > 0:
                data = []
                for bar in bars[ticker]:
                    data.append({
                        'timestamp': bar.timestamp,
                        'open': float(bar.open),
                        'high': float(bar.high),
                        'low': float(bar.low),
                        'close': float(bar.close),
                        'volume': int(bar.volume)
                    })
                return pd.DataFrame(data)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.debug(f"Bars error for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_previous_close(self, ticker: str) -> float:
        """Get previous day's closing price."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            # Get yesterday's daily bar
            end = datetime.now().replace(hour=0, minute=0)
            start = end - timedelta(days=5)
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            bars = self.data_api.get_stock_bars(request)
            
            if ticker in bars and len(bars[ticker]) > 0:
                return float(bars[ticker][-1].close)
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"Previous close error for {ticker}: {e}")
            return 0.0
    
    def get_avg_volume(self, ticker: str, days: int = 20) -> int:
        """Get average daily volume."""
        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            
            end = datetime.now()
            start = end - timedelta(days=days + 5)  # Extra buffer
            
            request = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end
            )
            
            bars = self.data_api.get_stock_bars(request)
            
            if ticker in bars and len(bars[ticker]) > 0:
                volumes = [int(bar.volume) for bar in bars[ticker]]
                return int(sum(volumes) / len(volumes))
            
            return 0
            
        except Exception as e:
            logger.debug(f"Avg volume error for {ticker}: {e}")
            return 0
    
    def analyze_ticker(self, ticker: str) -> PreMarketData:
        """
        Full pre-market analysis for a single ticker.
        
        Returns PreMarketData with all metrics.
        """
        pm = PreMarketData(ticker=ticker)
        
        try:
            # Get previous close
            pm.prev_close = self.get_previous_close(ticker)
            
            # Get current quote (pre-market or extended hours)
            quote = self.get_premarket_quote(ticker)
            if quote and quote['mid'] > 0:
                pm.premarket_price = quote['mid']
                pm.has_data = True
                pm.last_update = quote['timestamp']
            
            # Get pre-market bars for OHLC
            bars = self.get_premarket_bars(ticker)
            if not bars.empty:
                pm.premarket_open = float(bars.iloc[0]['open'])
                pm.premarket_high = float(bars['high'].max())
                pm.premarket_low = float(bars['low'].min())
                pm.premarket_volume = int(bars['volume'].sum())
                
                # Use latest bar close if no quote
                if pm.premarket_price == 0:
                    pm.premarket_price = float(bars.iloc[-1]['close'])
                    pm.has_data = True
                
                # Pre-market range
                if pm.premarket_low > 0:
                    pm.premarket_range_pct = (pm.premarket_high - pm.premarket_low) / pm.premarket_low * 100
                
                # Pre-market trend (compare open to latest)
                if pm.premarket_open > 0:
                    pm_change = (pm.premarket_price - pm.premarket_open) / pm.premarket_open * 100
                    if pm_change > 0.5:
                        pm.premarket_trend = "bullish"
                    elif pm_change < -0.5:
                        pm.premarket_trend = "bearish"
                    else:
                        pm.premarket_trend = "flat"
            
            # Gap calculation
            if pm.prev_close > 0 and pm.premarket_price > 0:
                pm.gap_pct = (pm.premarket_price - pm.prev_close) / pm.prev_close * 100
                if pm.gap_pct > 0.5:
                    pm.gap_direction = "up"
                elif pm.gap_pct < -0.5:
                    pm.gap_direction = "down"
                else:
                    pm.gap_direction = "flat"
            
            # Volume analysis
            pm.avg_volume = self.get_avg_volume(ticker)
            if pm.avg_volume > 0 and pm.premarket_volume > 0:
                # Pre-market is ~10% of day volume on average
                expected_pm_vol = pm.avg_volume * 0.1
                pm.volume_ratio = pm.premarket_volume / expected_pm_vol
            
            # Liquidity score (0-100)
            # Based on pre-market volume and bid-ask spread
            if pm.premarket_volume > 0:
                # Volume component (up to 50 points)
                vol_score = min(pm.volume_ratio * 25, 50)
                
                # Spread component (up to 50 points) - needs quote
                spread_score = 50  # Default
                if quote and quote['bid'] > 0 and quote['ask'] > 0:
                    spread_pct = (quote['ask'] - quote['bid']) / quote['mid'] * 100
                    pm.spread_pct = float(spread_pct)
                    if spread_pct < 0.1:
                        spread_score = 50
                    elif spread_pct < 0.3:
                        spread_score = 40
                    elif spread_pct < 0.5:
                        spread_score = 30
                    elif spread_pct < 1.0:
                        spread_score = 20
                    else:
                        spread_score = 10
                
                pm.liquidity_score = vol_score + spread_score

            decision = self.liquidity_gate.evaluate(
                price=pm.premarket_price or pm.prev_close,
                bid_price=(quote or {}).get("bid", 0.0),
                ask_price=(quote or {}).get("ask", 0.0),
                avg_volume=pm.avg_volume,
                session_volume=pm.premarket_volume,
            )
            pm.is_tradable = bool(decision.tradable)
            pm.liquidity_block_reason = str(decision.reason)
            if decision.spread_pct > 0:
                pm.spread_pct = float(decision.spread_pct)
            
        except Exception as e:
            logger.error(f"Pre-market analysis failed for {ticker}: {e}")
        
        return pm
    
    def analyze_watchlist(self, tickers: List[str]) -> List[PreMarketData]:
        """Analyze pre-market data for a list of tickers."""
        results = []
        for ticker in tickers:
            pm = self.analyze_ticker(ticker)
            results.append(pm)
            
            # Log significant gaps
            if pm.has_data:
                if abs(pm.gap_pct) > 2:
                    direction = "GAP UP" if pm.gap_pct > 0 else "GAP DOWN"
                    logger.info(f"   [{direction}] {ticker}: {pm.gap_pct:+.1f}%, PM trend: {pm.premarket_trend}")
        
        return results
    
    def get_gap_ups(self, tickers: List[str], min_gap: float = 1.0) -> List[PreMarketData]:
        """Get stocks gapping up in pre-market."""
        all_data = self.analyze_watchlist(tickers)
        return [pm for pm in all_data if pm.gap_pct >= min_gap]
    
    def get_gap_downs(self, tickers: List[str], min_gap: float = 1.0) -> List[PreMarketData]:
        """Get stocks gapping down in pre-market."""
        all_data = self.analyze_watchlist(tickers)
        return [pm for pm in all_data if pm.gap_pct <= -min_gap]


def demo():
    """Demo the pre-market analyzer (logging-only)."""
    logger.info("PreMarketAnalyzer demo disabled; use CLI/tests for execution.")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('premarket_analyzer.py loaded; use pytest/CLI for tests.')
