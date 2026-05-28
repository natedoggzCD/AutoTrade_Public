"""
Momentum Engine - News-driven momentum breakout and continuation discovery.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import pandas as pd
import numpy as np
from autotrade.utils.safe_logging import sanitize_for_console

from autotrade.analysis.news_sentiment import NewsSentimentAnalyzer
from autotrade.utils.local_data_provider import get_provider
from autotrade.signals.contracts import SignalAction, SignalFamily

logger = logging.getLogger("AutoTrade.MomentumEngine")

class SetupType(str, Enum):
    BREAKOUT = "breakout"
    CONTINUATION = "continuation"
    PULLBACK = "pullback"
    WATCH_ONLY = "watch_only"

def safe_cuda_check(obj):
    if obj is None or not hasattr(obj, 'cuda') or not callable(getattr(obj, 'cuda', None)):
        return None
    return obj.cuda()

class MomentumEngine:

    @staticmethod
    def safe_cuda_check(obj):
        if obj is None or not hasattr(obj, 'cuda') or not callable(getattr(obj, 'cuda', None)):
            return None
        return obj.cuda()


    @staticmethod
    def safe_cuda_check(obj):
        if obj is None or not hasattr(obj, 'cuda') or not callable(getattr(obj, 'cuda', None)):
            return None
        return obj.cuda()


    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer(max_news_age_days=1)
        self.data_provider = get_provider()

    @staticmethod
    def safe_cuda_check(obj):
        if obj is None or not hasattr(obj, 'cuda') or not callable(getattr(obj, 'cuda', None)):
            return None
        return obj.cuda()
    """
    Identifies high-conviction news-driven momentum setups.
    """
    
    def __init__(self):
        self.news_analyzer = NewsSentimentAnalyzer(max_news_age_days=1)
        self.data_provider = get_provider()

    @staticmethod
    def _extract_recent_close_pair(df: pd.DataFrame) -> Optional[Tuple[float, float]]:
        """
        Extract previous and latest close values with tolerant column handling.

        Returns:
            (prev_close, latest_close) or None if data is invalid.
        """
        if df is None or df.empty:
            return None

        close_col = None
        for candidate in ("close", "Close", "adj_close", "Adj Close"):
            if candidate in df.columns:
                close_col = candidate
                break

        if close_col is None:
            return None

        close_series = pd.to_numeric(df[close_col], errors="coerce").dropna()
        if len(close_series) < 2:
            return None

        prev_close = float(close_series.iloc[-2])
        latest_close = float(close_series.iloc[-1])
        if prev_close <= 0 or latest_close <= 0:
            return None

        return prev_close, latest_close
        
    def detect_setups(self, tickers: List[str]) -> List[Dict[str, Any]]:
        """
        Scan a list of tickers for news-driven momentum setups.
        """
        setups = []
        for ticker in tickers:
            try:
                setup = self._analyze_ticker(ticker)
                if setup and setup["setup_type"] != SetupType.WATCH_ONLY:
                    setups.append(setup)
            except Exception as e:
                logger.error(sanitize_for_console(f"Failed to analyze {ticker} for momentum: {e}"))
                
        return sorted(setups, key=lambda x: x["setup_confidence"], reverse=True)

    def _analyze_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Perform deep dive on a single ticker for news momentum."""
        # 1. Check News Catalyst
        news_result = self.news_analyzer.analyze_ticker(ticker)
        if news_result.get("news_count", 0) == 0 or news_result.get("sentiment") != "positive":
            return None
            
        # 2. Check Intraday Data
        df = self.data_provider.get_ticker_data(ticker, days=5)
        if df is None or df.empty:
            return None
            
        close_pair = self._extract_recent_close_pair(df)
        if close_pair is None:
            logger.debug(sanitize_for_console(f"Skipping {ticker}: invalid or missing close data"))
            return None

        prev_close, latest_price = close_pair
        if latest_price is None or prev_close is None:
            return None

        day_ret_pct = (latest_price / prev_close - 1) * 100
        if not np.isfinite(day_ret_pct):
            return None
        
        # 3. Classify Setup
        setup_type = SetupType.WATCH_ONLY
        confidence = 0
        
        # Simple classification logic (Phase 1)
        if day_ret_pct > 4.0:
            setup_type = SetupType.BREAKOUT
            confidence = min(90, 50 + day_ret_pct * 2)
        elif day_ret_pct > 1.0:
            setup_type = SetupType.CONTINUATION
            confidence = 60
        elif day_ret_pct < 0 and day_ret_pct > -3.0:
            setup_type = SetupType.PULLBACK
            confidence = 50
            
        return {
            "ticker": ticker,
            "setup_type": setup_type,
            "setup_confidence": confidence,
            "day_ret_pct": day_ret_pct,
            "sentiment_score": news_result.get("sentiment_score", 0),
            "catalyst": news_result.get("headlines", [""])[0] if news_result.get("headlines") else "Positive news"
        }
