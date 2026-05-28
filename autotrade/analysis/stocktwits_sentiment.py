"""
Stocktwits Sentiment Integration
=================================
Real-time social sentiment analysis from Stocktwits.

Features:
- Stocktwits API integration (no auth for basic access)
- Message volume and sentiment trending
- Bull/bear ratio calculation
- Watchlist tracking
- Sentiment alerts
- Integration with existing news_sentiment module

Usage:
    from autotrade.analysis.stocktwits_sentiment import StockTwitsSentiment
    
    st = StockTwitsSentiment()
    sentiment = st.get_sentiment("AAPL")
"""

import logging
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import threading
from collections import deque

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# Try curl_cffi for TLS fingerprint impersonation (bypasses 403)
try:
    from curl_cffi import requests as cffi_requests
    CURL_CFFI_AVAILABLE = True
except ImportError:
    CURL_CFFI_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StocktwitsMessage:
    """Single Stocktwits message."""
    id: int
    body: str
    sentiment: Optional[str]  # "Bullish" or "Bearish" or None
    created_at: datetime
    user: str
    followers: int = 0
    likes: int = 0
    
    @property
    def sentiment_value(self) -> float:
        """Convert sentiment to numeric value."""
        if self.sentiment == "Bullish":
            return 1.0
        elif self.sentiment == "Bearish":
            return -1.0
        return 0.0


@dataclass
class StocktwitsSentiment:
    """Sentiment analysis result from Stocktwits."""
    symbol: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Message counts
    total_messages: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0
    
    # Sentiment metrics
    sentiment_score: float = 0.0  # -100 to 100
    bull_bear_ratio: float = 1.0
    message_velocity: float = 0.0  # messages per hour
    
    # Trending info
    is_trending: bool = False
    trending_rank: Optional[int] = None
    watchlist_count: Optional[int] = None
    
    # Recent messages
    recent_messages: List[StocktwitsMessage] = field(default_factory=list)
    top_bullish: Optional[str] = None
    top_bearish: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'total_messages': self.total_messages,
            'bullish_count': self.bullish_count,
            'bearish_count': self.bearish_count,
            'sentiment_score': self.sentiment_score,
            'bull_bear_ratio': self.bull_bear_ratio,
            'message_velocity': self.message_velocity,
            'is_trending': self.is_trending,
            'top_bullish': self.top_bullish,
            'top_bearish': self.top_bearish
        }


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 200, period_seconds: int = 3600):
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls: deque = deque()
        self._lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Try to acquire a rate limit slot."""
        now = time.time()
        
        with self._lock:
            # Remove old calls
            cutoff = now - self.period_seconds
            while self.calls and self.calls[0] < cutoff:
                self.calls.popleft()
            
            # Check limit
            if len(self.calls) >= self.max_calls:
                return False
            
            self.calls.append(now)
            return True
    
    def wait_if_needed(self):
        """Wait if rate limited."""
        while not self.acquire():
            time.sleep(1)


class SentimentCache:
    """Cache for Stocktwits sentiment data."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 min default
        self._cache: Dict[str, Tuple[StocktwitsSentiment, float]] = {}
        self.ttl_seconds = ttl_seconds
        self._lock = threading.Lock()
    
    def get(self, symbol: str) -> Optional[StocktwitsSentiment]:
        """Get cached sentiment if valid."""
        with self._lock:
            if symbol in self._cache:
                sentiment, timestamp = self._cache[symbol]
                if time.time() - timestamp < self.ttl_seconds:
                    return sentiment
                del self._cache[symbol]
        return None
    
    def set(self, symbol: str, sentiment: StocktwitsSentiment):
        """Cache sentiment."""
        with self._lock:
            self._cache[symbol] = (sentiment, time.time())
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()


class StocktwitsSentimentAnalyzer:
    """
    Stocktwits sentiment analyzer.
    
    Uses the public Stocktwits API to get social sentiment data.
    No authentication required for basic symbol streams.
    """
    
    BASE_URL = "https://api.stocktwits.com/api/2"
    
    def __init__(
        self,
        cache_ttl: int = 300,
        rate_limit: int = 200,
        include_messages: bool = True
    ):
        """
        Initialize Stocktwits analyzer.
        
        Args:
            cache_ttl: Cache TTL in seconds
            rate_limit: Max API calls per hour
            include_messages: Whether to include recent messages
        """
        self.cache = SentimentCache(ttl_seconds=cache_ttl)
        self.rate_limiter = RateLimiter(max_calls=rate_limit)
        self.include_messages = include_messages
        
        # Stats
        self._api_calls = 0
        self._cache_hits = 0
        self._errors = 0
        
        logger.info(f"StocktwitsSentimentAnalyzer initialized (cache_ttl={cache_ttl}s)")
        
        # Create session with browser-like headers to avoid 403
        self._session = requests.Session()
        self._session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Origin': 'https://stocktwits.com',
            'Referer': 'https://stocktwits.com/',
            'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="121", "Chromium";v="121"',
            'Sec-Ch-Ua-Mobile': '?0',
            'Sec-Ch-Ua-Platform': '"Windows"',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site',
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with TLS fingerprint impersonation to bypass 403."""
        self.rate_limiter.wait_if_needed()
        
        url = f"{self.BASE_URL}/{endpoint}"
        if params:
            param_str = '&'.join(f"{k}={v}" for k, v in params.items())
            url = f"{url}?{param_str}"
        
        # Try curl_cffi first (impersonates Chrome TLS fingerprint)
        if CURL_CFFI_AVAILABLE:
            try:
                response = cffi_requests.get(url, impersonate="chrome120", timeout=10)
                if response.status_code == 200:
                    self._api_calls += 1
                    return response.json()
                elif response.status_code == 429:
                    logger.warning("Stocktwits rate limit hit, waiting...")
                    time.sleep(60)
                    return self._make_request(endpoint, params)
                else:
                    logger.warning(f"HTTP {response.status_code} from StockTwits")
                    self._errors += 1
                    return None
            except Exception as e:
                logger.debug(f"curl_cffi error: {e}, falling back to requests")
        
        # Fallback to regular requests with session
        if REQUESTS_AVAILABLE:
            try:
                response = self._session.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'application/json',
                    'Referer': 'https://stocktwits.com/',
                }, timeout=10)
                response.raise_for_status()
                self._api_calls += 1
                return response.json()
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:
                    logger.warning("Stocktwits rate limit hit, waiting...")
                    time.sleep(60)
                    return self._make_request(endpoint, params)
                logger.warning(f"HTTP error: {e}")
                self._errors += 1
                return None
                
            except Exception as e:
                logger.warning(f"Request error: {e}")
                self._errors += 1
                return None
        
        logger.warning("No HTTP library available")
        return None
    
    def get_symbol_stream(self, symbol: str, limit: int = 30) -> Optional[Dict]:
        """
        Get message stream for a symbol.
        
        Args:
            symbol: Stock ticker
            limit: Number of messages to fetch
            
        Returns:
            API response dict or None
        """
        return self._make_request(
            f"streams/symbol/{symbol}.json",
            params={"limit": limit}
        )
    
    def get_trending(self) -> Optional[List[Dict]]:
        """Get trending symbols."""
        data = self._make_request("trending/symbols.json")
        if data and 'symbols' in data:
            return data['symbols']
        return None
    
    def parse_messages(self, messages: List[Dict]) -> List[StocktwitsMessage]:
        """Parse raw API messages."""
        parsed = []
        
        for msg in messages:
            try:
                sentiment = None
                if 'entities' in msg and 'sentiment' in msg['entities']:
                    sentiment = msg['entities']['sentiment'].get('basic')
                
                created_at = datetime.strptime(
                    msg['created_at'],
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                
                parsed.append(StocktwitsMessage(
                    id=msg['id'],
                    body=msg.get('body', '')[:500],  # Limit length
                    sentiment=sentiment,
                    created_at=created_at,
                    user=msg.get('user', {}).get('username', 'unknown'),
                    followers=msg.get('user', {}).get('followers', 0),
                    likes=msg.get('likes', {}).get('total', 0)
                ))
                
            except Exception as e:
                logger.debug(f"Error parsing message: {e}")
                continue
        
        return parsed
    
    def calculate_sentiment(
        self,
        messages: List[StocktwitsMessage],
        weight_by_followers: bool = True
    ) -> Tuple[float, float]:
        """
        Calculate sentiment score from messages.
        
        Args:
            messages: List of parsed messages
            weight_by_followers: Weight by user followers
            
        Returns:
            Tuple of (sentiment_score, bull_bear_ratio)
        """
        if not messages:
            return 0.0, 1.0
        
        bullish = 0.0
        bearish = 0.0
        
        for msg in messages:
            weight = 1.0
            if weight_by_followers:
                # Log scale weight based on followers
                weight = 1 + min(5, (msg.followers / 1000) ** 0.5)
            
            if msg.sentiment == "Bullish":
                bullish += weight
            elif msg.sentiment == "Bearish":
                bearish += weight
        
        total = bullish + bearish
        if total == 0:
            return 0.0, 1.0
        
        # Sentiment score: -100 to 100
        sentiment_score = ((bullish - bearish) / total) * 100
        
        # Bull/bear ratio
        bull_bear_ratio = bullish / bearish if bearish > 0 else (10 if bullish > 0 else 1)
        
        return sentiment_score, bull_bear_ratio
    
    def calculate_velocity(self, messages: List[StocktwitsMessage]) -> float:
        """Calculate message velocity (messages per hour)."""
        if len(messages) < 2:
            return 0.0
        
        # Sort by time
        sorted_msgs = sorted(messages, key=lambda m: m.created_at)
        
        # Time span
        time_span = (sorted_msgs[-1].created_at - sorted_msgs[0].created_at).total_seconds()
        if time_span <= 0:
            return 0.0
        
        # Messages per hour
        return len(messages) / (time_span / 3600)
    
    def get_sentiment(
        self,
        symbol: str,
        use_cache: bool = True
    ) -> StocktwitsSentiment:
        """
        Get sentiment analysis for a symbol.
        
        Args:
            symbol: Stock ticker
            use_cache: Whether to use cached results
            
        Returns:
            StocktwitsSentiment object
        """
        symbol = symbol.upper()
        
        # Check cache
        if use_cache:
            cached = self.cache.get(symbol)
            if cached:
                self._cache_hits += 1
                return cached
        
        # Fetch from API
        data = self.get_symbol_stream(symbol)
        
        if not data or 'messages' not in data:
            return StocktwitsSentiment(symbol=symbol)
        
        # Parse messages
        messages = self.parse_messages(data['messages'])
        
        # Count sentiments
        bullish = sum(1 for m in messages if m.sentiment == "Bullish")
        bearish = sum(1 for m in messages if m.sentiment == "Bearish")
        neutral = len(messages) - bullish - bearish
        
        # Calculate metrics
        sentiment_score, bull_bear_ratio = self.calculate_sentiment(messages)
        velocity = self.calculate_velocity(messages)
        
        # Get symbol info
        symbol_info = data.get('symbol', {})
        is_trending = symbol_info.get('trending', False)
        watchlist_count = symbol_info.get('watchlist_count')
        
        # Top messages
        top_bullish = None
        top_bearish = None
        
        bullish_msgs = [m for m in messages if m.sentiment == "Bullish"]
        bearish_msgs = [m for m in messages if m.sentiment == "Bearish"]
        
        if bullish_msgs:
            top_bull = max(bullish_msgs, key=lambda m: m.likes + m.followers)
            top_bullish = top_bull.body[:200]
        
        if bearish_msgs:
            top_bear = max(bearish_msgs, key=lambda m: m.likes + m.followers)
            top_bearish = top_bear.body[:200]
        
        result = StocktwitsSentiment(
            symbol=symbol,
            total_messages=len(messages),
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            sentiment_score=sentiment_score,
            bull_bear_ratio=bull_bear_ratio,
            message_velocity=velocity,
            is_trending=is_trending,
            watchlist_count=watchlist_count,
            recent_messages=messages if self.include_messages else [],
            top_bullish=top_bullish,
            top_bearish=top_bearish
        )
        
        # Cache result
        self.cache.set(symbol, result)
        
        logger.info(
            f"Stocktwits {symbol}: score={sentiment_score:+.0f}, "
            f"ratio={bull_bear_ratio:.1f}, velocity={velocity:.1f}/hr"
        )
        
        return result
    
    def get_batch_sentiment(
        self,
        symbols: List[str]
    ) -> Dict[str, StocktwitsSentiment]:
        """
        Get sentiment for multiple symbols.
        
        Args:
            symbols: List of tickers
            
        Returns:
            Dict mapping symbol to sentiment
        """
        results = {}
        for symbol in symbols:
            try:
                results[symbol] = self.get_sentiment(symbol)
                time.sleep(0.5)  # Be nice to API
            except Exception as e:
                logger.warning(f"Error getting sentiment for {symbol}: {e}")
                results[symbol] = StocktwitsSentiment(symbol=symbol)
        
        return results
    
    def get_trending_symbols(self) -> List[Dict[str, Any]]:
        """
        Get currently trending symbols on Stocktwits.
        
        Returns:
            List of trending symbol info
        """
        trending = self.get_trending()
        
        if not trending:
            return []
        
        results = []
        for item in trending[:20]:  # Top 20
            results.append({
                'symbol': item.get('symbol'),
                'title': item.get('title'),
                'watchlist_count': item.get('watchlist_count'),
                'trending_score': item.get('trending_score', 0)
            })
        
        return results
    
    def is_abnormal_activity(
        self,
        symbol: str,
        velocity_threshold: float = 50,
        ratio_threshold: float = 3.0
    ) -> Tuple[bool, str]:
        """
        Check for abnormal social activity.
        
        Args:
            symbol: Stock ticker
            velocity_threshold: Messages/hour threshold
            ratio_threshold: Bull/bear ratio threshold
            
        Returns:
            Tuple of (is_abnormal, reason)
        """
        sentiment = self.get_sentiment(symbol)
        
        reasons = []
        
        if sentiment.message_velocity > velocity_threshold:
            reasons.append(f"High message velocity: {sentiment.message_velocity:.0f}/hr")
        
        if sentiment.bull_bear_ratio > ratio_threshold:
            reasons.append(f"Extreme bullish: {sentiment.bull_bear_ratio:.1f}x")
        elif sentiment.bull_bear_ratio < (1 / ratio_threshold):
            reasons.append(f"Extreme bearish: {sentiment.bull_bear_ratio:.2f}x")
        
        if sentiment.is_trending:
            reasons.append("Symbol is trending")
        
        if abs(sentiment.sentiment_score) > 70:
            direction = "bullish" if sentiment.sentiment_score > 0 else "bearish"
            reasons.append(f"Extreme {direction} sentiment: {sentiment.sentiment_score:+.0f}")
        
        is_abnormal = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else "Normal activity"
        
        return is_abnormal, reason
    
    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        return {
            'api_calls': self._api_calls,
            'cache_hits': self._cache_hits,
            'errors': self._errors,
            'cache_hit_rate': (
                self._cache_hits / (self._api_calls + self._cache_hits) * 100
                if (self._api_calls + self._cache_hits) > 0 else 0
            )
        }


# ============================================================
# Integration with news_sentiment module
# ============================================================

def combine_sentiment(
    finbert_score: float,
    stocktwits_sentiment: StocktwitsSentiment,
    finbert_weight: float = 0.6,
    stocktwits_weight: float = 0.4
) -> Dict[str, Any]:
    """
    Combine FinBERT and Stocktwits sentiment.
    
    Args:
        finbert_score: FinBERT sentiment (-1 to 1)
        stocktwits_sentiment: StocktwitsSentiment object
        finbert_weight: Weight for FinBERT
        stocktwits_weight: Weight for Stocktwits
        
    Returns:
        Combined sentiment analysis
    """
    # Normalize to same scale (-100 to 100)
    finbert_normalized = finbert_score * 100
    st_normalized = stocktwits_sentiment.sentiment_score
    
    # Weighted average
    combined_score = (
        finbert_normalized * finbert_weight +
        st_normalized * stocktwits_weight
    )
    
    # Agreement check
    finbert_direction = "bullish" if finbert_score > 0.1 else (
        "bearish" if finbert_score < -0.1 else "neutral"
    )
    st_direction = "bullish" if st_normalized > 10 else (
        "bearish" if st_normalized < -10 else "neutral"
    )
    
    agreement = finbert_direction == st_direction
    
    # Confidence based on agreement
    confidence = 80 if agreement else 50
    
    # Adjust for social velocity (high activity = more weight to social)
    if stocktwits_sentiment.message_velocity > 30:
        confidence += 10
    
    return {
        'combined_score': combined_score,
        'finbert_score': finbert_normalized,
        'stocktwits_score': st_normalized,
        'direction': "bullish" if combined_score > 10 else (
            "bearish" if combined_score < -10 else "neutral"
        ),
        'agreement': agreement,
        'confidence': min(100, confidence),
        'social_velocity': stocktwits_sentiment.message_velocity,
        'is_trending': stocktwits_sentiment.is_trending,
        'bull_bear_ratio': stocktwits_sentiment.bull_bear_ratio
    }


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('StocktwitsSentiment module loaded; use pytest/CLI for tests.')

