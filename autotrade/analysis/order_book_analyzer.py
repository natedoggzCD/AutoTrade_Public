
"""
Order Book Analyzer Utility.
Calculates Bid/Ask Imbalance and detects microstructure patterns like Iceberg orders.
"""

import logging
from typing import Dict, Any
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockSnapshotRequest, StockTradesRequest
from datetime import datetime, timedelta

logger = logging.getLogger("AutoTrade.OrderBookAnalyzer")

class OrderBookAnalyzer:
    @staticmethod
    def get_market_imbalance(symbol: str, data_client: StockHistoricalDataClient) -> Dict[str, Any]:
        """
        Calculate Bid/Ask Imbalance from the latest quote.
        Imbalance = bid_size / (bid_size + ask_size)
        Values > 0.5 indicate bullish bias (more buy interest).
        Values < 0.5 indicate bearish bias (more sell interest).
        """
        if data_client is None or not hasattr(data_client, "get_stock_snapshot"):
            logger.info(
                "L2 imbalance unavailable for %s: data client missing snapshot API",
                symbol,
            )
            return {
                "symbol": symbol,
                "imbalance": 0.5,
                "error": "data_client_unavailable",
                "degraded": True,
            }
        try:
            snapshot = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol))
            if symbol not in snapshot:
                return {"symbol": symbol, "imbalance": 0.5, "error": "No snapshot found"}
            
            quote = snapshot[symbol].latest_quote
            bid_size = quote.bid_size
            ask_size = quote.ask_size
            
            total_size = bid_size + ask_size
            if total_size == 0:
                return {"symbol": symbol, "imbalance": 0.5, "bid_size": 0, "ask_size": 0}
            
            imbalance = bid_size / total_size
            
            return {
                "symbol": symbol,
                "imbalance": float(imbalance),
                "bid_size": float(bid_size),
                "ask_size": float(ask_size),
                "bid_price": float(quote.bid_price),
                "ask_price": float(quote.ask_price),
                "spread": float(quote.ask_price - quote.bid_price),
                "spread_pct": float((quote.ask_price - quote.bid_price) / quote.ask_price * 100) if quote.ask_price > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Error calculating imbalance for {symbol}: {e}")
            return {"symbol": symbol, "imbalance": 0.5, "error": str(e)}

    @staticmethod
    def detect_iceberg(symbol: str, data_client: StockHistoricalDataClient, lookback_minutes: int = 5) -> Dict[str, Any]:
        """
        Heuristic Iceberg Detection:
        Looks for hidden liquidity by comparing traded volume at a price level vs. quoted size.
        If cumulative trade volume at a specific price exceeds the average quoted size by a large margin
        while the quote stays at that level, it suggests an iceberg.
        """
        if (
            data_client is None
            or not hasattr(data_client, "get_stock_trades")
            or not hasattr(data_client, "get_stock_snapshot")
        ):
            logger.info(
                "Iceberg detection unavailable for %s: data client missing trade/snapshot API",
                symbol,
            )
            return {
                "symbol": symbol,
                "iceberg_detected": False,
                "confidence": 0.0,
                "error": "data_client_unavailable",
                "degraded": True,
            }
        try:
            # 1. Get recent trades
            end = datetime.now()
            start = end - timedelta(minutes=lookback_minutes)
            trades_request = StockTradesRequest(
                symbol_or_symbols=symbol,
                start=start,
                end=end,
                limit=500
            )
            trades = data_client.get_stock_trades(trades_request)
            
            if symbol not in trades or trades[symbol].empty:
                return {"symbol": symbol, "iceberg_detected": False, "confidence": 0.0}
            
            df = trades[symbol]
            
            # 2. Get current snapshot for quoted size reference
            snapshot = data_client.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbol))
            if symbol not in snapshot:
                return {"symbol": symbol, "iceberg_detected": False}
                
            quote = snapshot[symbol].latest_quote
            bid_size = quote.bid_size
            ask_size = quote.ask_size
            
            # 3. Analyze trade clusters at bid/ask prices
            bid_price = quote.bid_price
            ask_price = quote.ask_price
            
            # Sum volume of trades happening AT the bid (likely selling into an iceberg)
            # and trades happening AT the ask (likely buying into an iceberg)
            # Use a small tolerance for rounding
            vol_at_bid = df[abs(df['price'] - bid_price) < 0.005]['size'].sum()
            vol_at_ask = df[abs(df['price'] - ask_price) < 0.005]['size'].sum()
            
            iceberg_side = None
            confidence = 0.0
            
            # Heuristic: If volume traded at bid is > 5x the currently visible bid_size
            if bid_size > 0 and vol_at_bid > bid_size * 5:
                iceberg_side = "bid"
                confidence = min(1.0, vol_at_bid / (bid_size * 20))
                
            elif ask_size > 0 and vol_at_ask > ask_size * 5:
                iceberg_side = "ask"
                confidence = min(1.0, vol_at_ask / (ask_size * 20))
                
            return {
                "symbol": symbol,
                "iceberg_detected": iceberg_side is not None,
                "side": iceberg_side,
                "confidence": float(confidence),
                "volume_at_level": float(vol_at_bid if iceberg_side == "bid" else vol_at_ask),
                "quoted_size": float(bid_size if iceberg_side == "bid" else ask_size)
            }
        except Exception as e:
            logger.error(f"Error detecting iceberg for {symbol}: {e}")
            return {"symbol": symbol, "iceberg_detected": False, "error": str(e)}
