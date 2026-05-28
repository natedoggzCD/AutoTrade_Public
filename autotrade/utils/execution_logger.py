from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

class OrderLifecycleLogger:
    def __init__(self):
        self._logs: Dict[str, Dict[str, Any]] = {}

    def log_signal(
        self,
        signal_id: str,
        symbol: str,
        arrival_price: float,
        nbbo: Optional[Tuple[float, float]] = None
    ) -> None:
        """Initialize a lifecycle log for a signal."""
        self._logs[signal_id] = {
            "symbol": symbol,
            "arrival_price": arrival_price,
            "nbbo": nbbo,
            "signal_time": datetime.now(),
            "events": {}
        }

    def log_event(
        self,
        signal_id: str,
        event_name: str,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Log a specific event in the order lifecycle."""
        if signal_id not in self._logs:
            raise KeyError(f"Signal ID {signal_id} not found in logs.")
        
        if timestamp is None:
            timestamp = datetime.now()
            
        self._logs[signal_id]["events"][event_name] = timestamp

    def get_log(self, signal_id: str) -> Dict[str, Any]:
        """Retrieve the lifecycle log for a given signal."""
        if signal_id not in self._logs:
            raise KeyError(f"Signal ID {signal_id} not found in logs.")
        return self._logs[signal_id]

    def get_all_logs(self) -> Dict[str, Dict[str, Any]]:
        """Retrieve all lifecycle logs."""
        return self._logs

    def save_logs(self, file_path: str) -> None:
        """Save logs to a JSON file."""
        import json
        import os
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w') as f:
                # Convert datetime objects to strings for JSON
                serializable_logs = {}
                for oid, data in self._logs.items():
                    s_data = data.copy()
                    if "signal_time" in s_data and isinstance(s_data["signal_time"], datetime):
                        s_data["signal_time"] = s_data["signal_time"].isoformat()
                    if "events" in s_data:
                        s_events = {}
                        for e_name, e_time in s_data["events"].items():
                            if isinstance(e_time, datetime):
                                s_events[e_name] = e_time.isoformat()
                            else:
                                s_events[e_name] = e_time
                        s_data["events"] = s_events
                    serializable_logs[oid] = s_data
                
                json.dump(serializable_logs, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save execution logs: {e}")

    def load_logs(self, file_path: str) -> None:
        """Load logs from a JSON file."""
        import json
        from dateutil.parser import parse
        try:
            if not os.path.exists(file_path):
                return
            with open(file_path, 'r') as f:
                loaded_data = json.load(f)
                for oid, data in loaded_data.items():
                    if "signal_time" in data and isinstance(data["signal_time"], str):
                        data["signal_time"] = parse(data["signal_time"])
                    if "events" in data:
                        for e_name, e_time in data["events"].items():
                            if isinstance(e_time, str):
                                data["events"][e_name] = parse(e_time)
                    self._logs[oid] = data
        except Exception as e:
            logger.error(f"Failed to load execution logs: {e}")

    def capture_arrival_price(self, ticker: str, context: Any) -> float:
        """
        Capture the arrival price benchmark (Bar-Close Benchmark).
        Takes the latest available close price from the context's price data.
        """
        import pandas as pd
        
        price_data = getattr(context, "price_data", None)
        if price_data is None or not isinstance(price_data, pd.DataFrame) or price_data.empty:
            return 0.0
            
        # Handle multi-index (symbol, timestamp) vs single index
        if isinstance(price_data.index, pd.MultiIndex):
            try:
                ticker_data = price_data.xs(ticker, level=0)
            except (KeyError, ValueError):
                return 0.0
        else:
            ticker_data = price_data
            
        if ticker_data.empty or "close" not in ticker_data.columns:
            return 0.0
            
        return float(ticker_data["close"].iloc[-1])

    def capture_nbbo(self, ticker: str, data_client: Any) -> Optional[Tuple[float, float]]:
        """
        Capture the National Best Bid and Offer (NBBO) at the current time.
        """
        try:
            from alpaca.data.requests import StockLatestQuoteRequest
            
            # This handles both SDK client and possible wrappers
            quote_response = data_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols=ticker)
            )
            
            # Standard Alpaca SDK response is a dict-like object mapping tickers to quotes
            if isinstance(quote_response, dict) and ticker in quote_response:
                quote = quote_response[ticker]
                bid = getattr(quote, "bid_price", None)
                ask = getattr(quote, "ask_price", None)
                if bid is not None and ask is not None:
                    return (float(bid), float(ask))
            
            # Alternative: MCP or direct dict response
            if isinstance(quote_response, dict):
                bid = quote_response.get("bid_price") or quote_response.get("bid")
                ask = quote_response.get("ask_price") or quote_response.get("ask")
                if bid is not None and ask is not None:
                    return (float(bid), float(ask))
                    
        except Exception:
            # Fallback or silent failure
            pass
            
        return None

    def calculate_is(self, execution_price: float, arrival_price: float, side: str) -> float:
        """
        Calculate total Implementation Shortfall (IS) in dollars.
        Positive value means slippage (cost), negative means improvement.
        """
        if arrival_price <= 0:
            return 0.0
        
        side = side.lower()
        if side == "buy":
            return execution_price - arrival_price
        elif side == "sell":
            return arrival_price - execution_price
        return 0.0

    def calculate_delay_cost(self, submission_price: float, arrival_price: float, side: str) -> float:
        """
        Calculate delay cost in dollars (submission price - arrival price).
        Measures the cost of delay between signal/decision and submission.
        """
        if arrival_price <= 0 or submission_price <= 0:
            return 0.0
            
        side = side.lower()
        if side == "buy":
            return submission_price - arrival_price
        elif side == "sell":
            return arrival_price - submission_price
        return 0.0

    def calculate_execution_cost(self, execution_price: float, submission_price: float, side: str) -> float:
        """
        Calculate execution cost in dollars (executed price - submission price).
        Measures the cost of execution relative to the submission price.
        """
        if execution_price <= 0 or submission_price <= 0:
            return 0.0
            
        side = side.lower()
        if side == "buy":
            return execution_price - submission_price
        elif side == "sell":
            return submission_price - execution_price
        return 0.0

    def calculate_effective_spread(self, execution_price: float, nbbo: Tuple[float, float], side: str) -> float:
        """
        Calculate the Effective Spread paid on an order.
        Eff. Spread = 2 * (Executed Price - Midpoint) * Side
        Side: 1 for Buy, -1 for Sell
        """
        if not nbbo or len(nbbo) != 2:
            return 0.0
            
        bid, ask = nbbo
        if bid <= 0 or ask <= 0:
            return 0.0
            
        midpoint = (bid + ask) / 2.0
        
        side_val = 1 if side.lower() == "buy" else -1
        return 2.0 * (execution_price - midpoint) * side_val
