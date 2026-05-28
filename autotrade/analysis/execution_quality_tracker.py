
"""
Execution Quality Tracker.
Calculates slippage and Implementation Shortfall (IS) benchmarks.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExecutionQualityTracker:
    """
    Tracks and benchmarks execution performance.
    """

    def calculate_slippage_bps(self, execution_price: float, benchmark_price: float, side: str) -> float:
        """
        Calculate slippage vs a benchmark (e.g., VWAP) in basis points.
        Positive value means worse than benchmark (higher price for buy, lower for sell).
        """
        if benchmark_price <= 0:
            return 0.0
            
        side = side.lower()
        if side == "buy":
            slippage = (execution_price - benchmark_price) / benchmark_price
        elif side == "sell":
            slippage = (benchmark_price - execution_price) / benchmark_price
        else:
            logger.warning(f"Unknown side '{side}', defaulting to 0 slippage")
            return 0.0
            
        return slippage * 10000.0

    def calculate_is_bps(self, execution_price: float, decision_price: float, side: str) -> float:
        """
        Calculate Implementation Shortfall (IS) vs decision price in basis points.
        Decision price is the price at the time the order was decided/generated.
        """
        # IS formula is technically broader (includes fees/missed opportunity),
        # but the price component is the primary intraday metric.
        return self.calculate_slippage_bps(execution_price, decision_price, side)
