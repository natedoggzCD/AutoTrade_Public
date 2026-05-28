
"""
VWAP Calculator Utility.
Calculates real-time VWAP and standard deviation bands from bar data.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class VWAPCalculator:
    """
    Calculates VWAP and bands for a given set of bars.
    """
    
    @staticmethod
    def calculate(bars: pd.DataFrame, window_minutes: Optional[int] = None) -> Dict[str, Any]:
        """
        Calculate VWAP from a DataFrame of bars.
        Expected columns: 'close', 'high', 'low', 'volume'
        Supports Intraday reset (uses only the last day of data).
        """
        if bars.empty:
            return {"vwap": 0.0, "std": 0.0, "last_price": 0.0}
            
        # Intraday Reset: Isolate only the last day's data
        if isinstance(bars.index, pd.DatetimeIndex):
            last_date = bars.index[-1].date()
            bars = bars[bars.index.date == last_date]
            
        if window_minutes:
            bars = bars.tail(window_minutes)
            
        tp = (bars['high'] + bars['low'] + bars['close']) / 3.0
        pv = tp * bars['volume']
        
        cum_pv = pv.sum()
        cum_vol = bars['volume'].sum()
        
        if cum_vol == 0:
            return {"vwap": 0.0, "std": 0.0, "last_price": bars['close'].iloc[-1]}
            
        vwap = cum_pv / cum_vol
        
        # Standard deviation (rolling)
        variance = ((tp - vwap)**2 * bars['volume']).sum() / cum_vol
        std_dev = np.sqrt(variance)
        
        results = {
            "vwap": float(vwap),
            "std": float(std_dev),
            "last_price": float(bars['close'].iloc[-1]),
            "upper_band": float(vwap + std_dev),
            "lower_band": float(vwap - std_dev),
            "upper_band_1": float(vwap + std_dev),
            "lower_band_1": float(vwap - std_dev),
            "upper_band_2": float(vwap + 2 * std_dev),
            "lower_band_2": float(vwap - 2 * std_dev),
            "upper_band_3": float(vwap + 3 * std_dev),
            "lower_band_3": float(vwap - 3 * std_dev)
        }
        return results

    @staticmethod
    def get_deviation(price: float, vwap_data: Dict[str, Any]) -> float:
        """Get price deviation from VWAP in standard deviations."""
        vwap = vwap_data.get("vwap", 0.0)
        std = vwap_data.get("std", 0.0)
        if std == 0:
            return 0.0
        return (price - vwap) / std
