"""
AutoTrade Utilities Package

Key exports:
- LocalDataProvider: Primary data source (local parquet with API fallback)
- get_data_source_mode(): Returns current data mode (local_only, live_prices, hybrid)
- is_market_hours(): Check if in regular trading hours
- is_extended_hours(): Check if in pre-market or after-hours
- NewsCache: Persistent news article cache
- DataSyncManager: DuckDB/parquet sync manager
"""

from __future__ import annotations

import importlib
from typing import Dict

__all__ = [
    'LocalDataProvider',
    'get_provider',
    'get_data_source_mode',
    'get_current_price',
    'get_sma',
    'get_ticker_data',
    'has_local_data',
    'bulk_sma_check',
    'is_market_hours',
    'is_extended_hours',
    'DataMode',
]

_LAZY_IMPORTS: Dict[str, str] = {
    'LocalDataProvider': 'autotrade.utils.local_data_provider',
    'get_provider': 'autotrade.utils.local_data_provider',
    'get_data_source_mode': 'autotrade.utils.local_data_provider',
    'get_current_price': 'autotrade.utils.local_data_provider',
    'get_sma': 'autotrade.utils.local_data_provider',
    'get_ticker_data': 'autotrade.utils.local_data_provider',
    'has_local_data': 'autotrade.utils.local_data_provider',
    'bulk_sma_check': 'autotrade.utils.local_data_provider',
    'is_market_hours': 'autotrade.utils.local_data_provider',
    'is_extended_hours': 'autotrade.utils.local_data_provider',
    'DataMode': 'autotrade.utils.local_data_provider',
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module = importlib.import_module(_LAZY_IMPORTS[name])
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'autotrade.utils' has no attribute '{name}'")


def __dir__():
    return sorted(list(__all__) + list(globals().keys()))
