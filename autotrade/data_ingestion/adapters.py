"""
Compatibility adapters over legacy utility modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from autotrade.data_ingestion.interfaces import HistoricalDataSource, RealtimePriceSource


class LocalDataProviderAdapter(HistoricalDataSource, RealtimePriceSource):
    """Adapter for `autotrade.utils.local_data_provider`."""

    def __init__(self):
        from autotrade.utils.local_data_provider import get_provider

        self._provider = get_provider()

    def get_ticker_data(
        self, ticker: str, days: int = 250, include_features: bool = True
    ) -> Optional[pd.DataFrame]:
        return self._provider.get_ticker_data(
            ticker=ticker,
            days=days,
            include_features=include_features,
        )

    def has_ticker(self, ticker: str) -> bool:
        return self._provider.has_ticker(ticker)

    def get_available_tickers(self) -> List[str]:
        return self._provider.get_available_tickers()

    def get_current_price(self, ticker: str) -> Optional[float]:
        price, _source = self._provider.get_current_price(ticker)
        return price

    def get_prices(self, tickers: List[str]) -> Dict[str, Optional[float]]:
        out: Dict[str, Optional[float]] = {}
        for ticker in tickers:
            price, _source = self._provider.get_current_price(ticker)
            out[ticker] = price
        return out


class DataSyncAdapter:
    """Adapter for `autotrade.utils.data_sync`."""

    def __init__(self):
        from autotrade.utils.data_sync import get_manager

        self._manager = get_manager()

    def check_sync_status(self) -> Dict[str, Any]:
        return self._manager.check_sync_status()

    def ensure_synced(self) -> bool:
        return self._manager.ensure_synced()

    def get_freshness(self) -> Dict[str, Any]:
        return self._manager.get_data_freshness()


class DataQualityAdapter:
    """Adapter for `autotrade.utils.data_quality`."""

    def validate_data_for_signals(self, as_of_date=None) -> Tuple[bool, Any]:
        from autotrade.utils.data_quality import validate_data_for_signals

        return validate_data_for_signals(as_of_date=as_of_date)

    def validate_signal_batch(self, signals: List[Dict]) -> Tuple[bool, Any]:
        from autotrade.utils.data_quality import validate_signal_batch

        return validate_signal_batch(signals=signals)

