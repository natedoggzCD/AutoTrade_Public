"""
Hourly Data Provider - Fast DuckDB-based hourly bar loading.
"""

import logging
from pathlib import Path
from typing import List, Optional, Union
from datetime import datetime, date

import duckdb
import pandas as pd

from autotrade.data_ingestion.paths import get_ingestion_paths

logger = logging.getLogger('AutoTrade.HourlyDataProvider')

class HourlyDataProvider:
    """
    Provides fast access to local hourly price data using DuckDB.
    """
    
    def __init__(self, parquet_path: Optional[Path] = None):
        paths = get_ingestion_paths()
        self.parquet_path = parquet_path or paths.hourly_prices_parquet
        self._conn = None
        
        if not self.parquet_path.exists():
            logger.warning(f"Hourly prices parquet not found at {self.parquet_path}")

    @property
    def conn(self):
        if self._conn is None:
            self._conn = duckdb.connect(database=':memory:')
        return self._conn

    def get_hourly_bars(
        self,
        ticker: str,
        start_date: Optional[Union[str, date]] = None,
        end_date: Optional[Union[str, date]] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """
        Fetch hourly bars for a specific ticker.
        """
        if not self.parquet_path.exists():
            return None
            
        ticker = ticker.upper()
        
        query = f"SELECT * FROM read_parquet('{self.parquet_path.as_posix()}') WHERE ticker = '{ticker}'"
        
        if start_date:
            query += f" AND Date >= '{start_date}'"
        if end_date:
            query += f" AND Date <= '{end_date}'"
            
        query += f" ORDER BY Date DESC LIMIT {limit}"
        
        try:
            df = self.conn.execute(query).fetchdf()
            if df.empty:
                return None
                
            # Normalize columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure 'date' column is datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                df.set_index('date', inplace=True)
                
            return df
        except Exception as e:
            logger.error(f"Failed to load hourly bars for {ticker}: {e}")
            return None

    def get_latest_hourly_bars(self, ticker: str, count: int = 24) -> Optional[pd.DataFrame]:
        """Get the N most recent hourly bars for a ticker."""
        return self.get_hourly_bars(ticker, limit=count)

# Singleton
_provider = None

def get_hourly_provider() -> HourlyDataProvider:
    global _provider
    if _provider is None:
        _provider = HourlyDataProvider()
    return _provider
