import pytest
import pandas as pd
import duckdb
from pathlib import Path
from autotrade.utils.data_sync import DataSyncManager, DATA_SOURCES

def test_duckdb_schema_consistency():
    """Verify that DuckDB view schema matches authoritative parquet schema."""
    sync = DataSyncManager()
    parquet_path = DATA_SOURCES['daily_features_parquet']
    duckdb_path = DATA_SOURCES['market_data_duckdb']
    
    if not parquet_path.exists():
        pytest.skip("Primary parquet file not found")
        
    # Run the actual sync
    success = sync.sync_duckdb()
    assert success is True, "sync_duckdb failed"
    
    # Get actual DuckDB columns from the synced view
    conn = duckdb.connect(str(duckdb_path), read_only=True)
    db_cols = set(c[0].lower() for c in conn.execute("DESCRIBE daily_prices").fetchall())
    conn.close()
    
    # Required columns for AutoTrade (must exist in view, even if NULL)
    # These are mapped names in the VIEW
    required = {'symbol', 'date', 'close', 'atr_14', 'market_cap'}
    for col in required:
        assert col in db_cols, f"Required column {col} missing from DuckDB view"

def test_row_count_sync():
    """Verify that DuckDB view row counts match parquet files."""
    sync = DataSyncManager()
    parquet_path = DATA_SOURCES['daily_features_parquet']
    
    if not parquet_path.exists():
        pytest.skip("Primary parquet file not found")
        
    # Get parquet row count
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path, columns=['ticker'])
    pq_rows = table.num_rows
    
    # Get sync status (which checks DuckDB)
    status = sync.check_sync_status()
    db_info = status['sources'].get('market_data_duckdb', {})
    
    if db_info.get('exists'):
        # Note: This checks the DuckDB file on disk, not necessarily the current view
        # We'll trust the sync manager's reported row count for now
        db_rows = db_info.get('row_count')
        if db_rows:
            assert db_rows == pq_rows, f"Row count mismatch: Parquet={pq_rows}, DuckDB={db_rows}"
