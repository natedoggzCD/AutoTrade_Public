import os
import pytest
from pathlib import Path
from config.config_loader import get_config
from autotrade.utils import alpaca_client_factory as acf
from autotrade.utils.financial_db import FinancialDB
import duckdb

def test_downday_root_connectivity():
    """Verify that DOWNDAY_ROOT is set and the directory is accessible."""
    downday_root = os.environ.get("DOWNDAY_ROOT")
    assert downday_root is not None, "DOWNDAY_ROOT environment variable is not set"
    
    root_path = Path(downday_root)
    assert root_path.exists(), f"DOWNDAY_ROOT directory does not exist: {downday_root}"
    assert root_path.is_dir(), f"DOWNDAY_ROOT is not a directory: {downday_root}"

def test_h5_file_exists():
    """Verify the authoritative HDF5 feature file exists in DOWNDAY_ROOT."""
    downday_root = os.environ.get("DOWNDAY_ROOT")
    h5_path = Path(downday_root) / "daily_features.h5"
    assert h5_path.exists(), f"Authoritative H5 file not found: {h5_path}"

def test_csv_data_exists():
    """Verify up-to-date CSV data sources exist in DOWNDAY_ROOT."""
    downday_root = os.environ.get("DOWNDAY_ROOT")
    root = Path(downday_root)
    
    daily_csv = root / "prices_daily.csv"
    hourly_csv = root / "prices_hourly.csv"
    
    assert daily_csv.exists(), f"Daily prices CSV not found: {daily_csv}"
    assert hourly_csv.exists(), f"Hourly prices CSV not found: {hourly_csv}"

def test_alpaca_connectivity():
    """Verify Alpaca API connectivity using credentials from config/.env."""
    try:
        client = acf.create_trading_client(validate_connection=True)
        account = client.get_account()
        assert account.status == "ACTIVE", f"Alpaca account status is {account.status}"
    except Exception as e:
        pytest.fail(f"Alpaca connectivity failed: {e}")

def test_financial_db_connectivity():
    """Verify connectivity to the SQLite financial database."""
    config = get_config()
    db_path = Path(config.project_root) / "data" / "financial.db"
    assert db_path.exists(), f"Financial DB not found at {db_path}"
    
    db = FinancialDB(db_path=db_path)
    # Simple query to verify connectivity
    try:
        # Just check if we can get a connection and a basic table exists
        with db._conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            assert len(tables) > 0, "No tables found in financial.db"
    except Exception as e:
        pytest.fail(f"Financial DB connectivity failed: {e}")

def test_duckdb_connectivity():
    """Verify DuckDB connectivity and accessibility."""
    config = get_config()
    duckdb_path = config.get_duckdb_path()
    
    # Ensure directory exists
    duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Connect (will create if doesn't exist, which is what the system does)
        conn = duckdb.connect(str(duckdb_path))
        # Check for presence of views/tables
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        # Even if empty, connection should succeed
    except Exception as e:
        pytest.fail(f"DuckDB connectivity failed: {e}")
