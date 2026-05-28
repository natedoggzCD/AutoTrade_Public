"""
Diagnostic Utility - System health and connectivity checks.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List
import duckdb
import sqlite3

try:
    from config.config_loader import get_config
except ImportError:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from config.config_loader import get_config
from autotrade.utils import alpaca_client_factory as acf
from autotrade.utils.financial_db import FinancialDB

logger = logging.getLogger('AutoTrade.Diagnostic')

def check_system_health() -> Dict[str, Any]:
    """
    Perform a comprehensive health check of the trading system.
    
    Returns:
        Dictionary with health status for each component.
    """
    results = {
        "status": "OK",
        "components": {},
        "issues": []
    }
    
    config = get_config()
    
    # 1. Check DownDay Root
    downday_root = os.environ.get("DOWNDAY_ROOT")
    if not downday_root:
        results["components"]["downday"] = "ERROR: DOWNDAY_ROOT not set"
        results["issues"].append("DOWNDAY_ROOT environment variable is missing")
    else:
        root_path = Path(downday_root)
        if not root_path.exists():
            results["components"]["downday"] = f"ERROR: Path {downday_root} does not exist"
            results["issues"].append(f"External data path {downday_root} is inaccessible")
        else:
            # Check for critical files
            missing_files = []
            for f in ["daily_features.h5", "prices_daily.csv", "prices_hourly.csv"]:
                if not (root_path / f).exists():
                    missing_files.append(f)
            
            if missing_files:
                results["components"]["downday"] = f"DEGRADED: Missing {', '.join(missing_files)}"
                results["issues"].append(f"Critical data files missing in DownDay root: {', '.join(missing_files)}")
            else:
                results["components"]["downday"] = "OK"

    # 2. Check Alpaca Connectivity
    try:
        client = acf.create_trading_client(validate_connection=True)
        account = client.get_account()
        results["components"]["alpaca"] = f"OK ({account.status})"
    except Exception as e:
        results["components"]["alpaca"] = f"ERROR: {str(e)}"
        results["issues"].append(f"Alpaca API connection failed: {e}")

    # 3. Check Financial DB (SQLite)
    try:
        db_path = Path(config.project_root) / "data" / "financial.db"
        if not db_path.exists():
            results["components"]["financial_db"] = "ERROR: File not found"
            results["issues"].append(f"Financial database file missing at {db_path}")
        else:
            db = FinancialDB(db_path=db_path)
            with db._conn() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                if not tables:
                    results["components"]["financial_db"] = "DEGRADED: No tables"
                    results["issues"].append("Financial database is empty (no tables found)")
                else:
                    results["components"]["financial_db"] = f"OK ({len(tables)} tables)"
    except Exception as e:
        results["components"]["financial_db"] = f"ERROR: {str(e)}"
        results["issues"].append(f"Financial database diagnostic failed: {e}")

    # 4. Check DuckDB
    try:
        duckdb_path = config.get_duckdb_path()
        duckdb_path.parent.mkdir(parents=True, exist_ok=True)
        conn = duckdb.connect(str(duckdb_path))
        tables = conn.execute("SHOW TABLES").fetchall()
        conn.close()
        results["components"]["duckdb"] = f"OK ({len(tables)} items)"
    except Exception as e:
        results["components"]["duckdb"] = f"ERROR: {str(e)}"
        results["issues"].append(f"DuckDB diagnostic failed: {e}")

    # Overall Status
    if any("ERROR" in str(v) for v in results["components"].values()):
        results["status"] = "CRITICAL"
    elif any("DEGRADED" in str(v) or results["issues"] for v in results["components"].values()):
        results["status"] = "WARNING"
        
    return results

if __name__ == "__main__":
    health = check_system_health()
    print(f"\nSystem Health: {health['status']}")
    for comp, stat in health['components'].items():
        print(f"  - {comp:12}: {stat}")
    
    if health['issues']:
        print("\nIssues Identified:")
        for issue in health['issues']:
            print(f"  [!] {issue}")
