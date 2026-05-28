import pytest
import os
import json
import subprocess
import sys
from datetime import datetime, timedelta
from autotrade.analysis.performance_reporting import PerformanceReporter

def test_performance_reporter_instantiation():
    """
    Verify that PerformanceReporter can be instantiated.
    """
    reporter = PerformanceReporter()
    assert reporter is not None

def test_performance_reporter_generate_weekly_summary(tmp_path):
    """
    Verify that PerformanceReporter generates a valid weekly summary from EOD files.
    """
    # Create dummy EOD review files
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    today = datetime.now()
    for i in range(3):
        date_str = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        file_path = data_dir / f"eod_review_{date_str}.json"
        content = {
            "date": date_str,
            "total_trades": 10,
            "win_rate": 0.6,
            "avg_pnl": 50.0,
            "score_buckets": {}
        }
        with open(file_path, "w") as f:
            json.dump(content, f)
            
    reporter = PerformanceReporter(data_dir=str(data_dir))
    summary = reporter.generate_weekly_summary()
    
    assert "win_rate" in summary
    assert "total_trades" in summary
    assert summary["total_trades"] == 30
    assert summary["win_rate"] == 0.6

def test_weekly_performance_tool_execution():
    """
    Verify that tools/weekly_performance.py executes without error.
    """
    # Set PYTHONPATH to include current directory
    env = os.environ.copy()
    env["PYTHONPATH"] = os.getcwd()
    
    # Just check if it can be called with --days
    result = subprocess.run(
        [sys.executable, "tools/weekly_performance.py", "--days", "7"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env
    )
    assert result.returncode == 0
    assert "PERFORMANCE SUMMARY" in result.stdout
