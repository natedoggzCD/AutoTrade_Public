
import sys
import unittest
from pathlib import Path
from datetime import datetime, timedelta

# Adjust path to find autotrade
sys.path.insert(0, str(Path(__file__).parent.parent))

from autotrade.backtesting.duckdb_backtester import DuckDBBacktester, BacktestResult
from autotrade.backtesting.hypothesis_engine import HypothesisEngine
from autotrade.utils.config_updater import SafeConfigUpdater
from autotrade.backtesting.comparison import StrategyComparator, ComparisonResult

class TestImprovementEngine(unittest.TestCase):
    def test_instantiation(self):
        # Test Backtester
        backtester = DuckDBBacktester("dummy_path.parquet")
        self.assertIsInstance(backtester, DuckDBBacktester)
        
        # Test Engine
        engine = HypothesisEngine(backtester)
        self.assertIsInstance(engine, HypothesisEngine)
        
        # Test Updater
        updater = SafeConfigUpdater()
        self.assertIsInstance(updater, SafeConfigUpdater)

    def test_comparison_logic(self):
        comp = StrategyComparator()
        
        b1 = BacktestResult(10, 50.0, 1.0, 2.0, 3.0, 10.0, -5.0, 1.5, 2.0, -10.0)
        # B2 is better (higher return, higher win rate)
        b2 = BacktestResult(10, 55.0, 1.2, 2.5, 4.0, 10.0, -5.0, 1.8, 2.0, -8.0)
        
        res = comp.compare(b1, b2)
        self.assertTrue(res.is_better)
        self.assertTrue(res.verdict.startswith("IMPROVED"))

if __name__ == '__main__':
    unittest.main()
