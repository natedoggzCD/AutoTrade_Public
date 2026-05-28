import unittest
import json
import pandas as pd
from pathlib import Path
from autotrade.analysis.post_market import PostMarketAnalyzer

class TestPostMarketEvaluation(unittest.TestCase):
    def setUp(self):
        self.analyzer = PostMarketAnalyzer()
        self.plan_data = {
            "signals": [
                {"symbol": "CNK", "entry_score": 75.0, "recommendation": "STRONG BUY"},
                {"symbol": "BORR", "entry_score": 0.0, "recommendation": "WATCH"},
                {"symbol": "TNK", "entry_score": 91.0, "recommendation": "STRONG BUY"}
            ]
        }
        self.execution_data = [
            {"symbol": "CNK", "entry_price": 30.0, "exit_price": 26.18, "pl_pct": -0.1273, "qty": 100},
            {"symbol": "BORR", "entry_price": 4.0, "exit_price": 4.14, "pl_pct": 0.035, "qty": 1000},
            {"symbol": "NE", "entry_price": 40.0, "exit_price": 41.54, "pl_pct": 0.0385, "qty": 100}
        ]

    def test_plan_adherence(self):
        adherence = self.analyzer.calculate_plan_adherence(self.plan_data, self.execution_data)
        # Verify CNK was an "Adhered" trade (it was in the plan)
        self.assertIn("CNK", adherence["adhered_symbols"])
        # Verify NE was a "Deviation" (it wasn't in the plan but was traded)
        self.assertIn("NE", adherence["deviations"])
        
    def test_alpha_attribution(self):
        metrics = self.analyzer.attribute_performance(self.plan_data, self.execution_data)
        # High score (CNK) vs Low score (BORR) performance delta
        self.assertLess(metrics["high_conviction_return"], metrics["low_conviction_return"])
        self.assertTrue(metrics["inverse_conviction_flag"])

if __name__ == "__main__":
    unittest.main()
