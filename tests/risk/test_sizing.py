import unittest
from autotrade.risk.sizing import RUnitSizer

class TestRiskSizing(unittest.TestCase):
    def setUp(self):
        # Setup sizer with 1% total risk per trade on 100k account
        # Set max_notional high (20%) so it doesn't interfere with R-unit tests
        self.sizer = RUnitSizer(total_equity=100000.0, risk_pct=0.01, max_notional_pct=0.20)

    def test_standard_sizing(self):
        # Entry 100, Stop 95 (Risk = 5 per share)
        # 1% of 100k = $1000 risk unit.
        # $1000 / 5 = 200 shares.
        qty = self.sizer.calculate_quantity(entry_price=100.0, stop_price=95.0)
        self.assertEqual(qty, 200)

    def test_wide_stop_reduces_size(self):
        # Entry 100, Stop 90 (Risk = 10 per share)
        # $1000 / 10 = 100 shares.
        qty = self.sizer.calculate_quantity(entry_price=100.0, stop_price=90.0)
        self.assertEqual(qty, 100)

    def test_tight_stop_increases_size_but_caps_notional(self):
        # Entry 100, Stop 99.5 (Risk = 0.5 per share)
        # $1000 / 0.5 = 2000 shares.
        # Notional = 2000 * 100 = 200k (2x leverage)
        # Sizer should cap at 5% max notional by default ($5000)
        # $5000 / 100 = 50 shares.
        self.sizer.max_notional_pct = 0.05
        qty = self.sizer.calculate_quantity(entry_price=100.0, stop_price=99.5)
        self.assertEqual(qty, 50)

if __name__ == "__main__":
    unittest.main()
