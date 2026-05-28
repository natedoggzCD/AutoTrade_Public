import unittest
from types import SimpleNamespace
from autotrade.core.day_manager import DayManager

class TestDayManagerFix(unittest.TestCase):
    def setUp(self):
        # We don't need a full init for these static-like methods
        self.dm = DayManager(dry_run=True)

    def test_position_qty_object(self):
        pos = SimpleNamespace(qty=10)
        self.assertEqual(self.dm._position_qty(pos), 10)

    def test_position_qty_dict(self):
        pos = {"qty": 20}
        self.assertEqual(self.dm._position_qty(pos), 20)

    def test_position_qty_none(self):
        self.assertEqual(self.dm._position_qty(None), 0)

    def test_position_float_object(self):
        pos = SimpleNamespace(avg_entry_price=150.5)
        self.assertEqual(self.dm._position_float(pos, "avg_entry_price"), 150.5)

    def test_position_float_dict(self):
        pos = {"avg_entry_price": 120.0}
        self.assertEqual(self.dm._position_float(pos, "avg_entry_price"), 120.0)

    def test_position_float_dict_fallback_avg_entry(self):
        pos = {"avg_entry": 85.5}
        # field is avg_entry_price, but it should fallback to avg_entry
        self.assertEqual(self.dm._position_float(pos, "avg_entry_price"), 85.5)

    def test_position_float_dict_fallback_avg_price(self):
        pos = {"avg_price": 75.77}
        self.assertEqual(self.dm._position_float(pos, "avg_entry_price"), 75.77)

    def test_position_float_none(self):
        self.assertEqual(self.dm._position_float(None, "any"), 0.0)

if __name__ == "__main__":
    unittest.main()
