import unittest
from autotrade.monitoring.halt_logic import HaltMonitor

class TestHaltFilter(unittest.TestCase):
    def setUp(self):
        self.monitor = HaltMonitor()

    def test_halt_detection(self):
        # Mocking or simulating a halt event
        symbol = "XYZ"
        self.monitor.mark_halted(symbol, reason="LULD")
        self.assertTrue(self.monitor.is_halted(symbol))
        
    def test_blacklist_integration(self):
        symbol = "XYZ"
        self.monitor.mark_halted(symbol)
        blacklist = self.monitor.get_blacklist()
        self.assertIn(symbol, blacklist)

    def test_halt_expiry(self):
        # Test that halts expire or are cleared
        symbol = "ABC"
        self.monitor.mark_halted(symbol)
        self.monitor.clear_halt(symbol)
        self.assertFalse(self.monitor.is_halted(symbol))

if __name__ == "__main__":
    unittest.main()
