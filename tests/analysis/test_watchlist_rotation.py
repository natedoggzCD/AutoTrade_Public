import unittest
import pandas as pd
from autotrade.analysis.rotation import RotationScheduler, WatchlistRotator
from autotrade.analysis.ranking import OpportunityScorer
from autotrade.monitoring.halt_logic import HaltMonitor

class TestWatchlistRotation(unittest.TestCase):
    def setUp(self):
        self.scorer = OpportunityScorer()
        self.halt_monitor = HaltMonitor()
        self.rotator = WatchlistRotator(
            scorer=self.scorer,
            halt_monitor=self.halt_monitor,
            min_delta=10.0,
            max_watchlist_size=5
        )

    def test_no_rotation_on_low_delta(self):
        # Current watchlist has symbols with score 50
        watchlist = pd.DataFrame({"symbol": ["A", "B", "C"], "opportunity_score": [50.0, 50.0, 50.0]})
        # Candidate has score 55 (Delta = 5, less than min_delta=10)
        candidates = pd.DataFrame({"symbol": ["D"], "opportunity_score": [55.0]})
        
        swaps = self.rotator.evaluate_rotation(watchlist, candidates)
        self.assertEqual(len(swaps), 0)

    def test_rotation_on_high_delta(self):
        # Current watchlist has score 40
        watchlist = pd.DataFrame({"symbol": ["A", "B", "C"], "opportunity_score": [40.0, 40.0, 40.0]})
        # Candidate has score 60 (Delta = 20, > min_delta=10)
        candidates = pd.DataFrame({"symbol": ["D"], "opportunity_score": [60.0]})
        
        swaps = self.rotator.evaluate_rotation(watchlist, candidates)
        self.assertEqual(len(swaps), 1)
        self.assertEqual(swaps[0]["remove"], "A") # Lowest score (or first in list)
        self.assertEqual(swaps[0]["add"], "D")

    def test_emergency_halt_rotation(self):
        watchlist = pd.DataFrame({"symbol": ["A", "B", "C"], "opportunity_score": [80.0, 80.0, 80.0]})
        candidates = pd.DataFrame({"symbol": ["D"], "opportunity_score": [70.0]})
        
        # Mark A as halted
        self.halt_monitor.mark_halted("A")
        
        # Even though Delta is negative (-10), A MUST be removed
        swaps = self.rotator.evaluate_rotation(watchlist, candidates)
        self.assertEqual(len(swaps), 1)
        self.assertEqual(swaps[0]["remove"], "A")
        self.assertEqual(swaps[0]["reason"], "HALTED")

    def test_rotation_scheduler_runs_only_on_interval_in_active_phases(self):
        scheduler = RotationScheduler(interval_cycles=15)

        self.assertFalse(scheduler.should_run(14, "CORE_TRADING"))
        self.assertTrue(scheduler.should_run(15, "CORE_TRADING"))
        self.assertTrue(scheduler.should_run(30, "RESEARCH"))
        self.assertFalse(scheduler.should_run(15, "WIND_DOWN"))

if __name__ == "__main__":
    unittest.main()
