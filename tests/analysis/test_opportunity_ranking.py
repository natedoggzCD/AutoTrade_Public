import unittest
import pandas as pd
from autotrade.analysis.ranking import OpportunityScorer

class TestOpportunityRanking(unittest.TestCase):
    def setUp(self):
        self.scorer = OpportunityScorer()

    def test_ranking_logic(self):
        # Create a mock dataframe of stock data
        data = {
            "symbol": ["AAPL", "TSLA", "NVDA", "CNK", "TNK"],
            "rvol": [1.0, 2.5, 3.0, 0.5, 4.0],  # Relative Volume
            "atr_14": [2.0, 5.0, 4.0, 1.0, 3.0], # ATR
            "intraday_range": [1.5, 6.0, 5.0, 0.4, 4.5], # Intraday High-Low
            "catalyst_score": [0, 50, 80, 10, 90] # Catalyst intensity
        }
        df = pd.DataFrame(data)
        
        # Rank the stocks
        ranked_df = self.scorer.rank_stocks(df)
        
        # TNK has highest RVOL, Catalyst, and Range Expansion (4.5/3.0 = 1.5x ATR)
        # Verify TNK is ranked high
        self.assertEqual(ranked_df.iloc[0]["symbol"], "TNK")
        
        # CNK has low RVOL and low catalyst, verify it is ranked lower
        self.assertNotIn("CNK", ranked_df.head(2)["symbol"].values)

    def test_score_components(self):
        # Test individual score component calculation
        rvol_score = self.scorer._calculate_rvol_score(3.0) # High RVOL
        self.assertGreater(rvol_score, 0)
        
        range_score = self.scorer._calculate_range_score(2.0, 1.0) # 2.0x ATR range
        self.assertGreater(range_score, 50)

if __name__ == "__main__":
    unittest.main()
