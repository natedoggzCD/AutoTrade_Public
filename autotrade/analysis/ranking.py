import pandas as pd
import numpy as np
from typing import Dict, List, Any


class OpportunityScorer:
    """
    Ranks watchlist candidates using dynamic intraday microstructure data.
    Focuses on Relative Volume (RVOL), Catalyst Intensity, and Range Expansion.
    """

    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {"rvol": 0.40, "catalyst": 0.35, "range": 0.25}

    def rank_stocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate a composite opportunity score for each symbol in the dataframe."""
        if df.empty:
            return df

        # Ensure required columns exist
        required_cols = ["rvol", "catalyst_score", "atr_14", "intraday_range"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        # Calculate component scores
        df["score_rvol"] = df["rvol"].apply(self._calculate_rvol_score)
        df["score_catalyst"] = df["catalyst_score"].apply(
            lambda x: min(100.0, float(x))
        )
        df["score_range"] = df.apply(
            lambda row: self._calculate_range_score(
                row["intraday_range"], row["atr_14"]
            ),
            axis=1,
        )

        # Calculate weighted composite score
        df["opportunity_score"] = (
            df["score_rvol"] * self.weights["rvol"]
            + df["score_catalyst"] * self.weights["catalyst"]
            + df["score_range"] * self.weights["range"]
        )

        # Sort by total score
        return df.sort_values(by="opportunity_score", ascending=False)

    def _calculate_rvol_score(self, rvol: float) -> float:
        """
        Normalize RVOL to a 0-100 scale.
        RVOL 1.0 (average) -> 20 score
        RVOL 2.0 (high) -> 70 score
        RVOL 3.0+ (exceptional) -> 100 score
        """
        if rvol <= 0:
            return 0.0
        # Sigmoid-like scaling for RVOL
        return min(100.0, (100 / (1 + np.exp(-2.5 * (rvol - 1.5)))))

    def _calculate_range_score(self, current_range: float, atr: float) -> float:
        """
        Calculates range expansion relative to average volatility.
        Range < 0.5x ATR -> Low score
        Range > 1.0x ATR -> Expansion score
        """
        if atr <= 0:
            return 0.0
        ratio = current_range / atr
        # Normalize: 1.0x ATR ratio -> 50 score, 2.0x ATR -> 100 score
        return min(100.0, max(0.0, ratio * 50))

    def get_summary_report(self, ranked_df: pd.DataFrame) -> str:
        """Generate a concise textual report of the top opportunities."""
        if ranked_df.empty:
            return "No opportunities identified."

        top_3 = ranked_df.head(3)
        report = "Top Intraday Opportunities:\n"
        for i, row in top_3.iterrows():
            report += f"- {row['symbol']}: Score={row['opportunity_score']:.1f} (RVOL={row['rvol']:.1f}x, Catalyst={row['catalyst_score']:.0f})\n"
        return report
