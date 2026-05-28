"""
Performance Reporting Module
============================
Aggregates daily review results into weekly and monthly reports.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class PerformanceReporter:
    """
    Aggregates trade performance data and generates summary reports.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def generate_weekly_summary(self, days: int = 7) -> Dict[str, Any]:
        """
        Aggregate results from the last N days.
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        relevant_files = []
        for file in self.data_dir.glob("eod_review_*.json"):
            try:
                # Extract date from filename: eod_review_YYYY-MM-DD.json
                date_part = file.stem.split("_")[-1]
                file_date = datetime.strptime(date_part, "%Y-%m-%d")
                if start_date <= file_date <= end_date:
                    relevant_files.append(file)
            except Exception:
                continue
                
        if not relevant_files:
            return {
                "status": "no_data",
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0
            }
            
        total_trades = 0
        total_win_count = 0
        pnl_sum = 0.0
        
        for file in relevant_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                    trades_count = data.get("total_trades", 0)
                    total_trades += trades_count
                    total_win_count += int(trades_count * data.get("win_rate", 0.0))
                    pnl_sum += data.get("avg_pnl", 0.0) * trades_count
            except Exception as e:
                logger.warning(f"Failed to read {file}: {e}")
                
        win_rate = total_win_count / total_trades if total_trades > 0 else 0.0
        avg_pnl = pnl_sum / total_trades if total_trades > 0 else 0.0
        
        summary = {
            "period": f"{start_date.date()} to {end_date.date()}",
            "total_trades": total_trades,
            "win_rate": round(win_rate, 3),
            "avg_pnl": round(avg_pnl, 2),
            "recommendations": self._generate_recommendations(win_rate, avg_pnl)
        }
        
        return summary

    def _generate_recommendations(self, win_rate: float, avg_pnl: float) -> List[str]:
        """
        Recommend threshold adjustments based on performance.
        """
        recs = []
        if win_rate < 0.45:
            recs.append("Increase min_signal_score by 5.0 (low win rate)")
        if avg_pnl < 0:
            recs.append("Increase min_conviction_score by 5.0 (negative P&L)")
        if win_rate > 0.60 and avg_pnl > 50:
            recs.append("Consider lowering thresholds slightly to increase volume")
            
        if not recs:
            recs.append("Maintain current research parameters")
            
        return recs
