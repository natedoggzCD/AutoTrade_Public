"""
Day Trade Tracker
=================
Legacy day-trade telemetry.

PDT limiting is disabled for this workflow. The tracker is kept only so older
call sites can record/report same-day activity without blocking runtime trades.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple

from config.config_loader import get_config, get_logging_config
from autotrade.utils.logging_utils import configure_logging, log_event

PROJECT_DIR = Path(os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2]))
TRACKER_FILE = PROJECT_DIR / 'day_trade_tracker.json'
logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    try:
        cfg = get_config()
        log_cfg = get_logging_config()
        configure_logging(
            cfg.logs_dir,
            level=log_cfg.level,
            filename=log_cfg.json_filename,
            max_bytes=log_cfg.max_bytes,
            backup_count=log_cfg.backup_count,
            console=log_cfg.console,
        )
    except Exception:
        logging.basicConfig(level=logging.INFO)


_configure_logging()


class DayTradeTracker:
    """Legacy tracker that never limits day trades."""
    
    def __init__(self, max_per_week: int = 999):
        self.max_per_week = max_per_week
        self.trades: List[Dict] = []
        self.load()
    
    def load(self):
        """Load trade history from disk."""
        if TRACKER_FILE.exists():
            with open(TRACKER_FILE, 'r') as f:
                data = json.load(f)
                self.trades = data.get('trades', [])
                self.max_per_week = data.get('max_per_week', 999)
    
    def save(self):
        """Save trade history to disk."""
        with open(TRACKER_FILE, 'w') as f:
            json.dump({
                'max_per_week': self.max_per_week,
                'trades': self.trades,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def _get_week_start(self) -> date:
        """Get the start of the current trading week (Monday)."""
        today = date.today()
        return today - timedelta(days=today.weekday())
    
    def get_trades_this_week(self) -> List[Dict]:
        """Get all day trades from this rolling 5-day week."""
        week_start = self._get_week_start()
        return [
            t for t in self.trades
            if datetime.fromisoformat(t['date']).date() >= week_start
        ]
    
    def get_count_this_week(self) -> int:
        """Get count of day trades this week."""
        return len(self.get_trades_this_week())
    
    def get_remaining(self) -> int:
        """Return an effectively unlimited day-trade budget."""
        return max(int(self.max_per_week or 999), 999)
    
    def can_day_trade(self, emergency: bool = False) -> Tuple[bool, str]:
        """
        Check if we can make a day trade.
        
        Args:
            emergency: If True, allows trade even at limit (for crash protection)
        
        Returns:
            (allowed, reason)
        """
        remaining = self.get_remaining()
        return True, f"PDT disabled; day trades unrestricted ({remaining} available)"
    
    def record_day_trade(self, ticker: str, side: str, shares: int, 
                         price: float, reason: str = ""):
        """Record a day trade."""
        self.trades.append({
            'date': datetime.now().isoformat(),
            'ticker': ticker,
            'side': side,
            'shares': shares,
            'price': price,
            'reason': reason
        })
        self.save()
        
        remaining = self.get_remaining()
        log_event(
            logger,
            'day_trade_recorded',
            ticker=ticker,
            side=side,
            shares=shares,
            price=price,
            remaining=remaining,
        )

    # Backwards-compatible alias (older code uses record_trade)
    def record_trade(self, ticker: str, side: str, shares: int,
                     price: float, reason: str = ""):
        self.record_day_trade(ticker, side, shares, price, reason)
    
    def cleanup_old_trades(self, days: int = 30):
        """Remove trades older than N days."""
        cutoff = datetime.now() - timedelta(days=days)
        self.trades = [
            t for t in self.trades
            if datetime.fromisoformat(t['date']) > cutoff
        ]
        self.save()
    
    def get_status(self) -> Dict:
        """Get current day trade status."""
        week_trades = self.get_trades_this_week()
        return {
            'week_start': self._get_week_start().isoformat(),
            'trades_this_week': len(week_trades),
            'max_per_week': self.max_per_week,
            'remaining': self.get_remaining(),
            'recent_trades': week_trades[-5:] if week_trades else []
        }
    
    def print_status(self):
        """Log current status."""
        status = self.get_status()
        log_event(logger, "day_trade_status", status=status)


# Convenience functions
_tracker = None

def get_tracker() -> DayTradeTracker:
    """Get or create the global tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = DayTradeTracker()
    return _tracker

def can_day_trade(emergency: bool = False) -> Tuple[bool, str]:
    """Check if we can make a day trade."""
    return get_tracker().can_day_trade(emergency)

def record_day_trade(ticker: str, side: str, shares: int, price: float, reason: str = ""):
    """Record a day trade."""
    get_tracker().record_day_trade(ticker, side, shares, price, reason)

def get_remaining_day_trades() -> int:
    """Get remaining day trades for the week."""
    return get_tracker().get_remaining()


if __name__ == '__main__':
    tracker = DayTradeTracker()
    tracker.cleanup_old_trades()
    tracker.print_status()

