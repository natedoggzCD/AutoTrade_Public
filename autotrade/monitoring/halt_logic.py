from typing import Dict, List, Optional
import time


class HaltMonitor:
    """
    Monitors market halts and LULD (Limit Up-Down) events.
    Automatically flags and blacklists symbols from active trading.
    """

    def __init__(self, expiry_seconds: int = 300):
        self.halted_symbols: Dict[str, Dict] = {}
        self.expiry_seconds = expiry_seconds

    def mark_halted(
        self, symbol: str, reason: str = "LULD", timestamp: Optional[float] = None
    ) -> None:
        """Mark a symbol as halted with a specific reason and timestamp."""
        ts = timestamp or time.time()
        self.halted_symbols[symbol] = {
            "reason": reason,
            "timestamp": ts,
            "expiry": ts + self.expiry_seconds,
        }

    def clear_halt(self, symbol: str) -> None:
        """Manually clear a halt status for a symbol."""
        if symbol in self.halted_symbols:
            del self.halted_symbols[symbol]

    def is_halted(self, symbol: str) -> bool:
        """Check if a symbol is currently halted, accounting for expiry."""
        if symbol not in self.halted_symbols:
            return False

        halt_data = self.halted_symbols[symbol]
        if time.time() > halt_data["expiry"]:
            del self.halted_symbols[symbol]
            return False

        return True

    def get_blacklist(self) -> List[str]:
        """Return a list of all currently halted symbols."""
        # Clean up expired halts first
        current_time = time.time()
        to_remove = [
            s for s, d in self.halted_symbols.items() if current_time > d["expiry"]
        ]
        for s in to_remove:
            del self.halted_symbols[s]

        return list(self.halted_symbols.keys())

    def update_from_feed(self, event_data: Dict) -> None:
        """Process incoming market feed events to detect halts."""
        # Placeholder for real-time feed integration (e.g., Polygon.io, Alpaca)
        pass
