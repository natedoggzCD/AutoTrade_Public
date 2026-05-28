"""
Position State - Core Data Structures for Portfolio Management
==============================================================
Tracks position lifecycle and conviction scoring.

Key Concepts:
- Hold phases are informational; PDT restrictions are disabled.
- Conviction: Need reasons TO hold, not just reasons to exit
- Rotation: After 1D hold, eligible for replacement by better candidates
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class HoldPhase(Enum):
    """Position lifecycle phases."""
    LOCKED = "locked"           # < 1D hold
    UNLOCKING = "unlocking"     # Within 1 hour of unlock (can sell at next open)
    UNLOCKED = "unlocked"       # > 1D hold, free to sell
    CRITICAL = "critical"       # Hard stop hit, MUST sell


class ConvictionLevel(Enum):
    """Conviction to hold a position."""
    STRONG_BUY = "strong_buy"   # +80-100: Should ADD to position
    BUY = "buy"                 # +60-79: Consider adding
    HOLD = "hold"               # +40-59: Keep position
    WEAK = "weak"               # +20-39: Consider trimming after unlock
    SELL = "sell"               # 0-19: Should exit when unlocked
    CRITICAL_EXIT = "critical"  # Negative: Exit NOW


class PositionAction(Enum):
    """Available actions for a position."""
    ADD = "add"                 # Scale into winner
    HOLD = "hold"               # Keep position unchanged
    TRIM = "trim"               # Reduce position size
    EXIT = "exit"               # Close entire position
    ROTATE = "rotate"           # Exit to free capital for better opportunity


@dataclass
class PDTStatus:
    """Pattern Day Trade rule tracking."""
    trades_used_today: int = 0
    trades_remaining: int = 10      # Per rolling 5-day week
    can_burn_day_trade: bool = True  # False if at limit
    emergency_only: bool = False     # True if at/near limit
    
    def can_sell_same_day(self, is_critical: bool = False) -> bool:
        """PDT is disabled; same-day sells are not blocked here."""
        return True


@dataclass
class PositionState:
    """
    Complete state for a single position.
    
    Tracks everything needed to make hold/sell/add decisions:
    - Entry info and current prices
    - Hold duration
    - Conviction scoring
    - Technical and sentiment context
    """
    # Identity
    symbol: str
    
    # Entry details
    entry_price: float
    entry_time: datetime
    qty: int
    
    # Current state
    current_price: float
    high_since_entry: float = 0.0
    low_since_entry: float = 0.0
    last_update: datetime = None
    
    # Computed P&L
    pnl_dollars: float = 0.0
    pnl_pct: float = 0.0
    
    # Hold phase
    hold_phase: HoldPhase = HoldPhase.LOCKED
    hold_minutes: float = 0.0
    unlock_time: datetime = None  # When position becomes unlocked
    
    # Conviction
    conviction_score: float = 50.0  # 0-100 scale
    conviction_level: ConvictionLevel = ConvictionLevel.HOLD
    conviction_reasons: List[str] = field(default_factory=list)
    
    # Recommended action
    recommended_action: PositionAction = PositionAction.HOLD
    action_confidence: float = 0.5
    action_reasons: List[str] = field(default_factory=list)
    
    # Technical context
    atr_pct: float = 2.0
    rsi: float = 50.0
    near_support: bool = False
    near_resistance: bool = False
    s1_price: float = 0.0
    r1_price: float = 0.0
    
    # Sentiment context
    sentiment_score: float = 0.0  # -1 to +1
    recent_news: List[str] = field(default_factory=list)
    
    # Sector/momentum context
    sector_momentum: float = 0.0  # -1 to +1
    relative_strength: float = 0.0  # vs SPY
    
    # Risk gate triggers
    risk_triggers: List[str] = field(default_factory=list)
    is_critical: bool = False
    
    def __post_init__(self):
        """Compute derived fields."""
        self.last_update = self.last_update or datetime.now()
        self.high_since_entry = self.high_since_entry or self.current_price
        self.low_since_entry = self.low_since_entry or self.current_price
        self._compute_pnl()
        self._compute_hold_phase()
    
    def _compute_pnl(self):
        """Compute P&L metrics."""
        self.pnl_dollars = (self.current_price - self.entry_price) * self.qty
        self.pnl_pct = ((self.current_price - self.entry_price) / self.entry_price) * 100
    
    def _compute_hold_phase(self):
        """Determine informational hold phase based on entry time."""
        now = datetime.now()
        self.hold_minutes = (now - self.entry_time).total_seconds() / 60
        
        # Calculate unlock time (next market open after 1 trading day)
        # Simple approximation: 24 hours from entry
        self.unlock_time = self.entry_time + timedelta(hours=24)
        
        # Adjust for weekends
        if self.unlock_time.weekday() == 5:  # Saturday
            self.unlock_time += timedelta(days=2)
        elif self.unlock_time.weekday() == 6:  # Sunday
            self.unlock_time += timedelta(days=1)
        
        # Determine phase
        if self.is_critical:
            self.hold_phase = HoldPhase.CRITICAL
        elif now >= self.unlock_time:
            self.hold_phase = HoldPhase.UNLOCKED
        elif now >= self.unlock_time - timedelta(hours=1):
            self.hold_phase = HoldPhase.UNLOCKING
        else:
            self.hold_phase = HoldPhase.LOCKED
    
    def update_price(self, new_price: float):
        """Update current price and track high/low."""
        self.current_price = new_price
        self.high_since_entry = max(self.high_since_entry, new_price)
        self.low_since_entry = min(self.low_since_entry, new_price)
        self.last_update = datetime.now()
        self._compute_pnl()
    
    def mark_critical(self, reason: str):
        """Mark position as critical."""
        self.is_critical = True
        self.hold_phase = HoldPhase.CRITICAL
        self.risk_triggers.append(reason)
        self.recommended_action = PositionAction.EXIT
        self.action_confidence = 0.95
        self.action_reasons.append(f"CRITICAL: {reason}")
    
    def set_conviction(self, score: float, reasons: List[str]):
        """Set conviction score and level."""
        self.conviction_score = max(0, min(100, score))
        self.conviction_reasons = reasons
        
        # Map score to level
        if score < 0:
            self.conviction_level = ConvictionLevel.CRITICAL_EXIT
        elif score < 20:
            self.conviction_level = ConvictionLevel.SELL
        elif score < 40:
            self.conviction_level = ConvictionLevel.WEAK
        elif score < 60:
            self.conviction_level = ConvictionLevel.HOLD
        elif score < 80:
            self.conviction_level = ConvictionLevel.BUY
        else:
            self.conviction_level = ConvictionLevel.STRONG_BUY
    
    def set_action(self, action: PositionAction, confidence: float, reasons: List[str]):
        """Set recommended action."""
        self.recommended_action = action
        self.action_confidence = confidence
        self.action_reasons = reasons
    
    def can_act(self, pdt_status: PDTStatus) -> bool:
        """Check if we can act on this position."""
        if self.hold_phase == HoldPhase.CRITICAL:
            return True  # Always act on critical
        if self.hold_phase == HoldPhase.UNLOCKED:
            return True  # Free to act after unlock
        if self.hold_phase == HoldPhase.LOCKED:
            return True
        return True  # UNLOCKING phase - can act
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat(),
            'qty': self.qty,
            'current_price': self.current_price,
            'pnl_pct': self.pnl_pct,
            'pnl_dollars': self.pnl_dollars,
            'hold_phase': self.hold_phase.value,
            'hold_minutes': self.hold_minutes,
            'unlock_time': self.unlock_time.isoformat() if self.unlock_time else None,
            'conviction_score': self.conviction_score,
            'conviction_level': self.conviction_level.value,
            'conviction_reasons': self.conviction_reasons,
            'recommended_action': self.recommended_action.value,
            'action_confidence': self.action_confidence,
            'action_reasons': self.action_reasons,
            'is_critical': self.is_critical,
            'risk_triggers': self.risk_triggers
        }


@dataclass
class PortfolioState:
    """
    Complete portfolio state for decision making.
    
    Tracks all positions and capital allocation.
    """
    positions: Dict[str, PositionState] = field(default_factory=dict)
    pdt_status: PDTStatus = field(default_factory=PDTStatus)
    
    # Capital metrics
    total_equity: float = 0.0
    cash_available: float = 0.0
    buying_power: float = 0.0
    positions_value: float = 0.0
    
    # Portfolio limits
    max_positions: int = 7
    position_size_target: float = 3000.0
    position_size_max: float = 6000.0
    
    # State tracking
    last_update: datetime = None
    
    def __post_init__(self):
        self.last_update = self.last_update or datetime.now()
    
    @property
    def position_count(self) -> int:
        return len(self.positions)
    
    @property
    def open_slots(self) -> int:
        return max(0, self.max_positions - self.position_count)
    
    @property
    def total_pnl(self) -> float:
        return sum(p.pnl_dollars for p in self.positions.values())
    
    @property
    def total_pnl_pct(self) -> float:
        if self.positions_value == 0:
            return 0.0
        return (self.total_pnl / self.positions_value) * 100
    
    def get_unlocked_positions(self) -> List[PositionState]:
        """Get positions that are past 1D hold (can sell freely)."""
        return [p for p in self.positions.values() 
                if p.hold_phase in (HoldPhase.UNLOCKED, HoldPhase.UNLOCKING)]
    
    def get_locked_positions(self) -> List[PositionState]:
        """Get positions still in 1D hold period."""
        return [p for p in self.positions.values() 
                if p.hold_phase == HoldPhase.LOCKED]
    
    def get_critical_positions(self) -> List[PositionState]:
        """Get positions that need immediate exit."""
        return [p for p in self.positions.values() 
                if p.hold_phase == HoldPhase.CRITICAL]
    
    def get_rotation_candidates(self) -> List[PositionState]:
        """Get unlocked positions with weak conviction (rotation candidates)."""
        return [p for p in self.get_unlocked_positions() 
                if p.conviction_level in (ConvictionLevel.WEAK, ConvictionLevel.SELL)]
    
    def get_add_candidates(self) -> List[PositionState]:
        """Get positions strong enough to add to."""
        return [p for p in self.positions.values() 
                if p.conviction_level in (ConvictionLevel.STRONG_BUY, ConvictionLevel.BUY)
                and p.pnl_pct > 3.0]  # Only add to winners
    
    def add_position(self, pos: PositionState):
        """Add or update a position."""
        self.positions[pos.symbol] = pos
        self.last_update = datetime.now()
    
    def remove_position(self, symbol: str):
        """Remove a position."""
        if symbol in self.positions:
            del self.positions[symbol]
        self.last_update = datetime.now()
    
    def summary(self) -> str:
        """Get portfolio summary string."""
        lines = [
            f"Portfolio: {self.position_count}/{self.max_positions} positions",
            f"P&L: ${self.total_pnl:,.2f} ({self.total_pnl_pct:+.1f}%)",
            "PDT: disabled",
            f"Unlocked: {len(self.get_unlocked_positions())} | Locked: {len(self.get_locked_positions())}",
        ]
        
        critical = self.get_critical_positions()
        if critical:
            lines.append(f"[CRITICAL] {[p.symbol for p in critical]}")
        
        return "\n".join(lines)


# Convenience functions
def create_position_state(
    symbol: str,
    entry_price: float,
    current_price: float,
    qty: int,
    entry_time: datetime = None
) -> PositionState:
    """Create a new position state."""
    return PositionState(
        symbol=symbol,
        entry_price=entry_price,
        current_price=current_price,
        qty=qty,
        entry_time=entry_time or datetime.now()
    )


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(__name__).info('position_state.py loaded; use pytest/CLI for tests.')
