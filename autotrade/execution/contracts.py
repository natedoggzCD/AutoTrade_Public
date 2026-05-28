from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]
ExecutionStatus = Literal[
    "submitted",
    "partial",
    "filled",
    "rejected",
    "canceled",
    "cannot_fill",
    "failed",
]


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: OrderSide
    qty: int
    action: str
    reason: str = ""
    reference_price: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderRequest:
    intent: OrderIntent
    order_type: OrderType = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    allow_partial_fill: bool = True
    urgency_tier: str = "normal"
    intended_price: Optional[float] = None
    decision_price: Optional[float] = None
    slippage_budget_bps: Optional[int] = None
    replace_count: int = 0
    record_journal: bool = False
    extended_hours: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderFill:
    qty: int
    price: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    fee: float = 0.0
    slippage_cost: float = 0.0
    spread_cost: float = 0.0


@dataclass(frozen=True)
class ExecutionError:
    code: str
    message: str
    retryable: bool = False
    raw: Any = None


@dataclass(frozen=True)
class ExecutionReport:
    order_id: str
    status: ExecutionStatus
    symbol: str
    side: OrderSide
    requested_qty: int
    filled_qty: int
    avg_fill_price: float
    fills: List[OrderFill] = field(default_factory=list)
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_spread: float = 0.0
    latency_ms: int = 0
    venue: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error: Optional[ExecutionError] = None
    raw: Any = None
    urgency_tier: str = "normal"
    intended_price: Optional[float] = None
    decision_price: Optional[float] = None
    slippage_bps: Optional[float] = None
    time_to_first_fill_ms: Optional[int] = None
    replace_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def remaining_qty(self) -> int:
        return max(0, int(self.requested_qty) - int(self.filled_qty))

    @property
    def is_terminal(self) -> bool:
        return self.status in {"filled", "rejected", "canceled", "cannot_fill", "failed"}
