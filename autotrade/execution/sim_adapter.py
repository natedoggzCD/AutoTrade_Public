from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np

from autotrade.backtesting.execution import ExecutionModel
from autotrade.execution.contracts import (
    ExecutionError,
    ExecutionReport,
    OrderFill,
    OrderRequest,
)


@dataclass(frozen=True)
class SimAdapterConfig:
    seed: int = 42
    latency_ms_min: int = 50
    latency_ms_max: int = 500
    partial_fill_enabled: bool = True
    cannot_fill_probability: float = 0.02
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    spread_pct: float = 0.001


class SimExecutionAdapter:
    def __init__(self, config: Optional[SimAdapterConfig] = None):
        self.config = config or SimAdapterConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._execution_model = ExecutionModel(
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            spread_pct=self.config.spread_pct,
            partial_fill_min=0.85 if self.config.partial_fill_enabled else 1.0,
            partial_fill_max=1.0,
            seed=self.config.seed,
        )
        self._counter = 0
        self._orders: Dict[str, ExecutionReport] = {}

    def _next_order_id(self) -> str:
        self._counter += 1
        return f"sim-{self._counter:06d}"

    def submit_order(self, request: OrderRequest) -> ExecutionReport:
        order_id = self._next_order_id()
        latency_ms = int(
            self._rng.integers(
                max(0, self.config.latency_ms_min),
                max(self.config.latency_ms_min, self.config.latency_ms_max) + 1,
            )
        )

        requested_qty = max(0, int(request.intent.qty or 0))
        ref_price = float(
            request.intent.reference_price
            or request.limit_price
            or 0.0
        )
        intended_price = float(
            request.intended_price
            or request.intent.reference_price
            or request.limit_price
            or 0.0
        )
        decision_price = float(
            request.decision_price
            or request.limit_price
            or request.intent.reference_price
            or 0.0
        )
        urgency_tier = str(request.urgency_tier or "normal").lower()
        replace_count = int(request.replace_count or 0)

        is_limit_waiting = False
        if request.order_type == "limit" and ref_price > 0 and request.limit_price is not None:
            if request.intent.side == "buy" and float(request.limit_price) < ref_price:
                is_limit_waiting = True
            if request.intent.side == "sell" and float(request.limit_price) > ref_price:
                is_limit_waiting = True

        if requested_qty <= 0:
            report = ExecutionReport(
                order_id=order_id,
                status="failed",
                symbol=request.intent.symbol,
                side=request.intent.side,
                requested_qty=0,
                filled_qty=0,
                avg_fill_price=0.0,
                latency_ms=latency_ms,
                venue="sim",
                urgency_tier=urgency_tier,
                intended_price=intended_price if intended_price > 0 else None,
                decision_price=decision_price if decision_price > 0 else None,
                replace_count=replace_count,
                error=ExecutionError(code="invalid_qty", message="qty must be > 0"),
                metadata={"order_type": request.order_type},
            )
            self._orders[order_id] = report
            return report

        if is_limit_waiting:
            report = ExecutionReport(
                order_id=order_id,
                status="submitted",
                symbol=request.intent.symbol,
                side=request.intent.side,
                requested_qty=requested_qty,
                filled_qty=0,
                avg_fill_price=0.0,
                latency_ms=latency_ms,
                venue="sim",
                urgency_tier=urgency_tier,
                intended_price=intended_price if intended_price > 0 else None,
                decision_price=decision_price if decision_price > 0 else None,
                replace_count=replace_count,
                metadata={"order_type": request.order_type, "limit_price": request.limit_price},
            )
            self._orders[order_id] = report
            return report

        cannot_fill = bool(self._rng.random() < float(self.config.cannot_fill_probability))
        if cannot_fill:
            report = ExecutionReport(
                order_id=order_id,
                status="cannot_fill",
                symbol=request.intent.symbol,
                side=request.intent.side,
                requested_qty=requested_qty,
                filled_qty=0,
                avg_fill_price=0.0,
                latency_ms=latency_ms,
                venue="sim",
                urgency_tier=urgency_tier,
                intended_price=intended_price if intended_price > 0 else None,
                decision_price=decision_price if decision_price > 0 else None,
                replace_count=replace_count,
                metadata={"order_type": request.order_type},
            )
            self._orders[order_id] = report
            return report

        fill_fraction = 1.0
        if request.allow_partial_fill and self.config.partial_fill_enabled:
            fill_fraction = self._execution_model.sample_fill_fraction(self._rng)

        filled_qty = int(requested_qty * fill_fraction)
        if filled_qty <= 0 and requested_qty > 0:
            filled_qty = 1
        filled_qty = min(filled_qty, requested_qty)

        fill_price = float(self._execution_model.effective_price(ref_price, request.intent.side))
        notional = float(fill_price * filled_qty)
        fees = float(self._execution_model.commission(notional))
        spread_cost = abs(notional) * float(self.config.spread_pct) / 2.0
        slippage_cost = abs(notional) * float(self.config.slippage_pct)

        status = "filled" if filled_qty >= requested_qty else "partial"
        slippage_bps = None
        if filled_qty > 0 and fill_price > 0 and intended_price > 0:
            side_sign = 1.0 if request.intent.side == "buy" else -1.0
            slippage_bps = (
                ((fill_price - intended_price) / intended_price) * 10000.0 * side_sign
            )
        fills = []
        if filled_qty > 0 and fill_price > 0:
            fills.append(
                OrderFill(
                    qty=filled_qty,
                    price=fill_price,
                    fee=fees,
                    spread_cost=spread_cost,
                    slippage_cost=slippage_cost,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        report = ExecutionReport(
            order_id=order_id,
            status=status,
            symbol=request.intent.symbol,
            side=request.intent.side,
            requested_qty=requested_qty,
            filled_qty=filled_qty,
            avg_fill_price=fill_price if filled_qty > 0 else 0.0,
            fills=fills,
            total_fees=fees,
            total_slippage=slippage_cost,
            total_spread=spread_cost,
            latency_ms=latency_ms,
            venue="sim",
            urgency_tier=urgency_tier,
            intended_price=intended_price if intended_price > 0 else None,
            decision_price=decision_price if decision_price > 0 else None,
            slippage_bps=slippage_bps,
            time_to_first_fill_ms=latency_ms if filled_qty > 0 else None,
            replace_count=replace_count,
            metadata={"order_type": request.order_type, "fill_fraction": fill_fraction},
        )
        self._orders[order_id] = report
        return report

    def cancel_order(self, order_id: str) -> ExecutionReport:
        existing = self._orders.get(order_id)
        if existing is None:
            return ExecutionReport(
                order_id=order_id,
                status="failed",
                symbol="",
                side="buy",
                requested_qty=0,
                filled_qty=0,
                avg_fill_price=0.0,
                venue="sim",
                error=ExecutionError(code="order_not_found", message=f"{order_id} not found"),
            )

        if existing.is_terminal:
            return existing

        canceled = ExecutionReport(
            order_id=existing.order_id,
            status="canceled",
            symbol=existing.symbol,
            side=existing.side,
            requested_qty=existing.requested_qty,
            filled_qty=existing.filled_qty,
            avg_fill_price=existing.avg_fill_price,
            fills=existing.fills,
            total_fees=existing.total_fees,
            total_slippage=existing.total_slippage,
            total_spread=existing.total_spread,
            latency_ms=existing.latency_ms,
            venue=existing.venue,
            urgency_tier=existing.urgency_tier,
            intended_price=existing.intended_price,
            decision_price=existing.decision_price,
            slippage_bps=existing.slippage_bps,
            time_to_first_fill_ms=existing.time_to_first_fill_ms,
            replace_count=existing.replace_count,
            metadata=dict(existing.metadata),
        )
        self._orders[order_id] = canceled
        return canceled
