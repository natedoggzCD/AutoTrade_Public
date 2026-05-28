import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


class RUnitSizer:
    """
    Calculates position size based on a fixed risk unit (1R).
    Ensures that the dollar loss on a stop-hit is consistent across trades.
    """

    def __init__(
        self,
        total_equity: float,
        risk_pct: float = 0.01,
        max_notional_pct: float = 0.05,
        min_qty: int = 1,
    ):
        self.total_equity = total_equity
        self.risk_pct = risk_pct  # e.g., 0.01 = 1% risk per trade
        self.max_notional_pct = (
            max_notional_pct  # e.g., 0.05 = max 5% of equity per position
        )
        self.min_qty = min_qty

    def calculate_quantity(
        self,
        entry_price: float,
        stop_price: float,
        *,
        side: str = "long",
        size_multiplier: float = 1.0,
        current_price: Optional[float] = None,
    ) -> int:
        """
        Calculates share quantity such that risk_per_share * qty = risk_unit.

        The notional cap is an absolute ceiling after size_multiplier is applied.
        This keeps stacked multipliers from scaling long or short positions beyond
        the configured concentration limit.
        """
        side_key = str(side or "long").lower().strip()
        sizing_price = float(current_price or entry_price)
        if entry_price <= 0 or stop_price <= 0 or sizing_price <= 0:
            logger.warning(
                f"Invalid pricing for sizing: entry={entry_price}, stop={stop_price}"
            )
            return 0
        if side_key == "long":
            risk_per_share = entry_price - stop_price
        elif side_key == "short":
            risk_per_share = stop_price - entry_price
        else:
            raise ValueError(f"Invalid side: {side}")
        if risk_per_share <= 0 or not math.isfinite(risk_per_share):
            logger.warning(
                "Invalid stop placement for %s: entry=%s, stop=%s",
                side_key,
                entry_price,
                stop_price,
            )
            return 0

        # 1. Calculate Dollar Risk Unit (1R)
        risk_unit = self.total_equity * self.risk_pct

        # 3. Raw R-Unit Quantity
        raw_qty = int(risk_unit / risk_per_share)
        try:
            multiplier = max(0.0, float(size_multiplier or 0.0))
        except (TypeError, ValueError):
            multiplier = 1.0
        scaled_qty = int(raw_qty * multiplier)

        # 4. Enforce Max Notional Cap (Portfolio Concentration Limit)
        max_notional = self.total_equity * self.max_notional_pct
        notional_qty = int(max_notional / sizing_price)

        final_qty = min(scaled_qty, notional_qty)

        # 5. Enforce Minimum Quantity
        if final_qty < self.min_qty:
            # If we can't afford the minimum quantity within risk limits, return 0
            # or optionally return min_qty if risk is very close.
            return 0

        return final_qty

    def get_risk_report(
        self,
        entry_price: float,
        stop_price: float,
        qty: int,
        *,
        side: str = "long",
    ) -> str:
        """Returns a string describing the risk profile of the calculated size."""
        side_key = str(side or "long").lower().strip()
        risk_per_share = (
            entry_price - stop_price if side_key == "long" else stop_price - entry_price
        )
        risk_total = risk_per_share * qty
        notional = entry_price * qty
        risk_basis_points = (risk_total / self.total_equity) * 10000

        return (
            f"Side: {side_key} | Size: {qty} shares | Notional: ${notional:.2f} ({notional / self.total_equity * 100:.1f}%) | "
            f"Risk: ${risk_total:.2f} ({risk_basis_points:.0f} bps)"
        )
