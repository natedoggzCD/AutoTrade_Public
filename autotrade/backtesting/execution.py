from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

Side = Literal["buy", "sell"]


@dataclass(frozen=True)
class ExecutionModel:
    """
    Lightweight execution model for backtests.

    - Spread: modeled as half-spread on each side (buy pays ask, sell hits bid)
    - Slippage: additional adverse move on each fill
    - Commission: charged on notional (both entry + exit)
    - Partial fills: simulated via a deterministic RNG (seeded)
    """

    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    spread_pct: float = 0.001

    partial_fill_min: float = 1.0
    partial_fill_max: float = 1.0

    seed: int = 42

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def effective_price(self, mid_price: float, side: Side) -> float:
        price = float(mid_price) if mid_price is not None else 0.0
        if price <= 0:
            return 0.0

        half_spread = self.spread_pct / 2.0
        slip = self.slippage_pct

        if side == "buy":
            return price * (1.0 + half_spread + slip)
        if side == "sell":
            return price * (1.0 - half_spread - slip)

        raise ValueError(f"Unknown side: {side}")

    def commission(self, notional: float) -> float:
        return abs(float(notional)) * float(self.commission_pct)

    def sample_fill_fraction(self, rng: np.random.Generator) -> float:
        lo = max(0.0, min(1.0, float(self.partial_fill_min)))
        hi = max(lo, min(1.0, float(self.partial_fill_max)))
        if hi - lo < 1e-9:
            return lo
        return float(rng.uniform(lo, hi))

