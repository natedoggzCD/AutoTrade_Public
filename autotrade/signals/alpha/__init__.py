"""
Alpha Signal Zoo
=================

This package contains the alpha family signal implementations.

Each module implements a specific alpha strategy:
- ts_momentum: Time-series momentum (12-1, 6-1 directional, MA trend)
- xs_momentum: Cross-sectional momentum (rank-and-rebalance)
- mean_reversion: Mean reversion (RSI, down-day exhaustion)
- breakout: Breakout/volatility expansion (squeeze to expansion)
- pullback: Pullback continuation (trend context + pullback reclaim)
- pairs: Stat-arb/pairs (spread z-score reversion)

Each signal model must implement the SignalModel protocol from interfaces.py.
"""

from autotrade.signals.alpha.ts_momentum import (
    TSMomentum12_1,
    TSMomentum6_1,
    MATrendSignal,
    register_ts_momentum_signals,
)
from autotrade.signals.alpha.xs_momentum import (
    XSMomentumSignal,
    register_xs_momentum_signals,
)
from autotrade.signals.alpha.mean_reversion import (
    RSISignal,
    DownDayExhaustionSignal,
    register_mean_reversion_signals,
)
from autotrade.signals.alpha.breakout import (
    SqueezeExpansionSignal,
    DonchianBreakoutSignal,
    register_breakout_signals,
)
from autotrade.signals.alpha.pullback import (
    PullbackContinuationSignal,
    register_pullback_signals,
)
from autotrade.signals.alpha.pairs import (
    PairsSignal,
    register_pairs_signals,
)
from autotrade.signals.alpha.fundamental import (
    SqueezeAlphaSource,
    register_fundamental_signals,
)
from autotrade.signals.alpha.pead import (
    PEADAlphaSource,
    register_pead_signals,
)
from autotrade.signals.alpha.inverse_etf import (
    InverseETFAlphaSource,
    register_inverse_etf_signals,
)

__all__ = [
    "TSMomentum12_1",
    "TSMomentum6_1",
    "MATrendSignal",
    "XSMomentumSignal",
    "RSISignal",
    "DownDayExhaustionSignal",
    "SqueezeExpansionSignal",
    "DonchianBreakoutSignal",
    "PullbackContinuationSignal",
    "PairsSignal",
    "PEADAlphaSource",
    "SqueezeAlphaSource",
    "InverseETFAlphaSource",
    "register_ts_momentum_signals",
    "register_xs_momentum_signals",
    "register_mean_reversion_signals",
    "register_breakout_signals",
    "register_pullback_signals",
    "register_pairs_signals",
    "register_fundamental_signals",
    "register_pead_signals",
    "register_inverse_etf_signals",
    "register_all_alpha_signals",
]


def register_all_alpha_signals() -> None:
    """Register all alpha family signals with the global registry."""
    register_ts_momentum_signals()
    register_xs_momentum_signals()
    register_mean_reversion_signals()
    register_breakout_signals()
    register_pullback_signals()
    register_pairs_signals()
    register_fundamental_signals()
    register_pead_signals()
    register_inverse_etf_signals()
