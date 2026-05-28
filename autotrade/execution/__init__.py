from autotrade.execution.contracts import (
    ExecutionError,
    ExecutionReport,
    OrderFill,
    OrderIntent,
    OrderRequest,
)
from autotrade.execution.sim_adapter import SimAdapterConfig, SimExecutionAdapter
from autotrade.execution.price_logic import OrderPriceCalculator


class _LiveTradingDisabled:
    """Placeholder â€” live execution modules are not included in this repo.

    This repo is a backtesting / alpha-mining environment only.
    AlpacaExecutionAdapter, ExecutionRouter, and related live-trading
    components have been intentionally removed.
    """

    def __init__(self, *a, **kw):
        raise NotImplementedError(
            "Live trading is disabled in this repo. "
            "Use SimExecutionAdapter for backtesting."
        )


AlpacaExecutionAdapter = _LiveTradingDisabled
ExecutionRouter = _LiveTradingDisabled
build_execution_router = _LiveTradingDisabled
make_order_request = _LiveTradingDisabled

__all__ = [
    "ExecutionError",
    "ExecutionReport",
    "OrderFill",
    "OrderIntent",
    "OrderRequest",
    "AlpacaExecutionAdapter",
    "OrderPriceCalculator",
    "SimAdapterConfig",
    "SimExecutionAdapter",
    "ExecutionRouter",
    "build_execution_router",
    "make_order_request",
]
