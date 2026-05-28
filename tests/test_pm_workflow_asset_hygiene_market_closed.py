from __future__ import annotations

from datetime import datetime

import pytest

from autotrade.core import pm_workflow as core_pm_workflow
from autotrade.execution import post_market_workflow as execution_pm_workflow


class _InactiveAssetClient:
    def get_asset(self, symbol):
        return type(
            "Asset",
            (),
            {
                "status": "inactive",
                "tradable": False,
            },
        )()


@pytest.mark.parametrize(
    "module",
    [core_pm_workflow, execution_pm_workflow],
)
def test_filter_plan_signals_defers_asset_lookup_when_market_closed(
    module, monkeypatch
):
    workflow = module.PostMarketWorkflow.__new__(module.PostMarketWorkflow)
    workflow._asset_eligibility_cache = {}
    workflow.client = _InactiveAssetClient()

    monkeypatch.setattr(
        module,
        "get_market_now",
        lambda: datetime(2026, 4, 16, 18, 30),
    )

    filtered, dropped = workflow._filter_plan_signals_by_asset_status(
        [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
    )

    assert [row["symbol"] for row in filtered] == ["AAPL", "MSFT"]
    assert dropped == []
