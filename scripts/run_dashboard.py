from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message='Field name "stop_price" shadows an attribute in parent')

from autotrade.monitoring.dashboard import PerformanceDashboard
from autotrade.monitoring.reporting import ReportingEngine


def build_dashboard() -> PerformanceDashboard:
    reporting_engine = ReportingEngine(output_dir=Path("logs/reports"))
    reporting_engine.logger.setLevel(logging.CRITICAL)
    reporting_engine.logger.disabled = True
    return PerformanceDashboard(
        reporting_engine=reporting_engine,
        host="127.0.0.1",
        port=8501,
    )


def main() -> int:
    dashboard = build_dashboard()
    payload = dashboard.render()
    print(json.dumps(payload["context"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
