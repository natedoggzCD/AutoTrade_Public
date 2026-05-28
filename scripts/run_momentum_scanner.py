from __future__ import annotations

import argparse
import json
import logging

from autotrade.utils.logging_utils import configure_logging
from autotrade.utils.momentum_scanner import MomentumScanner
from config.config_loader import get_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the live momentum scanner.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan cycle and print the resulting artifact summary.",
    )
    args = parser.parse_args()

    cfg = get_config()
    configure_logging(
        cfg.logs_dir,
        level=cfg.logging.level,
        filename=cfg.logging.json_filename,
        max_bytes=cfg.logging.max_bytes,
        backup_count=cfg.logging.backup_count,
        console=cfg.logging.console,
    )

    scanner = MomentumScanner(config=cfg)
    if args.once:
        payload = scanner.scan_once()
        print(json.dumps(payload, indent=2))
        return 0

    logging.getLogger(__name__).info("Starting momentum scanner loop")
    scanner.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
