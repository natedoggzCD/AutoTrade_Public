from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", message='Field name "stop_price" shadows an attribute in parent')

import streamlit as st

from scripts.run_dashboard import build_dashboard


def main() -> None:
    dashboard = build_dashboard()
    dashboard.render(st)


if __name__ == "__main__":
    main()
