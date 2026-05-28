# Signal Generation Pipeline

This directory contains the logic for identifying, generating, and validating trading signals.

## Signal Generation Hierarchy

AutoTrade follows a structured, multi-stage pipeline to ensure signal quality:

1.  **Screener V2 (Quantitative)**:
    -   Initial filtering based on technical criteria (price, volume, ATR).
    -   Reduces the universe to a manageable pool of high-probability candidates.
2.  **LLM Generation (Agentic Reasoning)**:
    -   Candidates from the screener are passed to the `AgenticSignalGenerator`.
    -   Uses local LLMs (via Ollama) to analyze news, social sentiment (Stocktwits), and chart patterns.
    -   Applies "Alpha Models" to score candidates based on specific strategies (Trend, Mean Reversion, etc.).
3.  **Backtest Validation**:
    -   Top-ranked signals are passed to the `StrategyValidator`.
    -   Performs a "Walk-Forward" validation to check historical performance of the specific setup.
    -   Signals that fail minimum win rate or profit factor thresholds are discarded.

## Primary vs. Fallback Pathways

-   **Primary (Full Agentic)**: Screener -> LLM Analysis -> Backtest Validation -> Premarket Manager.
-   **Fallback (Policy-Driven)**: If LLMs or Backtest components are unavailable, the system falls back to a policy-driven approach using quantitative technical indicators (e.g., VWAP, S/R levels) within the `PremarketManager`.

## Unified Premarket Handling

As of v0.12.0, all premarket analysis is consolidated into `autotrade/core/premarket_manager.py`. This module is responsible for:
-   Gap analysis and liquidity scoring.
-   VWAP tracking.
-   Technical S/R context estimation.
-   Integrating YouTube-based market intelligence.
