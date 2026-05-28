# Signal Generation Hierarchy

AutoTrade uses a multi-layered signal generation process to move from a universe of 4,000+ stocks to a high-conviction daily trade plan.

## 1. Discovery Layer (The "Universe")
*   **Universe Scanner** (`autotrade/signals/universe_scanner.py`):
    *   **Purpose**: Fast, full-coverage screening of the DownDay parquet dataset.
    *   **Technology**: DuckDB for SQL-speed queries.
    *   **Output**: Top ~200 candidates based on basic technical filters (price, volume, SMA alignment).

## 2. Quantitative Layer (The "Alpha Zoo")
*   **Modular Pipeline** (`autotrade/signals/pipeline.py`):
    *   **Purpose**: Orchestrates 14+ alpha models across 8 families.
    *   **Alpha Families**:
        *   `ts_momentum`: Time-series momentum.
        *   `xs_momentum`: Cross-sectional momentum.
        *   `mean_reversion`: RSI/Stochastic-based reversals.
        *   `breakout`: Donchian channel breakouts.
        *   `pullback`: EMA/SMA pullback setups.
        *   `pairs`: Statistical arbitrage pairs.
        *   `fundamental`: Squeeze and fundamental-based setups.
        *   `pead`: Post-Earnings Announcement Drift.
        *   `inverse_etf`: Market-hedge inverse ETF setups.
    *   **Regime Routing**: Dynamically adjusts family weights based on market regime (Trend, Chop, Crisis).

## 3. Technical Screening Layer
*   **Screener V2** (`autotrade/signals/screener_v2.py`):
    *   **Purpose**: Multi-factor technical scoring.
    *   **Scoring**: Combines SMA5 curl, EMA alignment, RSI momentum, and MACD confirmation.
    *   **Role**: Refines the candidates from the Discovery layer into a ranked list.

## 4. Agentic/Intelligence Layer
*   **Agentic Signal Generator** (`autotrade/signals/agentic_signal_generator.py`):
    *   **Purpose**: Final "deep" analysis of the top candidates.
    *   **Workflow**:
        1.  **Financial Check**: Earnings risk, balance sheet health.
        2.  **Sentiment**: News and StockTwits analysis via LLM.
        3.  **VLM Confirmation**: (Optional) Visual chart validation.
        4.  **LLM Synthesis**: Final setup quality score (0-100) and action recommendation.

## 5. Validation & Planning
*   **Signal Validator** (`autotrade/backtesting/signal_validator.py`):
    *   **Purpose**: Historical lookalike validation.
    *   **Rule**: Rejects signals that have poor historical win rates in similar market conditions.
*   **Unified Strategy**:
    *   **Purpose**: Finalizes the `morning_game_plan.json`.
    *   **Role**: Sets ATR-based stops and targets.

---

## Signal Flow Summary
`UniverseScanner (4000+)` -> `ScreenerV2 (~200)` -> `AlphaZoo (Enrichment)` -> `AgenticGenerator (~30)` -> `SignalValidator` -> `Trade Plan (~10-15)`
