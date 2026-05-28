# Module Index (AutoTrade)

This index summarizes where each module lives after the cleanup.

## Core Orchestration
- `autotrade/core/autonomous_agent.py` — time-aware orchestrator and scheduler
- `autotrade/core/agentic_orchestrator.py` — TaskRouter and multi-agent health monitor
- `autotrade/core/auto_fix_pipeline.py` — auto-fix patch + validate + rollback pipeline
- `autotrade/core/workflow_manager.py` — stateful workflow manager (phases, watchdog, retries)
- `autotrade/core/agentic_advisor.py` — LangGraph advisor bridge for Day Manager
- `autotrade/core/day_manager.py` — intraday position management
- `autotrade/core/pm_workflow.py` — post-market planning and next-day plan generation
- `autotrade/core/premarket_agent.py` — premarket screening and watchlist adjustments
- `autotrade/core/premarket_analyzer.py` — premarket data fetch/analysis
- `autotrade/core/overnight_agent.py` — post-market rotation planning
- `autotrade/core/master_supervisor.py` — multi-agent supervisor and health monitor
- `autotrade/core/orchestrator.py` — general task orchestrator
- `autotrade/core/state_manager.py` — persisted system state manager
- `autotrade/core/position_scheduler.py` — position entry/exit scheduling
- `autotrade/core/position_thesis.py` — trade thesis generator and manager
- `autotrade/core/premarket_manager.py` — premarket lifecycle management
- `autotrade/core/decision_claw.py` — LLM-based trade decision gating
- `autotrade/core/workflow_claw.py` — LLM-based workflow adjustment
- `autotrade/core/claw_command_watcher.py` — watches for agentic commands from external tools
- `autotrade/core/daily_lessons_analyzer.py` — analyzes performance and learns lessons
- `autotrade/core/daily_review.py` — daily performance review generator
- `autotrade/core/eod_review.py` — end-of-day summary and artifact generator
- `autotrade/core/microstructure.py` — order book and spread analysis core
- `autotrade/core/research_artifacts.py` — research metadata and artifact tracker
- `autotrade/core/signal_mining.py` — autonomous signal discovery agent
- `autotrade/core/threshold_alerts.py` — system-wide threshold alert manager

## Local Coding Agent
- `autotrade/core/local_coding_agent.py` — Cline/SWE-agent-inspired Plan/Act coding agent (local Ollama models)
  - Plan mode: glm-4.7-flash (30B MoE, 198K ctx) explores with read-only tools → presents checklist
  - Act mode: qwen2.5-coder:14b (fast code gen/repair) implements with write/exec tools
  - Heavy coder: qwen3-coder-next (80B MoE, 256K ctx) for complex multi-file tasks
  - Reviewer: phi4-reasoning:14b (chain-of-thought critic)
  - Repo-map, patch-only edits, verify-every-step, mode/tool permissions, command blocklist
  - Thinking model support: `think: false` + thinking field fallback
  - CLI: `python autotrade/core/local_coding_agent.py "prompts/dev/PROMPT.md"` (add `--auto` to skip approval)
- `autotrade/core/repo_map.py` — Aider-inspired AST-based symbol index with PageRank file ranking (551 lines)
  - Extracts classes, functions, methods from all Python files
  - Builds file-level call graph and ranks by PageRank
  - Generates token-budgeted context map (~2K tokens for 130+ files)
- `autotrade/core/staged_prompt_processor.py` — staged prompt analysis + execution for large dev prompts
  - Analyzer: glm-4.7-flash, Planner/Reasoner: phi4-reasoning:14b, Coder: qwen2.5-coder:14b
- `autotrade/core/project_chat.py` — interactive project-aware chat with RAG, multi-model, `/agent` and `/staged` commands
  - AGENTIC_MODELS: improve→qwen2.5-coder:14b, debug→qwen2.5-coder:14b, architect→glm-4.7-flash

## Signals + Strategy
- `autotrade/signals/agentic_signal_generator.py` — agentic signal generation
- `autotrade/signals/lessons_screener.py` — lessons-based screener
- `autotrade/signals/unified_strategy.py` — unified entry/exit logic + lessons filter
- `autotrade/signals/conviction_engine.py` — conviction scoring
- `autotrade/signals/trade_learner.py` — lesson learning + journaling
- `autotrade/signals/contracts.py` — typed signal contracts
- `autotrade/signals/interfaces.py` — signal protocol interfaces
- `autotrade/signals/baseline_signals.py` — frozen baseline wrappers
- `autotrade/signals/pipeline.py` — modular signal pipeline
- `autotrade/signals/registry.py` — central signal registry
- `autotrade/signals/regime_router.py` — regime + trend routing
- `autotrade/signals/screener_v2.py` — technical factor scoring engine
- `autotrade/signals/universe_scanner.py` — fast DuckDB universe discovery
- `autotrade/signals/vwap_universe_scanner.py` — VWAP-centric screening
- `autotrade/signals/momentum_engine.py` — intraday momentum tracker
- `autotrade/signals/support_resistance.py` — S/R level estimation
- `autotrade/signals/strategy_pool.py` — manager for validated signal strategies
- `autotrade/signals/inverse_etf_screener.py` — market-hedge screening

## Alpha Signal Zoo
- `autotrade/signals/alpha/ts_momentum.py` — time-series momentum
- `autotrade/signals/alpha/xs_momentum.py` — cross-sectional momentum
- `autotrade/signals/alpha/mean_reversion.py` — RSI stretch/snapback
- `autotrade/signals/alpha/breakout.py` — squeeze-to-expansion breakout
- `autotrade/signals/alpha/pullback.py` — daily trend + hourly pullback reclaim
- `autotrade/signals/alpha/pairs.py` — spread z-score reversion
- `autotrade/signals/alpha/fundamental.py` — squeeze and fundamental alpha
- `autotrade/signals/alpha/pead.py` — post-earnings announcement drift
- `autotrade/signals/alpha/inverse_etf.py` — hedging-ready inverse ETF signals

## Data Ingestion
- `autotrade/data_ingestion/schemas.py` — typed dataclasses (`DataRequest`, `DataFrameSnapshot`, `DataFreshnessStatus`, `IngestionHealthReport`)
- `autotrade/data_ingestion/interfaces.py` — protocol interfaces (`HistoricalDataSource`, `RealtimePriceSource`, `DataBootstrapper`)
- `autotrade/data_ingestion/bootstrap.py` — centralized parquet bootstrap + H5 conversion (`ensure_core_market_data_ready()`)
- `autotrade/data_ingestion/paths.py` — shared path resolution
- `autotrade/data_ingestion/errors.py` — typed ingestion exceptions
- `autotrade/data_ingestion/adapters.py` — compatibility adapters (`LocalDataProviderAdapter`, `DataSyncAdapter`, `DataQualityAdapter`)
- `autotrade/data_ingestion/gateway.py` — unified data access facade
- `autotrade/data_ingestion/fast_cache.py` — high-performance data caching layer
- `autotrade/data_ingestion/stream_bridge.py` — bridges historical and real-time data streams

## Feature Engineering
- `autotrade/feature_engineering/schemas.py` — typed dataclasses
- `autotrade/feature_engineering/interfaces.py` — protocol interfaces
- `autotrade/feature_engineering/technical.py` — core technical indicators
- `autotrade/feature_engineering/volatility_volume.py` — volatility/volume features
- `autotrade/feature_engineering/trend.py` — 3 orthogonal trend definitions
- `autotrade/feature_engineering/regime.py` — regime context features
- `autotrade/feature_engineering/momentum.py` — momentum features
- `autotrade/feature_engineering/reversal.py` — mean reversion features
- `autotrade/feature_engineering/breakout.py` — Donchian and squeeze features
- `autotrade/feature_engineering/pairs.py` — stat-arb features
- `autotrade/feature_engineering/vlm.py` — vision-language model features
- `autotrade/feature_engineering/pipeline.py` — multi-timeframe assembly
- `autotrade/feature_engineering/adapters.py` — compatibility adapters

## Execution
- `autotrade/execution/execution_engine.py` — main order execution engine
- `autotrade/execution/alpaca_adapter.py` — Alpaca API adapter
- `autotrade/execution/order_manager.py` — lifecycle manager for active orders
- `autotrade/execution/order_optimizer.py` — limit price and timing optimization
- `autotrade/execution/price_logic.py` — logic for slippage-controlled entry/exit
- `autotrade/execution/chase_logic.py` — intelligent bid/ask chasing
- `autotrade/execution/local_exit_manager.py` — manages local (non-server) exit triggers
- `autotrade/execution/router.py` — order routing logic
- `autotrade/execution/sim_adapter.py` — simulated execution adapter
- `autotrade/execution/post_market_workflow.py` — post-close settlement and reporting

## Risk + Portfolio
- `autotrade/risk/contracts.py` — typed risk dataclasses
- `autotrade/risk/interfaces.py` — protocol interfaces
- `autotrade/risk/policy_engine.py` — deterministic policy evaluator
- `autotrade/risk/risk_gate.py` — fast-path risk rules
- `autotrade/risk/portfolio_rotator.py` — rotation/replace logic
- `autotrade/risk/strategy_failsafe.py` — persisted failsafe state manager
- `autotrade/risk/position_state.py` — position/portfolio state models
- `autotrade/risk/day_trade_tracker.py` — PDT tracking
- `autotrade/risk/hedging_monitor.py` — portfolio-wide hedging and exposure monitor
- `autotrade/risk/inverse_etf_manager.py` — manages index-hedge positions
- `autotrade/risk/sizing.py` — position sizing logic

## Backtesting
- `autotrade/backtesting/engine.py` — core event-driven backtest engine
- `autotrade/backtesting/duckdb_backtester.py` — fast vectorized SQL-based backtester
- `autotrade/backtesting/strategy_backtester.py` — strategy-specific backtest runner
- `autotrade/backtesting/signal_validator.py` — historical lookalike signal validation
- `autotrade/backtesting/evaluation.py` — performance evaluation metrics
- `autotrade/backtesting/hypothesis_engine.py` — automated hypothesis testing
- `autotrade/backtesting/leakage_detector.py` — checks for look-ahead bias
- `autotrade/backtesting/monte_carlo.py` — equity curve simulation
- `autotrade/backtesting/statistical_controls.py` — significance and overfitting tests
- `autotrade/backtesting/artifact_persistence.py` — backtest result storage

## Replay
- `autotrade/replay/runtime_session_replay.py` — re-runs live sessions for debugging
- `autotrade/replay/minute_bar_archive.py` — DuckDB-backed minute bar storage
- `autotrade/replay/decision_claw_historical_eval.py` — backtests LLM decision logic
- `autotrade/replay/decision_claw_overnight_hold_eval.py` — evaluates overnight hold decisions

## Reasoning
- `autotrade/reasoning/daily_review_orchestrator.py` — orchestrates daily report reviews
- `autotrade/reasoning/sequential_engine.py` — step-by-step reasoning for trade setup analysis

## Market/News Analysis
- `autotrade/analysis/news_sentiment.py` — news sentiment wrapper
- `autotrade/analysis/finbert_analyzer.py` — FinBERT sentiment model
- `autotrade/analysis/stocktwits_sentiment.py` — Stocktwits sentiment
- `autotrade/analysis/searxng_client.py` — SearXNG search client
- `autotrade/analysis/financial_checks.py` — earnings and balance sheet hygiene
- `autotrade/analysis/market_regime.py` — regime classification (Trend/Chop/Crisis)
- `autotrade/analysis/pattern_playbooks.py` — technical pattern classification
- `autotrade/analysis/performance_reporting.py` — generate P&L and risk reports
- `autotrade/analysis/post_market.py` — post-market analysis runner
- `autotrade/analysis/ranking.py` — candidate ranking algorithms
- `autotrade/analysis/rotation.py` — sector rotation analysis
- `autotrade/analysis/sequential_outcome_scorer.py` — evaluates sequence of events for success probability
- `autotrade/analysis/sequential_shadow_runner.py` — runs shadow trades to test sequential reasoning
- `autotrade/analysis/sequential_shadow_schema.py` — schemas for shadow trade analysis
- `autotrade/analysis/execution_quality_tracker.py` — tracks slippage and fill quality
- `autotrade/analysis/order_book_analyzer.py` — Level 2 / order book pressure analysis

## Advisors
- `autotrade/advisors/position_advisor.py` — legacy advisor (non-agentic)

## Utilities
- `autotrade/utils/logging_utils.py` — JSON logging helpers
- `autotrade/utils/openai_client.py` — OpenAI fallback client
- `autotrade/utils/safe_logging.py` — Windows-safe logging
- `autotrade/utils/local_data_provider.py` — Local data (DownDay parquet) source
- `autotrade/utils/news_cache.py` — SQLite-backed news article cache
- `autotrade/utils/data_sync.py` — DuckDB/parquet sync manager
- `autotrade/utils/data_quality.py` — Data quality gate (freshness, completeness)
- `autotrade/utils/alpaca_client_factory.py` — centralized Alpaca client creation
- `autotrade/utils/market_time.py` — market-hours and holiday awareness
- `autotrade/utils/vwap_calculator.py` — high-fidelity VWAP computation
- `autotrade/utils/vlm_chart_generator.py` — generates 4-panel technical charts
- `autotrade/utils/vl_chart_validator.py` — validates charts via vision models or rules
- `autotrade/utils/youtube_rag.py` — RAG interface for YouTube transcript knowledge
- `autotrade/utils/mcp_client.py` — client for Model Context Protocol servers
- `autotrade/utils/agentic_log_router.py` — routes logs to specific agentic contexts
- `autotrade/utils/diagnostic.py` — system diagnostic utilities

## Monitoring + Observability
- `autotrade/monitoring/contracts.py` — system health and performance metric schemas
- `autotrade/monitoring/collector.py` — centralized metrics collector
- `autotrade/monitoring/dashboard.py` — real-time monitoring dashboard backend
- `autotrade/monitoring/alerts.py` — system-wide alerting logic
- `autotrade/monitoring/halt_logic.py` — trading halt detection and response
- `autotrade/monitoring/liquidity_gate.py` — liquidity-based execution gating
- `autotrade/monitoring/reporting.py` — system health reporting

## Config + Workflow
- `config/` — config loader + YAML settings
- `langgraph_workflow/` — LangGraph multi-agent pipeline
- `tools/` — generated analysis utilities
- `research/` — cached research + analysis artifacts
- `tests/` — validation scripts and smoke tests

## Entry Points
- **`autonomous_agent_main.bat`** — What user runs each day
