import pandas as pd
import yfinance as yf
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class PostMarketAnalyzer:
    """
    Evaluates intraday executions against the pre-market plan.
    Quantifies adherence, attributes performance, and identifies systemic failures.
    """

    def __init__(self):
        self.metrics = {}

    def evaluate_overnight_watchlist(self, watchlist: List[Dict], target_date: str) -> Dict:
        """
        Evaluates the performance of the top 50 symbols from the overnight watchlist.
        Calculates peak (OH) and close (OC) returns for the specified date.
        """
        top_50 = watchlist[:50]
        symbols = [item["symbol"] for item in top_50]
        
        day_value = date.fromisoformat(target_date)
        start_str = day_value.isoformat()
        end_str = (day_value + pd.Timedelta(days=1)).isoformat()

        logger.info(f"Evaluating overnight watchlist for {target_date} ({len(symbols)} symbols)")
        
        results = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Use 1m for precise session calculation
                df = ticker.history(start=start_str, end=end_str, interval="1m")
                
                if df.empty:
                    df_daily = ticker.history(start=start_str, end=end_str)
                    if df_daily.empty:
                        continue
                    o, h, l, c = df_daily['Open'].iloc[0], df_daily['High'].iloc[0], df_daily['Low'].iloc[0], df_daily['Close'].iloc[0]
                else:
                    if df.index.tz is None:
                        df.index = df.index.tz_localize("UTC").tz_convert("America/New_York")
                    else:
                        df.index = df.index.tz_convert("America/New_York")
                    
                    mask = (df.index.time >= datetime.strptime("09:30", "%H:%M").time()) & \
                           (df.index.time <= datetime.strptime("16:00", "%H:%M").time())
                    reg = df.loc[mask]
                    if reg.empty: reg = df
                    
                    o, h, l, c = reg['Open'].iloc[0], reg['High'].max(), reg['Low'].min(), reg['Close'].iloc[-1]

                results.append({
                    "symbol": symbol,
                    "perf_oc": (c - o) / o,
                    "perf_oh": (h - o) / o
                })
            except Exception as e:
                logger.warning(f"Failed to evaluate {symbol}: {e}")

        if not results:
            return {}

        df_res = pd.DataFrame(results)
        metrics = {
            "avg_peak_return": df_res["perf_oh"].mean(),
            "avg_close_return": df_res["perf_oc"].mean(),
            "win_rate": (df_res["perf_oc"] > 0).sum() / len(df_res),
            "hit_rate_2pct": (df_res["perf_oh"] >= 0.02).sum() / len(df_res),
            "top_performers": df_res.sort_values("perf_oh", ascending=False).head(5).to_dict(orient="records"),
            "bottom_performers": df_res.sort_values("perf_oc", ascending=True).head(5).to_dict(orient="records")
        }
        return metrics

    def calculate_plan_adherence(self, plan: Dict, executions: List[Dict]) -> Dict:
        """Compares actual trades with symbols listed in the pre-market plan."""
        # Check both top-level signals and regime_router signals
        signals = plan.get("signals", [])
        if not signals and "regime_router" in plan:
            signals = plan["regime_router"].get("signals", [])

        plan_symbols = {s.get("symbol", s.get("ticker")) for s in signals}
        execution_symbols = {e.get("symbol") for e in executions}

        adhered = execution_symbols.intersection(plan_symbols)
        deviations = execution_symbols.difference(plan_symbols)
        missed = plan_symbols.difference(execution_symbols)

        return {
            "adhered_symbols": list(adhered),
            "deviations": list(deviations),
            "missed_opportunities": list(missed),
            "adherence_score": len(adhered) / len(execution_symbols)
            if execution_symbols
            else 0.0,
        }

    def attribute_performance(self, plan: Dict, executions: List[Dict]) -> Dict:
        """Attributes returns to pre-market conviction scores."""
        # 1. Gather scores from signals or regime_router
        signals = plan.get("signals", [])
        if not signals and "regime_router" in plan:
            signals = plan["regime_router"].get("signals", [])

        plan_map = {
            s.get("symbol", s.get("ticker")): s.get(
                "entry_score", s.get("confidence", 0.0)
            )
            for s in signals
        }

        # 2. Add scores from plan 'positions' (overnight conviction)
        for p in plan.get("positions", []):
            symbol = p.get("symbol")
            if symbol not in plan_map:
                plan_map[symbol] = p.get("conviction_score", 0.0)

        high_conviction = []
        low_conviction = []

        for e in executions:
            symbol = e.get("symbol")
            score = plan_map.get(symbol, 0.0)
            returns = e.get("unrealized_plpc", e.get("pl_pct", 0.0))

            # Normalize returns (some files use % vs decimal)
            if abs(returns) > 1.0:
                returns = returns / 100.0

            if score >= 60:
                high_conviction.append(returns)
            else:
                low_conviction.append(returns)

        avg_high = (
            sum(high_conviction) / len(high_conviction) if high_conviction else 0.0
        )
        avg_low = sum(low_conviction) / len(low_conviction) if low_conviction else 0.0

        return {
            "high_conviction_return": avg_high,
            "low_conviction_return": avg_low,
            "inverse_conviction_flag": avg_low > avg_high,
            "performance_delta": avg_low - avg_high,
        }

    def synthesize_lesson(self, metrics: Dict) -> str:
        """Generates a concise textual lesson for the Knowledge Graph."""
        if metrics.get("inverse_conviction_flag"):
            return (
                f"INVERSE CONVICTION DETECTED: Low-score signals (Avg Return: {metrics['low_conviction_return'] * 100:.2f}%) "
                f"outperformed high-conviction signals (Avg Return: {metrics['high_conviction_return'] * 100:.2f}%). "
                f"Entry criteria for top-ranked symbols failed to filter for high-risk microstructure conditions."
            )
        return "Plan performance consistent with conviction scores."
