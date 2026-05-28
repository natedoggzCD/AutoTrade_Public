from typing import List, Dict, Any, Optional
import pandas as pd
from autotrade.analysis.ranking import OpportunityScorer
from autotrade.monitoring.halt_logic import HaltMonitor


class WatchlistRotator:
    """
    Manages dynamic intraday watchlist rotation.
    Decides when to swap symbols in and out of the active set based on opportunity delta.
    """

    def __init__(
        self,
        scorer: OpportunityScorer,
        halt_monitor: HaltMonitor,
        min_delta: float = 10.0,
        max_watchlist_size: int = 10,
    ):
        self.scorer = scorer
        self.halt_monitor = halt_monitor
        self.min_delta = min_delta
        self.max_watchlist_size = max_watchlist_size

    def evaluate_rotation(
        self, watchlist: pd.DataFrame, candidates: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """
        Evaluate if any symbols should be removed from the watchlist and replaced by candidates.
        Returns a list of swap instructions.
        """
        swaps = []
        if watchlist.empty or candidates.empty:
            return swaps

        # 1. Emergency removals (Halted symbols)
        for idx, row in watchlist.iterrows():
            symbol = row["symbol"]
            if self.halt_monitor.is_halted(symbol):
                # Find best candidate for replacement
                best_candidate = candidates.iloc[0]
                swaps.append(
                    {
                        "remove": symbol,
                        "add": best_candidate["symbol"],
                        "reason": "HALTED",
                        "delta": 0.0,  # Emergency swap
                    }
                )
                # Remove candidate from local list to prevent double allocation
                candidates = candidates.drop(best_candidate.name)

        # 2. Delta-based opportunistic rotation
        # Ensure opportunity_score exists to prevent KeyErrors
        for df_tmp in [watchlist, candidates]:
            if "opportunity_score" not in df_tmp.columns:
                df_tmp["opportunity_score"] = 0.0

        # Sort current watchlist by score (lowest first)
        watchlist_sorted = watchlist.sort_values(by="opportunity_score", ascending=True)
        # Sort candidates by score (highest first)
        candidates_sorted = candidates.sort_values(
            by="opportunity_score", ascending=False
        )

        for _, w_row in watchlist_sorted.iterrows():
            if candidates_sorted.empty:
                break

            best_candidate = candidates_sorted.iloc[0]
            delta = best_candidate["opportunity_score"] - w_row["opportunity_score"]

            if delta >= self.min_delta:
                # Already checked emergency, prevent duplicate removal
                if any(s["remove"] == w_row["symbol"] for s in swaps):
                    continue

                swaps.append(
                    {
                        "remove": w_row["symbol"],
                        "add": best_candidate["symbol"],
                        "reason": "OPPORTUNITY_DELTA",
                        "delta": float(delta),
                    }
                )
                # Remove candidate from consideration for next watchlist slot
                candidates_sorted = candidates_sorted.drop(best_candidate.name)

        return swaps

    def perform_swaps(
        self, current_watchlist: List[str], swaps: List[Dict[str, Any]]
    ) -> List[str]:
        """Apply swap instructions to a list of symbols."""
        updated = list(current_watchlist)
        for swap in swaps:
            if swap["remove"] in updated:
                idx = updated.index(swap["remove"])
                updated[idx] = swap["add"]
            else:
                # If remove isn't there (maybe it was manually changed), just append the add
                if swap["add"] not in updated:
                    updated.append(swap["add"])

        # Enforce max size
        return updated[: self.max_watchlist_size]


class RotationScheduler:
    """Cadence helper for periodic watchlist rescans."""

    def __init__(
        self,
        interval_cycles: int = 15,
        enabled_phases: Optional[List[str]] = None,
    ):
        self.interval_cycles = max(int(interval_cycles), 1)
        self.enabled_phases = {
            str(p) for p in (enabled_phases or ["CORE_TRADING", "RESEARCH"])
        }

    def should_run(self, cycle_count: int, current_phase: Any) -> bool:
        phase_name = getattr(current_phase, "name", str(current_phase or ""))
        if phase_name not in self.enabled_phases:
            return False
        return int(cycle_count or 0) % self.interval_cycles == 0
