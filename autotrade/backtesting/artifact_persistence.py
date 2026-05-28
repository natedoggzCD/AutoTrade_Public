from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ArtifactPaths:
    """Paths for persisted backtest artifacts."""

    run_dir: Path
    metrics_json: Path
    config_json: Path
    equity_plot: Path
    drawdown_plot: Path
    folds_json: Path
    diagnostics_json: Path


class ArtifactPersister:
    """
    Handles persistence of backtest result artifacts.

    Saves:
    - Metrics JSON
    - Config hash + seed
    - Equity and drawdown plots
    - Fold-level diagnostics
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        strategy_id: str = "default",
    ):
        """
        Initialize artifact persister.

        Args:
            base_dir: Base directory for artifacts. Defaults to data/backtest_artifacts.
            strategy_id: Strategy identifier for organizing artifacts
        """
        if base_dir is None:
            base_dir = Path("data") / "backtest_artifacts"

        self.base_dir = base_dir
        self.strategy_id = strategy_id

    def _ensure_run_dir(
        self,
        run_id: str,
        timestamp: str,
    ) -> Path:
        """Create and return run directory."""
        run_dir = (
            self.base_dir
            / self.strategy_id
            / timestamp[:10]
            / f"{run_id}_{timestamp[11:19].replace(':', '')}"
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _prepare_fold_data(
        self,
        folds: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """Prepare fold data for serialization."""
        fold_data = []
        for fold in folds:
            fold_dict = {
                "fold_index": fold.fold_index,
                "train_start": str(fold.train_start),
                "train_end": str(fold.train_end),
                "test_start": str(fold.test_start),
                "test_end": str(fold.test_end),
                "is_oos": fold.is_oos,
                "trades_count": fold.trades_count,
                "pnl_dollars": float(fold.pnl_dollars),
                "pnl_pct": float(fold.pnl_pct),
                "sharpe_ratio": float(fold.sharpe_ratio),
                "sortino_ratio": float(fold.sortino_ratio),
                "max_drawdown_pct": float(fold.max_drawdown_pct),
                "calmar_ratio": float(fold.calmar_ratio),
                "win_rate": float(fold.win_rate),
                "profit_factor": float(fold.profit_factor),
                "avg_trade_dollars": float(fold.avg_trade_dollars),
                "turnover": float(fold.turnover),
                "metrics": fold.metrics,
            }

            if hasattr(fold, "trades") and fold.trades:
                trade_summary = {
                    "total_trades": len(fold.trades),
                    "winning": sum(
                        1 for t in fold.trades if t.get("pnl_dollars", 0) > 0
                    ),
                    "losing": sum(
                        1 for t in fold.trades if t.get("pnl_dollars", 0) < 0
                    ),
                    "avg_pnl": float(
                        np.mean([t.get("pnl_dollars", 0) for t in fold.trades])
                    ),
                }
                fold_dict["trade_summary"] = trade_summary

            fold_data.append(fold_dict)

        return fold_data

    def _generate_equity_plot(
        self,
        folds: Sequence[Any],
        output_path: Path,
    ) -> Optional[str]:
        """Generate and save equity curve plot."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            equity_data = []
            for fold in folds:
                if hasattr(fold, "trades") and fold.trades:
                    cumulative = 0
                    for trade in fold.trades:
                        cumulative += trade.get("pnl_dollars", 0)
                        equity_data.append(
                            {
                                "date": trade.get("date", fold.test_start),
                                "equity": cumulative,
                                "fold": fold.fold_index,
                            }
                        )

            if not equity_data:
                logger.warning("No equity data to plot")
                return None

            df = pd.DataFrame(equity_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(df["date"], df["equity"], linewidth=1.5)
            ax.set_xlabel("Date")
            ax.set_ylabel("Cumulative PnL ($)")
            ax.set_title(f"Equity Curve - {self.strategy_id}")
            ax.grid(True, alpha=0.3)

            if "fold" in df.columns:
                fold_starts = df.groupby("fold")["date"].first()
                for fold_num in fold_starts.index:
                    ax.axvline(
                        x=fold_starts[fold_num],
                        color="gray",
                        linestyle="--",
                        alpha=0.3,
                    )

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            return str(output_path)

        except Exception as e:
            logger.warning(f"Failed to generate equity plot: {e}")
            return None

    def _generate_drawdown_plot(
        self,
        folds: Sequence[Any],
        output_path: Path,
    ) -> Optional[str]:
        """Generate and save drawdown plot."""
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            equity_data = []
            for fold in folds:
                if hasattr(fold, "trades") and fold.trades:
                    cumulative = 0
                    peak = 0
                    for trade in fold.trades:
                        cumulative += trade.get("pnl_dollars", 0)
                        peak = max(peak, cumulative)
                        drawdown = (peak - cumulative) / peak if peak > 0 else 0
                        equity_data.append(
                            {
                                "date": trade.get("date", fold.test_start),
                                "equity": cumulative,
                                "drawdown": drawdown * 100,
                                "fold": fold.fold_index,
                            }
                        )

            if not equity_data:
                logger.warning("No drawdown data to plot")
                return None

            df = pd.DataFrame(equity_data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            ax1.plot(df["date"], df["equity"], linewidth=1.5)
            ax1.set_ylabel("Cumulative PnL ($)")
            ax1.set_title(f"Equity & Drawdown - {self.strategy_id}")
            ax1.grid(True, alpha=0.3)

            ax2.fill_between(df["date"], 0, -df["drawdown"], alpha=0.5)
            ax2.set_xlabel("Date")
            ax2.set_ylabel("Drawdown (%)")
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150)
            plt.close()

            return str(output_path)

        except Exception as e:
            logger.warning(f"Failed to generate drawdown plot: {e}")
            return None

    def persist_artifact(
        self,
        artifact: Any,
        run_id: str,
        timestamp: str,
        config_hash: str,
        save_plots: bool = True,
    ) -> ArtifactPaths:
        """
        Persist backtest artifact to disk.

        Args:
            artifact: BacktestResultArtifact to persist
            run_id: Unique run identifier
            timestamp: Run timestamp
            config_hash: Configuration hash
            save_plots: Whether to generate plots

        Returns:
            Paths to persisted files
        """
        run_dir = self._ensure_run_dir(run_id, timestamp)

        metrics_json = run_dir / "metrics.json"
        with open(metrics_json, "w", encoding="utf-8") as f:
            json.dump(artifact.metrics or {}, f, indent=2)

        config_json = run_dir / "config.json"
        config_data = {
            "config_hash": config_hash,
            "run_id": run_id,
            "timestamp": timestamp,
            "strategy_id": artifact.strategy_id,
            "start_date": artifact.start_date,
            "end_date": artifact.end_date,
            "seed": artifact.request.get("seed", 42) if artifact.request else 42,
            "execution_config": artifact.execution_config,
            "cost_sensitivity": artifact.cost_sensitivity,
        }
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        folds_json = run_dir / "folds.json"
        fold_data = self._prepare_fold_data(artifact.folds)
        with open(folds_json, "w", encoding="utf-8") as f:
            json.dump(fold_data, f, indent=2)

        diagnostics_json = run_dir / "diagnostics.json"
        diagnostics = {
            "leakage_check_passed": artifact.leakage_check_passed,
            "leakage_warnings": artifact.leakage_warnings,
            "promotion_eligible": artifact.promotion_eligible,
            "promotion_reason_code": artifact.promotion_reason_code,
            "validation_notes": artifact.validation_notes,
        }
        with open(diagnostics_json, "w", encoding="utf-8") as f:
            json.dump(diagnostics, f, indent=2)

        equity_plot_path = run_dir / "equity_curve.png"
        drawdown_plot_path = run_dir / "drawdown.png"

        equity_path_str = None
        drawdown_path_str = None

        if save_plots:
            equity_path_str = self._generate_equity_plot(
                artifact.folds, equity_plot_path
            )
            drawdown_path_str = self._generate_drawdown_plot(
                artifact.folds, drawdown_plot_path
            )

        return ArtifactPaths(
            run_dir=run_dir,
            metrics_json=metrics_json,
            config_json=config_json,
            equity_plot=equity_plot_path if equity_path_str else Path(""),
            drawdown_plot=drawdown_plot_path if drawdown_path_str else Path(""),
            folds_json=folds_json,
            diagnostics_json=diagnostics_json,
        )


def create_persister(strategy_id: str = "default") -> ArtifactPersister:
    """Create an artifact persister with default settings."""
    return ArtifactPersister(strategy_id=strategy_id)
