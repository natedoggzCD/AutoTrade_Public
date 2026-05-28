"""
Signal Validator - Historical Similar-Signal Backtesting
=========================================================
Before a signal makes the watchlist, find historically similar technical
setups in the parquet and measure their forward returns. Reject signals
whose historical lookalikes failed.

Uses DuckDB for fast analytical queries against the daily_features parquet.

Usage:
    from autotrade.backtesting.signal_validator import SignalValidator
    v = SignalValidator()
    result = v.validate_signal(rsi=55, atr_percent=3.0)
    print(result)
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import pandas as pd

from config.config_loader import get_config

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validating a signal against historical similar setups."""

    similar_signals_found: int = 0
    historical_win_rate: float = 0.0
    avg_return_1d: float = 0.0
    avg_return_3d: float = 0.0
    avg_return_5d: float = 0.0
    avg_max_gain: float = 0.0
    avg_max_loss: float = 0.0
    confidence: float = 0.0
    backtest_score: float = 0.0

    def __repr__(self) -> str:
        return (
            f"ValidationResult(matches={self.similar_signals_found}, "
            f"win_rate={self.historical_win_rate:.1%}, "
            f"avg_5d={self.avg_return_5d:+.2f}%, "
            f"score={self.backtest_score:.0f})"
        )


class SignalValidator:
    """
    Validates trading signals by finding historically similar technical setups
    in the DownDay parquet and measuring their forward returns.

    Scoring (0-100, calibrated to data percentiles):
        40 pts: win rate (full credit at config win_rate_full_credit)
        30 pts: avg 5D return (full credit at config avg_return_full_credit)
        20 pts: confidence / sample size (full credit at config confidence_full_credit)
        10 pts: gain/loss ratio (full credit at config gl_ratio_full_credit)
    Penalties allow scores to go negative for actively bad setups.
    """

    def __init__(
        self,
        parquet_path: Optional[Path] = None,
        parent_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = get_config()
        self.logger = parent_logger or logger

        self.parquet_path = (
            Path(parquet_path) if parquet_path else self._resolve_parquet_path()
        )

        # Validation config
        val_cfg = self.config.signal_validation
        self.val_cfg = val_cfg
        self.lookback_days = val_cfg.lookback_days
        self.tolerance = val_cfg.tolerance
        self.min_similar_signals = val_cfg.min_similar_signals
        self.min_backtest_score = val_cfg.min_backtest_score

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Daily features parquet not found: {self.parquet_path}"
            )

        self.logger.info(
            f"SignalValidator initialized (lookback={self.lookback_days}d, "
            f"tolerance={self.tolerance:.0%}, min_score={self.min_backtest_score})"
        )

    def _resolve_parquet_path(self) -> Path:
        data_cfg = self.config.data
        base = Path(data_cfg.downday_root)
        rel = Path(data_cfg.daily_features_parquet)
        return rel if rel.is_absolute() else base / rel

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def validate_signal(
        self,
        rsi: float,
        atr_percent: float,
        volume_ratio: Optional[float] = None,
        weekly_return: Optional[float] = None,
    ) -> ValidationResult:
        """
        Validate a single signal by finding similar historical setups.

        Args:
            rsi: Current RSI value (e.g. 55)
            atr_percent: ATR as % of price (e.g. 3.0)
            volume_ratio: Volume / 20-day avg volume (optional)
            weekly_return: Weekly return % (optional)

        Returns:
            ValidationResult with historical stats and backtest score
        """
        results = self.validate_batch(
            [
                {
                    "rsi": rsi,
                    "atr_percent": atr_percent,
                    "volume_ratio": volume_ratio,
                    "weekly_return": weekly_return,
                }
            ]
        )
        return results[0]

    def validate_batch(self, candidates: List[Dict]) -> List[ValidationResult]:
        """
        Validate multiple candidates efficiently with a single DuckDB query.

        Each candidate dict should have: rsi, atr_percent.
        Optional: volume_ratio, weekly_return.

        Returns list of ValidationResult in same order as input candidates.
        """
        if not candidates:
            return []

        try:
            con = self._connect()
            results = self._run_batch_query(con, candidates)
            con.close()
            return results
        except Exception as e:
            self.logger.error(f"Batch validation failed: {e}")
            return [ValidationResult() for _ in candidates]

    def validate_and_filter(
        self,
        candidates: List[Any],
        logger: Optional[logging.Logger] = None,
        *,
        adjust_scores: bool = False,
        weight_signal: Optional[float] = None,
        weight_backtest: Optional[float] = None,
        portfolio_cfg: Optional[Any] = None,
        build_heatmap: bool = False,
        return_details: bool = False,
    ):
        """
        Validate candidates and filter out those below threshold.

        Stamps each candidate with backtest_score, historical_win_rate,
        similar_signals_found. Returns only candidates that pass.

        Non-fatal: returns all candidates if validation fails.
        """
        log = logger or self.logger
        val_cfg = self.val_cfg

        details = {
            "results": [],
            "heatmap": [],
            "rejected": 0,
            "errors": [],
        }

        if not candidates:
            return ([], details) if return_details else []

        if not getattr(val_cfg, "enabled", True):
            return (candidates, details) if return_details else candidates

        try:
            batch = [self._candidate_to_dict(c) for c in candidates]
            val_results = self.validate_batch(batch)
        except Exception as e:
            log.warning(f"Signal validation failed (non-fatal): {e}")
            details["errors"].append(str(e))
            return (candidates, details) if return_details else candidates

        filtered = []
        heatmap_buckets: Dict[str, Dict[str, float]] = {}
        w_sig = (
            weight_signal
            if weight_signal is not None
            else getattr(val_cfg, "score_weight_signal", 1.0)
        )
        w_bt = (
            weight_backtest
            if weight_backtest is not None
            else getattr(val_cfg, "score_weight_backtest", 0.0)
        )

        for cand, vr in zip(candidates, val_results):
            self._set_attr(cand, "backtest_score", vr.backtest_score)
            self._set_attr(cand, "historical_win_rate", vr.historical_win_rate)
            self._set_attr(cand, "similar_signals_found", vr.similar_signals_found)
            self._set_attr(cand, "avg_5d_return", vr.avg_return_5d)

            symbol = (
                self._get_attr(cand, "symbol", None)
                or self._get_attr(cand, "ticker", None)
                or "?"
            )

            if adjust_scores:
                base_score = self._get_attr(cand, "final_score", None)
                if base_score is None:
                    base_score = self._get_attr(cand, "entry_score", 0.0)
                blended = (base_score or 0.0) * w_sig + vr.backtest_score * w_bt
                self._set_attr(cand, "final_score", blended)

            if vr.backtest_score < val_cfg.min_backtest_score:
                log.info(
                    f"   [REJECTED] {symbol}: backtest_score={vr.backtest_score:.0f} < min {val_cfg.min_backtest_score} "
                    f"(win_rate={vr.historical_win_rate:.1%}, matches={vr.similar_signals_found})"
                )
                details["rejected"] += 1
                continue
            if vr.similar_signals_found < val_cfg.min_similar_signals:
                log.info(
                    f"   [REJECTED] {symbol}: only {vr.similar_signals_found} similar signals (< {val_cfg.min_similar_signals})"
                )
                details["rejected"] += 1
                continue

            if portfolio_cfg is not None:
                base_size = self._get_attr(
                    cand, "position_size", None
                ) or self._get_cfg_attr(portfolio_cfg, "position_size_target")
                if base_size:
                    mult = self._size_multiplier(
                        vr.backtest_score, vr.historical_win_rate
                    )
                    max_size = self._get_cfg_attr(portfolio_cfg, "position_size_max")
                    new_size = base_size * mult
                    if max_size:
                        new_size = min(new_size, max_size)
                    self._set_attr(cand, "position_size", new_size)

            if build_heatmap:
                scan = self._get_attr(cand, "scan_type", "") or "unknown"
                bucket = heatmap_buckets.setdefault(
                    scan,
                    {
                        "count": 0,
                        "sum_backtest": 0.0,
                        "sum_win_rate": 0.0,
                    },
                )
                bucket["count"] += 1
                bucket["sum_backtest"] += vr.backtest_score
                bucket["sum_win_rate"] += vr.historical_win_rate

            filtered.append(cand)

        if build_heatmap:
            for scan, stats in heatmap_buckets.items():
                cnt = stats["count"]
                avg_bt = stats["sum_backtest"] / cnt if cnt else 0.0
                avg_wr = stats["sum_win_rate"] / cnt if cnt else 0.0
                weight = 1.0
                if avg_wr >= 0.55:
                    weight = 1.5
                elif avg_wr >= 0.45:
                    weight = 1.2
                elif avg_wr < 0.35:
                    weight = 0.7
                details["heatmap"].append(
                    {
                        "scan_type": scan,
                        "count": cnt,
                        "avg_backtest_score": avg_bt,
                        "avg_win_rate": avg_wr,
                        "suggested_weight": weight,
                    }
                )

        details["results"] = val_results
        return (filtered, details) if return_details else filtered

    def _run_batch_query(
        self,
        con: duckdb.DuckDBPyConnection,
        candidates: List[Dict],
    ) -> List[ValidationResult]:
        """Execute efficient batch validation using DuckDB CTE + CROSS JOIN."""
        parquet = self.parquet_path.as_posix()
        tol_base = self.tolerance
        val_cfg = self.val_cfg

        tol_shrunk = max(
            val_cfg.adaptive_tolerance_min,
            tol_base * val_cfg.adaptive_tolerance_shrink_factor,
        )

        # Build candidate CTE values
        cte_rows = []
        for i, c in enumerate(candidates):
            rsi = c.get("rsi", 50.0)
            atr_pct = c.get("atr_percent", 3.0)
            ticker = c.get("ticker")
            ticker_sql = f"'{ticker}'" if ticker else "NULL"
            vol_ratio = c.get("volume_ratio")
            vol_sql = str(float(vol_ratio)) if vol_ratio is not None else "NULL"
            wk_ret = c.get("weekly_return")
            wk_sql = str(float(wk_ret)) if wk_ret is not None else "NULL"
            cte_rows.append(
                f"SELECT {i} AS cid, CAST({rsi} AS DOUBLE) AS rsi, "
                f"CAST({atr_pct} AS DOUBLE) AS atr_pct, "
                f"{ticker_sql} AS ticker_excl, "
                f"CAST({vol_sql} AS DOUBLE) AS vol_ratio_target, "
                f"CAST({wk_sql} AS DOUBLE) AS wk_return_target"
            )

        candidates_cte = " UNION ALL ".join(cte_rows)

        # Compute widest bounds for pre-filter
        all_rsi = [c.get("rsi", 50.0) for c in candidates]
        all_atr = [c.get("atr_percent", 3.0) for c in candidates]
        rsi_lo = min(all_rsi) * (1 - tol_base)
        rsi_hi = max(all_rsi) * (1 + tol_base)
        atr_lo = min(all_atr) * (1 - tol_base)
        atr_hi = max(all_atr) * (1 + tol_base)

        lookback = self.lookback_days

        ret_5d_expr = (
            "CASE WHEN b.close > 0 AND b.close_5d IS NOT NULL "
            "THEN ((b.close_5d - b.close) / b.close) * 100.0 ELSE NULL END"
        )

        if val_cfg.win_threshold_atr_relative:
            win_condition = f"({ret_5d_expr} IS NOT NULL AND {ret_5d_expr} >= c.atr_pct * {val_cfg.win_threshold_atr_multiple})"
        else:
            win_condition = f"({ret_5d_expr} IS NOT NULL AND {ret_5d_expr} >= {val_cfg.win_threshold_pct})"

        recency_clause = (
            ""
            if not val_cfg.recency_filter_enabled
            else f"AND Date >= CURRENT_DATE - INTERVAL '{val_cfg.recency_max_days} days'"
        )
        atr_filter_clause = (
            ""
            if not val_cfg.atr_filter_enabled
            else (
                f"AND CASE WHEN CAST(close AS DOUBLE) > 0 "
                f"THEN (CAST(atr_14 AS DOUBLE) / CAST(close AS DOUBLE)) * 100.0 ELSE 0.0 END "
                f"BETWEEN {val_cfg.min_atr_pct} AND {val_cfg.max_atr_pct}"
            )
        )
        liquidity_clause = (
            ""
            if not val_cfg.liquidity_filter_enabled
            else (
                f"AND close >= {val_cfg.min_price} "
                f"AND (COALESCE(CAST(volume AS DOUBLE), 0.0) * CAST(close AS DOUBLE)) >= {val_cfg.min_dollar_volume}"
            )
        )

        regime_clause = (
            ""
            if not val_cfg.regime_filter_enabled
            else (f"AND market_ret_20 >= {val_cfg.regime_min_trend}")
        )

        gl_ratio_clause = (
            ""
            if not val_cfg.gl_ratio_filter_enabled
            else (f"AND gl_ratio_row >= {val_cfg.gl_ratio_filter_min}")
        )

        tol_case = (
            f"CASE WHEN {str(val_cfg.adaptive_tolerance_enabled).upper()} AND pre_counts.n_matches >= {val_cfg.adaptive_tolerance_trigger_matches} "
            f"THEN {tol_shrunk} ELSE {tol_base} END"
        )

        query = f"""
        WITH candidates AS (
            {candidates_cte}
        ),
        -- Pre-filter: only rows within the widest bounds across all candidates
        base AS (
            SELECT
                ticker,
                Date AS dt,
                CAST(RSI_14 AS DOUBLE) AS rsi,
                CASE WHEN CAST(close AS DOUBLE) > 0
                     THEN (CAST(atr_14 AS DOUBLE) / CAST(close AS DOUBLE)) * 100.0
                     ELSE 0.0 END AS atr_pct,
                CAST(close AS DOUBLE) AS close,
                CAST(volume AS DOUBLE) AS volume,
                CAST(volume_ratio AS DOUBLE) AS vol_ratio,
                CASE WHEN LAG(CAST(close AS DOUBLE), 5) OVER w > 0
                     THEN (
                        (CAST(close AS DOUBLE) - LAG(CAST(close AS DOUBLE), 5) OVER w)
                        / LAG(CAST(close AS DOUBLE), 5) OVER w
                     ) * 100.0
                     ELSE NULL END AS wk_return,
                 CASE WHEN LAG(CAST(close AS DOUBLE), {val_cfg.regime_lookback_days}) OVER w > 0
                     THEN (
                        (CAST(close AS DOUBLE) - LAG(CAST(close AS DOUBLE), {val_cfg.regime_lookback_days}) OVER w)
                        / LAG(CAST(close AS DOUBLE), {val_cfg.regime_lookback_days}) OVER w
                     ) * 100.0
                     ELSE NULL END AS market_ret_20,
                -- Forward returns via LEAD
                LEAD(CAST(close AS DOUBLE), 1) OVER w AS close_1d,
                LEAD(CAST(close AS DOUBLE), 3) OVER w AS close_3d,
                LEAD(CAST(close AS DOUBLE), 5) OVER w AS close_5d,
                -- Max/min over next 5 days for gain/loss
                MAX(CAST(close AS DOUBLE)) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING
                ) AS max_5d,
                MIN(CAST(close AS DOUBLE)) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 1 FOLLOWING AND 5 FOLLOWING
                ) AS min_5d
            FROM parquet_scan('{parquet}')
            WHERE RSI_14 BETWEEN {rsi_lo} AND {rsi_hi}
              AND CAST(close AS DOUBLE) > 0
              AND CAST(atr_14 AS DOUBLE) > 0
              AND Date >= CURRENT_DATE - INTERVAL '{lookback} days'
              {recency_clause}
              AND CASE WHEN CAST(close AS DOUBLE) > 0
                       THEN (CAST(atr_14 AS DOUBLE) / CAST(close AS DOUBLE)) * 100.0
                       ELSE 0.0 END
                  BETWEEN {atr_lo} AND {atr_hi}
              {atr_filter_clause}
              {liquidity_clause}
            WINDOW w AS (PARTITION BY ticker ORDER BY Date)
        ),
        -- Cross join candidates with matching historical setups (base tolerance)
        matched_base AS (
            SELECT
                c.cid,
                b.*,
                c.rsi AS rsi_target,
                c.atr_pct AS atr_target,
                -- Forward return percentages
                CASE WHEN b.close > 0 AND b.close_1d IS NOT NULL
                     THEN ((b.close_1d - b.close) / b.close) * 100.0 ELSE NULL END AS ret_1d,
                CASE WHEN b.close > 0 AND b.close_3d IS NOT NULL
                     THEN ((b.close_3d - b.close) / b.close) * 100.0 ELSE NULL END AS ret_3d,
                 {ret_5d_expr} AS ret_5d,
                CASE WHEN b.close > 0 AND b.max_5d IS NOT NULL
                     THEN ((b.max_5d - b.close) / b.close) * 100.0 ELSE NULL END AS max_gain,
                CASE WHEN b.close > 0 AND b.min_5d IS NOT NULL
                     THEN ((b.min_5d - b.close) / b.close) * 100.0 ELSE NULL END AS max_loss,
                CASE WHEN {win_condition} THEN 1.0 ELSE 0.0 END AS is_win,
                CASE WHEN b.min_5d IS NOT NULL AND b.min_5d != 0 AND b.close > 0 AND b.min_5d != b.close
                     THEN ABS(((b.max_5d - b.close) / b.close) * 100.0) / NULLIF(ABS(((b.min_5d - b.close) / b.close) * 100.0), 0)
                     ELSE NULL END AS gl_ratio_row
            FROM candidates c
            CROSS JOIN base b
            WHERE b.rsi BETWEEN c.rsi * {1 - tol_base} AND c.rsi * {1 + tol_base}
              AND b.atr_pct BETWEEN c.atr_pct * {1 - tol_base} AND c.atr_pct * {1 + tol_base}
              AND b.close_5d IS NOT NULL
              AND (c.ticker_excl IS NULL OR b.ticker != c.ticker_excl)
              AND (c.vol_ratio_target IS NULL
                   OR b.vol_ratio BETWEEN c.vol_ratio_target * 0.7 AND c.vol_ratio_target * 1.3)
              AND (c.wk_return_target IS NULL
                   OR b.wk_return BETWEEN c.wk_return_target - 3.0 AND c.wk_return_target + 3.0)
              {regime_clause}
        ),
        -- Deduplicate: keep one row per ticker per week per candidate (pre-count)
        matched_deduped_pre AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY cid, ticker, DATE_TRUNC('week', dt)
                ORDER BY dt DESC
            ) AS rn
            FROM matched_base
        ),
        pre_counts AS (
            SELECT cid, COUNT(*) AS n_matches
            FROM matched_deduped_pre
            WHERE rn = 1
            GROUP BY cid
        ),
        -- Apply adaptive tolerance and optional gl-ratio filter
        matched_filtered AS (
            SELECT mb.*, {tol_case} AS tol_used
            FROM matched_base mb
            LEFT JOIN pre_counts ON pre_counts.cid = mb.cid
        ),
        matched AS (
            SELECT *,
                   ROW_NUMBER() OVER (
                       PARTITION BY cid, ticker, DATE_TRUNC('week', dt)
                       ORDER BY dt DESC
                   ) AS rn
            FROM matched_filtered
            WHERE rsi BETWEEN rsi_target * (1 - tol_used) AND rsi_target * (1 + tol_used)
              AND atr_pct BETWEEN atr_target * (1 - tol_used) AND atr_target * (1 + tol_used)
              {gl_ratio_clause}
        ),
        -- Deduplicate: keep one row per ticker per week per candidate
        matched_deduped AS (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY cid, ticker, DATE_TRUNC('week', dt)
                ORDER BY dt DESC
            ) AS rn
            FROM matched
        ),
        -- Aggregate per candidate
        agg AS (
            SELECT
                cid,
                COUNT(*) AS n_matches,
                AVG(is_win) AS win_rate,
                AVG(ret_1d) AS avg_ret_1d,
                AVG(ret_3d) AS avg_ret_3d,
                AVG(ret_5d) AS avg_ret_5d,
                AVG(max_gain) AS avg_max_gain,
                AVG(max_loss) AS avg_max_loss
            FROM matched_deduped
            WHERE rn = 1
            GROUP BY cid
        )
        SELECT * FROM agg ORDER BY cid
        """

        try:
            df = con.execute(query).fetchdf()
        except Exception as e:
            self.logger.error(f"DuckDB query error: {e}")
            return [ValidationResult() for _ in candidates]

        # Map results back to candidates
        results_map: Dict[int, dict] = {}
        for _, row in df.iterrows():
            results_map[int(row["cid"])] = row.to_dict()

        output = []
        for i in range(len(candidates)):
            if i in results_map:
                row = results_map[i]
                n = int(row["n_matches"])
                win_rate = float(row["win_rate"])
                avg_1d = float(row["avg_ret_1d"])
                avg_3d = float(row["avg_ret_3d"])
                avg_5d = float(row["avg_ret_5d"])
                avg_gain = float(row["avg_max_gain"])
                avg_loss = float(row["avg_max_loss"])

                confidence = self._compute_confidence(n)
                score = self._compute_score(
                    win_rate, avg_5d, confidence, avg_gain, avg_loss
                )

                result = ValidationResult(
                    similar_signals_found=n,
                    historical_win_rate=win_rate,
                    avg_return_1d=avg_1d,
                    avg_return_3d=avg_3d,
                    avg_return_5d=avg_5d,
                    avg_max_gain=avg_gain,
                    avg_max_loss=avg_loss,
                    confidence=confidence,
                    backtest_score=score,
                )
            else:
                result = ValidationResult()

            output.append(result)

        return output

    def _compute_confidence(self, matches: int) -> float:
        """Normalize confidence to 0-1 using calibrated match count."""
        full_credit = max(1.0, float(self.val_cfg.confidence_full_credit))
        return min(1.0, max(0.0, float(matches)) / full_credit)

    def _compute_score(
        self,
        win_rate: float,
        avg_5d_return: float,
        confidence: float,
        avg_max_gain: float,
        avg_max_loss: float,
    ) -> float:
        """Compute calibrated backtest score with penalty zone."""
        val_cfg = self.val_cfg

        # Win rate component (40 pts)
        wr_denom = max(1e-6, val_cfg.win_rate_full_credit)
        wr_score = min(1.0, max(0.0, win_rate) / wr_denom) * 40.0

        # Avg 5D return component (30 pts)
        ret_denom = max(1e-6, val_cfg.avg_return_full_credit)
        ret_score = min(1.0, max(0.0, avg_5d_return) / ret_denom) * 30.0

        # Confidence component (20 pts)
        conf_score = min(1.0, max(0.0, confidence)) * 20.0

        # Gain/loss ratio component (10 pts)
        gl_ratio = 0.0
        if avg_max_loss and abs(avg_max_loss) > 0.01:
            gl_ratio = abs(avg_max_gain) / abs(avg_max_loss)
        gl_denom = max(1e-6, val_cfg.gl_ratio_full_credit)
        gl_score = min(1.0, gl_ratio / gl_denom) * 10.0

        penalty = 0.0
        if win_rate < val_cfg.penalty_win_rate_threshold:
            penalty += val_cfg.penalty_win_rate_points
        if avg_5d_return < val_cfg.penalty_avg_return_threshold:
            penalty += val_cfg.penalty_avg_return_points
        if gl_ratio < val_cfg.penalty_gl_ratio_threshold:
            penalty += val_cfg.penalty_gl_ratio_points

        return wr_score + ret_score + conf_score + gl_score - penalty

    @staticmethod
    def _candidate_to_dict(candidate: Any) -> Dict:
        if isinstance(candidate, dict):
            return {
                "ticker": candidate.get("ticker") or candidate.get("symbol"),
                "rsi": candidate.get("rsi", 50.0),
                "atr_percent": candidate.get(
                    "atr_percent", candidate.get("atr_pct", 3.0)
                ),
                "volume_ratio": candidate.get("volume_ratio"),
                "weekly_return": candidate.get("weekly_return"),
            }

        return {
            "ticker": getattr(candidate, "symbol", None),
            "rsi": getattr(candidate, "rsi", 50.0),
            "atr_percent": getattr(candidate, "atr_percent", 3.0),
            "volume_ratio": getattr(candidate, "volume_ratio", None),
            "weekly_return": getattr(candidate, "weekly_return", None),
        }

    @staticmethod
    def _get_attr(candidate: Any, name: str, default=None):
        if isinstance(candidate, dict):
            return candidate.get(name, default)
        return getattr(candidate, name, default)

    @staticmethod
    def _set_attr(candidate: Any, name: str, value) -> None:
        if isinstance(candidate, dict):
            candidate[name] = value
        else:
            setattr(candidate, name, value)

    @staticmethod
    def _get_cfg_attr(cfg: Any, name: str, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(name, default)
        return getattr(cfg, name, default)

    @staticmethod
    def _size_multiplier(bt_score: float, win_rate: float) -> float:
        if bt_score >= 80 and win_rate >= 0.50:
            return 1.5
        if bt_score >= 60 and win_rate >= 0.40:
            return 1.0
        if bt_score >= 40 and win_rate >= 0.30:
            return 0.7
        return 0.0
