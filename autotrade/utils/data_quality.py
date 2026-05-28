"""
Data Quality Gate - Validates data freshness and quality before signal generation.

Prevents the system from generating signals when:
1. Data is stale (weekend, holiday, parquet not updated)
2. Technical indicators are uniform (pipeline broken)
3. Too many missing values in key columns
4. Signal batches show anomalous uniformity (flat scores)
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import duckdb

from config.config_loader import get_config
from autotrade.data_ingestion.bootstrap import ensure_core_market_data_ready
from autotrade.data_ingestion.paths import get_primary_parquet_path
from autotrade.utils.data_sync import expected_core_market_data_date

try:
    from autotrade.monitoring.collector import get_metrics_collector

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False

from autotrade.utils.market_time import (
    is_trading_day as market_time_is_trading_day,
)

logger = logging.getLogger(__name__)

PARQUET_PATH = get_primary_parquet_path()


@dataclass
class DataQualityReport:
    """Report from data quality validation."""

    is_fresh: bool  # Is data from the latest trading day?
    latest_date: Optional[date]  # Most recent date in parquet
    expected_date: Optional[date]  # What the latest date should be
    staleness_days: int  # How many trading days old
    is_trading_day: bool  # Is today a trading day?
    ticker_count: int  # Number of tickers with data on latest date
    pct_rsi_identical: float  # % of tickers with identical RSI (stale indicator)
    pct_missing_rsi: float  # % with NULL RSI_14
    pct_missing_atr: float  # % with NULL atr_14
    pct_missing_volume: float  # % with NULL or zero volume
    quality_score: float  # 0-100 overall quality score
    recommendation: str  # "PROCEED", "CAUTION", "ABORT"
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchQualityReport:
    """Report from signal batch quality validation."""

    is_valid: bool
    recommendation: str  # "REJECT", "WARN", "ACCEPT"
    issues: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


class DataQualityGate:
    """
    Validates data quality before signal generation.

    Key checks:
    1. Freshness: Is parquet data from the latest trading day?
    2. Completeness: Are key columns (RSI, ATR, Volume) populated?
    3. Uniformity: Are indicator values suspiciously identical?
    4. Coverage: Do we have data for enough tickers?
    """

    def __init__(
        self,
        parquet_path: str = str(PARQUET_PATH),
        min_quality_score: int = 50,
        caution_quality_score: int = 80,
        max_staleness_days: Optional[int] = None,
        min_ticker_count: int = 3000,
        max_pct_identical_rsi: float = 20.0,
    ):
        cfg_max_staleness = get_config().data.max_staleness_days
        self.parquet_path = Path(parquet_path)
        self.min_quality_score = min_quality_score
        self.caution_quality_score = caution_quality_score
        self.max_staleness_days = (
            cfg_max_staleness if max_staleness_days is None else max_staleness_days
        )
        self.min_ticker_count = min_ticker_count
        self.max_pct_identical_rsi = max_pct_identical_rsi

    def is_trading_day(self, check_date: date = None) -> bool:
        """Check if a given date is a US market trading day."""
        return market_time_is_trading_day(check_date)

    def _normalize_as_of_datetime(
        self, as_of_date: Optional[date | datetime] = None
    ) -> datetime:
        """Treat date-only inputs as pre-close session checks."""
        if isinstance(as_of_date, datetime):
            return as_of_date
        if isinstance(as_of_date, date):
            return datetime.combine(as_of_date, datetime.min.time())
        return datetime.now()

    def get_expected_latest_date(
        self, as_of_date: Optional[date | datetime] = None
    ) -> Optional[date]:
        """
        Calculate what the latest data date SHOULD be.

        Freshness is session-aware:
        - Before local market close, expect the previous trading day
        - After local market close, expect the current trading day
        - Monday intraday therefore expects Friday data
        """
        as_of_dt = self._normalize_as_of_datetime(as_of_date)
        expected = expected_core_market_data_date(reference_dt=as_of_dt)
        return date.fromisoformat(expected)

    def validate(self, as_of_date: Optional[date | datetime] = None) -> DataQualityReport:
        """
        Run all data quality checks and return a report.
        """
        issues = []
        stats = {}
        ensure_core_market_data_ready(fail_fast=False)

        # Check if parquet exists
        if not self.parquet_path.exists():
            return DataQualityReport(
                is_fresh=False,
                latest_date=None,
                expected_date=self.get_expected_latest_date(as_of_date),
                staleness_days=999,
                is_trading_day=self.is_trading_day(as_of_date),
                ticker_count=0,
                pct_rsi_identical=0,
                pct_missing_rsi=100,
                pct_missing_atr=100,
                pct_missing_volume=100,
                quality_score=0,
                recommendation="ABORT",
                issues=["Parquet file not found"],
                stats=stats,
            )

        # Get expected date
        expected_date = self.get_expected_latest_date(as_of_date)
        as_of_dt = self._normalize_as_of_datetime(as_of_date)
        is_trading_day = self.is_trading_day(as_of_dt.date())

        # Query parquet for latest date and quality metrics
        try:
            conn = duckdb.connect(database=":memory:")  # In-memory; read_parquet() in SQL handles the file path

            # Get latest date in parquet
            latest_date_result = conn.execute(
                """
                SELECT MAX(CAST(Date AS DATE)) as max_date
                FROM read_parquet(?)
            """,
                [str(self.parquet_path)],
            ).fetchone()

            latest_date = latest_date_result[0] if latest_date_result else None

            if latest_date:
                latest_date = (
                    latest_date.date() if hasattr(latest_date, "date") else latest_date
                )
            else:
                latest_date = None

            # Get quality metrics for latest date
            if latest_date:
                quality_result = conn.execute(
                    """
                    SELECT
                        COUNT(DISTINCT ticker) AS ticker_count,
                        COUNT(*) AS total_rows,
                        SUM(CASE WHEN rsi_14 IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_rsi_null,
                        SUM(CASE WHEN ABS(rsi_14 - 50.0) < 0.01 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_rsi_50,
                        STDDEV(rsi_14) AS rsi_stddev,
                        SUM(CASE WHEN atr_14 IS NULL OR atr_14 = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_atr_missing,
                        SUM(CASE WHEN volume IS NULL OR volume = 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_vol_missing,
                        SUM(CASE WHEN close IS NULL OR close <= 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_price_invalid
                    FROM read_parquet(?)
                    WHERE CAST(Date AS DATE) = ?
                """,
                    [str(self.parquet_path), latest_date],
                ).fetchone()

                ticker_count = quality_result[0] if quality_result else 0
                pct_rsi_null = quality_result[2] if quality_result else 100.0
                pct_rsi_50 = quality_result[3] if quality_result else 100.0
                rsi_stddev = quality_result[4] if quality_result else 0.0
                pct_atr_missing = quality_result[5] if quality_result else 100.0
                pct_vol_missing = quality_result[6] if quality_result else 100.0
                pct_price_invalid = quality_result[7] if quality_result else 100.0
            else:
                ticker_count = 0
                pct_rsi_null = 100.0
                pct_rsi_50 = 100.0
                rsi_stddev = 0.0
                pct_atr_missing = 100.0
                pct_vol_missing = 100.0
                pct_price_invalid = 100.0

            conn.close()

        except Exception as e:
            logger.error(f"Data quality check failed: {e}")
            return DataQualityReport(
                is_fresh=False,
                latest_date=None,
                expected_date=expected_date,
                staleness_days=999,
                is_trading_day=is_trading_day,
                ticker_count=0,
                pct_rsi_identical=100.0,
                pct_missing_rsi=100.0,
                pct_missing_atr=100.0,
                pct_missing_volume=100.0,
                quality_score=0,
                recommendation="ABORT",
                issues=[f"Failed to query parquet: {e}"],
                stats=stats,
            )

        # Calculate staleness
        staleness_days = 0
        if latest_date and expected_date:
            # Count trading days between latest_date and expected_date
            check = latest_date
            while check < expected_date:
                check += timedelta(days=1)
                if self.is_trading_day(check):
                    staleness_days += 1

        is_fresh = staleness_days <= self.max_staleness_days

        # Build stats dict
        stats = {
            "ticker_count": ticker_count,
            "pct_rsi_null": pct_rsi_null,
            "pct_rsi_50": pct_rsi_50,
            "rsi_stddev": rsi_stddev,
            "pct_atr_missing": pct_atr_missing,
            "pct_vol_missing": pct_vol_missing,
            "pct_price_invalid": pct_price_invalid,
        }

        # Check issues
        if not is_fresh:
            issues.append(
                f"Data stale: {staleness_days} trading days old (max: {self.max_staleness_days})"
            )

        if latest_date is None:
            issues.append("No data found in parquet")
        elif latest_date != expected_date and is_trading_day:
            issues.append(f"Expected data from {expected_date}, got {latest_date}")

        if ticker_count < self.min_ticker_count:
            issues.append(
                f"Low ticker count: {ticker_count} (min: {self.min_ticker_count})"
            )

        if pct_rsi_50 > self.max_pct_identical_rsi:
            issues.append(
                f"Suspicious RSI values: {pct_rsi_50:.1f}% have RSI=50 (stale indicator)"
            )

        if pct_rsi_null > 10:
            issues.append(f"Missing RSI: {pct_rsi_null:.1f}% of tickers")

        if pct_atr_missing > 10:
            issues.append(f"Missing ATR: {pct_atr_missing:.1f}% of tickers")

        if pct_vol_missing > 10:
            issues.append(f"Missing volume: {pct_vol_missing:.1f}% of tickers")

        # Calculate quality score
        quality_score = 100.0

        if not is_fresh:
            quality_score -= min(30, staleness_days * 10)

        quality_score -= (
            (ticker_count / self.min_ticker_count) * 10
            if ticker_count < self.min_ticker_count
            else 0
        )

        if pct_rsi_50 > self.max_pct_identical_rsi:
            quality_score -= min(20, (pct_rsi_50 - self.max_pct_identical_rsi))

        quality_score -= min(15, pct_rsi_null)
        quality_score -= min(10, pct_atr_missing)
        quality_score -= min(5, pct_vol_missing)

        quality_score = max(0, min(100, quality_score))

        # Determine recommendation
        if quality_score < self.min_quality_score:
            recommendation = "ABORT"
        elif quality_score < self.caution_quality_score:
            recommendation = "CAUTION"
        else:
            recommendation = "PROCEED"

        report = DataQualityReport(
            is_fresh=is_fresh,
            latest_date=latest_date,
            expected_date=expected_date,
            staleness_days=staleness_days,
            is_trading_day=is_trading_day,
            ticker_count=ticker_count,
            pct_rsi_identical=pct_rsi_50,
            pct_missing_rsi=pct_rsi_null,
            pct_missing_atr=pct_atr_missing,
            pct_missing_volume=pct_vol_missing,
            quality_score=quality_score,
            recommendation=recommendation,
            issues=issues,
            stats=stats,
        )

        if MONITORING_AVAILABLE:
            try:
                collector = get_metrics_collector()
                staleness_hours = staleness_days * 24 if staleness_days > 0 else 0.0
                collector.emit_data_freshness(
                    source="parquet",
                    age_hours=staleness_hours,
                    is_stale=not is_fresh,
                    record_count=ticker_count,
                    last_update=datetime.combine(latest_date, datetime.min.time())
                    if latest_date
                    else None,
                )
            except Exception:
                pass

        return report


class SignalBatchValidator:
    """
    Validates a batch of signals for quality before they reach the watchlist.

    Detects broken pipelines that produce:
    - All identical scores (score=60 for all)
    - All identical priorities (priority=99 for all)
    - All identical RSI values (stale data)
    - Too few or too many signals
    - Suspicious clustering
    """

    def __init__(
        self,
        min_score_stddev: float = 5.0,
        max_pct_same_priority: float = 10.0,
        min_signals: int = 5,
        max_signals: int = 200,
    ):
        self.min_score_stddev = min_score_stddev
        self.max_pct_same_priority = max_pct_same_priority
        self.min_signals = min_signals
        self.max_signals = max_signals

    def validate_batch(self, signals: List[Dict]) -> BatchQualityReport:
        """
        Run quality checks on a batch of signals.
        """
        issues = []
        stats = {}

        if not signals:
            return BatchQualityReport(
                is_valid=False,
                recommendation="REJECT",
                issues=["Empty signal batch"],
                stats={"count": 0},
            )

        count = len(signals)
        stats["count"] = count

        # Check signal count
        if count < self.min_signals:
            issues.append(f"Too few signals: {count} (min: {self.min_signals})")
        elif count > self.max_signals:
            issues.append(f"Too many signals: {count} (max: {self.max_signals})")

        # Extract scores
        scores = [s.get("score", 50) for s in signals]
        priorities = [
            s.get("priority")
            for s in signals
            if s.get("priority") not in (None, "", 0)
        ]
        rsi_values = [s.get("rsi", 50) for s in signals if s.get("rsi") is not None]

        # Score diversity
        if len(scores) > 1:
            score_mean = sum(scores) / len(scores)
            score_sq_diffs = [(s - score_mean) ** 2 for s in scores]
            score_stddev = (sum(score_sq_diffs) / len(scores)) ** 0.5
            stats["score_mean"] = score_mean
            stats["score_std"] = score_stddev
            stats["score_min"] = min(scores)
            stats["score_max"] = max(scores)

            if score_stddev < self.min_score_stddev:
                issues.append(
                    f"All scores identical or near-identical (std={score_stddev:.2f}, min={self.min_score_stddev})"
                )
        else:
            stats["score_mean"] = scores[0]
            stats["score_std"] = 0
            stats["score_min"] = scores[0]
            stats["score_max"] = scores[0]

        # Priority uniqueness
        if len(priorities) > 1:
            priority_counts = {}
            for p in priorities:
                priority_counts[p] = priority_counts.get(p, 0) + 1
            max_p_count = max(priority_counts.values())
            max_pct = (max_p_count / len(priorities)) * 100
            stats["unique_priorities"] = len(priority_counts)
            stats["max_priority_count"] = max_p_count
            stats["max_priority_pct"] = max_pct

            if max_pct > self.max_pct_same_priority:
                issues.append(f"Too many signals share same priority: {max_pct:.1f}%")
        else:
            stats["unique_priorities"] = len(priorities)

        # RSI diversity (if available)
        if len(rsi_values) > 1:
            rsi_mean = sum(rsi_values) / len(rsi_values)
            rsi_sq_diffs = [(r - rsi_mean) ** 2 for r in rsi_values]
            rsi_stddev = (sum(rsi_sq_diffs) / len(rsi_values)) ** 0.5
            stats["rsi_mean"] = rsi_mean
            stats["rsi_std"] = rsi_stddev

            if rsi_stddev < 2:
                issues.append(
                    f"All RSI values identical or near-identical (std={rsi_stddev:.2f})"
                )

        # Check for required fields
        missing_fields = []
        for i, s in enumerate(signals):
            if "symbol" not in s:
                missing_fields.append(f"Signal {i}: missing 'symbol'")
            if "score" not in s:
                missing_fields.append(f"Signal {i}: missing 'score'")

        if missing_fields:
            issues.extend(missing_fields[:5])  # Limit to 5

        # Determine recommendation
        if len(issues) >= 3:
            recommendation = "REJECT"
        elif len(issues) >= 1:
            recommendation = "WARN"
        else:
            recommendation = "ACCEPT"

        is_valid = recommendation != "REJECT"

        return BatchQualityReport(
            is_valid=is_valid,
            recommendation=recommendation,
            issues=issues,
            stats=stats,
        )

    def fix_flat_scores(self, signals: List[Dict]) -> List[Dict]:
        """
        Attempt to re-score signals if scores are flat.

        Uses quick heuristic scoring based on:
        - RSI distance from 50 (more extreme = higher priority)
        - ATR% (higher = more opportunity)
        - Volume ratio (higher = more conviction)
        """
        fixed = []

        for s in signals:
            score = s.get("score", 50)

            # Only fix if all scores are suspiciously similar
            if "score" not in s:
                continue

            # Calculate heuristic score
            heuristic = 50

            # RSI adjustment (extreme RSI = more conviction)
            rsi = s.get("rsi", 50)
            if rsi is not None:
                rsi_dist = abs(rsi - 50)
                heuristic += rsi_dist * 0.5  # RSI 30 or 70 gets +10

            # ATR% adjustment
            atr_pct = s.get("atr_pct", 2.0)
            if atr_pct is not None:
                heuristic += atr_pct * 2  # Higher ATR = more opportunity

            # Volume adjustment
            vol_ratio = s.get("volume_ratio", 1.0)
            if vol_ratio is not None:
                if vol_ratio > 1.5:
                    heuristic += 5
                elif vol_ratio < 0.7:
                    heuristic -= 5

            # Clamp score
            heuristic = max(20, min(95, heuristic))

            # Update signal
            s = s.copy()
            s["score"] = heuristic
            s["score_freshened"] = True
            s["original_score"] = score
            fixed.append(s)

        return fixed


def validate_data_for_signals(
    as_of_date: date = None,
) -> Tuple[bool, DataQualityReport]:
    """
    Convenience function to validate data before signal generation.

    Returns:
        (should_proceed, report)
    """
    gate = DataQualityGate()
    report = gate.validate(as_of_date)

    if report.recommendation == "ABORT":
        logger.warning(f"DATA QUALITY ABORT: {report.issues}")
        logger.warning(
            f"Latest data: {report.latest_date}, expected: {report.expected_date}"
        )
        logger.warning(
            f"Staleness: {report.staleness_days} days, quality: {report.quality_score:.0f}/100"
        )
        return False, report

    if report.recommendation == "CAUTION":
        logger.warning(f"DATA QUALITY CAUTION: {report.issues}")
        logger.info(f"Quality score: {report.quality_score:.0f}/100")

    if report.recommendation == "PROCEED":
        logger.info(
            f"Data quality OK: {report.quality_score:.0f}/100, {report.ticker_count} tickers, latest: {report.latest_date}"
        )

    return report.recommendation != "ABORT", report


def validate_signal_batch(signals: List[Dict]) -> Tuple[bool, BatchQualityReport]:
    """
    Convenience function to validate a signal batch.

    Returns:
        (is_valid, report)
    """
    validator = SignalBatchValidator()
    report = validator.validate_batch(signals)

    if report.recommendation == "REJECT":
        logger.error(f"Signal batch REJECTED: {report.issues}")
        return False, report

    if report.recommendation == "WARN":
        logger.warning(f"Signal batch warnings: {report.issues}")
        logger.info(
            f"Batch stats: {report.stats.get('count', 0)} signals, "
            f"score std: {report.stats.get('score_std', 0):.2f}"
        )

    if report.recommendation == "ACCEPT":
        logger.info(
            f"Signal batch valid: {report.stats.get('count', 0)} signals, "
            f"score range: {report.stats.get('score_min', 0):.1f}-{report.stats.get('score_max', 0):.1f}"
        )

    return report.is_valid, report
