"""
Market Regime Detector - Quantitative, data-driven regime classification.

Detects market conditions using:
1. Universe breadth (% of stocks advancing vs declining)
2. Cross-sectional momentum (are winners or losers leading?)
3. Volatility regime (expanding, contracting, stable)
4. Multi-day pattern detection (selloff -> reversal, grind up, distribution)

NO YouTube, NO LLM, NO sentiment. Pure price action.
"""

import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
import duckdb
import pandas as pd
from pathlib import Path
import yfinance as yf

from config.config_loader import get_config
from autotrade.data_ingestion.bootstrap import ensure_core_market_data_ready
from autotrade.utils.security_metadata import validate_ticker_sector_sidecar

logger = logging.getLogger(__name__)

# H11 (2026-05-21): per-process dedup keys for the missing sidecar WARNING.
# Keyed on (path, validation_reason) so a different failure mode re-arms.
_SEEN_SIDECAR_MISSING_KEYS: set = set()


SECTOR_ETF_MAP: Dict[str, str] = {
    "XLB": "materials",
    "XLC": "communication_services",
    "XLE": "energy",
    "XLF": "financials",
    "XLI": "industrials",
    "XLK": "technology",
    "XLP": "consumer_staples",
    "XLRE": "real_estate",
    "XLU": "utilities",
    "XLV": "healthcare",
    "XLY": "consumer_discretionary",
}


MACRO_REFERENCE_TICKERS: Dict[str, tuple[str, ...]] = {
    "GLD": ("GLD",),
    # Use Alpaca-tradable BTC proxies so macro reference collection stays
    # equity-safe while still surfacing a BTC momentum signal.
    "BTC": ("IBIT", "GBTC"),
}


class SectorAnalyzer:
    """
    Intraday-first macro/sector analyzer.

    Produces a sector_bias map with:
    - +15 for top 3 momentum sectors
    - -20 for bottom 3 momentum sectors
    """

    def __init__(
        self,
        symbols: Optional[Dict[str, str]] = None,
        leader_bonus: float = 15.0,
        laggard_penalty: float = -20.0,
    ) -> None:
        self.sector_map = dict(symbols or SECTOR_ETF_MAP)
        self.leader_bonus = float(leader_bonus)
        self.laggard_penalty = float(laggard_penalty)

    @staticmethod
    def _compute_momentum(etf_ticker: str) -> Optional[Dict[str, float]]:
        """Compute blended 1H/1D return momentum for one ETF ticker."""
        try:
            bars_1h = yf.Ticker(etf_ticker).history(period="7d", interval="1h")
            bars_1d = yf.Ticker(etf_ticker).history(period="7d", interval="1d")
            if bars_1h is None or bars_1h.empty or bars_1d is None or bars_1d.empty:
                return None

            close_1h = bars_1h["Close"].dropna()
            close_1d = bars_1d["Close"].dropna()
            if len(close_1h) < 2 or len(close_1d) < 2:
                return None

            ret_1h = float((close_1h.iloc[-1] / close_1h.iloc[-2] - 1.0) * 100.0)
            ret_1d = float((close_1d.iloc[-1] / close_1d.iloc[-2] - 1.0) * 100.0)
            # Intraday-first blend.
            momentum = (0.60 * ret_1h) + (0.40 * ret_1d)
            return {"ret_1h": ret_1h, "ret_1d": ret_1d, "momentum": momentum}
        except Exception:
            return None

    def analyze_sector_bias(self) -> Dict[str, Any]:
        """
        Return macro context with per-sector bias and macro references.
        """
        macro_refs: Dict[str, Dict[str, float]] = {}
        for ref_name, tickers in MACRO_REFERENCE_TICKERS.items():
            for ticker in tickers:
                m = self._compute_momentum(ticker)
                if m:
                    macro_refs[ref_name] = m
                    break

        sector_rows: List[Dict[str, Any]] = []
        for etf, sector_name in self.sector_map.items():
            m = self._compute_momentum(etf)
            if not m:
                continue
            sector_rows.append(
                {
                    "etf": etf,
                    "sector": sector_name,
                    "ret_1h": m["ret_1h"],
                    "ret_1d": m["ret_1d"],
                    "momentum": m["momentum"],
                }
            )

        sector_rows.sort(key=lambda row: float(row["momentum"]), reverse=True)
        leaders = sector_rows[:3]
        laggards = sector_rows[-3:] if len(sector_rows) >= 3 else []

        sector_bias: Dict[str, float] = {str(row["sector"]): 0.0 for row in sector_rows}
        for row in leaders:
            sector_bias[str(row["sector"])] = self.leader_bonus
        for row in laggards:
            sector_bias[str(row["sector"])] = self.laggard_penalty

        return {
            "generated_at": datetime.now().isoformat(),
            "macro_references": macro_refs,
            "sector_rankings": sector_rows,
            "leading_sectors": [str(row["sector"]) for row in leaders],
            "lagging_sectors": [str(row["sector"]) for row in laggards],
            "sector_bias": sector_bias,
        }


class MarketRegime(Enum):
    STRONG_RALLY = "strong_rally"  # Broad advance, high breadth
    RECOVERY = "recovery"  # Reversal from selloff
    SHORT_SQUEEZE = "short_squeeze"  # Sharp reversal, beaten stocks leading
    GRINDING_UP = "grinding_up"  # Slow steady advance
    NEUTRAL = "neutral"  # No clear direction
    ROTATION = "rotation"  # Some sectors up, others down
    DISTRIBUTION = "distribution"  # Topping, selling into strength
    SELLOFF = "selloff"  # Broad decline
    RISK_OFF = "risk_off"  # Sustained weakness, high risk
    CAPITULATION = "capitulation"  # Panic selling, extreme breadth
    CRASH = "crash"  # Severe multi-day decline


@dataclass
class RegimeAnalysis:
    regime: MarketRegime
    confidence: float  # 0-1
    breadth_pct_positive: float  # % of universe advancing today
    breadth_5d_trend: str  # "improving", "stable", "deteriorating"
    avg_universe_return_1d: float
    avg_universe_return_5d: float
    top_decile_return: float  # Top 10% avg return
    bottom_decile_return: float  # Bottom 10% avg return
    dispersion: float  # Std dev of returns (high = opportunity)
    consecutive_down_days: int
    consecutive_up_days: int
    volume_trend: str  # "increasing", "stable", "decreasing"
    leading_sectors: List[str] = field(default_factory=list)
    lagging_sectors: List[str] = field(default_factory=list)
    sector_momentum: Dict[str, float] = field(default_factory=dict)
    recommended_strategy: Dict[str, Any] = field(default_factory=dict)
    pattern_detected: Optional[str] = None
    detection_timestamp: datetime = field(default_factory=datetime.now)
    sources_degraded: bool = False
    stale_session: bool = False


class MarketRegimeDetector:
    def __init__(self, parquet_path: Optional[str] = None):
        """
        Initialize the Market Regime Detector.

        Args:
            parquet_path: Path to daily features parquet. If None, uses config.
        """
        config = get_config()
        self._data_cfg = config.data
        self._downday_root = Path(self._data_cfg.downday_root)
        raw_parquet = parquet_path or self._data_cfg.daily_features_parquet
        self.parquet_path = self._resolve_data_path(raw_parquet)
        self._h5_bootstrap_candidates = self._resolve_h5_candidates()
        self._last_analysis: Optional[RegimeAnalysis] = None
        self._last_analysis_time: Optional[datetime] = None
        self._last_analysis_session_key: Optional[str] = None
        self._cache_ttl_minutes = 5  # Cache regime for 5 minutes
        self._missing_parquet_last_log: Optional[datetime] = None
        self._missing_parquet_suppressed_count = 0
        self._missing_parquet_log_interval = timedelta(minutes=15)
        self._last_h5_bootstrap_attempt: Optional[datetime] = None
        self._h5_bootstrap_retry_interval = timedelta(minutes=15)
        self._h5_chunk_rows = 250_000
        self._parquet_column_cache: Dict[str, set[str]] = {}

    def _resolve_data_path(self, raw_path: str) -> Path:
        """Resolve a data path, preferring DownDay-root relative locations."""
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return candidate
        down_candidate = self._downday_root / candidate
        local_candidate = Path(raw_path)
        if down_candidate.exists() or not local_candidate.exists():
            return down_candidate
        return local_candidate

    def _resolve_h5_candidates(self) -> List[Path]:
        """Build an ordered list of H5 files that can bootstrap parquet."""
        cfg_h5 = Path(
            str(getattr(self._data_cfg, "daily_features_h5", "daily_features.h5"))
        )
        candidates: List[Path] = []
        if cfg_h5.is_absolute():
            candidates.append(cfg_h5)
        else:
            candidates.append(self._downday_root / cfg_h5)
            candidates.append(Path(cfg_h5))
        candidates.append(self._downday_root / "daily_features.h5")

        seen = set()
        deduped: List[Path] = []
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(path)
        return deduped

    @staticmethod
    def _normalize_bootstrap_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize H5 frame columns to expected parquet schema."""
        out = df.copy()
        if not isinstance(out, pd.DataFrame):
            raise ValueError("H5 payload is not a DataFrame")
        if out.empty:
            return out

        if isinstance(out.index, pd.MultiIndex) or out.index.name is not None:
            out = out.reset_index()

        canonical = {
            "date": "Date",
            "datetime": "Date",
            "timestamp": "Date",
            "ticker": "ticker",
            "symbol": "ticker",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
            "sector": "sector",
        }
        rename_map: Dict[str, str] = {}
        existing = set(out.columns)
        for col in out.columns:
            normalized = str(col).strip().lower()
            target = canonical.get(normalized)
            if target and target not in existing:
                rename_map[col] = target
        if rename_map:
            out = out.rename(columns=rename_map)

        if "sector" not in out.columns:
            out["sector"] = None

        required = ["Date", "ticker", "Open", "Close", "Volume"]
        missing = [c for c in required if c not in out.columns]
        if missing:
            raise ValueError(f"H5 data missing required columns: {missing}")

        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
        out = out.dropna(subset=["Date"])
        out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()
        out = out[out["ticker"] != ""]
        return out

    def _convert_h5_to_parquet(self, h5_path: Path, parquet_path: Path) -> bool:
        """
        Convert a daily_features H5 file to parquet.
        Uses chunked conversion for table-backed H5 stores when possible.
        """
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = parquet_path.with_suffix(".tmp.parquet")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

        with pd.HDFStore(str(h5_path), mode="r") as store:
            keys = list(store.keys())
            if not keys:
                raise ValueError(f"No datasets found in H5 file: {h5_path}")
            preferred_key = next(
                (k for k in keys if ("daily" in k.lower() or "feature" in k.lower())),
                keys[0],
            )
            storer = store.get_storer(preferred_key)
            nrows = int(getattr(storer, "nrows", 0) or 0)
            is_table = bool(getattr(storer, "is_table", False))

            # Use chunked conversion when H5 is table-backed to avoid large memory spikes.
            if is_table and nrows > 0:
                try:
                    import pyarrow as pa
                    import pyarrow.parquet as pq
                except Exception:
                    pa = None
                    pq = None
                if pa is not None and pq is not None:
                    writer = None
                    rows_written = 0
                    try:
                        for start in range(0, nrows, self._h5_chunk_rows):
                            stop = min(nrows, start + self._h5_chunk_rows)
                            chunk = store.select(preferred_key, start=start, stop=stop)
                            chunk = self._normalize_bootstrap_df(chunk)
                            if chunk.empty:
                                continue
                            table = pa.Table.from_pandas(chunk, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    str(tmp_path),
                                    table.schema,
                                    compression="snappy",
                                )
                            writer.write_table(table)
                            rows_written += len(chunk)
                    finally:
                        if writer is not None:
                            writer.close()

                    if rows_written <= 0:
                        raise ValueError(
                            f"No rows converted from H5 key {preferred_key}"
                        )
                    tmp_path.replace(parquet_path)
                    logger.info(
                        f"Initialized parquet from H5 (chunked): {parquet_path} "
                        f"(rows={rows_written}, source={h5_path}, key={preferred_key})"
                    )
                    return True

            # Fallback path: load selected key as full DataFrame.
            try:
                frame = store.select(preferred_key)
            except Exception:
                frame = store[preferred_key]
            frame = self._normalize_bootstrap_df(frame)
            if frame.empty:
                raise ValueError(f"H5 key {preferred_key} produced empty frame")
            frame.to_parquet(tmp_path, index=False)
            tmp_path.replace(parquet_path)
            logger.info(
                f"Initialized parquet from H5: {parquet_path} "
                f"(rows={len(frame)}, source={h5_path}, key={preferred_key})"
            )
            return True

    def _bootstrap_parquet_from_h5(self, parquet_path: Path) -> bool:
        """Attempt to create parquet from configured DownDay H5 source."""
        now_ts = datetime.now()
        if (
            self._last_h5_bootstrap_attempt is not None
            and (now_ts - self._last_h5_bootstrap_attempt)
            < self._h5_bootstrap_retry_interval
        ):
            return False
        self._last_h5_bootstrap_attempt = now_ts

        for h5_path in self._h5_bootstrap_candidates:
            if not h5_path.exists():
                continue
            logger.warning(
                f"Parquet missing ({parquet_path}). Initializing from H5 source: {h5_path}"
            )
            try:
                if self._convert_h5_to_parquet(h5_path, parquet_path):
                    return True
            except ImportError as e:
                logger.error(
                    "Failed to bootstrap parquet from H5 because PyTables is missing. "
                    "Install dependency 'tables'. "
                    f"Error: {e}"
                )
                return False
            except Exception as e:
                logger.error(f"H5 -> parquet bootstrap failed ({h5_path}): {e}")
        return False

    def _sector_lookup_path(self) -> Optional[Path]:
        """Return the ticker->sector parquet sidecar path if it exists."""
        candidate = (
            Path(__file__).resolve().parent.parent.parent
            / "data"
            / "ticker_sectors.parquet"
        )
        validation = validate_ticker_sector_sidecar(candidate)
        if validation.valid:
            return candidate
        # H11 (2026-05-21): dedup. The sidecar is checked many times per
        # cycle and was producing 158+ identical WARNINGs/session. Log the
        # actionable WARNING once per (path, reason) tuple per process,
        # then DEBUG for the rest. Regime falls back to degraded mode
        # (sector blending disabled) — the WARNING line makes that
        # explicit so operators can grep one line and know.
        sidecar_key = (str(candidate), str(validation.reason))
        if sidecar_key not in _SEEN_SIDECAR_MISSING_KEYS:
            _SEEN_SIDECAR_MISSING_KEYS.add(sidecar_key)
            logger.warning(
                "[REGIME] ticker sector sidecar unavailable or invalid: %s "
                "(%s) — regime running in degraded mode (sector blending "
                "disabled); subsequent occurrences will log at DEBUG",
                candidate,
                validation.reason,
            )
        else:
            logger.debug(
                "[REGIME] ticker sector sidecar unavailable or invalid: %s "
                "(%s) [dedup]",
                candidate,
                validation.reason,
            )
        return None

    def _get_parquet_columns(self, parquet_path: Path) -> set[str]:
        cache_key = str(parquet_path)
        if cache_key in self._parquet_column_cache:
            return self._parquet_column_cache[cache_key]

        columns: set[str] = set()
        try:
            import pyarrow.parquet as pq

            schema = pq.read_schema(parquet_path)
            columns = set(schema.names or [])
        except Exception:
            try:
                conn = duckdb.connect()
                df = conn.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)",
                    [str(parquet_path)],
                ).fetchdf()
                conn.close()
                columns = set(df.get("column_name", []).tolist())
            except Exception:
                columns = set()

        self._parquet_column_cache[cache_key] = columns
        return columns

    def detect_regime(
        self, as_of_date: Optional[str] = None, use_cache: bool = True
    ) -> RegimeAnalysis:
        """
        Detect current market regime from universe price data.

        Uses DuckDB to compute:
        1. Today's breadth (% advancing)
        2. 5-day breadth trend
        3. Cross-sectional return distribution
        4. Consecutive up/down day count
        5. Volume trend
        6. Sector relative strength

        Args:
            as_of_date: Date to analyze (YYYY-MM-DD). If None, uses latest date.
            use_cache: Whether to use cached result if recent.

        Returns:
            RegimeAnalysis with regime classification and metrics.
        """
        session_key = str(as_of_date or datetime.now().date())

        # Check cache
        if use_cache and self._last_analysis and self._last_analysis_time:
            if datetime.now() - self._last_analysis_time < timedelta(
                minutes=self._cache_ttl_minutes
            ):
                if self._last_analysis_session_key == session_key:
                    logger.debug(
                        f"Returning cached regime: {self._last_analysis.regime.value}"
                    )
                    return self._last_analysis

        # Compute breadth metrics
        metrics = self._compute_breadth_metrics(as_of_date)

        if metrics is None or metrics.get("total_stocks", 0) == 0:
            logger.warning(
                "Could not compute breadth metrics, defaulting to NEUTRAL (degraded)"
            )
            analysis = RegimeAnalysis(
                regime=MarketRegime.NEUTRAL,
                confidence=0.0,
                breadth_pct_positive=50.0,
                breadth_5d_trend="stable",
                avg_universe_return_1d=0.0,
                avg_universe_return_5d=0.0,
                top_decile_return=0.0,
                bottom_decile_return=0.0,
                dispersion=0.0,
                consecutive_down_days=0,
                consecutive_up_days=0,
                volume_trend="stable",
                recommended_strategy=self._get_neutral_strategy(),
                sources_degraded=True,
            )
            self._last_analysis = analysis
            self._last_analysis_time = datetime.now()
            self._last_analysis_session_key = session_key
            return analysis

        # Classify regime based on metrics
        regime, confidence, pattern = self._classify_regime(metrics)

        # Get strategy adjustments
        strategy = self.get_strategy_adjustments(regime, metrics)

        # Build analysis object
        analysis = RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            breadth_pct_positive=metrics.get("pct_advancing", 50.0),
            breadth_5d_trend=metrics.get("breadth_trend", "stable"),
            avg_universe_return_1d=metrics.get("avg_return_1d", 0.0),
            avg_universe_return_5d=metrics.get("avg_return_5d", 0.0),
            top_decile_return=metrics.get("top_decile", 0.0),
            bottom_decile_return=metrics.get("bottom_decile", 0.0),
            dispersion=metrics.get("return_dispersion", 0.0),
            consecutive_down_days=metrics.get("consecutive_down_days", 0),
            consecutive_up_days=metrics.get("consecutive_up_days", 0),
            volume_trend=metrics.get("volume_trend", "stable"),
            leading_sectors=metrics.get("leading_sectors", []),
            lagging_sectors=metrics.get("lagging_sectors", []),
            sector_momentum=metrics.get("sector_momentum", {}),
            recommended_strategy=strategy,
            pattern_detected=pattern,
            stale_session=bool(metrics.get("stale_session", False)),
        )

        # Log the detection
        logger.info(
            f"REGIME DETECTED: {regime.value} (confidence: {confidence:.0%}) | "
            f"Breadth: {metrics.get('pct_advancing', 0):.1f}% | "
            f"Pattern: {pattern or 'none'}"
        )

        # Update cache
        self._last_analysis = analysis
        self._last_analysis_time = datetime.now()
        self._last_analysis_session_key = session_key

        return analysis

    def _compute_breadth_metrics(
        self, as_of_date: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute breadth and momentum metrics from parquet data.

        Returns:
            Dictionary with breadth metrics or None if error.
        """
        try:
            parquet_path = Path(self.parquet_path)
            if not parquet_path.exists():
                try:
                    ensure_core_market_data_ready(
                        max_staleness_days=getattr(
                            self._data_cfg, "max_staleness_days", 1
                        ),
                        bootstrap_from_h5_enabled=getattr(
                            self._data_cfg, "bootstrap_from_h5_enabled", True
                        ),
                        fail_fast=False,
                        parquet_path=parquet_path,
                    )
                except Exception as exc:
                    logger.debug(f"Shared bootstrap gateway warning: {exc}")
                    self._bootstrap_parquet_from_h5(parquet_path)
            if not parquet_path.exists():
                now_ts = datetime.now()
                should_log = (
                    self._missing_parquet_last_log is None
                    or (now_ts - self._missing_parquet_last_log)
                    >= self._missing_parquet_log_interval
                )
                if should_log:
                    suffix = ""
                    if self._missing_parquet_suppressed_count > 0:
                        suffix = f" (suppressed repeats: {self._missing_parquet_suppressed_count})"
                    logger.error(f"Parquet file not found: {parquet_path}{suffix}")
                    self._missing_parquet_last_log = now_ts
                    self._missing_parquet_suppressed_count = 0
                else:
                    self._missing_parquet_suppressed_count += 1
                return None
            parquet_sql_path = parquet_path.as_posix().replace("'", "''")
            parquet_columns = self._get_parquet_columns(parquet_path)

            # Sector source: prefer parquet's own column, then a sidecar
            # ticker->sector lookup at data/ticker_sectors.parquet (built
            # by tools/build_ticker_sector_map.py). Fall back to NULL.
            sector_lookup_path = self._sector_lookup_path()
            if "sector" in parquet_columns:
                sector_select = "sector"
                sector_join = ""
            elif sector_lookup_path is not None:
                lookup_sql = sector_lookup_path.as_posix().replace("'", "''")
                sector_select = "ts.sector AS sector"
                sector_join = (
                    f"LEFT JOIN read_parquet('{lookup_sql}') ts "
                    "ON UPPER(ts.ticker) = UPPER(p.ticker)"
                )
            else:
                sector_select = "NULL::VARCHAR AS sector"
                sector_join = ""

            # DuckDB query for breadth metrics
            query = f"""
            WITH daily_returns AS (
                SELECT
                    CAST(p.Date AS DATE) AS date,
                    UPPER(p.ticker) AS ticker,
                    {sector_select},
                    (p.Close - p."Open") / NULLIF(p."Open", 0) * 100 AS intraday_return,
                    (p.Close - LAG(p.Close) OVER (PARTITION BY p.ticker ORDER BY p.Date))
                        / NULLIF(LAG(p.Close) OVER (PARTITION BY p.ticker ORDER BY p.Date), 0) * 100 AS daily_return,
                    p.Volume,
                    LAG(p.Volume) OVER (PARTITION BY p.ticker ORDER BY p.Date) AS prev_volume
                FROM read_parquet('{parquet_sql_path}') p
                {sector_join}
                WHERE CAST(p.Date AS DATE) >= CURRENT_DATE - INTERVAL '20 days'
            ),
            daily_breadth AS (
                SELECT
                    date,
                    COUNT(*) AS total_stocks,
                    SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) AS advancing,
                    SUM(CASE WHEN daily_return < 0 THEN 1 ELSE 0 END) AS declining,
                    SUM(CASE WHEN daily_return > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) AS pct_advancing,
                    AVG(daily_return) AS avg_return,
                    STDDEV(daily_return) AS return_dispersion,
                    PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY daily_return) AS top_decile,
                    PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY daily_return) AS bottom_decile,
                    AVG(Volume * 1.0 / NULLIF(prev_volume, 0)) AS avg_volume_ratio
                FROM daily_returns
                WHERE daily_return IS NOT NULL
                GROUP BY date
                ORDER BY date
            ),
            sector_returns AS (
                SELECT
                    date,
                    sector,
                    AVG(daily_return) AS sector_avg_return,
                    COUNT(*) AS sector_count
                FROM daily_returns
                WHERE daily_return IS NOT NULL AND sector IS NOT NULL
                GROUP BY date, sector
                HAVING COUNT(*) >= 5
            ),
            ranked_sectors AS (
                SELECT
                    date,
                    sector,
                    sector_avg_return,
                    ROW_NUMBER() OVER (PARTITION BY date ORDER BY sector_avg_return DESC) AS rank_asc,
                    ROW_NUMBER() OVER (PARTITION BY date ORDER BY sector_avg_return ASC) AS rank_desc
                FROM sector_returns
            ),
            latest_date AS (
                SELECT MAX(date) AS max_date FROM daily_breadth
            )
            SELECT 
                b.date,
                b.total_stocks,
                b.advancing,
                b.declining,
                b.pct_advancing,
                b.avg_return,
                b.return_dispersion,
                b.top_decile,
                b.bottom_decile,
                b.avg_volume_ratio,
                AVG(b.pct_advancing) OVER (ORDER BY b.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS breadth_5d_avg,
                AVG(b.avg_return) OVER (ORDER BY b.date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS avg_return_5d,
                LISTAGG(CASE WHEN rs.rank_asc <= 3 THEN rs.sector END, ', ') FILTER (WHERE rs.rank_asc <= 3) AS leading_sectors,
                LISTAGG(CASE WHEN rs.rank_desc <= 3 THEN rs.sector END, ', ') FILTER (WHERE rs.rank_desc <= 3) AS lagging_sectors
            FROM daily_breadth b
            LEFT JOIN ranked_sectors rs ON b.date = rs.date
            GROUP BY b.date, b.total_stocks, b.advancing, b.declining, b.pct_advancing,
                     b.avg_return, b.return_dispersion, b.top_decile, b.bottom_decile, b.avg_volume_ratio
            ORDER BY b.date DESC
            LIMIT 10
            """

            conn = duckdb.connect()
            df = conn.execute(query).fetchdf()
            conn.close()

            if df.empty:
                logger.warning("No data returned from breadth query")
                return None

            # Drop today's row when it represents an incomplete intraday snapshot.
            # The parquet is rewritten throughout the day and partial data
            # produces misleading "today's regime" classifications. We treat a
            # row as incomplete if its ticker count is < 50% of the median of
            # the prior closed sessions. We still capture the partial breadth
            # so the classifier can use it as a recovery hint when yesterday
            # was a real selloff.
            data_complete = True
            stale_session = False
            partial_today_breadth: Optional[float] = None
            partial_today_avg_return: Optional[float] = None
            if len(df) >= 2:
                latest_total = float(df.iloc[0].get("total_stocks") or 0.0)
                prior = df.iloc[1:6]["total_stocks"].astype(float)
                prior_median = float(prior.median()) if len(prior) else 0.0
                if (
                    prior_median > 0.0
                    and latest_total > 0.0
                    and latest_total < 0.5 * prior_median
                ):
                    partial_today_breadth = float(
                        df.iloc[0].get("pct_advancing") or 0.0
                    )
                    partial_today_avg_return = float(
                        df.iloc[0].get("avg_return") or 0.0
                    )
                    logger.info(
                        f"[REGIME] Partial-day row deferred "
                        f"(today_tickers={int(latest_total)} vs prior_median={int(prior_median)}, "
                        f"partial_breadth={partial_today_breadth:.1f}%, "
                        f"partial_avg_ret={partial_today_avg_return:.2f}%); "
                        f"using last closed session for base classification."
                    )
                    df = df.iloc[1:].reset_index(drop=True)
                    data_complete = False
                    stale_session = True
                    if df.empty:
                        return None

            # Get latest row
            latest = df.iloc[0].to_dict()

            # Compute consecutive days
            consecutive_down = 0
            consecutive_up = 0
            for _, row in df.iterrows():
                if row["pct_advancing"] < 50:
                    consecutive_down += 1
                    consecutive_up = 0
                elif row["pct_advancing"] > 50:
                    consecutive_up += 1
                    consecutive_down = 0
                else:
                    break

            # Determine breadth trend
            if len(df) >= 5:
                recent_breadth = df.iloc[:3]["pct_advancing"].mean()
                older_breadth = df.iloc[3:6]["pct_advancing"].mean()
                if recent_breadth > older_breadth + 5:
                    breadth_trend = "improving"
                elif recent_breadth < older_breadth - 5:
                    breadth_trend = "deteriorating"
                else:
                    breadth_trend = "stable"
            else:
                breadth_trend = "stable"

            # Determine volume trend
            vol_ratio = latest.get("avg_volume_ratio", 1.0)
            if vol_ratio > 1.2:
                volume_trend = "increasing"
            elif vol_ratio < 0.8:
                volume_trend = "decreasing"
            else:
                volume_trend = "stable"

            # Parse sectors
            leading_sectors = [
                s.strip()
                for s in str(latest.get("leading_sectors", "")).split(",")
                if s.strip()
            ]
            lagging_sectors = [
                s.strip()
                for s in str(latest.get("lagging_sectors", "")).split(",")
                if s.strip()
            ]

            # Always compute per-sector momentum from the SectorAnalyzer
            # (intraday sector ETF moves) so downstream sector-relative RS
            # has data even when the parquet's own sector ranking succeeds.
            # When parquet ranking failed (no `sector` column → ['None']),
            # also use the SectorAnalyzer for leading/lagging.
            sector_momentum: Dict[str, float] = {}
            try:
                bias_snapshot = get_current_sector_bias()
                rankings = bias_snapshot.get("sector_rankings") or []
                for row in rankings:
                    name = str(row.get("sector") or "").strip().lower()
                    if name:
                        sector_momentum[name] = float(row.get("momentum") or 0.0)
                if not leading_sectors or set(leading_sectors) == {"None"}:
                    leading_sectors = list(bias_snapshot.get("leading_sectors") or [])
                    lagging_sectors = list(bias_snapshot.get("lagging_sectors") or [])
            except Exception as exc:
                logger.debug(f"SectorAnalyzer momentum fetch failed: {exc}")

            return {
                "date": latest.get("date"),
                "total_stocks": latest.get("total_stocks", 0),
                "advancing": latest.get("advancing", 0),
                "declining": latest.get("declining", 0),
                "pct_advancing": latest.get("pct_advancing", 50.0),
                "avg_return_1d": latest.get("avg_return", 0.0),
                "avg_return_5d": latest.get("avg_return_5d", 0.0),
                "return_dispersion": latest.get("return_dispersion", 0.0),
                "top_decile": latest.get("top_decile", 0.0),
                "bottom_decile": latest.get("bottom_decile", 0.0),
                "avg_volume_ratio": vol_ratio,
                "breadth_5d_avg": latest.get("breadth_5d_avg", 50.0),
                "breadth_trend": breadth_trend,
                "consecutive_down_days": consecutive_down,
                "consecutive_up_days": consecutive_up,
                "volume_trend": volume_trend,
                "leading_sectors": leading_sectors,
                "lagging_sectors": lagging_sectors,
                "sector_momentum": sector_momentum,
                "data_complete": data_complete,
                "stale_session": stale_session,
                "partial_today_breadth": partial_today_breadth,
                "partial_today_avg_return": partial_today_avg_return,
            }

        except Exception as e:
            logger.error(f"Error computing breadth metrics: {e}")
            return None

    def _classify_regime(
        self, metrics: Dict[str, Any]
    ) -> tuple[MarketRegime, float, Optional[str]]:
        """
        Classify market regime based on breadth metrics.

        Classification logic:
        - breadth > 65% AND 5d_avg_return > 1%     -> STRONG_RALLY
        - breadth > 60% AND consecutive_down >= 3    -> RECOVERY / SHORT_SQUEEZE
        - breadth 45-55% AND dispersion < 1.5        -> NEUTRAL
        - breadth 45-55% AND dispersion > 3.0        -> ROTATION
        - breadth < 40% AND consecutive_down <= 2    -> SELLOFF
        - breadth < 30% AND volume_surge             -> CAPITULATION
        - breadth < 25% AND consecutive_down >= 4    -> CRASH

        Returns:
            Tuple of (regime, confidence, pattern_name)
        """
        breadth = metrics.get("pct_advancing", 50.0)
        avg_return_5d = metrics.get("avg_return_5d", 0.0)
        consecutive_down = metrics.get("consecutive_down_days", 0)
        consecutive_up = metrics.get("consecutive_up_days", 0)
        dispersion = metrics.get("return_dispersion", 0.0)
        volume_trend = metrics.get("volume_trend", "stable")
        bottom_decile = metrics.get("bottom_decile", 0.0)
        partial_breadth = metrics.get("partial_today_breadth")
        partial_avg_return = metrics.get("partial_today_avg_return")

        # Intraday recovery override: when the base row is a closed prior
        # selloff but today's partial tape is meaningfully positive, treat
        # the regime as RECOVERY rather than carrying forward yesterday's
        # SELLOFF/CAPITULATION/CRASH posture into a green session.
        if (
            partial_breadth is not None
            and partial_avg_return is not None
            and breadth < 45
            and partial_breadth >= 55
            and partial_avg_return >= 0.4
        ):
            return MarketRegime.RECOVERY, 0.75, "intraday_recovery_from_prior_weakness"

        pattern = None

        # Check for CRASH
        if breadth < 25 and consecutive_down >= 4:
            return MarketRegime.CRASH, 0.95, "multi_day_crash"

        # Check for CAPITULATION
        if breadth < 30 and volume_trend == "increasing" and consecutive_down >= 2:
            return MarketRegime.CAPITULATION, 0.90, "capitulation_volume"

        # Check for SHORT_SQUEEZE / RECOVERY
        if breadth > 60 and consecutive_down >= 3:
            if bottom_decile > 0:  # Most beaten stocks are rallying
                pattern = "short_squeeze"
                return MarketRegime.SHORT_SQUEEZE, 0.85, pattern
            else:
                pattern = "v_bottom_recovery"
                return MarketRegime.RECOVERY, 0.85, pattern

        # Check for STRONG_RALLY
        if breadth > 65 and avg_return_5d > 1.0:
            return MarketRegime.STRONG_RALLY, 0.85, "broad_advance"

        # Check for DISTRIBUTION
        if breadth < 45 and consecutive_up >= 3:
            return MarketRegime.DISTRIBUTION, 0.75, "distribution_top"

        # Check for SELLOFF
        if breadth < 40 and consecutive_down <= 2:
            return MarketRegime.SELLOFF, 0.80, "broad_decline"

        # Check for RISK_OFF (Sustained weakness but not yet a crash/capitulation)
        if breadth < 45 and consecutive_down >= 3:
            return MarketRegime.RISK_OFF, 0.85, "sustained_decline"

        # Check for GRINDING_UP
        if breadth > 55 and consecutive_up >= 2 and dispersion < 2.0:
            return MarketRegime.GRINDING_UP, 0.75, "steady_advance"

        # Check for ROTATION
        if 45 <= breadth <= 55 and dispersion > 3.0:
            return MarketRegime.ROTATION, 0.75, "sector_rotation"

        # Default to NEUTRAL
        confidence = 0.5 + (
            abs(breadth - 50) / 100
        )  # Higher confidence if away from 50
        return MarketRegime.NEUTRAL, confidence, None

    def detect_pattern(self, lookback_days: int = 10) -> Dict[str, Any]:
        """
        Detect multi-day patterns for actionable strategies.

        Patterns:
        - SHORT_SQUEEZE_SETUP: 3+ down days -> breadth flip > 60%
        - MOMENTUM_CONTINUATION: 3+ up days with expanding breadth
        - DISTRIBUTION_TOP: New highs decreasing while breadth flat/up
        - SECTOR_ROTATION: Some sectors +3%, others -3%
        - V_BOTTOM: Sharp selloff followed by high-volume reversal

        Returns:
            Dictionary with pattern details and recommended actions.
        """
        metrics = self._compute_breadth_metrics()
        if not metrics:
            return {"pattern": None, "confidence": 0.0}

        pattern_info = {
            "pattern": None,
            "confidence": 0.0,
            "evidence": {},
            "actions": [],
        }

        breadth = metrics.get("pct_advancing", 50.0)
        consecutive_down = metrics.get("consecutive_down_days", 0)
        consecutive_up = metrics.get("consecutive_up_days", 0)
        dispersion = metrics.get("return_dispersion", 0.0)
        volume_trend = metrics.get("volume_trend", "stable")

        # SHORT_SQUEEZE_SETUP detection
        if consecutive_down >= 3 and breadth > 60:
            pattern_info["pattern"] = "SHORT_SQUEEZE_SETUP"
            pattern_info["confidence"] = 0.85
            pattern_info["evidence"] = {
                "consecutive_down_days": consecutive_down,
                "current_breadth": breadth,
                "volume_trend": volume_trend,
            }
            pattern_info["actions"] = [
                "Scan for most beaten-down stocks (weekly_return < -5%)",
                "Filter: still has solid fundamentals",
                "Filter: volume surge on reversal day (>2x average)",
                "Enter with tight stops (1.5x ATR)",
                "Target: +2-3% (snapback, not long-term hold)",
            ]

        # MOMENTUM_CONTINUATION detection
        elif consecutive_up >= 3 and breadth > 55 and volume_trend == "increasing":
            pattern_info["pattern"] = "MOMENTUM_CONTINUATION"
            pattern_info["confidence"] = 0.80
            pattern_info["evidence"] = {
                "consecutive_up_days": consecutive_up,
                "breadth": breadth,
                "volume_trend": volume_trend,
            }
            pattern_info["actions"] = [
                "Buy breakouts, ride momentum",
                "Use momentum scan only",
                "Hold 3-5 days minimum",
            ]

        # DISTRIBUTION_TOP detection
        elif consecutive_up >= 2 and breadth < 45:
            pattern_info["pattern"] = "DISTRIBUTION_TOP"
            pattern_info["confidence"] = 0.75
            pattern_info["evidence"] = {
                "consecutive_up_days": consecutive_up,
                "breadth": breadth,
                "dispersion": dispersion,
            }
            pattern_info["actions"] = [
                "Tighten stops to 1.5x ATR",
                "Take 25% profit on all winners > +3%",
                "No new entries until breadth improves",
                "Increase cash reserve to 40%",
            ]

        # SECTOR_ROTATION detection
        elif 40 <= breadth <= 60 and dispersion > 3.0:
            pattern_info["pattern"] = "SECTOR_ROTATION"
            pattern_info["confidence"] = 0.70
            pattern_info["evidence"] = {
                "breadth": breadth,
                "dispersion": dispersion,
                "leading_sectors": metrics.get("leading_sectors", []),
                "lagging_sectors": metrics.get("lagging_sectors", []),
            }
            pattern_info["actions"] = [
                "Exit positions in lagging sectors",
                "Enter positions in leading sectors",
                "Use momentum scan only",
                "Hold period: 3-5 days",
            ]

        return pattern_info

    def get_strategy_adjustments(
        self, regime: MarketRegime, metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Return specific strategy parameter adjustments per regime.

        Returns dict compatible with trading_config.yaml overrides.

        Args:
            regime: The detected market regime
            metrics: Optional breadth metrics for fine-tuning

        Returns:
            Dictionary with strategy parameter overrides
        """
        config = get_config()

        # Try to load from config first
        try:
            regime_config = getattr(config, "regime_strategies", None)
            if regime_config and regime.value in regime_config:
                # Convert Pydantic model to dict
                cfg = regime_config[regime.value]
                if hasattr(cfg, "model_dump"):
                    return cfg.model_dump()
                elif hasattr(cfg, "dict"):
                    return cfg.dict()
                else:
                    return dict(cfg)
        except Exception as e:
            logger.debug(f"Could not load regime config: {e}")

        # Default strategy profiles
        strategies = {
            MarketRegime.STRONG_RALLY: {
                "scan_types": ["momentum_breakout", "earnings_momentum"],
                "min_atr_percent": 2.0,
                "position_size_multiplier": 1.2,
                "max_positions": 30,
                "hold_period_target": "3-5 days",
                "stop_multiplier": 2.0,
                "profit_target_multiplier": 3.0,
                "entry_preference": "buy_breakout",
                "cash_reserve_pct": 10,
            },
            MarketRegime.RECOVERY: {
                "scan_types": ["mean_reversion", "momentum_breakout"],
                "min_atr_percent": 2.5,
                "position_size_multiplier": 1.0,
                "max_positions": 25,
                "hold_period_target": "1-2 days",
                "stop_multiplier": 2.5,
                "profit_target_multiplier": 2.0,
                "entry_preference": "buy_dip",
                "cash_reserve_pct": 20,
            },
            MarketRegime.SHORT_SQUEEZE: {
                "scan_types": ["mean_reversion"],
                "min_atr_percent": 3.0,
                "position_size_multiplier": 0.8,
                "max_positions": 15,
                "hold_period_target": "1 day",
                "stop_multiplier": 1.5,
                "profit_target_multiplier": 1.5,
                "entry_preference": "buy_dip",
                "prefer_beaten_down": True,
                "cash_reserve_pct": 30,
            },
            MarketRegime.GRINDING_UP: {
                "scan_types": ["momentum_breakout"],
                "min_atr_percent": 2.0,
                "position_size_multiplier": 1.1,
                "max_positions": 25,
                "hold_period_target": "3-5 days",
                "stop_multiplier": 2.0,
                "profit_target_multiplier": 2.5,
                "entry_preference": "buy_breakout",
                "cash_reserve_pct": 15,
            },
            MarketRegime.NEUTRAL: self._get_neutral_strategy(),
            MarketRegime.ROTATION: {
                "scan_types": ["momentum_breakout"],
                "min_atr_percent": 2.5,
                "position_size_multiplier": 0.9,
                "max_positions": 20,
                "hold_period_target": "3-5 days",
                "stop_multiplier": 2.0,
                "profit_target_multiplier": 2.0,
                "entry_preference": "follow_sectors",
                "cash_reserve_pct": 25,
            },
            MarketRegime.DISTRIBUTION: {
                "scan_types": ["mean_reversion"],
                "min_atr_percent": 2.0,
                "position_size_multiplier": 0.6,
                "max_positions": 15,
                "hold_period_target": "1-2 days",
                "stop_multiplier": 1.5,
                "profit_target_multiplier": 1.5,
                "entry_preference": "tight_stops",
                "cash_reserve_pct": 40,
                "action": "tighten_stops",
            },
            MarketRegime.SELLOFF: {
                "scan_types": ["mean_reversion"],
                "min_atr_percent": 2.0,
                "position_size_multiplier": 0.5,
                "max_positions": 10,
                "hold_period_target": "1 day",
                "stop_multiplier": 1.5,
                "profit_target_multiplier": 1.5,
                "cash_reserve_pct": 50,
            },
            MarketRegime.RISK_OFF: {
                "scan_types": ["mean_reversion"],
                "min_atr_percent": 2.0,
                "position_size_multiplier": 0.3,
                "max_positions": 5,
                "hold_period_target": "1 day",
                "stop_multiplier": 1.2,
                "profit_target_multiplier": 1.2,
                "cash_reserve_pct": 60,
                "action": "tighten_stops",
            },
            MarketRegime.CAPITULATION: {
                "scan_types": [],
                "position_size_multiplier": 0.0,
                "max_positions": 0,
                "action": "exit_all_losers",
                "cash_reserve_pct": 80,
            },
            MarketRegime.CRASH: {
                "scan_types": [],
                "position_size_multiplier": 0.0,
                "max_positions": 0,
                "action": "exit_all_positions",
                "cash_reserve_pct": 90,
            },
        }

        return strategies.get(regime, self._get_neutral_strategy())

    def _get_neutral_strategy(self) -> Dict[str, Any]:
        """Return neutral/default strategy."""
        return {
            "scan_types": ["momentum_breakout", "mean_reversion"],
            "min_atr_percent": 3.0,
            "position_size_multiplier": 1.0,
            "max_positions": 25,
            "hold_period_target": "3 days",
            "stop_multiplier": 2.0,
            "profit_target_multiplier": 2.5,
            "entry_preference": "standard",
            "cash_reserve_pct": 20,
        }

    def get_regime_signal(self) -> Dict[str, Any]:
        """
        Get a simplified signal dict for integration with other systems.

        Returns:
            Dict with regime info for use in signals, conviction, etc.
        """
        analysis = self.detect_regime()

        return {
            "regime": analysis.regime.value,
            "confidence": analysis.confidence,
            "breadth_pct": analysis.breadth_pct_positive,
            "pattern": analysis.pattern_detected,
            "strategy_adjustments": analysis.recommended_strategy,
            "timestamp": analysis.detection_timestamp.isoformat(),
        }


# Singleton instance for easy access
_detector: Optional[MarketRegimeDetector] = None
_sector_analyzer: Optional[SectorAnalyzer] = None


def get_regime_detector(parquet_path: Optional[str] = None) -> MarketRegimeDetector:
    """
    Get or create the singleton MarketRegimeDetector instance.

    Args:
        parquet_path: Optional custom parquet path

    Returns:
        MarketRegimeDetector instance
    """
    global _detector
    if _detector is None or parquet_path is not None:
        _detector = MarketRegimeDetector(parquet_path)
    return _detector


def detect_current_regime() -> RegimeAnalysis:
    """
    Convenience function to detect current regime.

    Returns:
        RegimeAnalysis object
    """
    detector = get_regime_detector()
    return detector.detect_regime()


def get_sector_analyzer() -> SectorAnalyzer:
    """Get or create singleton SectorAnalyzer."""
    global _sector_analyzer
    if _sector_analyzer is None:
        _sector_analyzer = SectorAnalyzer()
    return _sector_analyzer


def get_current_sector_bias() -> Dict[str, Any]:
    """Convenience helper for macro + sector bias snapshot."""
    analyzer = get_sector_analyzer()
    return analyzer.analyze_sector_bias()


def get_current_regime_signal() -> Dict[str, Any]:
    """
    Convenience function to get current regime as signal dict.

    Returns:
        Dict with regime information
    """
    detector = get_regime_detector()
    return detector.get_regime_signal()
