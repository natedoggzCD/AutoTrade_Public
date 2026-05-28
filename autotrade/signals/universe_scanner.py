"""
Universe Scanner - Fast DuckDB-based stock screening.

Scans the FULL DownDay universe using DuckDB for fast, full-coverage
candidate generation with unique scores and clear ranking factors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Optional, Tuple

import duckdb
import pandas as pd

from config.config_loader import get_config
from autotrade.feature_engineering.adapters import get_universe_scanner_adapter
from autotrade.signals.support_resistance import estimate_sr_levels
from autotrade.signals.universe_filters import (
    DEFAULT_MAX_MARKET_CAP,
    DEFAULT_MIN_MARKET_CAP,
    HARD_BLOCK_MEGA_CAP,
)
from autotrade.utils.security_metadata import get_nasdaq_screener_path


@dataclass
class ScanResult:
    """Container for scan results."""

    data: pd.DataFrame
    latest_date: Optional[date]
    elapsed_sec: float


class UniverseScanner:
    """
    DuckDB-powered full-universe scanner.

    Produces fast scans across the full 4,450-ticker DownDay universe
    and computes unique, rankable scores directly in SQL.
    """

    def __init__(
        self,
        parquet_path: Optional[Path] = None,
        predictions_db_path: Optional[Path] = None,
        parent_logger: Optional[logging.Logger] = None,
    ) -> None:
        self.config = get_config()
        self.logger = parent_logger or logging.getLogger(__name__)

        self.parquet_path = (
            Path(parquet_path) if parquet_path else self._resolve_parquet_path()
        )
        # Legacy predictions DB input is intentionally ignored.
        self.predictions_db_path = None
        if predictions_db_path:
            self.logger.info(
                "Ignoring legacy predictions_db_path override; scanner is parquet-only."
            )

        self.scan_cfg = self.config.universe_scanner
        self.feature_cfg = self.config.feature_engineering
        self.exclude_symbols = {
            "AAPL",
            "MSFT",
            "NVDA",
            "TSLA",
            "AMZN",
            "GOOGL",
            "META",
            "SPY",
            "QQQ",
            "DIA",
            "IWM",
            "VXX",
            "UVXY",
            "NEE",
            "SO",  # Exclude mega-cap utilities that violate small/mid-cap focus
        }
        self.exclude_symbols.update(HARD_BLOCK_MEGA_CAP)
        self._feature_adapter = None

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Daily features parquet not found: {self.parquet_path}"
            )

        if self.feature_cfg.enabled:
            try:
                self._feature_adapter = get_universe_scanner_adapter(
                    config=self.feature_cfg.model_dump()
                )
            except Exception as e:
                self.logger.warning(
                    "Universe scanner feature adapter unavailable, falling back: %s",
                    e,
                )

    def _resolve_parquet_path(self) -> Path:
        data_cfg = self.config.data
        base = Path(data_cfg.downday_root)
        rel = Path(data_cfg.daily_features_parquet)
        return rel if rel.is_absolute() else base / rel

    def _connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect()

    def _resolve_security_metadata_path(self) -> Path:
        return get_nasdaq_screener_path()

    def _get_latest_date(self, con: duckdb.DuckDBPyConnection) -> Optional[date]:
        df = con.execute(
            "SELECT MAX(Date) AS max_date FROM parquet_scan(?)",
            [str(self.parquet_path)],
        ).fetchdf()
        if df.empty or df.iloc[0]["max_date"] is None:
            return None
        max_ts = pd.to_datetime(df.iloc[0]["max_date"])
        return max_ts.date()

    def _is_data_stale(self, latest: Optional[date]) -> bool:
        if latest is None:
            return True
        max_age = int(self.scan_cfg.max_data_age_days)
        age_days = (datetime.now().date() - latest).days
        if age_days > max_age:
            self.logger.warning(
                "Universe scanner data stale: latest=%s age=%s days (max=%s).",
                latest,
                age_days,
                max_age,
            )
            return True
        return False

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        # Accept legacy `support` key as alias for pullback.
        pullback_weight = float(weights.get("pullback", weights.get("support", 0.10)))
        base = {
            "momentum": float(weights.get("momentum", 0.30)),
            "volume": float(weights.get("volume", 0.25)),
            "volatility": float(weights.get("volatility", 0.20)),
            "trend": float(weights.get("trend", 0.15)),
            "pullback": pullback_weight,
        }
        total = sum(base.values())
        if total <= 0:
            return {
                "momentum": 0.30,
                "volume": 0.25,
                "volatility": 0.20,
                "trend": 0.15,
                "pullback": 0.10,
            }
        return {k: v / total for k, v in base.items()}

    def _build_scored_query(
        self,
        lookback_days: int,
        has_sector: bool,
        has_market_cap: bool,
        parquet_has_sector: bool,
        parquet_has_market_cap: bool,
        metadata_path: Optional[Path],
    ) -> Tuple[str, Dict[str, float]]:
        cfg = self.scan_cfg
        weights = self._normalize_weights(cfg.score_weights)

        preferred_low, preferred_high = cfg.preferred_atr_range
        atr_mid = (preferred_low + preferred_high) / 2.0

        min_price = cfg.min_price
        max_price = cfg.max_price
        min_avg_volume = cfg.min_avg_volume
        min_mkt_cap = getattr(cfg, "min_market_cap", DEFAULT_MIN_MARKET_CAP)
        max_mkt_cap = getattr(cfg, "max_market_cap", DEFAULT_MAX_MARKET_CAP)
        post_spike_volume = float(
            getattr(cfg, "post_spike_volume_threshold", 5.0) or 5.0
        )
        post_spike_range_atr = float(
            getattr(cfg, "post_spike_range_atr_threshold", 1.5) or 1.5
        )

        metadata_join = ""
        if metadata_path and metadata_path.exists():
            metadata_join = f"""
            LEFT JOIN (
                SELECT
                    UPPER(Symbol) AS meta_ticker,
                    NULLIF(TRIM(Sector), '') AS sector,
                    TRY_CAST(REPLACE(CAST("Market Cap" AS VARCHAR), ',', '') AS DOUBLE) AS market_cap
                FROM read_csv_auto('{metadata_path.as_posix()}', HEADER=TRUE)
            ) meta
                ON UPPER(p.ticker) = meta.meta_ticker
            """

        if has_sector:
            if parquet_has_sector and metadata_join:
                sector_select = "COALESCE(p.sector, meta.sector) AS sector,"
            elif parquet_has_sector:
                sector_select = "p.sector AS sector,"
            else:
                sector_select = "meta.sector AS sector,"
        else:
            sector_select = ""

        if has_market_cap:
            if parquet_has_market_cap and metadata_join:
                market_cap_select = (
                    "COALESCE(p.market_cap, meta.market_cap) AS market_cap,"
                )
            elif parquet_has_market_cap:
                market_cap_select = "p.market_cap AS market_cap,"
            else:
                market_cap_select = "meta.market_cap AS market_cap,"
        else:
            market_cap_select = ""

        sector_scored = "sector," if has_sector else ""
        market_cap_scored = "l.market_cap," if has_market_cap else ""
        sector_momentum = (
            "AVG(weekly_return) OVER (PARTITION BY sector) AS sector_momentum,"
            if has_sector
            else "CAST(NULL AS DOUBLE) AS sector_momentum,"
        )

        sector_bonus = (
            "CASE WHEN sector_momentum > 3 THEN 10 ELSE 0 END" if has_sector else "0"
        )
        market_cap_filter = (
            f"AND (l.market_cap IS NULL OR l.market_cap BETWEEN {min_mkt_cap} AND {max_mkt_cap})"
            if has_market_cap
            else ""
        )

        exclude_list = ", ".join([f"'{s}'" for s in sorted(self.exclude_symbols)])
        exclude_clause = f"AND ticker NOT IN ({exclude_list})" if exclude_list else ""

        query = f"""
        WITH base AS (
            SELECT
                ticker,
                {sector_select}
                Date,
                Open,
                High,
                Low,
                Close,
                Volume,
                RSI_14,
                RSI_14_lag_1,
                SMA_20,
                atr_14,
                {market_cap_select}
                AVG(Volume) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS volume_sma_20,
                MIN(Low) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS support_ref,
                MAX(High) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ) AS resistance_ref,
                (Close / NULLIF(LAG(Close, {lookback_days}) OVER (PARTITION BY ticker ORDER BY Date), 0) - 1) * 100
                    AS weekly_return,
                Volume / NULLIF(AVG(Volume) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                ), 0) AS volume_ratio_20d,
                (Open - LAG(Close) OVER (PARTITION BY ticker ORDER BY Date))
                    / NULLIF(LAG(Close) OVER (PARTITION BY ticker ORDER BY Date), 0) * 100 AS gap_percent,
                AVG(Close) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 49 PRECEDING AND CURRENT ROW
                ) AS SMA_50,
                AVG(Close) OVER (
                    PARTITION BY ticker ORDER BY Date
                    ROWS BETWEEN 199 PRECEDING AND CURRENT ROW
                ) AS SMA_200,
                ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY Date DESC) AS rn
            FROM parquet_scan('{self.parquet_path.as_posix()}') p
            {metadata_join}
            WHERE Close IS NOT NULL AND Volume IS NOT NULL
              {exclude_clause}
        ),
        latest AS (
            SELECT * FROM base WHERE rn = 1
        ),
        levels AS (
            SELECT
                l.*,
                COALESCE(l.support_ref, l.Close - COALESCE(l.atr_14, l.Close * 0.03) * 1.5) AS s1_price_calc,
                COALESCE(l.resistance_ref, l.Close + COALESCE(l.atr_14, l.Close * 0.03) * 2.0) AS r1_price_calc
            FROM latest l
        ),
        scored AS (
            SELECT
                l.ticker,
                {sector_scored}
                l.Date,
                l.Open,
                l.High,
                l.Low,
                l.Close,
                l.Close AS price,
                l.Volume,
                l.RSI_14,
                l.RSI_14 AS rsi,
                l.RSI_14_lag_1,
                l.SMA_20,
                l.SMA_50,
                l.SMA_200,
                l.atr_14,
                {market_cap_scored}
                l.weekly_return,
                l.volume_ratio_20d,
                l.gap_percent,
                l.s1_price_calc AS s1_price,
                GREATEST(20, LEAST(95, 100 - ABS((l.Close - l.s1_price_calc) / NULLIF(l.Close, 0) * 100) * 15)) AS s1_strength,
                l.r1_price_calc AS r1_price,
                GREATEST(20, LEAST(95, 100 - ABS((l.r1_price_calc - l.Close) / NULLIF(l.Close, 0) * 100) * 15)) AS r1_strength,
                (l.r1_price_calc + COALESCE(l.atr_14, l.Close * 0.03) * 0.75) AS r2_price,
                CASE WHEN COALESCE(l.atr_14, 0) > 0 THEN (l.Close - l.s1_price_calc) / l.atr_14 ELSE NULL END AS support_dist_atr,
                CASE WHEN COALESCE(l.atr_14, 0) > 0 THEN (l.r1_price_calc - l.Close) / l.atr_14 ELSE NULL END AS resistance_dist_atr,
                COALESCE(
                    GREATEST(COALESCE(l.atr_14, 0) * 3.0, l.Close * 0.015)
                    / NULLIF(GREATEST(COALESCE(l.atr_14, 0) * 2.0, l.Close * 0.010), 0),
                    1.5
                ) AS rr_ratio,
                LEAST(
                    10,
                    GREATEST(
                        0,
                        5
                        + COALESCE(l.weekly_return, 0) * 0.4
                        + COALESCE((l.Close / NULLIF(l.SMA_20, 0) - 1) * 100, 0) * 0.15
                        + (COALESCE(l.volume_ratio_20d, 1) - 1) * 1.5
                    )
                ) AS bullish_score,
                COALESCE((l.atr_14 / NULLIF(l.Close, 0)) * 100, 0) AS atr_percent_db,
                COALESCE((l.Close - l.s1_price_calc) / NULLIF(l.Close, 0) * 100, 0) AS distance_to_s1_pct,
                COALESCE((l.r1_price_calc - l.Close) / NULLIF(l.Close, 0) * 100, 0) AS distance_to_r1_pct,
                GREATEST(0.02, LEAST(1.25, COALESCE((l.atr_14 / NULLIF(l.Close, 0)) * 100, 0) * 0.05)) AS spread_pct,
                COALESCE((l.atr_14 / NULLIF(l.Close, 0)) * 100, 0) AS atr_percent,
                COALESCE(l.volume_ratio_20d, 0) AS volume_ratio,
                COALESCE(l.volume_sma_20, l.Volume) AS avg_volume,
                {sector_momentum}
                COALESCE(PERCENT_RANK() OVER (ORDER BY COALESCE(l.weekly_return, 0)) * 100, 0) AS momentum_component,
                COALESCE(PERCENT_RANK() OVER (ORDER BY COALESCE(l.volume_ratio_20d, 0)) * 100, 0) AS volume_component,
                COALESCE(
                    (1 - PERCENT_RANK() OVER (ORDER BY ABS(COALESCE((l.atr_14 / NULLIF(l.Close, 0)) * 100, 0) - {atr_mid}))) * 100,
                    0
                ) AS volatility_component,
                (
                    CASE WHEN l.Close > l.SMA_20 THEN 40 ELSE 0 END +
                    CASE WHEN l.Close > l.SMA_50 THEN 30 ELSE 0 END +
                    CASE WHEN l.Close > l.SMA_200 THEN 30 ELSE 0 END
                ) AS trend_component,
                (
                    CASE
                        WHEN l.RSI_14 IS NULL THEN 50
                        WHEN l.RSI_14 BETWEEN 35 AND 65 THEN 100 - ABS(l.RSI_14 - 50) * 2
                        WHEN l.RSI_14 < 35 THEN 55
                        ELSE 35
                    END
                ) AS pullback_component
            FROM levels l
            WHERE l.Close BETWEEN {min_price} AND {max_price}
              AND COALESCE(l.volume_sma_20, l.Volume) >= {min_avg_volume}
              {market_cap_filter}
              {exclude_clause}
              AND NOT (
                  COALESCE(l.volume_ratio_20d, 0) >= {post_spike_volume}
                  AND COALESCE((l.High - l.Low) / NULLIF(l.Close, 0), 0)
                      >= {post_spike_range_atr} * COALESCE(l.atr_14 / NULLIF(l.Close, 0), 0)
              )
        )
        SELECT
            *,
            (
                momentum_component * {weights["momentum"]} +
                volume_component * {weights["volume"]} +
                volatility_component * {weights["volatility"]} +
                trend_component * {weights["trend"]} +
                pullback_component * {weights["pullback"]}
            ) AS score_base,
            (
                LEAST(
                    100,
                    (
                        momentum_component * {weights["momentum"]} +
                        volume_component * {weights["volume"]} +
                        volatility_component * {weights["volatility"]} +
                        trend_component * {weights["trend"]} +
                        pullback_component * {weights["pullback"]}
                    ) + {sector_bonus}
                )
                + (ABS(hash(ticker)) % 1000) / 10000.0
            ) AS score
        FROM scored
        """
        return query, weights

    def _run_union_scan(self, max_candidates: int, lookback_days: int) -> ScanResult:
        start = datetime.now()
        con = self._connect()

        latest = self._get_latest_date(con)
        if self._is_data_stale(latest):
            return ScanResult(
                pd.DataFrame(), latest, (datetime.now() - start).total_seconds()
            )

        metadata_path = self._resolve_security_metadata_path()
        metadata_has_sector = False
        metadata_has_market_cap = False

        # Detect optional parquet columns
        try:
            cols_df = con.execute(
                "DESCRIBE SELECT * FROM parquet_scan(?)", [str(self.parquet_path)]
            ).fetchdf()
            cols = {c.lower() for c in cols_df["column_name"].tolist()}
            parquet_has_sector = "sector" in cols
            parquet_has_market_cap = "market_cap" in cols
        except Exception:
            parquet_has_sector = False
            parquet_has_market_cap = False

        if metadata_path.exists():
            try:
                metadata_cols = {
                    str(c).strip().lower()
                    for c in pd.read_csv(metadata_path, nrows=0).columns
                }
                metadata_has_sector = "sector" in metadata_cols
                metadata_has_market_cap = (
                    "market cap" in metadata_cols or "market_cap" in metadata_cols
                )
            except Exception as e:
                self.logger.warning(
                    "Universe scanner metadata CSV unreadable, continuing without metadata: %s",
                    e,
                )

        has_sector = parquet_has_sector or metadata_has_sector
        has_market_cap = parquet_has_market_cap or metadata_has_market_cap

        if not has_sector:
            self.logger.debug(
                "Universe scanner: sector column not found, skipping sector momentum boost."
            )
        elif not parquet_has_sector and metadata_has_sector:
            self.logger.debug(
                "Universe scanner: sector sourced from %s.",
                metadata_path.name,
            )
        if not has_market_cap:
            self.logger.debug(
                "Universe scanner: market_cap column not found, skipping market cap filter."
            )
        elif not parquet_has_market_cap and metadata_has_market_cap:
            self.logger.debug(
                "Universe scanner: market_cap sourced from %s.",
                metadata_path.name,
            )

        scored_query, _ = self._build_scored_query(
            lookback_days,
            has_sector,
            has_market_cap,
            parquet_has_sector,
            parquet_has_market_cap,
            metadata_path if metadata_path.exists() else None,
        )

        min_atr = self.scan_cfg.min_atr_percent
        pref_low, pref_high = self.scan_cfg.preferred_atr_range

        scan_sql_parts = []

        # Dynamic params
        cfg = self.scan_cfg
        vol_ratio = cfg.min_volume_ratio
        rsi_min = cfg.rsi_min
        rsi_max = cfg.rsi_max
        wk_ret = cfg.min_weekly_return
        gap_pct = cfg.min_gap_percent

        if "momentum_breakout" in self.scan_cfg.scan_types:
            scan_sql_parts.append(f"""
                SELECT *, 'momentum_breakout' AS scan_type
                FROM ({scored_query})
                WHERE weekly_return > {wk_ret}
                  AND RSI_14 BETWEEN {rsi_min} AND {rsi_max}
                  AND volume_ratio >= {vol_ratio}
                  AND Close > SMA_20
                  AND atr_percent >= {min_atr}
            """)

        if "mean_reversion" in self.scan_cfg.scan_types:
            scan_sql_parts.append(f"""
                SELECT *, 'mean_reversion' AS scan_type
                FROM ({scored_query})
                WHERE weekly_return < -3
                  AND RSI_14 < 35
                  AND volume_ratio >= {max(2.0, vol_ratio)}
                  AND (
                        (SMA_200 IS NOT NULL AND ABS(Close - SMA_200) / NULLIF(Close, 0) * 100 <= 3)
                        OR (support_dist_atr IS NOT NULL AND support_dist_atr <= 1.0)
                  )
                  AND atr_percent >= {min_atr}
            """)

        if "earnings_momentum" in self.scan_cfg.scan_types:
            scan_sql_parts.append(f"""
                SELECT *, 'earnings_momentum' AS scan_type
                FROM ({scored_query})
                WHERE gap_percent > {gap_pct}
                  AND volume_ratio >= {max(3.0, vol_ratio)}
                  AND (RSI_14 - COALESCE(RSI_14_lag_1, RSI_14)) > 0
                  AND atr_percent BETWEEN {pref_low} AND {pref_high}
            """)

        if not scan_sql_parts:
            return ScanResult(
                pd.DataFrame(), latest, (datetime.now() - start).total_seconds()
            )

        union_sql = " UNION ALL ".join(scan_sql_parts)

        final_sql = f"""
            WITH combined AS (
                {union_sql}
            ),
            ranked AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY ticker
                           ORDER BY score DESC, volume_ratio DESC, atr_percent DESC, rr_ratio DESC
                       ) AS rn
                FROM combined
            )
            SELECT *
            FROM ranked
            WHERE rn = 1
            ORDER BY score DESC, volume_ratio DESC, atr_percent DESC, rr_ratio DESC
            LIMIT {max_candidates}
        """

        df = con.execute(final_sql).fetchdf()
        con.close()

        if not df.empty:
            df["date"] = pd.to_datetime(df["Date"]).dt.date

        elapsed = (datetime.now() - start).total_seconds()
        return ScanResult(df, latest, elapsed)

    def _enhance_scan_sr_levels(
        self,
        df: pd.DataFrame,
        *,
        lookback_bars: int = 140,
    ) -> pd.DataFrame:
        """
        Improve candidate S/R levels using pivot clustering on recent OHLCV.

        Uses only technical price/volume data from the local parquet source.
        """
        if df.empty or "ticker" not in df.columns:
            return df

        tickers = df["ticker"].dropna().astype(str).str.upper().unique().tolist()
        if not tickers:
            return df

        tickers_sql = ", ".join("'" + t.replace("'", "''") + "'" for t in tickers)
        if not tickers_sql:
            return df

        con = self._connect()
        try:
            history_sql = f"""
            WITH hist AS (
                SELECT
                    ticker,
                    Date,
                    Open,
                    High,
                    Low,
                    Close,
                    Volume,
                    atr_14,
                    ROW_NUMBER() OVER (
                        PARTITION BY ticker ORDER BY Date DESC
                    ) AS rn
                FROM parquet_scan('{self.parquet_path.as_posix()}')
                WHERE ticker IN ({tickers_sql})
                  AND Close IS NOT NULL
                  AND High IS NOT NULL
                  AND Low IS NOT NULL
            )
            SELECT
                ticker,
                Date,
                Open,
                High,
                Low,
                Close,
                Volume,
                atr_14
            FROM hist
            WHERE rn <= {max(40, int(lookback_bars))}
            ORDER BY ticker, Date
            """
            hist_df = con.execute(history_sql).fetchdf()
        except Exception as e:
            self.logger.warning(
                "SR enhancement history query failed, using base levels: %s", e
            )
            con.close()
            return df
        finally:
            try:
                con.close()
            except Exception:
                pass

        if hist_df.empty:
            return df

        sr_by_ticker: Dict[str, Dict[str, float]] = {}
        for ticker, group in hist_df.groupby("ticker"):
            try:
                sr = estimate_sr_levels(group, lookback_bars=min(140, len(group)))
                if sr:
                    sr_by_ticker[str(ticker).upper()] = sr
            except Exception as e:
                self.logger.debug("SR enhancement skipped for %s: %s", ticker, e)

        if not sr_by_ticker:
            return df

        enhanced = df.copy()
        fields = [
            "s1_price",
            "s1_strength",
            "r1_price",
            "r1_strength",
            "support_dist_atr",
            "resistance_dist_atr",
            "distance_to_s1_pct",
            "distance_to_r1_pct",
            "sr_quality_score",
        ]

        for idx, row in enhanced.iterrows():
            ticker = str(row.get("ticker", "")).upper()
            sr = sr_by_ticker.get(ticker)
            if not sr:
                continue
            for field in fields:
                val = sr.get(field)
                if val is None:
                    continue
                if field in {"s1_price", "r1_price"} and float(val) <= 0:
                    continue
                enhanced.at[idx, field] = float(val)

        return enhanced

    def combined_scan(
        self, max_candidates: Optional[int] = None, lookback_days: int = 5
    ) -> pd.DataFrame:
        """
        Run all enabled scans, merge and dedupe.

        Returns DataFrame with ticker, score, and ranking factors.
        """
        max_candidates = max_candidates or self.scan_cfg.max_candidates
        result = self._run_union_scan(
            max_candidates=max_candidates, lookback_days=lookback_days
        )

        if result.data.empty:
            self.logger.warning("Universe scanner returned no candidates.")
            return result.data

        result.data = self._enhance_scan_sr_levels(result.data)

        if self._feature_adapter is not None:
            try:
                result.data = self._feature_adapter.enrich_scan_results(result.data)
            except Exception as e:
                self.logger.warning(
                    "Universe scanner shared feature enrichment failed, continuing: %s",
                    e,
                )

        self.logger.info(
            "Universe scan complete: %s candidates | latest=%s | elapsed=%.2fs",
            len(result.data),
            result.latest_date,
            result.elapsed_sec,
        )

        return result.data

    def scan_momentum_breakouts(
        self, lookback_days: int = 5, min_volume: int = 100000
    ) -> pd.DataFrame:
        """Find stocks with momentum breakout characteristics."""
        df = self.combined_scan(
            max_candidates=self.scan_cfg.max_candidates, lookback_days=lookback_days
        )
        if df.empty:
            return df
        filtered = df[df["scan_type"] == "momentum_breakout"].copy()
        if "avg_volume" in filtered.columns:
            filtered = filtered[filtered["avg_volume"] >= min_volume]
        return filtered

    def scan_mean_reversion(self, lookback_days: int = 5) -> pd.DataFrame:
        """Find oversold stocks bouncing off support."""
        df = self.combined_scan(
            max_candidates=self.scan_cfg.max_candidates, lookback_days=lookback_days
        )
        return df[df["scan_type"] == "mean_reversion"].copy() if not df.empty else df

    def scan_earnings_momentum(self) -> pd.DataFrame:
        """Find stocks with post-earnings momentum."""
        df = self.combined_scan(
            max_candidates=self.scan_cfg.max_candidates, lookback_days=5
        )
        return df[df["scan_type"] == "earnings_momentum"].copy() if not df.empty else df
