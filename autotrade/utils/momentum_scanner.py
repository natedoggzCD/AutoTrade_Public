"""
Live momentum scanner artifact generator.

Builds a continuously refreshed watchlist of off-plan momentum names using the
local universe, intraday market data, and catalyst enrichment.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence
from zoneinfo import ZoneInfo
from collections import Counter

try:
    import duckdb
except Exception:  # pragma: no cover - dependency is optional at import time
    duckdb = None

import pandas as pd

from config.config_loader import get_config
from autotrade.utils.alpaca_client_factory import (
    create_data_client,
    resolve_alpaca_credentials,
)
from autotrade.utils.news_aggregator import NewsAggregator

try:
    from autotrade.utils.intraday_data_provider import get_intraday_bars_batch

    INTRADAY_BATCH_AVAILABLE = True
except Exception:  # pragma: no cover - fallback is load-time only
    get_intraday_bars_batch = None
    INTRADAY_BATCH_AVAILABLE = False


logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
CT = ZoneInfo("America/Chicago")
PROJECT_DIR = Path(
    os.environ.get("AUTOTRADE_ROOT", Path(__file__).resolve().parents[2])
)
DEFAULT_EXCLUDE = {"SPY", "QQQ"}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.astimezone(ET)


def _parse_clock(value: Any, default_hour: int, default_minute: int) -> dt_time:
    text = str(value or "").strip()
    if not text:
        return dt_time(default_hour, default_minute)
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return dt_time(hour, minute)
    except Exception:
        pass
    return dt_time(default_hour, default_minute)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically with retries for Windows file lock contention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")

    import time
    max_retries = 5
    retry_delay = 0.5

    last_exc = None
    for attempt in range(max_retries):
        try:
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=False)
                handle.flush()
                os.fsync(handle.fileno())

            # Attempt atomic replace. On Windows, this can still fail if another 
            # process has the target file open for reading.
            if path.exists():
                # On Windows, replace() can raise PermissionError if 'path' is open.
                # Try os.replace which is generally more atomic and robust on POSIX,
                # but still has limitations on Windows.
                os.replace(str(tmp_path), str(path))
            else:
                tmp_path.rename(path)

            return # Success

        except (PermissionError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))
                continue

    logger.error(f"[MOMENTUM] Failed to write watchlist to {path} after {max_retries} attempts: {last_exc}")
    # Final cleanup attempt
    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass



def _safe_path(value: Any) -> Path:
    path = Path(str(value or "").strip() or "data/momentum_watchlist_live.json")
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def _normalize_bar_frame(bars: Any) -> pd.DataFrame:
    if isinstance(bars, pd.DataFrame):
        frame = bars.copy()
    else:
        frame = pd.DataFrame(list(bars or []))
    if frame.empty:
        return frame
    frame.columns = [str(col).lower() for col in frame.columns]
    if isinstance(frame.index, pd.DatetimeIndex):
        idx = pd.to_datetime(frame.index, errors="coerce")
        if idx.tz is not None:
            idx = idx.tz_convert(ET)
        frame.index = idx
    elif "timestamp" in frame.columns:
        ts = pd.to_datetime(frame["timestamp"], errors="coerce")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert(ET)
        frame.index = ts
    frame = frame.dropna(subset=[c for c in ("open", "high", "low", "close", "volume") if c in frame.columns])
    if not {"open", "high", "low", "close", "volume"}.issubset(frame.columns):
        return pd.DataFrame()
    return frame.sort_index()


def load_momentum_watchlist(
    path: Optional[Path] = None,
    *,
    now_et: Optional[datetime] = None,
    config: Optional[Any] = None,
) -> Dict[str, Any]:
    cfg = config or get_config()
    scanner_cfg = getattr(cfg, "momentum_scanner", None)
    if scanner_cfg is None or not bool(getattr(scanner_cfg, "enabled", True)):
        return {
            "loaded": False,
            "stale": False,
            "reason": "disabled",
            "artifact_path": str(_safe_path(path or "data/momentum_watchlist_live.json")),
            "generated_at_et": "",
            "symbols": [],
        }

    artifact_path = _safe_path(path or getattr(scanner_cfg, "artifact_path", None))
    if not artifact_path.exists():
        return {
            "loaded": False,
            "stale": False,
            "reason": "missing",
            "artifact_path": str(artifact_path),
            "generated_at_et": "",
            "symbols": [],
        }

    try:
        with open(artifact_path, encoding="utf-8") as handle:
            payload = json.load(handle) or {}
    except Exception as exc:
        logger.warning("Momentum watchlist artifact unreadable: %s", exc)
        return {
            "loaded": False,
            "stale": False,
            "reason": f"read_error:{type(exc).__name__}",
            "artifact_path": str(artifact_path),
            "generated_at_et": "",
            "symbols": [],
        }

    current_ts = (now_et or datetime.now(ET)).astimezone(ET)
    generated_at = _parse_iso_dt(payload.get("generated_at_et"))
    status = str(payload.get("status", "") or "").strip().lower()
    stale_after_seconds = int(getattr(scanner_cfg, "stale_after_seconds", 420) or 420)
    stale = bool(
        generated_at is None
        or (current_ts - generated_at).total_seconds() > max(30, stale_after_seconds)
    )
    age_seconds = (
        None
        if generated_at is None
        else max(0.0, (current_ts - generated_at).total_seconds())
    )
    symbols = list(payload.get("symbols", []) or [])
    if stale:
        symbols = []
    if not status and generated_at is not None and symbols:
        # Backward compatibility for older artifacts written before explicit
        # status/candidate_count fields were added.
        status = "ok"
    loaded = generated_at is not None and not stale and status == "ok"

    return {
        "loaded": loaded,
        "stale": stale,
        "reason": "stale" if stale else (status if status != "ok" else ""),
        "artifact_path": str(artifact_path),
        "generated_at_et": payload.get("generated_at_et", ""),
        "generated_at_ct": payload.get("generated_at_ct", ""),
        "session": str(payload.get("session", "")),
        "status": status,
        "candidate_count": _as_int(payload.get("candidate_count"), len(symbols)),
        "raw_candidate_count": _as_int(payload.get("raw_candidate_count"), 0),
        "scan_count": _as_int(payload.get("scan_count"), 0),
        "age_seconds": age_seconds,
        "symbols": symbols,
        "raw": payload,
    }


class MomentumScanner:
    def __init__(
        self,
        *,
        config: Optional[Any] = None,
        output_path: Optional[Path] = None,
        news_aggregator: Optional[NewsAggregator] = None,
        data_client: Any = None,
        intraday_fetcher: Optional[Callable[..., Dict[str, pd.DataFrame]]] = None,
        universe_rows: Optional[Sequence[Mapping[str, Any]]] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or get_config()
        self.scanner_cfg = self.config.momentum_scanner
        self.output_path = _safe_path(output_path or self.scanner_cfg.artifact_path)
        self.news_aggregator = news_aggregator or NewsAggregator()
        self.data_client = data_client if data_client is not None else self._build_data_client()
        self.intraday_fetcher = intraday_fetcher or get_intraday_bars_batch
        self._universe_rows_override = list(universe_rows) if universe_rows is not None else None
        self._sleep_fn = sleep_fn

    @staticmethod
    def _now_et() -> datetime:
        return datetime.now(ET)

    def _build_data_client(self) -> Any:
        try:
            creds = resolve_alpaca_credentials(require=False)
            if creds is None:
                return None
            return create_data_client(
                api_key=creds.api_key,
                secret_key=creds.secret_key,
                paper=bool(creds.paper),
                require_credentials=False,
            )
        except Exception as exc:
            logger.debug("Momentum scanner Alpaca client unavailable: %s", exc)
            return None

    def _resolve_daily_features_path(self) -> Path:
        data_cfg = self.config.data
        rel = Path(data_cfg.daily_features_parquet)
        if rel.is_absolute():
            return rel
        return Path(data_cfg.downday_root) / rel

    def _resolve_session(self, now_et: datetime) -> str:
        local_time = now_et.timetz()
        now_ct = now_et.astimezone(CT)
        start_time_ct = _parse_clock(
            getattr(self.scanner_cfg, "start_time_ct", "06:00"), 6, 0
        )
        if dt_time(4, 0, tzinfo=ET) <= local_time < dt_time(9, 30, tzinfo=ET):
            if now_ct.timetz().replace(tzinfo=None) < start_time_ct:
                return "before_start_window"
            return "premarket"
        if dt_time(9, 30, tzinfo=ET) <= local_time < dt_time(16, 0, tzinfo=ET):
            return "intraday"
        return "outside"

    def _sleep_seconds_for_session(self, session: str) -> int:
        if session == "premarket":
            return int(getattr(self.scanner_cfg, "premarket_interval_seconds", 90) or 90)
        if session == "intraday":
            return int(getattr(self.scanner_cfg, "intraday_interval_seconds", 180) or 180)
        return int(
            getattr(self.scanner_cfg, "sleep_outside_window_seconds", 300) or 300
        )

    def _build_rejection_summary(
        self, candidates: Sequence[Mapping[str, Any]]
    ) -> Dict[str, int]:
        counts: Counter[str] = Counter()
        for row in candidates or []:
            for reason in list(row.get("rejection_reasons", []) or []):
                label = str(reason or "").strip()
                if label:
                    counts[label] += 1
        return dict(counts.most_common(10))

    def _load_universe_rows(self) -> List[Dict[str, Any]]:
        if self._universe_rows_override is not None:
            return [dict(row) for row in self._universe_rows_override]
        if duckdb is None:
            return []

        parquet_path = self._resolve_daily_features_path()
        if not parquet_path.exists():
            logger.warning("Momentum scanner parquet missing: %s", parquet_path)
            return []

        con = duckdb.connect()
        try:
            query = """
                WITH base AS (
                    SELECT
                        UPPER(ticker) AS ticker,
                        CAST(Date AS DATE) AS trade_date,
                        Close,
                        Volume,
                        atr_14,
                        AVG(Volume) OVER (
                            PARTITION BY ticker ORDER BY Date
                            ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                        ) AS avg_volume_20,
                        (Close / NULLIF(LAG(Close, 5) OVER (PARTITION BY ticker ORDER BY Date), 0) - 1) * 100
                            AS weekly_return,
                        ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY Date DESC) AS rn
                    FROM parquet_scan(?)
                    WHERE ticker IS NOT NULL
                      AND Close IS NOT NULL
                      AND Volume IS NOT NULL
                )
                SELECT
                    ticker,
                    trade_date,
                    Close AS prev_close,
                    COALESCE(avg_volume_20, Volume) AS avg_volume,
                    atr_14,
                    weekly_return,
                    CASE
                        WHEN Close > 0 THEN COALESCE((atr_14 / Close) * 100, 0)
                        ELSE 0
                    END AS atr_percent
                FROM base
                WHERE rn = 1
            """
            frame = con.execute(query, [str(parquet_path)]).fetchdf()
        finally:
            con.close()

        if frame.empty:
            return []

        cfg = self.scanner_cfg
        frame["ticker"] = frame["ticker"].astype(str).str.upper()
        frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce").fillna(0.0)
        frame["avg_volume"] = pd.to_numeric(frame["avg_volume"], errors="coerce").fillna(0.0)
        frame["atr_14"] = pd.to_numeric(frame["atr_14"], errors="coerce").fillna(0.0)
        frame["atr_percent"] = pd.to_numeric(frame["atr_percent"], errors="coerce").fillna(0.0)
        frame["weekly_return"] = pd.to_numeric(frame["weekly_return"], errors="coerce").fillna(0.0)
        frame = frame[
            (frame["prev_close"] >= float(cfg.min_price))
            & (frame["prev_close"] <= float(cfg.max_price))
            & (frame["avg_volume"] >= float(cfg.min_avg_volume))
        ]
        frame = frame[~frame["ticker"].isin(DEFAULT_EXCLUDE)]
        # Universe pre-ranking: liquidity x volatility only. Intentionally
        # excludes weekly_return — that biased the scan window toward names
        # that already had last week's move (consolidating/fading by intraday
        # 10:30) and starved today's actual breakouts from being seen.
        frame["selection_score"] = (
            frame["avg_volume"].clip(lower=1.0).pow(0.35)
            * frame["atr_percent"].clip(lower=0.5)
        )
        if not frame.empty and "selection_score" in frame.columns:
            frame = frame.sort_values("selection_score", ascending=False)
        return frame.to_dict(orient="records")

    def _select_symbols(self, rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        max_symbols = int(getattr(self.scanner_cfg, "max_live_scan_symbols", 400) or 400)
        selected = [dict(row) for row in rows[: max(1, max_symbols)]]
        return [row for row in selected if str(row.get("ticker", "")).strip()]

    def _fetch_bars_map(self, symbols: Sequence[str], minutes_back: int) -> Dict[str, pd.DataFrame]:
        if not symbols or not INTRADAY_BATCH_AVAILABLE or self.intraday_fetcher is None:
            return {}
        batch_size = max(1, int(getattr(self.scanner_cfg, "alpaca_batch_size", 25) or 25))
        pause_seconds = max(
            0.0, float(getattr(self.scanner_cfg, "alpaca_pause_seconds", 1.0) or 0.0)
        )
        result: Dict[str, pd.DataFrame] = {}
        for offset in range(0, len(symbols), batch_size):
            batch = list(symbols[offset : offset + batch_size])
            try:
                bars_map = self.intraday_fetcher(
                    batch,
                    self.data_client,
                    minutes_back=minutes_back,
                    interval="1m",
                    max_batch=batch_size,
                )
            except Exception as exc:
                logger.debug("Momentum scanner batch failed: %s", exc)
                bars_map = {}
            for ticker, bars in (bars_map or {}).items():
                result[str(ticker).upper()] = _normalize_bar_frame(bars)
            if pause_seconds > 0 and offset + batch_size < len(symbols):
                self._sleep_fn(pause_seconds)
        return result

    def _score_symbol(
        self,
        universe_row: Mapping[str, Any],
        bars: pd.DataFrame,
        *,
        session: str,
        now_et: datetime,
    ) -> Optional[Dict[str, Any]]:
        frame = _normalize_bar_frame(bars)
        if frame.empty or len(frame) < 8:
            return None

        ticker = str(universe_row.get("ticker") or "").upper().strip()
        if not ticker:
            return None

        frame = frame.tail(max(20, int(getattr(self.scanner_cfg, "momentum_min_bars", 12) or 12)))
        first_open = _as_float(frame["open"].iloc[0], 0.0)
        last_price = _as_float(frame["close"].iloc[-1], 0.0)
        high_price = _as_float(frame["high"].max(), 0.0)
        total_volume = _as_int(frame["volume"].sum(), 0)
        prev_close = _as_float(universe_row.get("prev_close"), 0.0)
        avg_volume = _as_float(universe_row.get("avg_volume"), 0.0)
        atr_14 = _as_float(universe_row.get("atr_14"), 0.0)
        weekly_return = _as_float(universe_row.get("weekly_return"), 0.0)
        if first_open <= 0 or last_price <= 0 or prev_close <= 0:
            return None

        close_diff = frame["close"].diff().dropna()
        positive_steps = int((close_diff > 0).sum())
        trend_strength = positive_steps / max(1, len(close_diff))
        short_now = _as_float(frame["close"].tail(5).mean(), last_price)
        short_prev = _as_float(frame["close"].iloc[-10:-5].mean(), first_open)
        short_momentum = (
            ((short_now - short_prev) / short_prev) * 100 if short_prev > 0 else 0.0
        )

        gain_pct = ((last_price - first_open) / first_open) * 100
        gap_pct = ((last_price - prev_close) / prev_close) * 100
        pullback_pct = ((high_price - last_price) / high_price) * 100 if high_price > 0 else 0.0

        if session == "premarket":
            premarket_start = now_et.replace(hour=4, minute=0, second=0, microsecond=0)
            open_time = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
            total_window = max((open_time - premarket_start).total_seconds(), 1.0)
            elapsed = max(0.0, min((now_et - premarket_start).total_seconds(), total_window))
            progress = elapsed / total_window
            # Scale expected premarket participation from 3% near 04:00 ET up to
            # 8% into the open so legitimate runners are not rejected just
            # because overall tape participation is still thin before the bell.
            expected_fraction = 0.03 + (0.05 * progress)
        else:
            expected_fraction = 0.20
        volume_ratio = total_volume / max(1.0, avg_volume * expected_fraction)
        min_trend = float(getattr(self.scanner_cfg, "min_trend_strength", 0.55))
        min_rel_volume = float(getattr(self.scanner_cfg, "min_rel_volume", 1.5))
        fade_reject_pct = float(getattr(self.scanner_cfg, "fade_reject_pct", 4.0))

        rejection_reasons: List[str] = []
        if trend_strength < min_trend:
            rejection_reasons.append(f"trend_strength<{min_trend:.2f}")
        if volume_ratio < min_rel_volume:
            rejection_reasons.append(f"rel_volume<{min_rel_volume:.2f}")
        if pullback_pct > fade_reject_pct:
            rejection_reasons.append(f"pullback>{fade_reject_pct:.1f}")

        if session == "premarket":
            min_gain = float(getattr(self.scanner_cfg, "min_premarket_gain_pct", 2.5))
            min_volume = int(getattr(self.scanner_cfg, "min_premarket_volume", 100000))
            if gain_pct < min_gain:
                rejection_reasons.append(f"premarket_gain<{min_gain:.1f}")
            if total_volume < min_volume:
                rejection_reasons.append(f"premarket_volume<{min_volume}")
        else:
            min_gain = float(getattr(self.scanner_cfg, "min_intraday_gain_pct", 4.0))
            min_short = float(getattr(self.scanner_cfg, "min_short_momentum_pct", 0.4))
            if gap_pct < min_gain:
                rejection_reasons.append(f"intraday_gain<{min_gain:.1f}")
            if short_momentum < min_short:
                rejection_reasons.append(f"short_momentum<{min_short:.1f}")

        base_score = (
            45.0
            + min(25.0, max(0.0, gain_pct) * 3.5)
            + min(18.0, max(0.0, volume_ratio - 1.0) * 9.0)
            + min(15.0, max(0.0, trend_strength - 0.50) * 40.0)
            + min(10.0, max(0.0, short_momentum) * 5.0)
            + min(6.0, max(0.0, weekly_return) * 0.35)
            + min(5.0, _as_float(universe_row.get("atr_percent"), 0.0) * 0.25)
            - min(12.0, max(0.0, pullback_pct - 1.0) * 3.0)
        )
        score = _clamp(base_score, 0.0, 100.0)
        min_score = float(getattr(self.scanner_cfg, "min_score", 75.0))
        if score < min_score:
            rejection_reasons.append(f"scanner_score<{min_score:.0f}")

        stop_buffer = max(atr_14 * 1.5, last_price * 0.02)
        target_buffer = max(atr_14 * 3.0, last_price * 0.04)
        entry_price = round(last_price, 4)
        stop_loss = round(max(0.01, last_price - stop_buffer), 4)
        target = round(last_price + target_buffer, 4)
        risk_reward = target_buffer / max(stop_buffer, 1e-9)
        priority_score = (
            score
            + max(0.0, volume_ratio - 1.0) * 18.0
            + max(0.0, trend_strength - 0.5) * 20.0
            + max(0.0, short_momentum) * 6.0
            - max(0.0, pullback_pct - 1.0) * 4.0
        )

        return {
            "symbol": ticker,
            "ticker": ticker,
            "phase_seen": session,
            "first_seen_et": now_et.isoformat(),
            "last_seen_et": now_et.isoformat(),
            "score": round(score, 2),
            "confidence": round(score, 2),
            "qualified_for_entry": len(rejection_reasons) == 0,
            "rejection_reasons": rejection_reasons,
            "entry_source": "momentum_scanner",
            "source_bucket": "watchlist",
            "reserved_slot_class": "momentum_scanner",
            "intraday_reserve": True,
            "setup_type": "continuation",
            "scan_type": "momentum_breakout",
            "recommendation": "watch",
            "action": "watch",
            "entry_price": entry_price,
            "current_price": entry_price,
            "price": entry_price,
            "stop_loss": stop_loss,
            "target": target,
            "risk_reward": round(risk_reward, 2),
            "priority_score": round(priority_score, 2),
            "premarket_gain_pct": round(gain_pct, 2) if session == "premarket" else round(max(0.0, gap_pct), 2),
            "session_gain_pct": round(gap_pct, 2),
            "day_return": round(gap_pct, 2),
            "short_momentum": round(short_momentum, 2),
            "pullback_pct": round(pullback_pct, 2),
            "premarket_volume": total_volume if session == "premarket" else 0,
            "session_volume": total_volume,
            "volume_ratio": round(volume_ratio, 3),
            "rel_volume": round(volume_ratio, 3),
            "trend_strength": round(trend_strength, 3),
            "weekly_return": round(weekly_return, 2),
            "atr_14": round(atr_14, 4),
            "atr_percent": round(_as_float(universe_row.get("atr_percent"), 0.0), 3),
            "avg_volume": _as_int(avg_volume, 0),
            "catalyst_score": 0.0,
            "catalyst_summary": "",
            "has_catalyst": False,
            "news_sentiment": 0.0,
            "sentiment_score": 0.0,
            "liquidity_score": round(
                _clamp(min(100.0, 30.0 + volume_ratio * 20.0 + _as_float(universe_row.get("atr_percent")) * 3.0), 0.0, 100.0),
                2,
            ),
        }

    def _enrich_candidates(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        max_news = max(0, int(getattr(self.scanner_cfg, "max_news_candidates", 25) or 0))
        require_catalyst = bool(
            getattr(self.scanner_cfg, "require_catalyst_for_extreme_gap_names", True)
        )
        extreme_gap_pct = float(getattr(self.scanner_cfg, "extreme_gap_pct", 12.0))

        ranked = sorted(candidates, key=lambda row: row.get("score", 0.0), reverse=True)
        for row in ranked[:max_news]:
            ticker = str(row.get("ticker") or "").upper()
            if not ticker:
                continue
            try:
                news = self.news_aggregator.collect(ticker)
            except Exception as exc:
                logger.debug("Momentum scanner news failed for %s: %s", ticker, exc)
                news = {}

            sentiment = _as_float(news.get("sentiment_score"), 0.0)
            catalyst_score = _as_float(news.get("catalyst_score"), 0.0)
            has_catalyst = bool(news.get("has_catalyst"))
            offering_flag = bool(news.get("has_offering") or news.get("has_dilution"))
            row["news_sentiment"] = round(sentiment, 3)
            row["sentiment_score"] = round(sentiment, 3)
            row["has_catalyst"] = has_catalyst
            row["catalyst_score"] = round(catalyst_score, 2)
            row["catalyst_tags"] = list(news.get("catalyst_tags", []) or [])
            row["catalyst_note"] = str(news.get("catalyst_note", "") or "")
            row["catalyst_summary"] = row["catalyst_note"] or "; ".join(row["catalyst_tags"][:3])
            if offering_flag:
                row["rejection_reasons"].append("negative_catalyst")
            if require_catalyst and abs(_as_float(row.get("session_gain_pct"), 0.0)) >= extreme_gap_pct and not has_catalyst:
                row["rejection_reasons"].append("extreme_gap_without_catalyst")
            row["score"] = round(
                _clamp(
                    _as_float(row.get("score"), 0.0)
                    + sentiment * 8.0
                    + min(10.0, catalyst_score * 4.0),
                    0.0,
                    100.0,
                ),
                2,
            )
            row["qualified_for_entry"] = len(row["rejection_reasons"]) == 0
            if has_catalyst:
                row["scan_type"] = "earnings_momentum"
        return ranked

    def scan_once(self, now_et: Optional[datetime] = None) -> Dict[str, Any]:
        now_ts = (now_et or self._now_et()).astimezone(ET)
        now_ct = now_ts.astimezone(CT)
        session = self._resolve_session(now_ts)
        sleep_seconds = self._sleep_seconds_for_session(session)
        payload: Dict[str, Any] = {
            "generated_at_et": now_ts.isoformat(),
            "generated_at_ct": now_ct.isoformat(),
            "artifact_path": str(self.output_path),
            "session": session,
            "scanner_mode": "daemon",
            "scan_count": 0,
            "candidate_count": 0,
            "raw_candidate_count": 0,
            "universe_size_considered": 0,
            "rejection_summary": {},
            "next_scan_due_ct": (
                now_ct + pd.Timedelta(seconds=max(5, sleep_seconds))
            ).isoformat(),
            "symbols": [],
        }
        if not bool(getattr(self.scanner_cfg, "enabled", True)):
            payload["status"] = "disabled"
            _atomic_write_json(self.output_path, payload)
            return payload
        if session in {"outside", "before_start_window"}:
            payload["status"] = (
                "before_start_window" if session == "before_start_window" else "outside_window"
            )
            _atomic_write_json(self.output_path, payload)
            return payload

        universe_rows = self._load_universe_rows()
        selected_rows = self._select_symbols(universe_rows)
        payload["scan_count"] = len(selected_rows)
        payload["universe_size_considered"] = len(selected_rows)
        if not selected_rows:
            payload["status"] = "empty_universe"
            _atomic_write_json(self.output_path, payload)
            return payload

        symbols = [str(row.get("ticker") or "").upper() for row in selected_rows]
        minutes_back = int(
            getattr(
                self.scanner_cfg,
                "scan_minutes_back_premarket" if session == "premarket" else "scan_minutes_back_intraday",
                390,
            )
            or 390
        )
        bars_map = self._fetch_bars_map(symbols, minutes_back)
        candidates: List[Dict[str, Any]] = []
        for row in selected_rows:
            ticker = str(row.get("ticker") or "").upper()
            bars = bars_map.get(ticker)
            if bars is None:
                continue
            candidate = self._score_symbol(row, bars, session=session, now_et=now_ts)
            if candidate is not None:
                candidates.append(candidate)

        enriched = self._enrich_candidates(candidates)
        max_candidates = max(1, int(getattr(self.scanner_cfg, "max_candidates", 50) or 50))
        symbols_out = [
            row
            for row in sorted(
                enriched,
                key=lambda item: (
                    item.get("priority_score", item.get("score", 0.0)),
                    item.get("score", 0.0),
                ),
                reverse=True,
            )
            if row.get("qualified_for_entry")
        ][:max_candidates]

        payload.update(
            {
                "status": "ok",
                "candidate_count": len(symbols_out),
                "raw_candidate_count": len(candidates),
                "rejection_summary": self._build_rejection_summary(enriched),
                "symbols": symbols_out,
            }
        )
        _atomic_write_json(self.output_path, payload)
        return payload

    def run_forever(self) -> None:
        while True:
            now_ts = self._now_et()
            session = self._resolve_session(now_ts)
            try:
                self.scan_once(now_et=now_ts)
            except Exception as exc:  # pragma: no cover - safety loop
                logger.exception("Momentum scanner cycle failed: %s", exc)
            sleep_for = self._sleep_seconds_for_session(session)
            self._sleep_fn(max(5, sleep_for))


def run_momentum_scanner_once(now_et: Optional[datetime] = None) -> Dict[str, Any]:
    return MomentumScanner().scan_once(now_et=now_et)
