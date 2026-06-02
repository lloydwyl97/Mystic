"""
Shared historical-context rules for any path that can influence live/paper trading.

Aligned with the AI signal generator: minimum **1m ingest depth** (resampled to strategy primary),
optional MTF coverage,
explicit feature-store fallback (off by default), and helpers for ATR from real OHLCV.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from typing import Any


def min_ohlcv_bars_for_signal() -> int:
    try:
        return max(200, int(os.getenv("MIN_OHLCV_BARS_FOR_SIGNAL", "200")))
    except (ValueError, TypeError):
        return 200


def ohlcv_fetch_limit_1m() -> int:
    try:
        cap = int(os.getenv("OHLCV_FETCH_LIMIT", "1000"))
        need = min_ohlcv_bars_for_signal()
        return min(1000, max(need, cap))
    except (ValueError, TypeError):
        return 1000


def min_primary_bars_for_strategy(strategy_id: str) -> int:
    """Minimum **primary** (5m/15m) bars after resample — depth equivalent to MIN_OHLCV_BARS_FOR_SIGNAL on 1m."""
    from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy

    psec = primary_bar_seconds_for_strategy(strategy_id)
    need1m = min_ohlcv_bars_for_signal()
    wall_sec = need1m * 60
    return max(40, int(wall_sec // psec) + 10)


def ohlcv_1m_fetch_limit_for_primary(strategy_id: str) -> int:
    """How many 1m bars to pull so that resampling yields enough primary bars."""
    from backend.config.ai_primary_clock import primary_bar_seconds_for_strategy

    psec = primary_bar_seconds_for_strategy(strategy_id)
    need_p = min_primary_bars_for_strategy(strategy_id)
    # Each primary bar needs psec/60 one-minute rows + margin
    est = int(need_p * (psec // 60) + min_ohlcv_bars_for_signal() // 2 + 120)
    return min(1500, max(ohlcv_fetch_limit_1m(), est))


def feature_store_ohlcv_fallback_enabled() -> bool:
    """SQLite feature_ohlcv is sparse (ingestor writes ~1 row/pass); default off so it cannot fake deep history."""
    return os.getenv("FEATURE_STORE_OHLCV_FALLBACK_ENABLED", "false").lower() in ("1", "true", "yes")


def mtf_history_gate_enabled() -> bool:
    return os.getenv("MTF_HISTORY_GATE_ENABLED", "true").lower() in ("1", "true", "yes")


def mtf_min_bars_per_timeframe() -> int:
    try:
        return max(1, int(os.getenv("MTF_MIN_BARS_PER_TIMEFRAME", "20")))
    except (ValueError, TypeError):
        return 20


def mtf_fetch_limit() -> int:
    try:
        return max(mtf_min_bars_per_timeframe(), int(os.getenv("MTF_FETCH_LIMIT", "50")))
    except (ValueError, TypeError):
        return 50


def mtf_required_ok_count(total_timeframes: int) -> int:
    try:
        n = int(os.getenv("MTF_REQUIRED_OK_COUNT", "0"))
    except (ValueError, TypeError):
        n = 0
    if n <= 0:
        return total_timeframes
    return min(max(1, n), total_timeframes)


def feature_store_rows_to_ohlcv(rows: list[dict[str, Any]]) -> list[list]:
    """feature_store OHLCV dict rows -> [ts_ms, o, h, l, c, v] (same shape as live_market_data.get_ohlcv)."""
    out: list[list] = []
    for r in rows:
        ts_raw = r.get("ts")
        if not ts_raw or not isinstance(ts_raw, str):
            continue
        try:
            s = ts_raw.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ts_ms = int(dt.timestamp() * 1000)
        except (ValueError, TypeError, OSError):
            continue
        out.append(
            [
                ts_ms,
                float(r.get("open", 0)),
                float(r.get("high", 0)),
                float(r.get("low", 0)),
                float(r.get("close", 0)),
                float(r.get("volume", 0)),
            ],
        )
    return out


async def evaluate_day_active_timeframe_coverage(ccxt_symbol: str) -> tuple[bool, list[str], dict[str, Any]]:
    """
    DAY readiness: every ``DAY_ACTIVE_TIMEFRAMES`` plus month-from-daily + 1m depth for FEATURE_MAPPING.
    """
    from backend.config.day_active_timeframes import DAY_ACTIVE_TIMEFRAMES
    from backend.services.day_active_market_bundle import async_fetch_day_active_ohlcv_bundle, validate_day_active_bundle
    from backend.services.live_market_data import live_market_data_service

    meta: dict[str, Any] = {"timeframes": list(DAY_ACTIVE_TIMEFRAMES)}
    if live_market_data_service is None:
        return False, ["no_live_market_data_service"], meta
    bundle = await async_fetch_day_active_ohlcv_bundle(live_market_data_service, ccxt_symbol)
    ok, missing = validate_day_active_bundle(bundle)
    meta["bar_counts"] = {tf: len(bundle.get(tf) or []) if isinstance(bundle.get(tf), list) else 0 for tf in DAY_ACTIVE_TIMEFRAMES}
    meta["month_vec_ok"] = bool(bundle.get("_month_vec"))
    return ok, missing, meta


async def evaluate_multi_timeframe_coverage(ccxt_symbol: str) -> tuple[int, int, dict[str, int]]:
    from backend.services.day_active_market_bundle import async_read_cached_day_active_bundle
    from backend.services.live_market_data import live_market_data_service

    timeframes = ["1m", "5m", "15m", "1h", "4h", "1d"]
    min_each = mtf_min_bars_per_timeframe()

    cached = await async_read_cached_day_active_bundle(ccxt_symbol)
    if isinstance(cached, dict) and cached:
        bar_counts = {tf: len(cached.get(tf) or []) if isinstance(cached.get(tf), list) else 0 for tf in timeframes}
        ok = sum(1 for tf in timeframes if bar_counts.get(tf, 0) >= min_each)
        return ok, len(timeframes), bar_counts

    if live_market_data_service is None:
        return 0, 6, {}

    lim = mtf_fetch_limit()

    async def _count(tf: str) -> tuple[str, int]:
        try:
            o = await live_market_data_service.get_ohlcv(ccxt_symbol, tf, limit=lim)
            n = len(o) if isinstance(o, list) else 0
            return tf, n
        except Exception:
            return tf, 0

    pairs = await asyncio.gather(*[_count(tf) for tf in timeframes])
    bar_counts = dict(pairs)
    ok = sum(1 for tf in timeframes if bar_counts.get(tf, 0) >= min_each)
    return ok, len(timeframes), bar_counts


async def trading_history_context_ok(ccxt_symbol: str) -> tuple[bool, str]:
    """
    True when the **DAY-active** timeframe contract + month-from-daily + 1m indicator depth passes.
    (Mystic live universe is DAY-only; this replaces the thin 1m+6-TF probe.)
    """
    ok, missing, _meta = await evaluate_day_active_timeframe_coverage(ccxt_symbol)
    if ok:
        return True, "ok"
    tail = ";".join(missing[:24]) if missing else "unknown_day_gate_failure"
    return False, tail


def atr_from_ohlcv_wilder(ohlcv: list[list], period: int = 14) -> float:
    """Wilder-style ATR from exchange OHLCV rows [ts,o,h,l,c,v]."""
    if not ohlcv or len(ohlcv) < period + 1:
        return 0.0
    true_ranges: list[float] = []
    for i in range(1, len(ohlcv)):
        high = float(ohlcv[i][2])
        low = float(ohlcv[i][3])
        prev_close = float(ohlcv[i - 1][4])
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return 0.0
    return sum(true_ranges[-period:]) / period
