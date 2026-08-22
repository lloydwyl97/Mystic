"""Fetch & validate DAY multi-timeframe OHLCV bundles (Binance.US, CCXT intervals)."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from typing import Any

import numpy as np

from backend.config.day_active_timeframes import (
    DAY_ACTIVE_TIMEFRAMES,
    DAY_FEATURE_BUILDER_MIN_1M_BARS,
    DAY_MONTH_CONTEXT_MIN_1D_BARS,
    fetch_limit_for_day_tf,
    min_bars_for_day_tf,
)
from backend.config.mystic_api_schedule import DAY_BUNDLE_CACHE_TTL_SEC

logger = logging.getLogger(__name__)

DAY_BUNDLE_CACHE_PREFIX = "day_active_bundle:"
_stagger_applied: set[str] = set()
_fetch_locks: dict[str, asyncio.Lock] = {}
_lock_dict_guard = asyncio.Lock()

# Tiered per-timeframe refresh intervals (seconds).
#
# The bundle used to be refetched wholesale (all 10 TFs) every ~60s, gated by a
# single DAY_BUNDLE_CACHE_TTL_SEC. Slower timeframes cannot possibly change
# faster than their own candle-close cadence (a 1d candle closes once/day, a
# 1w candle once/week) — refetching them on the same cadence as 1m wasted the
# large majority of DAY's own klines weight budget for zero freshness benefit.
# 1m/5m keep their original fast cadence; slower TFs get progressively longer
# tiers, all still far fresher than each TF's own candle-close rate. Configurable
# per-TF via env for operational tuning without a code change.
_TF_REFRESH_TIER_SEC: dict[str, float] = {
    "1m": float(os.getenv("DAY_TF_REFRESH_1M_SEC", str(DAY_BUNDLE_CACHE_TTL_SEC))),
    "5m": float(os.getenv("DAY_TF_REFRESH_5M_SEC", "120")),
    "15m": float(os.getenv("DAY_TF_REFRESH_15M_SEC", "300")),
    "30m": float(os.getenv("DAY_TF_REFRESH_30M_SEC", "600")),
    "1h": float(os.getenv("DAY_TF_REFRESH_1H_SEC", "900")),
    "4h": float(os.getenv("DAY_TF_REFRESH_4H_SEC", "1800")),
    "8h": float(os.getenv("DAY_TF_REFRESH_8H_SEC", "3600")),
    "12h": float(os.getenv("DAY_TF_REFRESH_12H_SEC", "3600")),
    "1d": float(os.getenv("DAY_TF_REFRESH_1D_SEC", "3600")),
    "1w": float(os.getenv("DAY_TF_REFRESH_1W_SEC", "3600")),
}


def _tf_refresh_interval_sec(tf: str) -> float:
    return _TF_REFRESH_TIER_SEC.get(tf, DAY_BUNDLE_CACHE_TTL_SEC)


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _normalize_ccxt_symbol(ccxt_symbol: str) -> str:
    s = str(ccxt_symbol or "").strip().upper()
    if "/" in s:
        return s
    if s.endswith("USDT") and len(s) > 4:
        return f"{s[:-4]}/USDT"
    return s


def _bundle_cache_key(ccxt_symbol: str) -> str:
    return f"{DAY_BUNDLE_CACHE_PREFIX}{_normalize_ccxt_symbol(ccxt_symbol).replace('/', '')}"


async def apply_day_bundle_stagger(role: str) -> None:
    """One-time startup phase offset so bundle consumers do not align on the same second."""
    role_key = (role or "").strip().lower()
    if role_key in _stagger_applied:
        return
    _stagger_applied.add(role_key)
    env_key = f"DAY_BUNDLE_STAGGER_{role_key.upper()}_SEC"
    default = {"signal": "0", "portfolio": "20", "ai_context": "25", "readiness": "40", "learning": "50"}.get(role_key, "0")
    try:
        delay = max(0.0, float(os.getenv(env_key, default)))
    except (TypeError, ValueError):
        delay = 0.0
    if delay > 0:
        logger.info("DAY_BUNDLE_STAGGER role=%s sleep=%.1fs", role_key, delay)
        await asyncio.sleep(delay)


async def _get_fetch_lock(ccxt_symbol: str) -> asyncio.Lock:
    sym = _normalize_ccxt_symbol(ccxt_symbol)
    async with _lock_dict_guard:
        if sym not in _fetch_locks:
            _fetch_locks[sym] = asyncio.Lock()
        return _fetch_locks[sym]


def _bundle_cache_usable(bundle: dict[str, list[list]], fetched_at: float) -> bool:
    if time.time() - fetched_at > DAY_BUNDLE_CACHE_TTL_SEC:
        return False
    rows_1m = bundle.get("1m")
    n1 = len(rows_1m) if isinstance(rows_1m, list) else 0
    if n1 < min(30, DAY_FEATURE_BUILDER_MIN_1M_BARS):
        return False
    for tf in ("5m", "15m", "1h", "1d"):
        rows = bundle.get(tf)
        if not isinstance(rows, list) or len(rows) < 5:
            return False
    return True


async def _read_bundle_cache_full(
    ccxt_symbol: str,
) -> tuple[dict[str, list[list]], dict[str, float], float] | None:
    """Internal: returns (bundle, tf_fetched_at, fetched_at) or None.

    Used by the fetch path to determine which timeframes are still within
    their own refresh tier and can be reused without hitting Binance again.
    """
    try:
        from backend.config.redis_config import get_shared_redis_async

        r = await get_shared_redis_async()
        raw = await r.get(_bundle_cache_key(ccxt_symbol))
        if not raw:
            return None
        payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        fetched_at = float(payload.get("fetched_at") or 0)
        bundle_raw = payload.get("bundle")
        if not isinstance(bundle_raw, dict):
            return None
        bundle: dict[str, list[list]] = {}
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle_raw.get(tf)
            bundle[tf] = list(rows) if isinstance(rows, list) else []
        if "_month_vec" in bundle_raw and isinstance(bundle_raw.get("_month_vec"), list):
            bundle["_month_vec"] = list(bundle_raw["_month_vec"])
        tf_fetched_at_raw = payload.get("tf_fetched_at") or {}
        tf_fetched_at = {tf: float(tf_fetched_at_raw.get(tf, fetched_at)) for tf in DAY_ACTIVE_TIMEFRAMES}
        return bundle, tf_fetched_at, fetched_at
    except Exception as exc:
        logger.debug("DAY_BUNDLE_CACHE_READ_FAIL %s: %s", ccxt_symbol, exc)
        return None


async def _read_bundle_cache(ccxt_symbol: str) -> dict[str, list[list]] | None:
    full = await _read_bundle_cache_full(ccxt_symbol)
    if full is None:
        return None
    bundle, _tf_fetched_at, fetched_at = full
    validate_day_active_bundle(bundle)
    if not _bundle_cache_usable(bundle, fetched_at):
        return None
    logger.debug("DAY_BUNDLE_CACHE_HIT %s age=%.1fs", _normalize_ccxt_symbol(ccxt_symbol), time.time() - fetched_at)
    return bundle


async def _write_bundle_cache(
    ccxt_symbol: str,
    bundle: dict[str, list[list]],
    *,
    tf_fetched_at: dict[str, float] | None = None,
) -> None:
    try:
        from backend.config.redis_config import get_shared_redis_async

        now = time.time()
        serializable = {tf: bundle.get(tf) or [] for tf in DAY_ACTIVE_TIMEFRAMES if isinstance(bundle.get(tf), list)}
        validate_day_active_bundle(bundle)
        if bundle.get("_month_vec"):
            serializable["_month_vec"] = bundle["_month_vec"]
        tf_ts = {tf: float((tf_fetched_at or {}).get(tf, now)) for tf in DAY_ACTIVE_TIMEFRAMES}
        payload = json.dumps(
            {
                "fetched_at": now,
                "ccxt_symbol": _normalize_ccxt_symbol(ccxt_symbol),
                "bundle": serializable,
                "tf_fetched_at": tf_ts,
            }
        )
        r = await get_shared_redis_async()
        # Key TTL must safely outlive the longest per-TF refresh tier (up to
        # 3600s) or slow-TF cache entries would be evicted before their
        # intended lifetime, forcing an unnecessary refetch anyway.
        await r.set(_bundle_cache_key(ccxt_symbol), payload, ex=max(DAY_BUNDLE_CACHE_TTL_SEC * 2, 7200))
    except Exception as exc:
        logger.debug("DAY_BUNDLE_CACHE_WRITE_FAIL %s: %s", ccxt_symbol, exc)


async def async_read_cached_day_active_bundle(ccxt_symbol: str) -> dict[str, list[list]] | None:
    """Read-only cache lookup (no Binance fetch)."""
    return await _read_bundle_cache(ccxt_symbol)


def read_cached_day_active_bundle_sync(ccxt_symbol: str) -> dict[str, list[list]] | None:
    """Sync read-only cache lookup for callers in non-async contexts (e.g. the DAY
    candidate ranking path in portfolio_engine.py). No Binance fetch, no freshness
    validation beyond raw TTL — callers needing strict staleness checks should use
    the async path instead."""
    try:
        from backend.config.redis_config import get_shared_redis_sync

        r = get_shared_redis_sync()
        if not r:
            return None
        raw = r.get(_bundle_cache_key(ccxt_symbol))
        if not raw:
            return None
        payload = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        bundle_raw = payload.get("bundle")
        if not isinstance(bundle_raw, dict):
            return None
        bundle: dict[str, list[list]] = {}
        for tf in DAY_ACTIVE_TIMEFRAMES:
            rows = bundle_raw.get(tf)
            bundle[tf] = list(rows) if isinstance(rows, list) else []
        return bundle
    except Exception as exc:
        logger.debug("DAY_BUNDLE_CACHE_SYNC_READ_FAIL %s: %s", ccxt_symbol, exc)
        return None


def month_context_four_from_daily(ohlcv_1d: list[list]) -> tuple[list[float] | None, str | None]:
    """
    Calendar-month-style context derived **only** from real native **1d** candles.
    Uses the last DAY_MONTH_CONTEXT_MIN_1D_BARS closes (needs full window).
    """
    if not ohlcv_1d or len(ohlcv_1d) < DAY_MONTH_CONTEXT_MIN_1D_BARS:
        return None, (f"month_context_needs_{DAY_MONTH_CONTEXT_MIN_1D_BARS}_daily_bars_have_{len(ohlcv_1d or [])}")
    w = DAY_MONTH_CONTEXT_MIN_1D_BARS
    window = ohlcv_1d[-w:]
    closes = [_safe_float(r[4], 0.0) for r in window if isinstance(r, (list, tuple)) and len(r) > 4]
    highs = [_safe_float(r[2], 0.0) for r in window if isinstance(r, (list, tuple)) and len(r) > 2]
    lows = [_safe_float(r[3], 0.0) for r in window if isinstance(r, (list, tuple)) and len(r) > 3]
    if len(closes) < w or len(highs) < w or len(lows) < w:
        return None, "month_context_malformed_1d_window"
    c0 = closes[-22] if len(closes) >= 22 else closes[0]
    c_now = closes[-1]
    if c0 <= 0 or c_now <= 0:
        return None, "month_context_bad_close"
    log_ret_month = math.log(c_now / c0)

    lc = np.log(np.asarray(closes[-21:], dtype=np.float64) + 1e-12)
    lr = np.diff(lc)
    vol = float(np.std(lr, ddof=0)) if lr.size >= 2 else 0.0
    mean_c = float(np.mean(closes)) + 1e-12
    hl_range = (float(np.max(highs)) - float(np.min(lows))) / mean_c
    peak = np.maximum.accumulate(np.asarray(closes, dtype=np.float64))
    dd = float(np.min((np.asarray(closes) - peak) / (peak + 1e-12)))
    out = [
        _safe_float(log_ret_month),
        _safe_float(vol),
        _safe_float(hl_range),
        _safe_float(dd),
    ]
    for i, v in enumerate(out):
        out[i] = float(max(-6.0, min(6.0, v)))
    return out, None


def validate_day_active_bundle(bundle: dict[str, list[list]]) -> tuple[bool, list[str]]:
    """Return (ok, missing_reasons) — empty list reasons => ok."""
    missing: list[str] = []
    for tf in DAY_ACTIVE_TIMEFRAMES:
        rows = bundle.get(tf)
        need = min_bars_for_day_tf(tf)
        n = len(rows) if isinstance(rows, list) else 0
        if n < need:
            missing.append(f"missing_tf:{tf}_bars_{n}_need_{need}")

    rows_1m = bundle.get("1m")
    n1 = len(rows_1m) if isinstance(rows_1m, list) else 0
    if n1 < DAY_FEATURE_BUILDER_MIN_1M_BARS:
        missing.append(f"missing_indicator_primary_1m_bars_{n1}_need_{DAY_FEATURE_BUILDER_MIN_1M_BARS}")

    d1 = bundle.get("1d") or []
    mn, mn_err = month_context_four_from_daily(d1 if isinstance(d1, list) else [])
    if mn is None and mn_err:
        missing.append(f"month:{mn_err}")
    bundle["_month_vec"] = mn  # type: ignore[index]
    return len(missing) == 0, missing


async def _fetch_day_active_ohlcv_bundle_raw(
    svc: Any,
    ccxt_symbol: str,
    *,
    prior_bundle: dict[str, list[list]] | None = None,
    prior_tf_fetched_at: dict[str, float] | None = None,
) -> tuple[dict[str, list[list]], dict[str, float]]:
    """Pull DAY_ACTIVE_TF from the live service (exchange-native only).

    Timeframes still within their own tiered refresh interval
    (``_TF_REFRESH_TIER_SEC``) are reused from ``prior_bundle`` instead of
    being refetched — slower TFs cannot change faster than their own
    candle-close cadence, so this eliminates redundant Binance calls with no
    freshness loss. Returns (bundle, tf_fetched_at) for the caller to persist.
    """
    sym = _normalize_ccxt_symbol(ccxt_symbol)
    out: dict[str, list[list]] = {}
    tf_fetched_at: dict[str, float] = {}
    now = time.time()
    critical_tfs = {"1m", "5m", "15m"}
    prior_bundle = prior_bundle or {}
    prior_tf_fetched_at = prior_tf_fetched_at or {}
    for tf in DAY_ACTIVE_TIMEFRAMES:
        prior_rows = prior_bundle.get(tf)
        prior_ts = prior_tf_fetched_at.get(tf, 0.0)
        still_fresh = isinstance(prior_rows, list) and len(prior_rows) >= min_bars_for_day_tf(tf) and (now - prior_ts) < _tf_refresh_interval_sec(tf)
        if still_fresh:
            out[tf] = prior_rows
            tf_fetched_at[tf] = prior_ts
            continue
        lim = fetch_limit_for_day_tf(tf)
        rows: list[list] = []
        for attempt in range(2):
            try:
                raw = await svc.get_ohlcv(sym, tf, lim)
                rows = list(raw) if isinstance(raw, list) else []
            except Exception as e:
                logger.debug("DAY_BUNDLE_TF_FAIL %s %s %s attempt=%s", sym, tf, e, attempt + 1)
                rows = []
            if rows or tf not in critical_tfs or attempt == 1:
                break
            await asyncio.sleep(1.5)
        need = min_bars_for_day_tf(tf)
        fetched_n = len(rows) if isinstance(rows, list) else 0
        prior_n = len(prior_rows) if isinstance(prior_rows, list) else 0
        # A failed/empty live fetch is not proof the market has no bars.
        # Keep a complete prior TF so a rate-limit or 451 cannot flap the
        # DAY active contract while Redis/context already hold those TFs.
        if fetched_n < need and prior_n >= need:
            logger.warning(
                "DAY_BUNDLE_TF_EMPTY_REUSE_PRIOR %s %s fetched=%s need=%s prior=%s",
                sym,
                tf,
                fetched_n,
                need,
                prior_n,
            )
            out[tf] = prior_rows
            tf_fetched_at[tf] = prior_ts or now
        else:
            out[tf] = rows
            tf_fetched_at[tf] = now
    return out, tf_fetched_at


async def async_fetch_day_active_ohlcv_bundle(
    svc: Any,
    ccxt_symbol: str,
    *,
    force_refresh: bool = False,
) -> dict[str, list[list]]:
    """
    Fetch DAY bundle with shared Redis cache.

    Signal generator should pass ``force_refresh=True`` (primary writer).
    Portfolio, AI context, readiness, and training reference reads use cache.
    """
    sym = _normalize_ccxt_symbol(ccxt_symbol)
    if not force_refresh:
        cached = await _read_bundle_cache(sym)
        if cached is not None:
            return cached

    lock = await _get_fetch_lock(sym)
    async with lock:
        if not force_refresh:
            cached = await _read_bundle_cache(sym)
            if cached is not None:
                return cached

        # Even on force_refresh (primary writer, called every DAY_AI_SIGNAL_LOOP_SEC),
        # still consult per-TF timestamps so slow timeframes within their own tier
        # are reused rather than unconditionally refetched — force_refresh only
        # means "don't trust the overall fast-tier cache gate", not "refetch everything".
        prior_full = await _read_bundle_cache_full(sym)
        prior_bundle = prior_full[0] if prior_full else None
        prior_tf_fetched_at = prior_full[1] if prior_full else None

        logger.debug("DAY_BUNDLE_FETCH %s force=%s", sym, force_refresh)
        bundle, tf_fetched_at = await _fetch_day_active_ohlcv_bundle_raw(
            svc,
            sym,
            prior_bundle=prior_bundle,
            prior_tf_fetched_at=prior_tf_fetched_at,
        )
        ok, _ = validate_day_active_bundle(dict(bundle))
        if ok or len(bundle.get("1m") or []) >= 30:
            await _write_bundle_cache(sym, bundle, tf_fetched_at=tf_fetched_at)
        return bundle


async def async_fetch_day_active_ohlcv_bundle_asof(svc: Any, ccxt_symbol: str, end_time_ms: int) -> dict[str, list[list]]:
    """
    DAY_ACTIVE_TF OHLC windows ending at ``end_time_ms`` (Binance klines ``endTime``).

    Builds real point-in-time multi-TF snapshots for DAY v5 RF training anchors without slicing a
    too-short concurrent bundle.
    """
    end_ms = max(1, int(end_time_ms))
    sym = _normalize_ccxt_symbol(ccxt_symbol)

    async def _one_tf(tf: str) -> tuple[str, list[list]]:
        try:
            lim = fetch_limit_for_day_tf(tf)
            raw = await svc.get_ohlcv(sym, tf, lim, end_time_ms=end_ms)
            return tf, list(raw) if isinstance(raw, list) else []
        except Exception as e:
            logger.debug("DAY_BUNDLE_ASOF_TF_FAIL %s %s et=%s %s", sym, tf, end_ms, e)
            return tf, []

    pairs = await asyncio.gather(*(_one_tf(tf) for tf in DAY_ACTIVE_TIMEFRAMES))
    return dict(pairs)


__all__ = [
    "apply_day_bundle_stagger",
    "async_fetch_day_active_ohlcv_bundle",
    "async_fetch_day_active_ohlcv_bundle_asof",
    "async_read_cached_day_active_bundle",
    "month_context_four_from_daily",
    "read_cached_day_active_bundle_sync",
    "validate_day_active_bundle",
]
