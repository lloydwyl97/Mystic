"""Validated bid/ask book path. Does not change live DAY sizing.

Live liquidity/sizing still reads the existing fallback path.
This module writes `market_book:{SYMBOL}` and logs shadow size vs current size.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from backend.config.execution_cost_model import (
    RECORDED_FULL_SPREAD_PCT,
    shadow_sizing_enabled,
)
from backend.services.day_liquidity_gate import (
    DEFAULT_TYPICAL_SPREAD_BPS,
    FALLBACK_TYPICAL_SPREAD_BPS,
    _score_to_size_factor,
    _spread_credit,
    compute_liquidity_quality,
)

logger = logging.getLogger(__name__)

BOOK_KEY_PREFIX = "market_book:"
BOOK_TTL_SEC = 120


def _api(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def book_redis_key(symbol: str) -> str:
    return f"{BOOK_KEY_PREFIX}{_api(symbol)}"


def book_payload(
    *,
    bid: float,
    ask: float,
    source: str,
    timestamp: float | None = None,
) -> dict[str, Any]:
    ts = float(timestamp or time.time())
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    spread_bps = ((ask - bid) / mid * 10000.0) if mid > 0 else 0.0
    age = max(0.0, time.time() - ts)
    return {
        "bid": bid,
        "ask": ask,
        "midpoint": mid,
        "spread_bps": round(spread_bps, 6),
        "timestamp": ts,
        "source": source,
        "freshness_sec": round(age, 3),
        "fresh": age <= 30.0,
    }


def write_market_book(redis_client: Any, symbol: str, payload: dict[str, Any]) -> None:
    """Write the isolated book hash. Never overwrite market:{base} strings."""
    if redis_client is None:
        return
    key = book_redis_key(symbol)
    mapping = {k: str(v) for k, v in payload.items()}
    try:
        redis_client.hset(key, mapping=mapping)
        redis_client.expire(key, BOOK_TTL_SEC)
    except TypeError:
        pipe = redis_client.pipeline()
        for k, v in mapping.items():
            pipe.hset(key, k, v)
        pipe.expire(key, BOOK_TTL_SEC)
        pipe.execute()


async def write_market_book_async(redis_client: Any, symbol: str, payload: dict[str, Any]) -> None:
    if redis_client is None:
        return
    key = book_redis_key(symbol)
    mapping = {k: str(v) for k, v in payload.items()}
    try:
        await redis_client.hset(key, mapping=mapping)
        await redis_client.expire(key, BOOK_TTL_SEC)
    except TypeError:
        pipe = redis_client.pipeline()
        for k, v in mapping.items():
            pipe.hset(key, k, v)
        await pipe.expire(key, BOOK_TTL_SEC)
        await pipe.execute()


def read_market_book(redis_client: Any, symbol: str) -> dict[str, Any] | None:
    if redis_client is None:
        return None
    raw = redis_client.hgetall(book_redis_key(symbol))
    if not raw:
        return None
    decoded = {(k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
    try:
        bid = float(decoded.get("bid") or 0)
        ask = float(decoded.get("ask") or 0)
        ts = float(decoded.get("timestamp") or 0)
    except (TypeError, ValueError):
        return None
    if bid <= 0 or ask <= 0:
        return None
    return book_payload(bid=bid, ask=ask, source=str(decoded.get("source") or "market_book"), timestamp=ts)


def tape_median_spread_bps(symbol: str) -> float:
    return float(RECORDED_FULL_SPREAD_PCT.get(_api(symbol), 0.0002)) * 10000.0


def _typical_bps(symbol: str) -> float:
    slash = symbol if "/" in symbol else f"{_api(symbol)[:-4]}/USDT"
    return float(DEFAULT_TYPICAL_SPREAD_BPS.get(slash, FALLBACK_TYPICAL_SPREAD_BPS))


def shadow_liquidity_compare(
    *,
    symbol: str,
    current_decision_data: dict[str, Any],
    real_spread_bps: float,
    current_notional_usd: float,
) -> dict[str, Any]:
    """Compare fallback liquidity credit/size vs real-spread credit/size. Log only."""
    current = compute_liquidity_quality(current_decision_data, symbol)
    typical = _typical_bps(symbol)
    proposed_credit, proposed_reason = _spread_credit(real_spread_bps, typical)
    # Same depth term as current so the only delta is spread.
    depth_credit = float((current.get("liquidity_components") or {}).get("depth_imbalance", {}).get("credit") or 0.6)
    proposed_score = max(0.0, min(1.0, proposed_credit * 0.70 + depth_credit * 0.30))
    proposed_size_factor = _score_to_size_factor(proposed_score)
    current_factor = float(current.get("liquidity_quality_size_factor") or 1.0)
    current_credit = float((current.get("liquidity_components") or {}).get("spread_vs_typical", {}).get("credit") or 0.0)
    proposed_notional = float(current_notional_usd) * proposed_size_factor / current_factor if current_factor > 0 else float(current_notional_usd)
    delta_usd = proposed_notional - float(current_notional_usd)
    delta_pct = (delta_usd / current_notional_usd * 100.0) if current_notional_usd else 0.0
    return {
        "shadow": True,
        "live_sizing_unchanged": True,
        "symbol": _api(symbol),
        "current_fallback_liquidity_credit": current_credit,
        "proposed_real_spread_liquidity_credit": proposed_credit,
        "proposed_spread_reason": proposed_reason,
        "current_position_size_usd": round(float(current_notional_usd), 4),
        "proposed_position_size_usd": round(proposed_notional, 4),
        "difference_usd": round(delta_usd, 4),
        "difference_pct": round(delta_pct, 4),
        "current_size_factor": current_factor,
        "proposed_size_factor": proposed_size_factor,
        "real_spread_bps": real_spread_bps,
        "current_spread_bps": current.get("liquidity_spread_bps"),
    }


def log_spread_shadow(
    *,
    symbol: str,
    current_decision_data: dict[str, Any],
    current_notional_usd: float,
    redis_client: Any = None,
) -> dict[str, Any] | None:
    if not shadow_sizing_enabled():
        return None
    book = read_market_book(redis_client, symbol) if redis_client is not None else None
    if book and book.get("spread_bps"):
        real_bps = float(book["spread_bps"])
        source = "market_book"
    else:
        real_bps = tape_median_spread_bps(symbol)
        source = "decision_book_tape_median"
    cmp_ = shadow_liquidity_compare(
        symbol=symbol,
        current_decision_data=current_decision_data,
        real_spread_bps=real_bps,
        current_notional_usd=current_notional_usd,
    )
    cmp_["spread_source"] = source
    logger.warning("SPREAD_SHADOW %s", json.dumps(cmp_, separators=(",", ":")))
    return cmp_
