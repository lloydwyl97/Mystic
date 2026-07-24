"""
Market-Role Context API endpoints.

GET /api/context/market-role          — all four coins' live role context
GET /api/context/market-role/{symbol} — one symbol's live role context
GET /api/context/market-role/summary  — compact summary for dashboard widget
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import redis.asyncio as redis_async
from fastapi import APIRouter, HTTPException

from backend.config.redis_config import get_shared_redis_async
from backend.config.trading_universe import get_trading_symbols
from backend.services.ai_decision_contract import REDIS_KEY_AI_CONTEXT
from backend.services.market_role_intelligence import (
    MARKET_ROLES,
    get_cached_role_context,
)

router = APIRouter()
logger = logging.getLogger(__name__)

_WATCHED_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]


async def _read_role_context_from_redis(symbol: str, redis_client: redis_async.Redis) -> dict[str, Any]:
    """
    Read the ai_context:{symbol} hash and extract role-intelligence fields.
    Returns a structured dict with clear source/freshness metadata.
    """
    sym = symbol.upper()
    result: dict[str, Any] = {
        "symbol": sym,
        "market_role": MARKET_ROLES.get(sym, "unknown"),
        "source_status": "unavailable",
        "freshness_seconds": None,
        "error": None,
    }

    try:
        raw = await redis_client.hgetall(REDIS_KEY_AI_CONTEXT.format(symbol=sym))
        if not raw:
            result["error"] = "no_context_in_redis"
            return result

        payload: dict[str, str] = {
            (k.decode() if isinstance(k, bytes) else k): (v.decode() if isinstance(v, bytes) else v)
            for k, v in raw.items()
        }

        # Compute freshness
        ts_raw = payload.get("ts_utc", "")
        freshness_sec: float | None = None
        if ts_raw:
            try:
                from datetime import datetime, timezone

                tnorm = str(ts_raw).replace("Z", "+00:00")
                t = datetime.fromisoformat(tnorm)
                if t.tzinfo is None:
                    t = t.replace(tzinfo=timezone.utc)
                freshness_sec = round(max(0.0, (datetime.now(timezone.utc) - t.astimezone(timezone.utc)).total_seconds()), 1)
            except Exception:
                pass

        # Parse role_intel_json if present
        role_intel: dict[str, Any] = {}
        raw_role = payload.get("ctx_role_intel_json", "{}")
        if raw_role and raw_role != "{}":
            try:
                role_intel = json.loads(raw_role)
            except Exception:
                role_intel = {}

        def _f(key: str, default: Any = None) -> Any:
            v = payload.get(key)
            if v is None:
                return default
            try:
                return float(v)
            except (TypeError, ValueError):
                return v

        result.update({
            "market_role": role_intel.get("market_role") or MARKET_ROLES.get(sym, "unknown"),
            "role_code": role_intel.get("role_code"),
            # Relative strength
            "rs_btc_24h": _f("ctx_rs_btc"),
            "rs_eth_24h": _f("ctx_rs_eth"),
            "rs_short_1h": role_intel.get("rs_short_1h"),
            "rs_medium_4h": role_intel.get("rs_medium_4h"),
            # Cross-asset
            "btc_correlation": role_intel.get("btc_correlation"),
            "btc_beta": role_intel.get("btc_beta"),
            # Composite scores
            "momentum_score": role_intel.get("momentum_score"),
            "volatility_score": role_intel.get("volatility_score"),
            "volume_accel": role_intel.get("volume_accel"),
            # Catalyst
            "catalyst_score": role_intel.get("catalyst_score"),
            "catalyst_source": role_intel.get("catalyst_source", "unavailable"),
            "catalyst_category": role_intel.get("catalyst_category"),
            # Market conditions
            "market_regime": payload.get("ctx_market_regime", "unknown"),
            "risk_regime": role_intel.get("risk_regime", "neutral"),
            "fear_greed": _f("ctx_sentiment_fear_greed"),
            "btc_dominance_proxy": _f("ctx_btc_dominance_proxy"),
            # Microstructure
            "spread_pct": _f("ctx_spread_pct"),
            "depth_imbalance": _f("ctx_depth_imbalance"),
            "volume_24h_usd": _f("ctx_volume_24h_usd"),
            "liquidity_tier": payload.get("ctx_liquidity_tier"),
            # Ranking
            "role_ranking_delta": _f("ctx_role_ranking_delta", 0.0),
            "ctx_multiplier": _f("ctx_multiplier", 1.0),
            # Source / freshness
            "source_status": role_intel.get("source_status", "live") if role_intel else "partial",
            "freshness_seconds": freshness_sec,
            "context_ts_utc": ts_raw,
            # Consumer confirmation
            "day_consumed": freshness_sec is not None and freshness_sec < 300,
            "scalp_consumed": freshness_sec is not None and freshness_sec < 120,
        })

    except Exception as exc:
        logger.debug("market_context_endpoint %s error: %s", symbol, exc)
        result["error"] = str(exc)[:120]

    return result


@router.get("/api/context/market-role")
async def get_all_market_role_contexts() -> dict[str, Any]:
    """Live structured market-role context for all four core symbols."""
    redis_client = None
    try:
        redis_client = await get_shared_redis_async()
    except Exception:
        pass

    symbols = _WATCHED_SYMBOLS
    results: list[dict[str, Any]] = []

    for sym in symbols:
        if redis_client is not None:
            ctx = await _read_role_context_from_redis(sym, redis_client)
        else:
            # Fall back to in-process cache
            cached = get_cached_role_context(sym)
            if cached:
                ctx = cached.to_dict()
                ctx["day_consumed"] = True
                ctx["scalp_consumed"] = True
            else:
                ctx = {"symbol": sym, "market_role": MARKET_ROLES.get(sym, "unknown"), "source_status": "unavailable"}
        results.append(ctx)

    # Global regime from BTC context
    btc_ctx = next((r for r in results if r.get("symbol") == "BTCUSDT"), {})
    global_regime = btc_ctx.get("market_regime", "unknown")

    return {
        "ok": True,
        "global_market_regime": global_regime,
        "symbols": {r["symbol"]: r for r in results},
        "generated_at": time.time(),
        "watched_symbols": symbols,
    }


@router.get("/api/context/market-role/summary")
async def get_market_role_summary() -> dict[str, Any]:
    """
    Compact summary for the dashboard widget.
    Returns only the most relevant fields for display.
    """
    redis_client = None
    try:
        redis_client = await get_shared_redis_async()
    except Exception:
        pass

    summary: list[dict[str, Any]] = []
    for sym in _WATCHED_SYMBOLS:
        if redis_client is not None:
            ctx = await _read_role_context_from_redis(sym, redis_client)
        else:
            ctx = {"symbol": sym, "market_role": MARKET_ROLES.get(sym, "unknown"), "source_status": "unavailable"}

        def _fmt(v: Any, decimals: int = 3) -> Any:
            if v is None:
                return None
            try:
                return round(float(v), decimals)
            except (TypeError, ValueError):
                return v

        summary.append({
            "symbol": sym,
            "role": ctx.get("market_role", "unknown"),
            "regime": ctx.get("market_regime", "unknown"),
            "rs_btc": _fmt(ctx.get("rs_btc_24h")),
            "momentum": _fmt(ctx.get("momentum_score")),
            "volatility": _fmt(ctx.get("volatility_score")),
            "btc_corr": _fmt(ctx.get("btc_correlation")),
            "btc_beta": _fmt(ctx.get("btc_beta")),
            "catalyst": _fmt(ctx.get("catalyst_score")),
            "catalyst_src": ctx.get("catalyst_source", "unavailable"),
            "rank_delta": _fmt(ctx.get("role_ranking_delta"), 4),
            "freshness_s": ctx.get("freshness_seconds"),
            "source": ctx.get("source_status", "unavailable"),
        })

    btc_item = next((s for s in summary if s["symbol"] == "BTCUSDT"), {})
    return {
        "ok": True,
        "global_regime": btc_item.get("regime", "unknown"),
        "coins": summary,
        "ts": time.time(),
    }


@router.get("/api/context/market-role/{symbol}")
async def get_single_market_role_context(symbol: str) -> dict[str, Any]:
    """Live role context for one symbol."""
    sym = symbol.upper().replace("/", "").replace("-", "")
    if not sym.endswith("USDT"):
        sym = f"{sym}USDT"
    if sym not in _WATCHED_SYMBOLS and sym not in get_trading_symbols():
        raise HTTPException(status_code=404, detail=f"Symbol {sym} not in trading universe")

    redis_client = None
    try:
        redis_client = await get_shared_redis_async()
    except Exception:
        pass

    if redis_client is not None:
        ctx = await _read_role_context_from_redis(sym, redis_client)
    else:
        cached = get_cached_role_context(sym)
        ctx = cached.to_dict() if cached else {"symbol": sym, "source_status": "unavailable"}

    return {"ok": True, **ctx}


__all__ = ["router"]
