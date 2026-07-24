"""
Market-Role Context API endpoints.

GET /api/context/market-role                   — all four coins' live role context
GET /api/context/market-role/summary           — compact summary for dashboard widget
GET /api/context/market-role/ranking-breakdown — DAY + SCALP score breakdown for all four coins
GET /api/context/market-role/{symbol}          — one symbol's live role context
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
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
_DB_PATH = os.getenv("TRADING_DB_PATH", "/home/mystic/mystic/mystic_trading.db")


def _freshness_sec(ts_raw: str) -> float | None:
    if not ts_raw:
        return None
    try:
        tnorm = str(ts_raw).replace("Z", "+00:00")
        t = datetime.fromisoformat(tnorm)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return round(max(0.0, (datetime.now(timezone.utc) - t.astimezone(timezone.utc)).total_seconds()), 1)
    except Exception:
        return None


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

        ts_raw = payload.get("ts_utc", "")
        fresh_sec = _freshness_sec(ts_raw)

        # Parse role_intel_json
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
            "rs_btc_24h": _f("ctx_rs_btc"),
            "rs_eth_24h": _f("ctx_rs_eth"),
            "rs_short_1h": role_intel.get("rs_short_1h"),
            "rs_medium_4h": role_intel.get("rs_medium_4h"),
            "btc_correlation": role_intel.get("btc_correlation"),
            "btc_beta": role_intel.get("btc_beta"),
            "momentum_score": role_intel.get("momentum_score"),
            "volatility_score": role_intel.get("volatility_score"),
            "volume_accel": role_intel.get("volume_accel"),
            "catalyst_score": role_intel.get("catalyst_score"),
            "catalyst_source": role_intel.get("catalyst_source", "unavailable"),
            "catalyst_category": role_intel.get("catalyst_category"),
            "market_regime": payload.get("ctx_market_regime", "unknown"),
            "risk_regime": role_intel.get("risk_regime", "neutral"),
            "fear_greed": _f("ctx_sentiment_fear_greed"),
            "btc_dominance_proxy": _f("ctx_btc_dominance_proxy"),
            "spread_pct": _f("ctx_spread_pct"),
            "depth_imbalance": _f("ctx_depth_imbalance"),
            "volume_24h_usd": _f("ctx_volume_24h_usd"),
            "liquidity_tier": payload.get("ctx_liquidity_tier"),
            "live_context_adjustment": role_intel.get("live_context_adjustment", _f("ctx_role_ranking_delta", 0.0)),
            "role_ranking_delta": _f("ctx_role_ranking_delta", 0.0),
            "ctx_multiplier": _f("ctx_multiplier", 1.0),
            "source_status": role_intel.get("source_status", "live") if role_intel else "partial",
            "freshness_seconds": fresh_sec,
            "context_ts_utc": ts_raw,
            "day_consumed": fresh_sec is not None and fresh_sec < 300,
            "scalp_consumed": fresh_sec is not None and fresh_sec < 120,
        })

    except Exception as exc:
        logger.debug("market_context_endpoint %s error: %s", symbol, exc)
        result["error"] = str(exc)[:120]

    return result


def _build_day_breakdown(sym: str, ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Build DAY ranking breakdown from decision_data injected fields.
    Falls back to live context if no in-flight candidate is available.
    """
    # Fetch learned stats
    day_stats = {"sample_count": 0, "learned_adjustment": 0.0, "confidence": 0.0, "confidence_status": "insufficient_data"}
    try:
        from backend.services.market_role_outcome_learner import get_learning_stats as _gls
        s = _gls(_DB_PATH, sym, "day")
        day_stats = {
            "sample_count": s.sample_count,
            "learned_adjustment": s.learned_adjustment,
            "confidence": s.confidence,
            "confidence_status": s.confidence_status,
        }
    except Exception:
        pass

    live_adj = float(ctx.get("live_context_adjustment") or ctx.get("role_ranking_delta") or 0.0)
    learned_adj = day_stats["learned_adjustment"] if day_stats["sample_count"] >= 10 else 0.0

    # Base rank is not known without a live candidate; report "N/A" and components only
    return {
        "symbol": sym,
        "strategy": "day",
        "base_rank_score": "N/A (no active candidate)",
        "live_context_adjustment": round(live_adj, 6),
        "learned_adjustment": round(learned_adj, 6),
        "final_rank_score": "base + live_adj + learned_adj (resolved at trade time)",
        "learning_sample_count": day_stats["sample_count"],
        "learning_confidence": round(day_stats["confidence"], 4),
        "learning_confidence_status": day_stats["confidence_status"],
        "note": "final DAY rank_score = BuyCandidate.rank_score() which fuses base, live, and learned at selection time",
    }


def _build_scalp_breakdown(sym: str) -> dict[str, Any]:
    """
    Build SCALP ranking breakdown using cached role context + learned stats.
    """
    scalp_stats = {"sample_count": 0, "learned_adjustment": 0.0, "confidence": 0.0, "confidence_status": "insufficient_data"}
    try:
        from backend.services.market_role_outcome_learner import get_learning_stats as _gls
        s = _gls(_DB_PATH, sym, "scalp")
        scalp_stats = {
            "sample_count": s.sample_count,
            "learned_adjustment": s.learned_adjustment,
            "confidence": s.confidence,
            "confidence_status": s.confidence_status,
        }
    except Exception:
        pass

    live_adj = 0.0
    raw_delta = 0.0
    try:
        from backend.services.market_role_intelligence import get_cached_role_context as _gcrc
        _rctx = _gcrc(sym)
        if _rctx is not None:
            raw_delta = _rctx.live_ranking_delta()
            live_adj = round(max(-0.04, min(0.04, raw_delta * (0.04 / 0.06))), 5)
    except Exception:
        pass

    learned_adj = scalp_stats["learned_adjustment"] if scalp_stats["sample_count"] >= 10 else 0.0

    return {
        "symbol": sym,
        "strategy": "scalp",
        "base_candidate_score": "N/A (resolved per setup signal at execution time)",
        "live_context_adjustment": round(live_adj, 5),
        "learned_adjustment": round(learned_adj, 5),
        "final_candidate_score": "base + live_adj + learned_adj (resolved per tick in rank_setup_signal)",
        "learning_sample_count": scalp_stats["sample_count"],
        "learning_confidence": round(scalp_stats["confidence"], 4),
        "learning_confidence_status": scalp_stats["confidence_status"],
        "raw_live_delta_before_scale": round(raw_delta, 6),
        "note": "SCALP adjustments are capped: live ±0.04, learned ±0.02, combined ±0.06",
    }


@router.get("/api/context/market-role")
async def get_all_market_role_contexts() -> dict[str, Any]:
    """Live structured market-role context for all four core symbols."""
    redis_client = None
    try:
        redis_client = await get_shared_redis_async()
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    for sym in _WATCHED_SYMBOLS:
        if redis_client is not None:
            ctx = await _read_role_context_from_redis(sym, redis_client)
        else:
            cached = get_cached_role_context(sym)
            if cached:
                ctx = cached.to_dict()
                ctx["day_consumed"] = True
                ctx["scalp_consumed"] = True
            else:
                ctx = {"symbol": sym, "market_role": MARKET_ROLES.get(sym, "unknown"), "source_status": "unavailable"}
        results.append(ctx)

    btc_ctx = next((r for r in results if r.get("symbol") == "BTCUSDT"), {})
    global_regime = btc_ctx.get("market_regime", "unknown")

    return {
        "ok": True,
        "global_market_regime": global_regime,
        "symbols": {r["symbol"]: r for r in results},
        "generated_at": time.time(),
        "watched_symbols": _WATCHED_SYMBOLS,
    }


@router.get("/api/context/market-role/summary")
async def get_market_role_summary() -> dict[str, Any]:
    """Compact summary for the dashboard widget."""
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
            "rs_short_1h": _fmt(ctx.get("rs_short_1h")),
            "momentum": _fmt(ctx.get("momentum_score")),
            "volatility": _fmt(ctx.get("volatility_score")),
            "btc_corr": _fmt(ctx.get("btc_correlation")),
            "btc_beta": _fmt(ctx.get("btc_beta")),
            "catalyst": _fmt(ctx.get("catalyst_score")),
            "catalyst_src": ctx.get("catalyst_source", "unavailable"),
            "rank_delta": _fmt(ctx.get("role_ranking_delta"), 4),
            "live_context_adj": _fmt(ctx.get("live_context_adjustment"), 4),
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


@router.get("/api/context/market-role/ranking-breakdown")
async def get_ranking_breakdown() -> dict[str, Any]:
    """
    DAY and SCALP ranking score breakdown for all four coins.

    Returns per-symbol:
      - base_rank_score / base_candidate_score
      - live_context_adjustment  (data-driven, ±0.06 DAY / ±0.04 SCALP)
      - learned_adjustment       (±0.02 from outcome history; 0 until MIN_SAMPLES=10)
      - final_rank_score / final_candidate_score
      - learning_sample_count
      - learning_confidence
      - learning_confidence_status  ("insufficient_data" / "low_confidence" / "confident")

    Note: base scores are resolved per-candidate at trade selection time.
    This endpoint shows the context adjustment components that are always present.
    """
    redis_client = None
    try:
        redis_client = await get_shared_redis_async()
    except Exception:
        pass

    breakdown: dict[str, Any] = {}
    for sym in _WATCHED_SYMBOLS:
        ctx: dict[str, Any] = {}
        if redis_client is not None:
            ctx = await _read_role_context_from_redis(sym, redis_client)
        else:
            cached = get_cached_role_context(sym)
            if cached:
                ctx = cached.to_dict()

        breakdown[sym] = {
            "day": _build_day_breakdown(sym, ctx),
            "scalp": _build_scalp_breakdown(sym),
            "live_context": {
                "rs_short_1h": ctx.get("rs_short_1h"),
                "rs_medium_4h": ctx.get("rs_medium_4h"),
                "momentum_score": ctx.get("momentum_score"),
                "volatility_score": ctx.get("volatility_score"),
                "volume_accel": ctx.get("volume_accel"),
                "btc_correlation": ctx.get("btc_correlation"),
                "btc_beta": ctx.get("btc_beta"),
                "catalyst_score": ctx.get("catalyst_score"),
                "market_regime": ctx.get("market_regime"),
                "freshness_seconds": ctx.get("freshness_seconds"),
                "source_status": ctx.get("source_status"),
            },
        }

    return {
        "ok": True,
        "symbols": breakdown,
        "note": "base scores resolved at candidate selection; adjustment components shown here",
        "bounds": {
            "day_live_context_adjustment": "±0.06",
            "day_learned_adjustment": "±0.02",
            "day_combined_cap": "±0.08",
            "scalp_live_context_adjustment": "±0.04",
            "scalp_learned_adjustment": "±0.02",
            "scalp_combined_cap": "±0.06",
        },
        "min_samples_for_learning": 10,
        "generated_at": time.time(),
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

    # Append ranking breakdown inline
    ctx["ranking_breakdown"] = {
        "day": _build_day_breakdown(sym, ctx),
        "scalp": _build_scalp_breakdown(sym),
    }

    return {"ok": True, **ctx}


__all__ = ["router"]
