#!/usr/bin/env python3
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import httpx

from backend.config.redis_config import get_shared_redis_async
from backend.services.ttl_cache import TTLCache

logger = logging.getLogger(__name__)
_cache = TTLCache(ttl_seconds=300)

MARKET_REGIME_GLOBAL_KEY = "market_regime:global"
# Successful Fear&Greed writes: keep short TTL so values stay fresh
_REGIME_REDIS_TTL_OK = int(os.getenv("MARKET_REGIME_REDIS_TTL", "300"))
# When API or Redis partial-write fails, seed so dashboard / AI always have a readable key
_REGIME_REDIS_TTL_FALLBACK = int(os.getenv("MARKET_REGIME_FALLBACK_TTL", "3600"))


async def _persist_regime_to_redis(
    score: float,
    raw_value: float,
    regime: str,
    *,
    source: str,
    ttl_seconds: int,
) -> bool:
    """Write canonical shape to Redis. Returns True if a value was stored."""
    try:
        redis_client = get_shared_redis_async()
        if redis_client is None:
            logger.warning("market_regime: Redis client unavailable; cannot persist %s", MARKET_REGIME_GLOBAL_KEY)
            return False
        regime_data = {
            "score": float(score),
            "raw_value": float(raw_value),
            "regime": str(regime).lower(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
        }
        await redis_client.setex(
            MARKET_REGIME_GLOBAL_KEY,
            max(60, int(ttl_seconds)),
            json.dumps(regime_data),
        )
        logger.info("Stored market regime (%s): %s (score: %.3f)", source, regime_data["regime"], score)
        return True
    except Exception as e:
        logger.warning("Failed to persist market regime to Redis (%s): %s", source, e)
        return False


async def seed_market_regime_for_dashboard() -> None:
    """
    Ensure ``market_regime:global`` exists before traffic hits the dashboard.
    Call once from app lifespan (local + production) so regime badge and AI paths always have a readable key.
    """
    redis_client = get_shared_redis_async()
    if redis_client is None:
        logger.warning("REDIS_SEED: shared Redis client is None; skipping market_regime seed")
        return

    try:
        await regime_score()
    except Exception as e:
        logger.warning("REDIS_SEED: regime_score raised during seed: %s", e)

    try:
        raw = await redis_client.get(MARKET_REGIME_GLOBAL_KEY)
    except Exception as e:
        logger.warning("REDIS_SEED: could not read %s: %s", MARKET_REGIME_GLOBAL_KEY, e)
        raw = None

    if raw:
        logger.info("REDIS_SEED: %s already populated after regime_score()", MARKET_REGIME_GLOBAL_KEY)
        return

    ok = await _persist_regime_to_redis(
        0.0,
        50.0,
        "sideways",
        source="mystic_startup_default",
        ttl_seconds=_REGIME_REDIS_TTL_FALLBACK,
    )
    if ok:
        logger.info("REDIS_SEED: wrote default %s for dashboard readability", MARKET_REGIME_GLOBAL_KEY)


async def regime_score() -> float:
    """
    Returns market regime score in [-1,1] and stores to Redis.
    Uses Fear & Greed Index from alternative.me (free, no key required)
    """
    key = "regime"
    cached = _cache.get(key)
    if cached is not None:
        return float(cached)

    # Try configured API first
    url = os.getenv("REGIME_API_URL", "")
    api_key = os.getenv("REGIME_API_KEY", "")

    # Default to free Fear & Greed API when no override URL is configured.
    if not url:
        url = "https://api.alternative.me/fng/"
        api_key = None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {"token": api_key} if api_key else None
            r = await client.get(url, params=params)
            r.raise_for_status()
            data = r.json()

            # Alternative.me format
            raw = float(data["data"][0].get("value", 50.0)) if "data" in data and len(data["data"]) > 0 else float(data.get("score") or data.get("value") or 50.0)

            raw = max(0.0, min(100.0, raw))
            s = (raw - 50.0) / 50.0  # Map to [-1, 1]
            _cache.set(key, s)

            regime = "bull" if s > 0.2 else ("bear" if s < -0.2 else "sideways")
            await _persist_regime_to_redis(
                s,
                raw,
                regime,
                source="fear_greed_api",
                ttl_seconds=_REGIME_REDIS_TTL_OK,
            )

            return s

    except Exception as e:
        logger.exception("Failed to fetch market regime: %s", e)
        _cache.set(key, 0.0)
        # Always leave Redis readable for dashboard / portfolio_engine sync
        await _persist_regime_to_redis(
            0.0,
            50.0,
            "sideways",
            source="api_error_fallback",
            ttl_seconds=_REGIME_REDIS_TTL_FALLBACK,
        )
        return 0.0


async def get_regime_snapshot_for_signal(redis_client: Any | None) -> dict[str, Any]:
    """
    Regime label + score [-1, 1] for ML `ai_signal:*` payloads and engine explainability.

    Prefers Redis ``market_regime:global``; falls back to ``regime_score()`` (which
    refreshes Redis when the API succeeds).
    """
    if redis_client is not None:
        try:
            raw = await redis_client.get(MARKET_REGIME_GLOBAL_KEY)
            if raw:
                data = json.loads(raw if isinstance(raw, str) else raw.decode())
                return {
                    "regime_label": str(data.get("regime", "sideways")).lower(),
                    "regime_score": float(data.get("score", 0.0)),
                }
        except Exception as e:
            logger.debug("get_regime_snapshot_for_signal: redis read failed: %s", e)

    s = float(await regime_score())
    lab = "bull" if s > 0.2 else ("bear" if s < -0.2 else "sideways")
    return {"regime_label": lab, "regime_score": s}
