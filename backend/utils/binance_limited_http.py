"""Binance.US REST helpers that consume the shared Redis weight limiter before httpx."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from backend.config.mystic_api_schedule import (
    BINANCE_LIMITER_CONSUME_TIMEOUT_SEC,
)
from backend.utils.binance_weight_limiter import BinanceWeightLimiter

logger = logging.getLogger(__name__)

BINANCEUS_API_BASE = "https://api.binance.us"


async def limited_binance_get(
    endpoint: str,
    *,
    params: dict[str, Any] | None = None,
    timeout_sec: float = 8.0,
    consume_timeout: float | None = None,
) -> httpx.Response | None:
    """
    GET against Binance.US with limiter consume. ``endpoint`` is the path only, e.g.
    ``/api/v3/ticker/price``.
    """
    path = endpoint if endpoint.startswith("/") else f"/{endpoint}"
    url = f"{BINANCEUS_API_BASE}{path}"
    wait = consume_timeout if consume_timeout is not None else BINANCE_LIMITER_CONSUME_TIMEOUT_SEC
    try:
        limiter = await BinanceWeightLimiter.create()
        await limiter.consume(path, timeout=wait)
    except Exception as exc:
        logger.debug("LIMITED_HTTP consume failed path=%s err=%s", path, exc)
        return None
    try:
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            return await client.get(url, params=params or {})
    except Exception as exc:
        logger.debug("LIMITED_HTTP request failed path=%s err=%s", path, exc)
        return None
