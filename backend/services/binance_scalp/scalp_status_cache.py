"""Cached read-only scalp status snapshots — fast /api/scalp/status responses."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_build_lock = threading.Lock()
_mem_cache: dict[str, Any] = {"payload": None, "built_at": 0.0, "warm_rounds": -1}


def status_cache_ttl_sec() -> float:
    raw = os.getenv("SCALP_STATUS_CACHE_TTL_SEC", "20")
    try:
        return max(5.0, float(raw))
    except (TypeError, ValueError):
        return 20.0


def _redis_cache_key(warm_rounds: int) -> str:
    from backend.services.binance_scalp.config import get_scalp_config

    cfg = get_scalp_config()
    prefix = (cfg.redis_key_prefix or "scalp").strip().rstrip(":")
    return f"{prefix}:status:snapshot:w{int(warm_rounds)}"


def _load_redis_cache(warm_rounds: int) -> dict[str, Any] | None:
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        raw = client.get(_redis_cache_key(warm_rounds))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("scalp status redis cache read skipped: %s", exc)
        return None


def _save_redis_cache(warm_rounds: int, payload: dict[str, Any], ttl_sec: float) -> None:
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        client.setex(_redis_cache_key(warm_rounds), max(5, int(ttl_sec)), json.dumps(payload, separators=(",", ":")))
    except Exception as exc:
        logger.debug("scalp status redis cache write skipped: %s", exc)


def get_cached_scalp_status(*, warm_rounds: int = 0, force_refresh: bool = False) -> dict[str, Any]:
    from backend.services.binance_scalp.status_snapshot import build_scalp_status

    ttl = status_cache_ttl_sec()
    now = time.time()
    wr = int(warm_rounds)

    if not force_refresh:
        cached = _load_redis_cache(wr)
        if cached and (now - float(cached.get("_cached_at") or 0)) < ttl:
            payload = dict(cached)
            payload["cache_hit"] = True
            payload["cache_backend"] = "redis"
            payload["cache_age_sec"] = round(now - float(cached.get("_cached_at") or now), 2)
            return payload

        with _lock:
            if (
                _mem_cache.get("payload") is not None
                and int(_mem_cache.get("warm_rounds") or -1) == wr
                and (now - float(_mem_cache.get("built_at") or 0)) < ttl
            ):
                payload = dict(_mem_cache["payload"])
                payload["cache_hit"] = True
                payload["cache_backend"] = "memory"
                payload["cache_age_sec"] = round(now - float(_mem_cache["built_at"]), 2)
                return payload

    def _stale_fallback(exc: Exception) -> dict[str, Any] | None:
        logger.warning("scalp status build failed (warm=%s): %s", wr, exc)
        with _lock:
            stale_src = _mem_cache.get("payload")
        if stale_src:
            stale = dict(stale_src)
            stale["cache_hit"] = True
            stale["cache_stale"] = True
            stale["cache_backend"] = "memory_stale"
            stale["status_error"] = str(exc)[:240]
            stale.pop("_cached_at", None)
            return stale
        redis_stale = _load_redis_cache(wr)
        if redis_stale:
            stale = dict(redis_stale)
            stale["cache_hit"] = True
            stale["cache_stale"] = True
            stale["cache_backend"] = "redis_stale"
            stale["status_error"] = str(exc)[:240]
            stale.pop("_cached_at", None)
            return stale
        return None

    with _build_lock:
        # Re-check cache after waiting — another thread may have refreshed.
        now = time.time()
        if not force_refresh:
            with _lock:
                if (
                    _mem_cache.get("payload") is not None
                    and int(_mem_cache.get("warm_rounds") or -1) == wr
                    and (now - float(_mem_cache.get("built_at") or 0)) < ttl
                ):
                    payload = dict(_mem_cache["payload"])
                    payload["cache_hit"] = True
                    payload["cache_backend"] = "memory"
                    payload["cache_age_sec"] = round(now - float(_mem_cache["built_at"]), 2)
                    return payload

        try:
            payload = build_scalp_status(warm_rounds=wr)
        except Exception as exc:
            fallback = _stale_fallback(exc)
            if fallback is not None:
                return fallback
            raise

    payload["cache_hit"] = False
    payload["cache_age_sec"] = 0.0
    payload["cache_ttl_sec"] = ttl
    payload["cache_backend"] = "none"
    payload["_cached_at"] = now

    store = dict(payload)
    _save_redis_cache(wr, store, ttl)
    with _lock:
        _mem_cache["payload"] = store
        _mem_cache["built_at"] = now
        _mem_cache["warm_rounds"] = wr
    out = dict(payload)
    out.pop("_cached_at", None)
    return out


def invalidate_scalp_status_cache() -> None:
    with _lock:
        _mem_cache["payload"] = None
        _mem_cache["built_at"] = 0.0
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        for wr in (0, 6, 12):
            client.delete(_redis_cache_key(wr))
    except Exception:
        pass


__all__ = ["get_cached_scalp_status", "invalidate_scalp_status_cache", "status_cache_ttl_sec"]
