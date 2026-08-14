"""Cached read-only scalp status snapshots — fast /api/scalp/status responses.

GET path must NEVER call build_scalp_status / REST depth / klines / strategy router.
The paper runner publishes a Redis snapshot each tick; the API only reads it.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

# API-facing snapshot key (warm=0). Runner publishes here each tick.
_API_SNAPSHOT_WARM = 0
_LAST_PUBLISHED_PAYLOAD: dict[str, Any] | None = None

# Minimum TTL must outlive long first-tick kline warm-up (often minutes).
_DEFAULT_SNAPSHOT_TTL_SEC = 300.0
_MIN_SNAPSHOT_TTL_SEC = 180.0


def status_cache_ttl_sec() -> float:
    raw = os.getenv("SCALP_STATUS_CACHE_TTL_SEC", str(int(_DEFAULT_SNAPSHOT_TTL_SEC)))
    try:
        return max(_MIN_SNAPSHOT_TTL_SEC, float(raw))
    except (TypeError, ValueError):
        return _DEFAULT_SNAPSHOT_TTL_SEC


def status_stale_sec() -> float:
    raw = os.getenv("SCALP_STATUS_STALE_SEC", "240")
    try:
        return max(60.0, float(raw))
    except (TypeError, ValueError):
        return 240.0


def api_status_snapshot_redis_key() -> str:
    """Single shared Redis key for runner publish + API read."""
    from backend.services.binance_scalp.config import get_scalp_config
    from backend.services.binance_scalp.redis_keys import api_status_snapshot_key

    cfg = get_scalp_config()
    return api_status_snapshot_key(cfg.redis_key_prefix)


def _redis_cache_key(warm_rounds: int = _API_SNAPSHOT_WARM) -> str:
    _ = warm_rounds
    return api_status_snapshot_redis_key()


def publish_status_snapshot(payload: dict[str, Any], *, ttl_sec: float | None = None) -> None:
    """Writer-side publish (runner tick / heartbeat). Never call from GET handlers."""
    global _LAST_PUBLISHED_PAYLOAD
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        ttl = int(ttl_sec if ttl_sec is not None else status_cache_ttl_sec())
        ttl = max(int(_MIN_SNAPSHOT_TTL_SEC), ttl)
        now = time.time()
        store = dict(payload)
        store["_cached_at"] = float(store.get("_cached_at") or now)
        store["updated_at_epoch"] = float(store.get("updated_at_epoch") or store["_cached_at"])
        store["updated_at"] = store.get("updated_at") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(store["updated_at_epoch"]))
        store["age_seconds"] = 0.0
        store["snapshot_available"] = True
        store["stale"] = False
        store.setdefault("reason", "OK")
        store.setdefault("runner_active", True)
        store.setdefault("snapshot_source", "runner_tick")
        key = _redis_cache_key(_API_SNAPSHOT_WARM)
        client.setex(key, ttl, json.dumps(store, separators=(",", ":")))
        _LAST_PUBLISHED_PAYLOAD = dict(store)
    except Exception as exc:
        logger.warning("scalp status redis publish skipped: %s", exc)


def refresh_status_snapshot_heartbeat(*, reason: str = "runner_heartbeat") -> bool:
    """Re-publish last snapshot with fresh timestamps so TTL cannot expire mid-tick."""
    global _LAST_PUBLISHED_PAYLOAD
    base = dict(_LAST_PUBLISHED_PAYLOAD or {})
    if not base:
        base = {
            "overall_decision": "SCANNING",
            "top_blocker": reason,
            "operational_summary": {"operational_mode": "heartbeat", "open_count": 0},
            "snapshot_source": "runner_heartbeat",
            "pnl_summary": {"engine": "scalp", "note": "pnl_omitted_on_fast_path"},
        }
    now = time.time()
    base["_cached_at"] = now
    base["updated_at_epoch"] = now
    base["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))
    base["heartbeat_epoch"] = now
    base["snapshot_source"] = str(base.get("snapshot_source") or "runner_heartbeat")
    if reason and reason not in ("runner_heartbeat",):
        base["top_blocker"] = base.get("top_blocker") or reason
    try:
        publish_status_snapshot(base, ttl_sec=status_cache_ttl_sec())
        return True
    except Exception:
        return False


def read_published_scalp_status() -> dict[str, Any] | None:
    """Read-only Redis snapshot for GET /api/scalp/status — no rebuild."""
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        raw = client.get(_redis_cache_key(_API_SNAPSHOT_WARM))
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except Exception as exc:
        logger.debug("scalp status redis cache read skipped: %s", exc)
        return None


def get_cached_scalp_status(*, warm_rounds: int = 0, force_refresh: bool = False) -> dict[str, Any]:
    """
    Fast GET path: return published Redis snapshot only.

    ``force_refresh`` / ``warm_rounds`` are ignored for rebuild — diagnostics that
    need a full rebuild must call ``build_scalp_status`` from a non-GET tool.
    """
    del force_refresh  # GET must never rebuild
    _ = warm_rounds
    now = time.time()
    stale_after = status_stale_sec()
    cached = read_published_scalp_status()
    if not cached:
        return {
            "snapshot_available": False,
            "stale": True,
            "reason": "SCALP_STATUS_SNAPSHOT_MISSING",
            "cache_hit": False,
            "cache_backend": "none",
            "overall_decision": "DEGRADED",
            "top_blocker": "SCALP_STATUS_SNAPSHOT_MISSING",
            "note": "Runner has not published a status snapshot yet — retry shortly.",
        }

    cached_at = float(cached.get("_cached_at") or cached.get("updated_at_epoch") or 0.0)
    age = (now - cached_at) if cached_at > 0 else None
    out = dict(cached)
    out.pop("_cached_at", None)
    out["cache_hit"] = True
    out["cache_backend"] = "redis"
    out["snapshot_available"] = True
    out["cache_age_sec"] = round(age, 2) if age is not None else None
    out["age_seconds"] = round(age, 2) if age is not None else None

    if age is not None and age > stale_after:
        out["stale"] = True
        out["reason"] = "STALE"
        out["overall_decision"] = out.get("overall_decision") or "DEGRADED"
        out["note"] = f"Scalp status snapshot stale ({age:.0f}s) — showing last published state."
    else:
        out["stale"] = False
        out.setdefault("reason", "OK")
    return out


def invalidate_scalp_status_cache() -> None:
    try:
        import redis

        from backend.services.binance_scalp.config import get_scalp_config

        cfg = get_scalp_config()
        client = redis.from_url(cfg.redis_url, decode_responses=True)
        client.delete(_redis_cache_key(_API_SNAPSHOT_WARM))
    except Exception:
        pass


def build_runner_api_status_payload(
    *,
    runner_state: dict[str, Any] | None,
    last_decision: dict[str, Any] | None,
    entry_armed: bool,
    open_count: int,
    products: list[str] | tuple[str, ...],
    scalp_live: bool,
    scalp_paper_enabled: bool,
    pnl_summary: dict[str, Any] | None = None,
    snapshot_source: str = "runner_tick",
    open_symbols: list[str] | None = None,
    last_fill: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Lightweight API snapshot assembled from runner tick state (no REST/klines)."""
    now = time.time()
    rs = dict(runner_state or {})
    ld = dict(last_decision or {})
    decision = str(ld.get("decision") or "")
    overall = {
        "ENTER": "READY",
        "BLOCKED": "BLOCKED",
        "NO_SIGNAL": "NO_SETUP",
        "WAITING": "WAITING",
    }.get(decision, "SCANNING")
    if rs.get("operational_mode") == "max_open_positions_reached" or open_count > 0 and str(rs.get("entry_blocked_reason") or "") == "MAX_OPEN_POSITIONS":
        overall = "WAITING_FOR_EXIT"
    top_blocker = str(ld.get("reason") or rs.get("entry_blocked_reason") or "") or None
    open_syms = list(open_symbols or rs.get("open_symbols") or [])
    selected = ld.get("symbol") or ld.get("selected_symbol") or (open_syms[0] if open_syms else None)
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
        "_cached_at": now,
        "updated_at_epoch": now,
        "age_seconds": 0.0,
        "overall_decision": overall,
        "decision_source": "engine_canonical",
        "canonical_engine_decision": ld or None,
        "top_blocker": top_blocker,
        "selected_symbol": selected,
        "operational_summary": {
            "operational_mode": rs.get("operational_mode") or "entry_scan_active",
            "open_count": open_count,
            "max_open_positions": rs.get("max_open_positions"),
            "entry_blocked_reason": rs.get("entry_blocked_reason"),
            "open_slot_held": bool(open_count > 0),
            "open_symbols": open_syms,
        },
        "open_slot_held": bool(open_count > 0),
        "runner_state": rs,
        "entry_armed": bool(entry_armed),
        "open_scalp_positions": int(open_count),
        "open_positions_summary": [{"symbol": s, "status": "OPEN"} for s in open_syms],
        "last_fill": last_fill,
        "scalp_engaged": True,
        "scalp_live": bool(scalp_live),
        "scalp_paper_enabled": bool(scalp_paper_enabled),
        "products": list(products),
        "snapshot_source": snapshot_source,
        "snapshot_available": True,
        "stale": False,
        "reason": "OK",
        "runner_active": True,
        "pnl_summary": pnl_summary or {"engine": "scalp", "note": "pnl_omitted_on_fast_path"},
        "heartbeat_epoch": float(rs.get("updated_at_epoch") or now),
        "circuit_breaker_open": str(ld.get("reason") or rs.get("entry_blocked_reason") or "") == "SCALP_CIRCUIT_BREAKER_OPEN",
        "scalp_entry_enabled": not bool(rs.get("entry_blocked_reason") or (str(ld.get("reason") or "") in {"SCALP_CIRCUIT_BREAKER_OPEN", "SCALP_PAPER_DISABLED", "FEE_MODEL_UNVERIFIED"})),
        "scalp_exit_enabled": True,
    }


__all__ = [
    "api_status_snapshot_redis_key",
    "build_runner_api_status_payload",
    "get_cached_scalp_status",
    "invalidate_scalp_status_cache",
    "publish_status_snapshot",
    "read_published_scalp_status",
    "refresh_status_snapshot_heartbeat",
    "status_cache_ttl_sec",
    "status_stale_sec",
]
