"""Cross-process internal task-health heartbeats.

Motivation: on 2026-07-02 the order_book_collector's WebSocket receive loop
silently stopped processing messages for ~8 hours. The OS process (uvicorn)
stayed up, the socket stayed connected, and no exception was ever raised —
so nothing in systemd/process-level monitoring could have detected it. The
only thing that would have caught it is a heartbeat emitted from *inside*
the loop doing real work, checked against an expected cadence.

This module is a thin, generic, Redis-backed heartbeat registry so any
long-running loop (in any of Mystic's separate OS processes) can call
``beat(name)`` each time it does real work, and a single place (this module,
queried from the always-on backend API process) can report which internal
tasks have gone silent — independent of whether the owning OS process is
still technically "running".

Deliberately minimal: no new tables, no schema, just short-TTL Redis keys.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any

_HEARTBEAT_KEY_PREFIX = "task_heartbeat:"
_DEFAULT_TTL_SEC = 600

# Expected max silence (seconds) before a registered task is considered STALE.
# Conservative thresholds — tuned to the task's normal cadence with headroom
# for transient network/exchange hiccups, not tight enough to false-alarm.
CRITICAL_TASK_THRESHOLDS_SEC: dict[str, float] = {
    "order_book_collector:ws_messages": 60.0,
    "agg_trade_collector:ws_messages": 180.0,
    "live_market_data:ohlcv_loop": 180.0,
    "scalp_runner:tick": 60.0,
}


@dataclass(frozen=True)
class TaskHealth:
    name: str
    last_beat_epoch: float | None
    age_sec: float | None
    threshold_sec: float | None
    status: str  # "OK" | "STALE" | "UNKNOWN" (never beaten) | "UNMONITORED" (no threshold registered)
    extra: dict[str, Any]


def _heartbeat_mapping(extra: dict[str, Any] | None) -> dict[str, str]:
    now = time.time()
    mapping: dict[str, str] = {"last_beat_epoch": str(now)}
    if extra:
        for k, v in extra.items():
            with contextlib.suppress(Exception):
                mapping[str(k)] = str(v)
    return mapping


async def beat(task_name: str, redis_client: Any, *, extra: dict[str, Any] | None = None, ttl_sec: int = _DEFAULT_TTL_SEC) -> None:
    """Record that ``task_name`` did real work right now (async redis client). Best-effort, never raises."""
    if redis_client is None:
        return
    try:
        key = f"{_HEARTBEAT_KEY_PREFIX}{task_name}"
        mapping = _heartbeat_mapping(extra)
        pipe = redis_client.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl_sec)
        await pipe.execute()
    except Exception:
        # Heartbeat plumbing must never take down the task it's monitoring.
        pass


def beat_sync(task_name: str, redis_client: Any, *, extra: dict[str, Any] | None = None, ttl_sec: int = _DEFAULT_TTL_SEC) -> None:
    """Sync-redis-client variant of ``beat`` for non-asyncio loops (e.g. the scalp paper runner)."""
    if redis_client is None:
        return
    try:
        key = f"{_HEARTBEAT_KEY_PREFIX}{task_name}"
        mapping = _heartbeat_mapping(extra)
        pipe = redis_client.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, ttl_sec)
        pipe.execute()
    except Exception:
        pass


def _status_for(age_sec: float | None, threshold_sec: float | None) -> str:
    if age_sec is None:
        return "UNKNOWN"
    if threshold_sec is None:
        return "UNMONITORED"
    return "OK" if age_sec <= threshold_sec else "STALE"


async def get_task_health(redis_client: Any, *, task_names: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Report health for known critical tasks plus any other tasks that have ever beaten.

    Returns a dict with per-task status and an overall summary so callers
    (dashboard, /api/system/task-health) don't need their own threshold logic.
    """
    now = time.time()
    names = list(task_names) if task_names is not None else list(CRITICAL_TASK_THRESHOLDS_SEC.keys())

    results: dict[str, TaskHealth] = {}
    if redis_client is not None:
        try:
            discovered_keys: set[str] = set()
            cursor = 0
            while True:
                cursor, batch = await redis_client.scan(cursor=cursor, match=f"{_HEARTBEAT_KEY_PREFIX}*", count=100)
                for k in batch:
                    ks = k.decode() if isinstance(k, bytes) else str(k)
                    discovered_keys.add(ks[len(_HEARTBEAT_KEY_PREFIX) :])
                if cursor == 0:
                    break
            for n in discovered_keys:
                if n not in names:
                    names.append(n)
        except Exception:
            pass

    for name in names:
        threshold = CRITICAL_TASK_THRESHOLDS_SEC.get(name)
        last_beat: float | None = None
        extra: dict[str, Any] = {}
        if redis_client is not None:
            try:
                raw = await redis_client.hgetall(f"{_HEARTBEAT_KEY_PREFIX}{name}")
                if raw:
                    norm = {(k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else v) for k, v in raw.items()}
                    if norm.get("last_beat_epoch") is not None:
                        with contextlib.suppress(ValueError, TypeError):
                            last_beat = float(norm["last_beat_epoch"])
                    extra = {k: v for k, v in norm.items() if k != "last_beat_epoch"}
            except Exception:
                pass
        age = (now - last_beat) if last_beat is not None else None
        results[name] = TaskHealth(
            name=name,
            last_beat_epoch=last_beat,
            age_sec=round(age, 1) if age is not None else None,
            threshold_sec=threshold,
            status=_status_for(age, threshold),
            extra=extra,
        )

    stale = [n for n, h in results.items() if h.status == "STALE"]
    unknown_critical = [n for n in CRITICAL_TASK_THRESHOLDS_SEC if results.get(n) and results[n].status == "UNKNOWN"]
    overall = "OK"
    if stale or unknown_critical:
        overall = "DEGRADED"

    return {
        "overall_status": overall,
        "stale_tasks": stale,
        "unknown_critical_tasks": unknown_critical,
        "tasks": {
            n: {
                "status": h.status,
                "age_sec": h.age_sec,
                "threshold_sec": h.threshold_sec,
                "last_beat_epoch": h.last_beat_epoch,
                "extra": h.extra,
            }
            for n, h in results.items()
        },
        "checked_at_epoch": now,
    }


__all__ = [
    "CRITICAL_TASK_THRESHOLDS_SEC",
    "TaskHealth",
    "beat",
    "beat_sync",
    "get_task_health",
]
