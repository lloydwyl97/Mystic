"""Scalp runtime control keys — Redis only, scalp: namespace."""

from __future__ import annotations

import redis

from backend.services.binance_scalp.redis_keys import assert_key_allowed, normalize_prefix

CONTROL_TTL_SEC = 3600
ENTRY_ARMED_NAME = "entry_armed"


def control_key(prefix: str, name: str) -> str:
    key = f"{normalize_prefix(prefix)}:control:{name}"
    assert_key_allowed(key, prefix=prefix)
    return key


def is_entry_armed(client: redis.Redis, *, prefix: str = "scalp") -> bool:
    raw = client.get(control_key(prefix, ENTRY_ARMED_NAME))
    return str(raw or "0").strip() == "1"


def set_entry_armed(
    client: redis.Redis,
    *,
    prefix: str = "scalp",
    armed: bool,
    persistent: bool = False,
) -> None:
    key = control_key(prefix, ENTRY_ARMED_NAME)
    if armed:
        if persistent:
            client.set(key, "1")
        else:
            client.setex(key, CONTROL_TTL_SEC, "1")
    else:
        client.set(key, "0")


def clear_entry_armed(client: redis.Redis, *, prefix: str = "scalp") -> None:
    client.delete(control_key(prefix, ENTRY_ARMED_NAME))
