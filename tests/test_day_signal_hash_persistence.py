"""DAY ai_signal:day:* hashes must stay populated every cycle (no HLEN flap)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.live_strategy_contracts import redis_ai_signal_key


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.ttls: dict[str, int] = {}
        self.deleted: list[str] = []

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hset(self, key: str, mapping: dict[str, Any] | None = None, **kwargs):
        cur = self.hashes.setdefault(key, {})
        if mapping:
            cur.update({str(k): str(v) for k, v in mapping.items()})
        cur.update({str(k): str(v) for k, v in kwargs.items()})
        return 1

    async def hlen(self, key: str) -> int:
        return len(self.hashes.get(key, {}))

    async def expire(self, key: str, ttl: int) -> bool:
        self.ttls[key] = int(ttl)
        return True

    async def delete(self, *keys: str) -> int:
        n = 0
        for k in keys:
            self.deleted.append(k)
            if k in self.hashes:
                del self.hashes[k]
                n += 1
        return n

    def pipeline(self, transaction: bool = True):
        redis_self = self

        class _Pipe:
            def __init__(self):
                self.ops = []

            def hmset(self, key, mapping):
                self.ops.append(("hmset", key, mapping))

            def expire(self, key, ttl):
                self.ops.append(("expire", key, ttl))

            async def execute(self):
                for op in self.ops:
                    if op[0] == "hmset":
                        await redis_self.hset(op[1], mapping=op[2])
                    elif op[0] == "expire":
                        await redis_self.expire(op[1], op[2])
                return [True] * len(self.ops)

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

        return _Pipe()


def _make_generator(fake: _FakeRedis):
    from backend.services.ai_signal_generator import RealTimeAISignalGenerator

    gen = RealTimeAISignalGenerator.__new__(RealTimeAISignalGenerator)
    gen.redis = fake
    gen.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    gen.enabled_strategies = ["day"]
    return gen


@pytest.mark.asyncio
async def test_hold_nosignal_still_nonempty_hash() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    await gen._publish_degraded_day_signal_hash("day", "BTCUSDT", reason="NO_SIGNAL")
    key = redis_ai_signal_key("day", "BTCUSDT")
    assert await fake.hlen(key) > 0
    h = await fake.hgetall(key)
    assert h["symbol"] == "BTCUSDT"
    assert h["signal"] in ("HOLD", "NO_SIGNAL") or h["side"] == "HOLD"
    assert "timestamp" in h


@pytest.mark.asyncio
async def test_preserve_never_deletes_stale_hash() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    key = redis_ai_signal_key("day", "ETHUSDT")
    fake.hashes[key] = {
        "symbol": "ETHUSDT",
        "signal": "BUY",
        "side": "BUY",
        "timestamp": str(0.0),  # ancient
        "writer_timestamp": str(0.0),
        "confidence": "0.5",
    }
    with patch("backend.services.ai_signal_generator.MAX_SIGNAL_AGE_SEC", 180):
        await gen._preserve_existing_signal_ttl("day", "ETHUSDT", skip_reason="DAY_ACTIVE_CONTRACT_FAIL")
    assert key not in fake.deleted
    assert await fake.hlen(key) > 0
    h = await fake.hgetall(key)
    assert h.get("signal_content_stale") == "1"
    assert float(fake.ttls[key]) >= 180


@pytest.mark.asyncio
async def test_preserve_missing_hash_publishes_degraded() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    await gen._preserve_existing_signal_ttl("day", "SOLUSDT", skip_reason="DAY_ACTIVE_SKIP_NO_LIVE")
    key = redis_ai_signal_key("day", "SOLUSDT")
    assert await fake.hlen(key) > 0
    assert "/" not in key


@pytest.mark.asyncio
async def test_all_four_symbols_published_each_cycle() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    for sym in gen.symbols:
        await gen._publish_degraded_day_signal_hash("day", sym, reason="HOLD")
    for sym in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]:
        key = redis_ai_signal_key("day", sym)
        assert await fake.hlen(key) > 0
        assert key == f"ai_signal:day:{sym}"


@pytest.mark.asyncio
async def test_db_failure_does_not_delete_redis_signal() -> None:
    fake = _FakeRedis()
    key = redis_ai_signal_key("day", "XRPUSDT")
    fake.hashes[key] = {
        "symbol": "XRPUSDT",
        "signal": "HOLD",
        "side": "HOLD",
        "timestamp": str(1_000_000_000.0),
        "confidence": "0.1",
    }
    # Simulate the production contract: redis already published; DB fails afterward.
    before = dict(fake.hashes[key])
    # No delete path may run on DB failure — assert helper does not wipe.
    gen = _make_generator(fake)
    await gen._preserve_existing_signal_ttl("day", "XRPUSDT", skip_reason="db_degraded_test")
    assert key not in fake.deleted
    assert await fake.hlen(key) > 0
    assert fake.hashes[key]["symbol"] == before["symbol"]


@pytest.mark.asyncio
async def test_ttl_exceeds_expected_publish_interval() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    with patch("backend.services.ai_signal_generator.AI_SIGNAL_REDIS_TTL_SEC", 300):
        with patch("backend.services.ai_signal_generator.MAX_SIGNAL_AGE_SEC", 180):
            ttl = gen._signal_redis_ttl_sec()
    assert ttl >= 600
    await gen._publish_degraded_day_signal_hash("day", "BTCUSDT", reason="HOLD")
    key = redis_ai_signal_key("day", "BTCUSDT")
    assert fake.ttls[key] >= 180


@pytest.mark.asyncio
async def test_no_slash_variant_keys() -> None:
    fake = _FakeRedis()
    gen = _make_generator(fake)
    await gen._publish_degraded_day_signal_hash("day", "BTC/USDT", reason="HOLD")
    assert "ai_signal:day:BTC/USDT" not in fake.hashes
    assert await fake.hlen(redis_ai_signal_key("day", "BTCUSDT")) > 0


def test_redis_ai_signal_key_canonical() -> None:
    assert redis_ai_signal_key("day", "btc/usdt") == "ai_signal:day:BTCUSDT"
    assert redis_ai_signal_key("day", "ETHUSDT") == "ai_signal:day:ETHUSDT"
