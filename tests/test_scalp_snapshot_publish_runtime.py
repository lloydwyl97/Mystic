"""SCALP runner must publish the same Redis snapshot key the API reads."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.services.binance_scalp import scalp_status_cache
from backend.services.binance_scalp.redis_keys import (
    API_STATUS_SNAPSHOT_KEY,
    api_status_snapshot_key,
    status_snapshot_key,
)


def test_api_and_runner_share_same_snapshot_key() -> None:
    assert api_status_snapshot_key("scalp") == API_STATUS_SNAPSHOT_KEY
    assert status_snapshot_key("scalp", 0) == API_STATUS_SNAPSHOT_KEY
    assert scalp_status_cache.api_status_snapshot_redis_key() == API_STATUS_SNAPSHOT_KEY


def test_status_cache_ttl_at_least_180() -> None:
    assert scalp_status_cache.status_cache_ttl_sec() >= 180.0


def test_publish_status_snapshot_writes_api_key() -> None:
    store: dict[str, tuple[str, int]] = {}

    class _FakeRedis:
        def setex(self, key, ttl, value):
            store["key"] = key
            store["ttl"] = int(ttl)
            store["value"] = value

    fake = _FakeRedis()
    with (
        patch("redis.from_url", return_value=fake),
        patch(
            "backend.services.binance_scalp.config.get_scalp_config",
            return_value=MagicMock(redis_url="redis://localhost:6379/0", redis_key_prefix="scalp"),
        ),
    ):
        scalp_status_cache.publish_status_snapshot(
            {"overall_decision": "SCANNING", "snapshot_source": "runner_warm"},
            ttl_sec=90,
        )
    assert store["key"] == API_STATUS_SNAPSHOT_KEY
    assert store["ttl"] >= 180
    payload = json.loads(store["value"])
    assert payload["snapshot_available"] is True
    assert payload["runner_active"] is True


def test_refresh_heartbeat_republishes_last_payload() -> None:
    calls: list[dict] = []

    def _pub(payload, *, ttl_sec=None):
        calls.append({"payload": dict(payload), "ttl": ttl_sec})

    with patch.object(scalp_status_cache, "publish_status_snapshot", side_effect=_pub):
        scalp_status_cache._LAST_PUBLISHED_PAYLOAD = {
            "overall_decision": "WAITING_FOR_EXIT",
            "open_slot_held": True,
            "snapshot_source": "runner_tick",
        }
        ok = scalp_status_cache.refresh_status_snapshot_heartbeat(reason="runner_heartbeat")
    assert ok is True
    assert calls
    assert calls[0]["payload"]["open_slot_held"] is True
    assert calls[0]["payload"]["updated_at_epoch"] > 0


def test_build_payload_marks_open_slot_held() -> None:
    payload = scalp_status_cache.build_runner_api_status_payload(
        runner_state={"entry_blocked_reason": "MAX_OPEN_POSITIONS", "open_symbols": ["BTCUSDT"]},
        last_decision={"decision": "BLOCKED", "reason": "MAX_OPEN_POSITIONS", "symbol": "BTCUSDT"},
        entry_armed=True,
        open_count=1,
        products=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"],
        scalp_live=False,
        scalp_paper_enabled=True,
        snapshot_source="runner_tick",
        open_symbols=["BTCUSDT"],
    )
    assert payload["open_slot_held"] is True
    assert payload["snapshot_available"] is True
    assert payload["runner_active"] is True
    assert payload["snapshot_source"] == "runner_tick"
    assert payload["selected_symbol"] == "BTCUSDT"


def test_api_reads_published_key_without_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    monkeypatch.setattr(
        scalp_status_cache,
        "read_published_scalp_status",
        lambda: {
            "_cached_at": now,
            "snapshot_source": "runner_tick",
            "overall_decision": "READY",
            "open_slot_held": False,
        },
    )
    out = scalp_status_cache.get_cached_scalp_status()
    assert out["snapshot_available"] is True
    assert out["stale"] is False
    assert out["snapshot_source"] == "runner_tick"


def test_api_missing_snapshot_degraded_no_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scalp_status_cache, "read_published_scalp_status", lambda: None)
    out = scalp_status_cache.get_cached_scalp_status()
    assert out["snapshot_available"] is False
    assert out["reason"] == "SCALP_STATUS_SNAPSHOT_MISSING"


def test_paper_engine_warm_and_tick_publish_same_key() -> None:
    from backend.services.binance_scalp.paper_engine import BinanceScalpPaperEngine

    published: list[str] = []

    def _pub(payload, *, ttl_sec=None):
        published.append(str(payload.get("snapshot_source")))

    eng = MagicMock(spec=BinanceScalpPaperEngine)
    eng.config = MagicMock(
        redis_key_prefix="scalp",
        products=["BTCUSDT"],
        scalp_live=False,
        scalp_paper_enabled=True,
        max_open_positions=4,
        database_path=":memory:",
    )
    eng._redis = MagicMock()
    eng._redis.get.return_value = None
    eng._entry_armed_ok = lambda: True
    eng._cached_open_symbols = []
    eng._publish_api_status_snapshot = BinanceScalpPaperEngine._publish_api_status_snapshot.__get__(eng, BinanceScalpPaperEngine)

    with patch(
        "backend.services.binance_scalp.scalp_status_cache.publish_status_snapshot",
        side_effect=_pub,
    ):
        eng._publish_api_status_snapshot(
            open_rows=[],
            epoch=time.time(),
            entry_blocked_reason="WARM_COMPLETE",
            snapshot_source="runner_warm",
        )
        eng._publish_api_status_snapshot(
            open_rows=[],
            epoch=time.time(),
            entry_blocked_reason="MAX_OPEN_POSITIONS",
            snapshot_source="runner_tick",
        )
        eng._publish_api_status_snapshot(
            open_rows=[],
            epoch=time.time(),
            entry_blocked_reason="TICK_ERROR:RuntimeError",
            snapshot_source="runner_error",
        )
    assert "runner_warm" in published
    assert "runner_tick" in published
    assert "runner_error" in published
