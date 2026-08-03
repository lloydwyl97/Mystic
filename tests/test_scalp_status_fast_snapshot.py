"""GET /api/scalp/status must stay fast — Redis snapshot only, no rebuild."""

from __future__ import annotations

import inspect
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from backend.endpoints import scalp_status_endpoints
from backend.services.binance_scalp import scalp_status_cache


def test_status_route_source_never_builds() -> None:
    source = inspect.getsource(scalp_status_endpoints.scalp_status)
    assert "build_scalp_pnl_summary" not in source
    assert "fetch_depth_sync" not in source
    assert "get_cached_scalp_status(" in source
    assert "build_scalp_status(" not in source


def test_cache_helper_never_calls_build(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"build": 0, "depth": 0}

    def _boom_build(*_a, **_k):
        called["build"] += 1
        raise AssertionError("build_scalp_status must not run on GET")

    monkeypatch.setattr(
        "backend.services.binance_scalp.status_snapshot.build_scalp_status",
        _boom_build,
        raising=False,
    )
    monkeypatch.setattr(scalp_status_cache, "read_published_scalp_status", lambda: None)
    out = scalp_status_cache.get_cached_scalp_status()
    assert out["snapshot_available"] is False
    assert out["reason"] == "SCALP_STATUS_SNAPSHOT_MISSING"
    assert called["build"] == 0


def test_redis_snapshot_present_returns_quickly(monkeypatch: pytest.MonkeyPatch) -> None:
    now = time.time()
    monkeypatch.setattr(
        scalp_status_cache,
        "read_published_scalp_status",
        lambda: {
            "_cached_at": now,
            "overall_decision": "READY",
            "entry_armed": True,
            "open_scalp_positions": 0,
            "pnl_summary": {"engine": "scalp", "today": {"sells": 0}},
        },
    )
    monkeypatch.setattr(scalp_status_endpoints, "_scalp_runner_active", lambda: True)
    t0 = time.time()
    result = scalp_status_endpoints.scalp_status()
    elapsed = time.time() - t0
    assert elapsed < 0.5
    assert result["runner_active"] is True
    assert result["snapshot_available"] is True
    assert result["stale"] is False
    assert result["overall_decision"] == "READY"


def test_redis_snapshot_missing_degraded_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scalp_status_cache, "read_published_scalp_status", lambda: None)
    monkeypatch.setattr(scalp_status_endpoints, "_scalp_runner_active", lambda: True)
    t0 = time.time()
    result = scalp_status_endpoints.scalp_status()
    assert time.time() - t0 < 0.5
    assert result["runner_active"] is True
    assert result["snapshot_available"] is False
    assert result["stale"] is True
    assert result["reason"] == "SCALP_STATUS_SNAPSHOT_MISSING"


def test_redis_snapshot_stale_degraded_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scalp_status_cache,
        "read_published_scalp_status",
        lambda: {
            "_cached_at": time.time() - 9999,
            "overall_decision": "SCANNING",
            "pnl_summary": {"engine": "scalp"},
        },
    )
    monkeypatch.setattr(scalp_status_endpoints, "_scalp_runner_active", lambda: True)
    result = scalp_status_endpoints.scalp_status()
    assert result["stale"] is True
    assert result["reason"] == "STALE"
    assert result["snapshot_available"] is True


def test_runner_inactive_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scalp_status_endpoints, "_scalp_runner_active", lambda: False)
    t0 = time.time()
    result = scalp_status_endpoints.scalp_status()
    assert time.time() - t0 < 0.5
    assert result["runner_active"] is False
    assert result["reason"] == "RUNNER_INACTIVE"


def test_get_status_does_not_write_db(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(scalp_status_endpoints, "_scalp_runner_active", lambda: True)
    monkeypatch.setattr(
        scalp_status_cache,
        "read_published_scalp_status",
        lambda: {"_cached_at": time.time(), "overall_decision": "READY", "pnl_summary": {}},
    )
    with patch("sqlite3.connect") as connect:
        scalp_status_endpoints.scalp_status()
        connect.assert_not_called()


def test_publish_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    store: dict[str, str] = {}

    class FakeRedis:
        def setex(self, key, ttl, val):
            store[key] = val

        def get(self, key):
            return store.get(key)

    fake = FakeRedis()
    monkeypatch.setattr(
        "backend.services.binance_scalp.config.get_scalp_config",
        lambda: MagicMock(redis_url="redis://localhost:6379/0", redis_key_prefix="scalp"),
    )
    with patch("redis.from_url", return_value=fake):
        payload = scalp_status_cache.build_runner_api_status_payload(
            runner_state={"updated_at_epoch": time.time(), "operational_mode": "entry_scan_active"},
            last_decision={"decision": "BLOCKED", "reason": "TARGET_NOT_REACHABLE"},
            entry_armed=True,
            open_count=0,
            products=["BTCUSDT"],
            scalp_live=False,
            scalp_paper_enabled=True,
            pnl_summary={"engine": "scalp"},
        )
        scalp_status_cache.publish_status_snapshot(payload, ttl_sec=60)
        assert store
        got = scalp_status_cache.read_published_scalp_status()
        assert got is not None
        assert got["snapshot_source"] == "runner_tick"
