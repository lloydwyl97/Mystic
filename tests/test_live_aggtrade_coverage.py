"""Regression coverage for live aggTrade infrastructure used by SCALP V2.

Restored from test_aggtrade_collector_supervision.py (deleted in commit 87ef12e
as part of paper-runner removal).  These tests cover live production modules —
AggTradeCollector, structural_tape, structural_breaker — NOT the retired paper
runner.  They were incorrectly classified as paper-only.
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from unittest import mock

from backend.services.agg_trade_collector import AggTradeCollector
from backend.services.binance_scalp.structural_breaker import (
    StructuralBreakerState,
    default_thresholds,
    evaluate,
)
from backend.services.binance_scalp.structural_tape import parse_agg_payload, publish_trade_event, tape_freshness
from backend.services.task_health_monitor import CRITICAL_TASK_THRESHOLDS_SEC


def _agg_msg(symbol: str, agg_id: int, *, t_ms: int = 1_700_000_000_123) -> str:
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@aggTrade",
            "data": {
                "e": "aggTrade",
                "s": symbol,
                "a": agg_id,
                "p": "100.0",
                "q": "0.01",
                "m": True,
                "T": t_ms,
            },
        }
    )


class _FakeWS:
    def __init__(self, recvs: list):
        self._recvs = list(recvs)
        self.closed = False

    async def recv(self):
        if not self._recvs:
            await asyncio.sleep(3600)
            raise TimeoutError("hung")
        item = self._recvs.pop(0)
        if isinstance(item, BaseException):
            raise item
        if item == "__idle__":
            await asyncio.sleep(3600)
            raise TimeoutError("idle")
        return item

    async def close(self):
        self.closed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def test_combined_url_has_four_subscriptions():
    c = AggTradeCollector()
    c.symbols = ["BTC", "ETH", "SOL", "XRP"]
    url = c.combined_stream_url()
    for name in ("btcusdt@aggTrade", "ethusdt@aggTrade", "solusdt@aggTrade", "xrpusdt@aggTrade"):
        assert name in url


def test_websocket_disconnect_reconnects_without_synthetic_trades():
    c = AggTradeCollector()
    c.symbols = ["BTC", "ETH", "SOL", "XRP"]
    c.idle_reconnect_sec = 0.05
    c.is_running = True
    urls: list[str] = []
    published: list = []

    first = _FakeWS([ConnectionError("no close frame received or sent")])
    second = _FakeWS([_agg_msg("BTCUSDT", 88)])
    sockets = [first, second]

    def _connect(url, **_kwargs):
        urls.append(url)
        return sockets.pop(0)

    def _publish(_r, ev):
        published.append(ev)
        c.is_running = False

    async def _run():
        with (
            mock.patch("backend.services.agg_trade_collector.websockets.connect", side_effect=_connect),
            mock.patch("backend.services.microstructure_engine.record_agg_trade"),
            mock.patch(
                "backend.services.binance_scalp.structural_tape.publish_trade_event",
                side_effect=_publish,
            ),
            mock.patch("backend.config.redis_config.get_shared_redis_sync", return_value=object()),
            mock.patch.object(c, "_heartbeat_throttled", new=mock.AsyncMock()),
            mock.patch("backend.services.agg_trade_collector._RECONNECT_MIN_SEC", 0.01),
            mock.patch("backend.services.agg_trade_collector._RECONNECT_MAX_SEC", 0.01),
        ):
            await asyncio.wait_for(c._run_websocket(), timeout=1.0)

    asyncio.run(_run())
    assert len(urls) >= 2
    for url in urls:
        for name in ("btcusdt@aggTrade", "ethusdt@aggTrade", "solusdt@aggTrade", "xrpusdt@aggTrade"):
            assert name in url
    assert c.stats["reconnects"] >= 1
    # All published trades must have a real agg_id — no synthetic fills
    assert all(ev.agg_id == 88 for ev in published)
    assert not any(ev.agg_id <= 0 for ev in published)


def test_connect_exception_does_not_kill_loop():
    c = AggTradeCollector()
    c.symbols = ["BTC"]
    c.idle_reconnect_sec = 0.05
    c.is_running = True
    hits = {"n": 0}

    def _connect(*_a, **_k):
        hits["n"] += 1
        if hits["n"] == 1:
            raise OSError("Name or service not known")
        c.is_running = False
        return _FakeWS([])

    async def _run():
        with (
            mock.patch("backend.services.agg_trade_collector.websockets.connect", side_effect=_connect),
            mock.patch("backend.services.agg_trade_collector._RECONNECT_MIN_SEC", 0.01),
            mock.patch("backend.services.agg_trade_collector._RECONNECT_MAX_SEC", 0.01),
        ):
            await asyncio.wait_for(c._run_websocket(), timeout=1.0)

    asyncio.run(_run())
    assert hits["n"] >= 2
    assert c.stats["reconnects"] >= 1


def test_missing_exchange_timestamp_is_not_synthesized():
    """Payload without T field must not produce a synthetic timestamp."""
    ev = parse_agg_payload({"s": "BTCUSDT", "a": 1, "p": "1", "q": "1", "m": True}, recv_ts=time.time())
    assert ev is None
    ev2 = parse_agg_payload({"s": "BTCUSDT", "a": 2, "p": "1", "q": "1", "m": True, "T": 1_700_000_000_000}, recv_ts=123.0)
    assert ev2 is not None
    assert ev2.trade_ts == 1_700_000_000.0
    assert ev2.recv_ts == 123.0


def test_fresh_key_expires_and_tape_is_stale_without_new_prints():
    store: dict[str, tuple[str, int | None]] = {}

    class _R:
        def xadd(self, *_a, **_k):
            return "1-0"

        def set(self, k, v, ex=None):
            store[k] = (v, ex)

        def get(self, k):
            return store.get(k, (None, None))[0]

    r = _R()
    ev = parse_agg_payload({"s": "BTCUSDT", "a": 9, "p": "1", "q": "1", "m": True, "T": 1_700_000_000_000}, recv_ts=10.0)
    assert ev is not None
    publish_trade_event(r, ev)
    assert store["scalp:tape:fresh:BTCUSDT"][1] == 30
    view = tape_freshness(r, "BTCUSDT", now=1_700_000_000.0 + 1.0, stale_sec=15.0)
    assert view["fresh"] is True
    gone = type(r)()
    stale = tape_freshness(gone, "BTCUSDT", now=time.time(), stale_sec=15.0)
    assert stale["fresh"] is False


def test_stale_tape_opens_breaker_and_fresh_tape_closes_it():
    now = datetime(2026, 8, 31, tzinfo=timezone.utc)
    th = default_thresholds(consec=8, daily_loss_usd=25, timeout_rate=0.5, adverse_rate=0.8, recovery_sec=1800)
    prior = StructuralBreakerState(open=False, reason="", tripped_at="", recovery_until="", stats={}, thresholds=th)
    opened = evaluate(
        consec_losses=0,
        daily_pnl=0.0,
        rolling_pnl=[],
        timeout_rate=0.0,
        adverse_1s_rate=0.0,
        tape_stale=True,
        now=now,
        prior=prior,
    )
    assert opened.open is True
    assert opened.reason == "STALE_TRADE_STREAM"
    recovered = evaluate(
        consec_losses=0,
        daily_pnl=0.0,
        rolling_pnl=[],
        timeout_rate=0.0,
        adverse_1s_rate=0.0,
        tape_stale=False,
        now=now,
        prior=opened,
    )
    assert recovered.open is False


def test_aggtrade_heartbeat_is_monitored():
    """AggTradeCollector WS messages must be in CRITICAL_TASK_THRESHOLDS_SEC."""
    assert "agg_trade_collector:ws_messages" in CRITICAL_TASK_THRESHOLDS_SEC
    assert CRITICAL_TASK_THRESHOLDS_SEC["agg_trade_collector:ws_messages"] == 180.0
