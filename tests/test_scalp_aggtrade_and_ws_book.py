"""SCALP tape parse + WS book publish. DAY untouched."""

from __future__ import annotations

import asyncio
import json
from unittest import mock

from backend.services.agg_trade_collector import AggTradeCollector
from backend.services.binance_scalp.market_reader import (
    SCALP_WS_DEPTH_MAX_AGE_SEC,
    _read_ws_depth,
    publish_ws_depth,
)


def test_aggtrade_accepts_combined_and_raw_payloads():
    rec = []

    def _record(symbol, qty, is_buyer_maker, ts=None):
        rec.append((symbol, qty, is_buyer_maker, ts))

    c = AggTradeCollector()

    async def _run():
        with mock.patch("backend.services.microstructure_engine.record_agg_trade", _record), mock.patch.object(
            c, "_heartbeat_throttled", new=mock.AsyncMock()
        ):
            await c._process_message(
                json.dumps(
                    {
                        "stream": "btcusdt@aggTrade",
                        "data": {"s": "BTCUSDT", "q": "0.01", "m": True, "T": 1_000_000},
                    }
                )
            )
            await c._process_message(
                json.dumps({"e": "aggTrade", "s": "ETHUSDT", "q": "0.2", "m": False, "T": 2_000_000})
            )
            await c._process_message(json.dumps({"result": None, "id": 1}))

    asyncio.run(_run())
    assert rec[0][0] == "BTC" and rec[0][2] is True
    assert rec[1][0] == "ETH" and rec[1][2] is False
    assert c.stats["trades_processed"] == 2


def test_publish_ws_depth_readable_under_max_age():
    store = {}

    class _R:
        def set(self, k, v, ex=None):
            store[k] = v

        def get(self, k):
            return store.get(k)

    fake = _R()
    with mock.patch("backend.services.binance_scalp.market_reader.redis.from_url", return_value=fake):
        publish_ws_depth("BTC", [[100.0, 1.0]], [[100.1, 1.0]])
    got = _read_ws_depth(fake, "BTCUSDT")
    assert got is not None
    bids, asks, age = got
    assert bids[0][0] == 100.0
    assert asks[0][0] == 100.1
    assert 0 <= age <= SCALP_WS_DEPTH_MAX_AGE_SEC
