"""SCALP momentum needs samples between 60s evaluations or 15s/30s change is always 0."""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from backend.services.binance_scalp.market_reader import ScalpMarketReader
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.scalp_strategy_router import ScalpStrategyRouter


class _FakeRedis:
    def __init__(self, books: dict[str, tuple[float, float]]):
        self.books = books

    def get(self, key: str):
        sym = key.rsplit(":", 1)[-1]
        if sym not in self.books:
            return None
        bid, ask = self.books[sym]
        return json.dumps({"fetched_at": time.time(), "bids": [[bid, 1.0]], "asks": [[ask, 1.0]]})


def _reader(books) -> ScalpMarketReader:
    reader = ScalpMarketReader.__new__(ScalpMarketReader)
    reader._redis = _FakeRedis(books)
    return reader


def test_top_of_book_reads_websocket_only(monkeypatch):
    import backend.services.binance_scalp.market_reader as mr

    monkeypatch.setattr(mr, "fetch_depth_sync", lambda *_a, **_k: pytest.fail("REST depth must not be called"))
    reader = _reader({"BTCUSDT": (100.0, 100.2)})
    assert reader.read_top_of_book("BTCUSDT") == (100.0, pytest.approx(100.1))
    assert reader.read_top_of_book("ETHUSDT") is None


def test_sampled_history_gives_nonzero_short_momentum():
    tracker = MomentumTracker()
    router = ScalpStrategyRouter.__new__(ScalpStrategyRouter)
    router.config = SimpleNamespace(products=["BTCUSDT", "ETHUSDT"])
    router.momentum = tracker
    books = {"BTCUSDT": (100.0, 100.2)}
    router.reader = _reader(books)

    t0 = 1_000.0
    for i in range(7):
        books["BTCUSDT"] = (100.0 + i * 0.1, 100.2 + i * 0.1)
        assert router.sample_momentum(epoch=t0 + i * 5) == 1

    now = t0 + 30
    diag = tracker.diagnostics("BTCUSDT", now, 100.6, 100.7)
    assert diag.bid_change_15s > 0
    assert diag.mid_change_30s > 0


def test_single_sample_per_minute_leaves_short_momentum_zero():
    tracker = MomentumTracker()
    tracker.record("BTCUSDT", 1_000.0, 100.0, 100.1)
    diag = tracker.diagnostics("BTCUSDT", 1_060.0, 101.0, 101.1)
    assert diag.bid_change_15s == 0.0
    assert diag.mid_change_30s == 0.0
