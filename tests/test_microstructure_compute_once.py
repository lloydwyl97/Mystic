"""One feature computation per order-book update, with byte-identical outputs."""

from __future__ import annotations

import asyncio
import importlib

import pytest

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


@pytest.fixture(autouse=True)
def mod(monkeypatch):
    m = importlib.import_module("backend.services.microstructure_engine")
    m._STATE.clear()
    for k in m._COMPUTE_STATS:
        m._COMPUTE_STATS[k] = 0
    monkeypatch.setattr(m, "_features_from_redis", lambda _symbol: {})
    monkeypatch.setattr(m, "_persist_row", lambda *_a, **_k: None)
    monkeypatch.setattr(m, "_attach_cross_market", lambda out, _symbol: out.update({"xm_marker": 1.0}))
    monkeypatch.setattr(m, "_now", lambda: 1_000_100.0)
    yield m
    m._STATE.clear()


class _Pipe:
    def __init__(self, sink):
        self.sink = sink

    def hset(self, key, mapping):
        self.sink.setdefault(key, {}).update(mapping)

    def expire(self, *_a):
        pass

    async def execute(self):
        return True


class _Redis:
    def __init__(self):
        self.data: dict[str, dict] = {}

    def pipeline(self, transaction=True):
        return _Pipe(self.data)


def _book(px: float, bid_sz: float, ask_sz: float):
    bids = [(px - 0.01 * (i + 1), bid_sz + 0.1 * i) for i in range(20)]
    asks = [(px + 0.01 * (i + 1), ask_sz + 0.05 * i) for i in range(20)]
    return bids, asks


def _fresh(m, symbol):
    """Features computed from scratch, bypassing the per-version cache."""
    st = m._STATE[m._base(symbol)]
    st.feat_key = st.feat_base = None
    out, needs_overlay = m._compute_book_features(symbol, st)
    if needs_overlay:
        m._overlay_redis_tape(out, symbol)
    m._attach_cross_market(out, symbol)
    return out


def _legacy_delta(m, feats):
    import math

    if not feats or feats.get("data_age_sec", 999) > 10.0:
        return 0.0
    ofi_5s = float(feats.get("ofi_5s", 0.0))
    agg_flow_5s = float(feats.get("agg_flow_imbalance_5s", 0.0))
    mp_pressure = float(feats.get("microprice_pressure", 0.0))
    absorp = float(feats.get("bid_absorption_score", 0.0)) - float(feats.get("ask_absorption_score", 0.0))
    adverse = float(feats.get("adverse_selection_score", 0.0))
    ofi_signed = math.tanh(ofi_5s / 5.0) if abs(ofi_5s) > 1e-9 else 0.0
    signal = (0.32 * ofi_signed) + (0.22 * agg_flow_5s) + (0.22 * math.tanh(mp_pressure * 500.0)) + (0.14 * max(-1.0, min(1.0, absorp))) - (0.10 * (2.0 * adverse - 0.5))
    cap = m._RANKING_DELTA_CAP
    return round(max(-cap, min(cap, signal * cap)), 6)


def _drive(m, symbol, n, redis, *, trades=True):
    """Simulate the collector: record a book, then publish it, n times."""
    for i in range(n):
        t = 1_000_000.0 + i * 0.25
        if trades and i % 3 == 0:
            m.record_agg_trade(symbol, 0.5 + i * 0.01, bool(i % 2), ts=t - 0.1)
        bids, asks = _book(100.0 + 0.01 * (i % 7), 2.0 + (i % 5) * 0.3, 1.5 + (i % 4) * 0.2)
        m.record_snapshot(symbol, bids, asks, ts=t)
        asyncio.run(m.publish_to_redis_async(symbol, redis))


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_one_feature_computation_per_book_update(mod, symbol):
    redis = _Redis()
    _drive(mod, symbol, 60, redis)  # 15s of books: crosses the 5s DB-persist cadence
    stats = mod.compute_stats()
    assert stats["orderbook_messages"] == 60
    assert stats["feature_computations"] == 60
    assert stats["feature_computations_per_message"] == 1.0
    assert stats["feature_reuses"] > 0  # persist + publish reused the same snapshot


def test_cached_features_equal_fresh_computation(mod):
    for i in range(40):
        t = 1_000_010.0 + i * 0.25
        mod.record_agg_trade("BTCUSDT", 0.3 + i * 0.02, bool(i % 3), ts=t - 0.05)
        bids, asks = _book(100.0 + 0.02 * (i % 5), 1.0 + i * 0.05, 2.0 - i * 0.01)
        mod.record_snapshot("BTCUSDT", bids, asks, ts=t)
        cached_a = mod.compute_features("BTCUSDT")
        cached_b = mod.compute_features("BTCUSDT")
        fresh = _fresh(mod, "BTCUSDT")
        assert cached_a == fresh
        assert cached_b == fresh
        assert list(cached_a) == list(fresh)


def test_new_book_or_trade_never_reuses_stale_features(mod):
    bids, asks = _book(100.0, 2.0, 1.0)
    mod.record_snapshot("ETHUSDT", bids, asks, ts=1_000_000.0)
    first = mod.compute_features("ETHUSDT")
    assert mod._COMPUTE_STATS["feature_computations"] == 1

    bids, asks = _book(100.5, 1.0, 3.0)
    mod.record_snapshot("ETHUSDT", bids, asks, ts=1_000_000.5)
    second = mod.compute_features("ETHUSDT")
    assert mod._COMPUTE_STATS["feature_computations"] == 2
    assert second["best_bid"] != first["best_bid"]
    assert second == _fresh(mod, "ETHUSDT")

    mod.record_agg_trade("ETHUSDT", 4.0, False, ts=1_000_000.6)
    third = mod.compute_features("ETHUSDT")
    assert mod._COMPUTE_STATS["feature_computations"] == 3
    assert third == _fresh(mod, "ETHUSDT")


def test_reuse_refreshes_time_dependent_fields(mod, monkeypatch):
    bids, asks = _book(100.0, 2.0, 1.0)
    mod.record_snapshot("SOLUSDT", bids, asks, ts=1_000_000.0)
    a = mod.compute_features("SOLUSDT")
    monkeypatch.setattr(mod, "_now", lambda: 1_000_200.0)
    calls = []
    monkeypatch.setattr(mod, "_attach_cross_market", lambda out, s: (calls.append(s), out.update({"xm_marker": 2.0})))
    b = mod.compute_features("SOLUSDT")
    assert mod._COMPUTE_STATS["feature_computations"] == 1
    assert b["data_age_sec"] == pytest.approx(200.0)
    assert a["data_age_sec"] == pytest.approx(100.0)
    assert b["xm_marker"] == 2.0 and calls == ["SOLUSDT"]
    assert mod.get_microstructure_ranking_delta("SOLUSDT") == 0.0  # stale book -> no delta


def test_redis_tape_overlay_is_reread_on_reuse(mod, monkeypatch):
    bids, asks = _book(100.0, 2.0, 1.0)
    mod.record_snapshot("BTCUSDT", bids, asks, ts=1_000_000.0)
    assert mod.compute_features("BTCUSDT")["agg_flow_imbalance_5s"] == 0.0
    monkeypatch.setattr(mod, "_features_from_redis", lambda _s: {"data_age_sec": 0.01, "agg_flow_imbalance_5s": 0.42})
    again = mod.compute_features("BTCUSDT")
    assert mod._COMPUTE_STATS["feature_computations"] == 1
    assert again["agg_flow_imbalance_5s"] == 0.42
    assert again == _fresh(mod, "BTCUSDT")


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_published_values_match_uncached_legacy_path(mod, monkeypatch, symbol):
    """The Redis hashes the SCALP micro learner reads carry identical values."""
    monkeypatch.setattr(mod, "_now", lambda: 1_000_010.2)
    redis = _Redis()
    _drive(mod, symbol, 41, redis)
    fresh = _fresh(mod, symbol)
    legacy_delta = _legacy_delta(mod, fresh)
    base = mod._base(symbol)
    full = redis.data[f"microstructure:{base}"]
    expected = {k: str(v) for k, v in fresh.items() if k != "symbol"}
    expected["ranking_delta"] = str(legacy_delta)
    assert full == expected
    assert redis.data[f"orderbook:{base}"]["microstructure_ranking_delta"] == str(legacy_delta)
    assert mod.ranking_delta_from_features(fresh) == legacy_delta
    assert mod.get_microstructure_ranking_delta(symbol) == legacy_delta


def test_get_stats_exposes_instrumentation(mod):
    _drive(mod, "XRPUSDT", 3, _Redis())
    stats = mod.get_stats()
    for key in ("orderbook_messages", "feature_computations", "feature_computations_per_message"):
        assert key in stats
