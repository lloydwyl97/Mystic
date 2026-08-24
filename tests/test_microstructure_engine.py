"""Unit tests for backend.services.microstructure_engine.

Covers: multi-depth imbalance, microprice, snapshot-OFI, aggressor volume,
queue-dynamics inference, imbalance slope/persistence/reversal, and the
bounded ranking delta (must never exceed its cap and must be an EV/ranking
input only — no boolean/gate return type anywhere in this module).
"""

from __future__ import annotations

import importlib
import time

import pytest


@pytest.fixture(autouse=True)
def _fresh_state(monkeypatch):
    """Each test gets an isolated symbol-state dict (module uses process-global state)."""
    mod = importlib.import_module("backend.services.microstructure_engine")
    mod._STATE.clear()
    monkeypatch.setattr(mod, "_features_from_redis", lambda symbol: {})
    yield mod
    mod._STATE.clear()


def _book(bid_px, bid_sz, ask_px, ask_sz, depth=20, spacing=0.01):
    bids = [(bid_px - i * spacing, bid_sz) for i in range(depth)]
    asks = [(ask_px + i * spacing, ask_sz) for i in range(depth)]
    return bids, asks


def test_imbalance_at_depth_bid_heavy():
    from backend.services import microstructure_engine as m

    bids = [(100.0, 5.0), (99.9, 5.0), (99.8, 5.0)]
    asks = [(100.1, 1.0), (100.2, 1.0), (100.3, 1.0)]
    obi = m.imbalance_at_depth(bids, asks, 3)
    assert 0.5 < obi <= 1.0


def test_imbalance_at_depth_balanced():
    from backend.services import microstructure_engine as m

    bids = [(100.0, 5.0)]
    asks = [(100.1, 5.0)]
    obi = m.imbalance_at_depth(bids, asks, 1)
    assert obi == pytest.approx(0.0, abs=1e-9)


def test_imbalance_at_depth_empty_is_zero():
    from backend.services import microstructure_engine as m

    assert m.imbalance_at_depth([], [], 5) == 0.0


def test_microprice_weighted_toward_thin_side():
    from backend.services import microstructure_engine as m

    # Thin ask (size 1) vs thick bid (size 9) -> microprice should be pulled
    # toward the ask because the thin side is more likely to be walked through.
    mp = m.microprice(bid_px=100.0, bid_sz=9.0, ask_px=101.0, ask_sz=1.0)
    mid = 100.5
    assert mp > mid  # weighted toward ask price since ask side is thin


def test_microprice_symmetric_when_equal_sizes():
    from backend.services import microstructure_engine as m

    mp = m.microprice(bid_px=100.0, bid_sz=5.0, ask_px=101.0, ask_sz=5.0)
    assert mp == pytest.approx(100.5, abs=1e-9)


def test_microprice_degenerate_inputs_return_mid_or_zero():
    from backend.services import microstructure_engine as m

    assert m.microprice(0.0, 5.0, 101.0, 5.0) == 0.0
    assert m.microprice(100.0, 0.0, 101.0, 0.0) == pytest.approx(100.5, abs=1e-9)


def test_record_snapshot_rejects_bad_input():
    from backend.services import microstructure_engine as m

    m.record_snapshot("BTCUSDT", [], [])
    assert m.compute_features("BTCUSDT") == {}

    m.record_snapshot("BTCUSDT", [(0.0, 1.0)], [(1.0, 1.0)])
    assert m.compute_features("BTCUSDT") == {}


def test_compute_features_populates_all_depths():
    from backend.services import microstructure_engine as m

    bids, asks = _book(100.0, 5.0, 100.2, 3.0)
    m.record_snapshot("ETHUSDT", bids, asks, ts=time.time())
    feats = m.compute_features("ETHUSDT")
    assert feats["symbol"] == "ETH"
    for d in m.DEPTH_LEVELS:
        assert f"obi_l{d}" in feats
    assert feats["microprice"] > 0
    assert feats["spread_pct"] > 0


def test_ofi_increment_positive_when_bid_grows_and_ask_shrinks():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids1, asks1 = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("SOLUSDT", bids1, asks1, ts=t0)

    # Bid size increases (more buy interest resting), ask size decreases.
    bids2, asks2 = _book(100.0, 8.0, 100.2, 2.0)
    m.record_snapshot("SOLUSDT", bids2, asks2, ts=t0 + 0.1)

    feats = m.compute_features("SOLUSDT")
    assert feats["ofi_1s"] > 0


def test_ofi_increment_negative_when_ask_grows_and_bid_shrinks():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids1, asks1 = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("XRPUSDT", bids1, asks1, ts=t0)

    bids2, asks2 = _book(100.0, 2.0, 100.2, 8.0)
    m.record_snapshot("XRPUSDT", bids2, asks2, ts=t0 + 0.1)

    feats = m.compute_features("XRPUSDT")
    assert feats["ofi_1s"] < 0


def test_agg_trade_buy_sell_semantics():
    from backend.services import microstructure_engine as m

    bids, asks = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("BTCUSDT", bids, asks)

    now = time.time()
    # isBuyerMaker=False => aggressive BUY
    m.record_agg_trade("BTCUSDT", qty=2.0, is_buyer_maker=False, ts=now)
    # isBuyerMaker=True => aggressive SELL
    m.record_agg_trade("BTCUSDT", qty=1.0, is_buyer_maker=True, ts=now)

    feats = m.compute_features("BTCUSDT")
    assert feats["agg_buy_vol_5s"] == pytest.approx(2.0)
    assert feats["agg_sell_vol_5s"] == pytest.approx(1.0)
    assert feats["agg_flow_imbalance_5s"] > 0  # net buy pressure


def test_queue_dynamics_detects_addition_and_removal():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids1, asks1 = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("BTCUSDT", bids1, asks1, ts=t0)

    # Best bid size grows (replenishment/addition); best ask size shrinks (depletion).
    bids2 = [(100.0, 9.0)] + bids1[1:]
    asks2 = [(100.2, 1.0)] + asks1[1:]
    m.record_snapshot("BTCUSDT", bids2, asks2, ts=t0 + 0.5)

    feats = m.compute_features("BTCUSDT")
    assert feats["bid_depth_added_1s"] > 0
    assert feats["bid_replenished_1s"] > 0
    assert feats["ask_depth_removed_1s"] > 0


def test_persistence_and_reversal_all_same_sign():
    from backend.services import microstructure_engine as m

    persistence, reversals = m._persistence_and_reversals([0.2, 0.3, 0.1, 0.25])
    assert persistence == pytest.approx(1.0)
    assert reversals == 0


def test_persistence_and_reversal_alternating():
    from backend.services import microstructure_engine as m

    persistence, reversals = m._persistence_and_reversals([0.2, -0.2, 0.2, -0.2])
    assert reversals == 3
    assert 0.0 <= persistence <= 1.0


def test_linreg_slope_increasing_series():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    points = [(t0, 0.0), (t0 + 1, 1.0), (t0 + 2, 2.0)]
    slope = m._linreg_slope(points)
    assert slope == pytest.approx(1.0, abs=1e-6)


def test_linreg_slope_needs_two_points():
    from backend.services import microstructure_engine as m

    assert m._linreg_slope([(1.0, 1.0)]) == 0.0
    assert m._linreg_slope([]) == 0.0


def test_ranking_delta_bounded_and_never_raises():
    from backend.services import microstructure_engine as m

    # No data at all -> 0.0, no exception.
    assert m.get_microstructure_ranking_delta("DOGEUSDT") == 0.0

    t0 = time.time()
    bids, asks = _book(100.0, 50.0, 100.2, 1.0)
    for i in range(20):
        b, a = _book(100.0 + i * 0.001, 50.0 + i, 100.2, max(0.1, 1.0 - i * 0.02))
        m.record_snapshot("BTCUSDT", b, a, ts=t0 + i * 0.1)
    delta = m.get_microstructure_ranking_delta("BTCUSDT")
    assert -0.03 - 1e-9 <= delta <= 0.03 + 1e-9


def test_ranking_delta_return_type_is_float_not_bool_never_a_gate():
    from backend.services import microstructure_engine as m

    delta = m.get_microstructure_ranking_delta("BTCUSDT")
    assert isinstance(delta, float)
    assert not isinstance(delta, bool)


def test_compute_features_missing_symbol_is_empty_dict():
    from backend.services import microstructure_engine as m

    assert m.compute_features("NOSUCHSYMBOL") == {}


def test_features_from_redis_used_when_local_state_empty(monkeypatch):
    from backend.services import microstructure_engine as m

    m._STATE.clear()
    monkeypatch.setattr(
        m,
        "_features_from_redis",
        lambda symbol: {"symbol": "BTC", "ofi_5s": 1.25, "data_age_sec": 0.01, "source": "redis"},
    )
    feats = m.compute_features("BTCUSDT")
    assert feats["source"] == "redis"
    assert feats["ofi_5s"] == 1.25


def test_stale_redis_features_are_not_authoritative(monkeypatch):
    from backend.services import microstructure_engine as m

    class _R:
        def hgetall(self, key):
            return {"data_age_sec": "99", "ofi_5s": "9"}

    monkeypatch.setattr("backend.config.redis_config.get_shared_redis_sync", lambda: _R())
    assert m._features_from_redis("ETHUSDT") == {}


def test_get_stats_reports_tracked_symbols():
    from backend.services import microstructure_engine as m

    bids, asks = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("BTCUSDT", bids, asks)
    stats = m.get_stats()
    assert "BTC" in stats["symbols_tracked"]
    assert stats["depth_samples"]["BTC"] >= 1


def test_binance_is_buyer_maker_true_is_aggressive_sell():
    from backend.services.microstructure_engine import aggressor_is_sell

    assert aggressor_is_sell(True) is True
    assert aggressor_is_sell(False) is False


def test_binance_aggtrade_fixture_direction():
    """Official Binance.US/com aggTrade ``m`` flag (buyer is maker)."""
    from backend.services.microstructure_engine import aggressor_is_sell

    sell_aggressor = {
        "e": "aggTrade",
        "E": 1672515782136,
        "s": "BTCUSDT",
        "a": 26129,
        "p": "0.001",
        "q": "100",
        "f": 100,
        "l": 105,
        "T": 1672515782136,
        "m": True,
    }
    buy_aggressor = dict(sell_aggressor)
    buy_aggressor["m"] = False
    assert aggressor_is_sell(bool(sell_aggressor["m"])) is True
    assert aggressor_is_sell(bool(buy_aggressor["m"])) is False


def test_redis_tape_overlay_when_local_book_has_no_trades(monkeypatch):
    from backend.services import microstructure_engine as m

    bids, asks = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("ETHUSDT", bids, asks, ts=time.time())
    monkeypatch.setattr(
        m,
        "_features_from_redis",
        lambda symbol: {
            "symbol": "ETH",
            "data_age_sec": 0.01,
            "agg_flow_imbalance_5s": 0.42,
            "trade_count_5s": 7,
            "signed_volume_5s": 1.25,
            "flow_acceleration": 0.11,
            "bid_absorption_score": 0.31,
            "ask_absorption_score": 0.0,
            "ofi_5s": 99.0,
        },
    )
    feats = m.compute_features("ETHUSDT")
    assert feats["agg_flow_imbalance_5s"] == 0.42
    assert feats["trade_count_5s"] == 7
    assert feats["signed_volume_5s"] == 1.25
    assert feats["flow_acceleration"] == 0.11
    assert feats["bid_absorption_score"] == 0.31
    assert feats["source"] == "local_book+redis_tape"
    assert feats["ofi_5s"] != 99.0


def test_local_trades_are_not_overwritten_by_redis(monkeypatch):
    from backend.services import microstructure_engine as m

    bids, asks = _book(100.0, 5.0, 100.2, 5.0)
    now = time.time()
    m.record_snapshot("BTCUSDT", bids, asks, ts=now)
    m.record_agg_trade("BTCUSDT", qty=3.0, is_buyer_maker=False, ts=now)
    monkeypatch.setattr(
        m,
        "_features_from_redis",
        lambda symbol: {"agg_flow_imbalance_5s": -1.0, "trade_count_5s": 99, "data_age_sec": 0.01},
    )
    feats = m.compute_features("BTCUSDT")
    assert feats["trade_count_5s"] == 1
    assert feats["agg_flow_imbalance_5s"] == pytest.approx(1.0)


def test_bid_absorption_rises_on_aggressive_sells_stable_price_replenish():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids1, asks1 = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("SOLUSDT", bids1, asks1, ts=t0)
    m.record_agg_trade("SOLUSDT", qty=2.0, is_buyer_maker=True, ts=t0 + 0.1)
    bids2, asks2 = _book(100.0, 8.0, 100.2, 5.0)
    m.record_snapshot("SOLUSDT", bids2, asks2, ts=t0 + 0.2)
    feats = m.compute_features("SOLUSDT")
    assert feats["bid_absorption_score"] > 0.0
    assert feats["ask_absorption_score"] == 0.0


def test_ask_absorption_rises_on_aggressive_buys_stable_price_replenish():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids1, asks1 = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("SOLUSDT", bids1, asks1, ts=t0)
    m.record_agg_trade("SOLUSDT", qty=2.0, is_buyer_maker=False, ts=t0 + 0.1)
    bids2, asks2 = _book(100.0, 5.0, 100.2, 8.0)
    m.record_snapshot("SOLUSDT", bids2, asks2, ts=t0 + 0.2)
    feats = m.compute_features("SOLUSDT")
    assert feats["ask_absorption_score"] > 0.0
    assert feats["bid_absorption_score"] == 0.0


def test_no_aggressive_flow_leaves_absorption_neutral():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    bids, asks = _book(100.0, 5.0, 100.2, 5.0)
    m.record_snapshot("XRPUSDT", bids, asks, ts=t0)
    m.record_snapshot("XRPUSDT", bids, asks, ts=t0 + 0.2)
    feats = m.compute_features("XRPUSDT")
    assert feats["bid_absorption_score"] == 0.0
    assert feats["ask_absorption_score"] == 0.0
    assert feats["trade_count_5s"] == 0
    assert feats["agg_flow_imbalance_5s"] == 0.0


def test_ofi_depth_normalized_xrp_comparable_to_btc():
    from backend.services import microstructure_engine as m

    t0 = time.time()
    btc1, a1 = _book(80000.0, 1.0, 80001.0, 1.0)
    btc2, a2 = _book(80000.0, 1.6, 80001.0, 0.4)
    m.record_snapshot("BTCUSDT", btc1, a1, ts=t0)
    m.record_snapshot("BTCUSDT", btc2, a2, ts=t0 + 0.1)
    x1, xa1 = _book(1.50, 5000.0, 1.51, 5000.0)
    x2, xa2 = _book(1.50, 8000.0, 1.51, 2000.0)
    m.record_snapshot("XRPUSDT", x1, xa1, ts=t0)
    m.record_snapshot("XRPUSDT", x2, xa2, ts=t0 + 0.1)
    btc = m.compute_features("BTCUSDT")["ofi_1s"]
    xrp = m.compute_features("XRPUSDT")["ofi_1s"]
    assert btc > 0 and xrp > 0
    assert abs(btc - xrp) / max(abs(btc), abs(xrp)) < 0.15
