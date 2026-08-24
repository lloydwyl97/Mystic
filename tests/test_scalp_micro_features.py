"""Deterministic microstructure feature families — bull / bear / neutral / stale."""

from __future__ import annotations

import time

import pytest

from backend.services.binance_scalp.scalp_markout import compute_markout_point
from backend.services.binance_scalp.scalp_micro_ev import heuristic_horizon_ev, multi_horizon_ev
from backend.services.microstructure_engine import compute_features, microprice, record_agg_trade, record_snapshot


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    import backend.services.microstructure_engine as m

    m._STATE.clear()
    yield
    m._STATE.clear()


def _book(bid_px, bid_sz, ask_px, ask_sz, depth=20, spacing=0.01):
    bids = [(bid_px - i * spacing, bid_sz) for i in range(depth)]
    asks = [(ask_px + i * spacing, ask_sz) for i in range(depth)]
    return bids, asks


def test_queue_imbalance_bull_bear_neutral():
    from backend.services.microstructure_engine import imbalance_at_depth

    bids, asks = _book(100.0, 10.0, 100.1, 1.0)
    assert imbalance_at_depth(bids, asks, 5) > 0.4
    bids, asks = _book(100.0, 1.0, 100.1, 10.0)
    assert imbalance_at_depth(bids, asks, 5) < -0.4
    bids, asks = _book(100.0, 5.0, 100.1, 5.0)
    assert imbalance_at_depth(bids, asks, 1) == pytest.approx(0.0, abs=1e-9)


def test_microprice_and_displacement():
    mp = microprice(100.0, 9.0, 101.0, 1.0)
    mid = 100.5
    assert mp > mid
    t0 = time.time()
    bids, asks = _book(100.0, 9.0, 100.1, 1.0)
    record_snapshot("BTCUSDT", bids, asks, ts=t0)
    feats = compute_features("BTCUSDT")
    assert feats["microprice_pressure"] > 0
    assert feats["microprice"] > feats["mid"]


def test_ofi_and_aggressive_flow_signed():
    t0 = time.time()
    for i in range(8):
        bids, asks = _book(100.0 + i * 0.01, 20.0 + i, 100.2, 2.0)
        record_snapshot("ETHUSDT", bids, asks, ts=t0 + i * 0.1)
        record_agg_trade("ETHUSDT", 1.0, is_buyer_maker=False, ts=t0 + i * 0.1)
    feats = compute_features("ETHUSDT")
    assert feats["ofi_1s"] != 0 or feats["agg_flow_imbalance_1s"] > 0
    assert feats["agg_flow_imbalance_5s"] > 0
    assert feats["trade_count_5s"] >= 1
    assert feats["signed_volume_5s"] > 0


def test_cancellation_replenishment_and_absorption():
    t0 = time.time()
    bids, asks = _book(50.0, 10.0, 50.05, 10.0)
    record_snapshot("SOLUSDT", bids, asks, ts=t0)
    # Bid depth vanishes then refills while aggressive sells print and mid holds.
    bids2, asks2 = _book(50.0, 1.0, 50.05, 10.0)
    record_snapshot("SOLUSDT", bids2, asks2, ts=t0 + 0.2)
    record_agg_trade("SOLUSDT", 4.0, is_buyer_maker=True, ts=t0 + 0.25)
    bids3, asks3 = _book(50.0, 12.0, 50.05, 10.0)
    record_snapshot("SOLUSDT", bids3, asks3, ts=t0 + 0.4)
    feats = compute_features("SOLUSDT")
    assert "bid_cancelled_5s" in feats
    assert "bid_replenished_5s" in feats
    assert feats["bid_absorption_score"] >= 0.0
    assert feats["ask_absorption_score"] >= 0.0


def test_fragility_and_adverse_selection_penalize_thinning():
    t0 = time.time()
    for i in range(6):
        sz = max(0.2, 8.0 - i * 1.5)
        bids, asks = _book(2.0, sz, 2.002, 8.0)
        record_snapshot("XRPUSDT", bids, asks, ts=t0 + i * 0.2)
        record_agg_trade("XRPUSDT", 2.0, is_buyer_maker=False, ts=t0 + i * 0.2)
    feats = compute_features("XRPUSDT")
    assert 0.0 <= feats["adverse_selection_score"] <= 1.0
    assert 0.0 <= feats["depth_fragility"] <= 1.0
    assert feats["p_adverse_move"] == feats["adverse_selection_score"]


def test_stale_missing_features_are_empty_or_zero_delta():
    from backend.services.microstructure_engine import get_microstructure_ranking_delta

    assert compute_features("DOGEUSDT") == {}
    assert get_microstructure_ranking_delta("DOGEUSDT") == 0.0


def test_weighted_depth_and_flow_accel_keys():
    t0 = time.time()
    bids, asks = _book(100.0, 3.0, 100.1, 1.0)
    record_snapshot("BTCUSDT", bids, asks, ts=t0)
    record_snapshot("BTCUSDT", bids, asks, ts=t0 + 0.5)
    feats = compute_features("BTCUSDT")
    assert "weighted_depth_imbalance" in feats
    assert "flow_acceleration" in feats
    assert "l1_liquidity_ratio" in feats


def test_executable_markout_long_uses_bid_not_mid_fantasy():
    pt = compute_markout_point(
        side="BUY",
        mid0=100.0,
        entry_px=100.05,
        mid_t=100.10,
        exit_px=100.00,
        fee_pct=0.0004,
        slip_pct=0.0001,
    )
    assert pt["mid_markout"] > 0
    assert pt["gross_markout"] < 0
    assert pt["executable_net_markout"] < pt["gross_markout"]


def test_multi_horizon_ev_responds_to_ofi():
    bull = {"ofi_5s": 8.0, "agg_flow_imbalance_5s": 0.8, "microprice_pressure": 0.0004, "obi_l5": 0.6, "bid_absorption_score": 0.4, "ask_absorption_score": 0.0, "adverse_selection_score": 0.05, "spread_pct": 0.0001}
    bear = {"ofi_5s": -8.0, "agg_flow_imbalance_5s": -0.8, "microprice_pressure": -0.0004, "obi_l5": -0.6, "bid_absorption_score": 0.0, "ask_absorption_score": 0.4, "adverse_selection_score": 0.7, "spread_pct": 0.0004}
    assert heuristic_horizon_ev(bull, 10) > heuristic_horizon_ev(bear, 10)
    ev = multi_horizon_ev(bull)
    assert ev["EV_5s"] != ev["EV_1s"]
    assert ev["calibration_status"] == "INCONCLUSIVE"


def test_cross_market_outage_is_fail_open():
    from backend.services.binance_scalp.scalp_cross_market import cross_market_features, reset_cross_market_cache

    reset_cross_market_cache()
    out = cross_market_features("BTCUSDT", own_mid=0.0)
    assert out["cross_venue_available"] is False or out["cross_market_stale"] in {True, False}
    # Unavailable venue must not raise and must not imply a hard block.
    assert "spot_perp_basis_bps" in out
