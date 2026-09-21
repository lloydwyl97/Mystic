"""Failsafe trips only on complete fresh NLE, never on partial cash."""

from __future__ import annotations

from backend.services.canonical_failsafe_equity import build_canonical_nle, decide_account_failsafe


def _complete_balances(usdt="220.00", btc="0.00002922"):
    return [
        {"asset": "USDT", "free": usdt, "locked": "0"},
        {"asset": "BTC", "free": btc, "locked": "0"},
        {"asset": "ETH", "free": "0.000135", "locked": "0"},
        {"asset": "SOL", "free": "0.0121544", "locked": "0"},
        {"asset": "XRP", "free": "0.34804", "locked": "0"},
    ]


def _bids():
    return {"BTCUSDT": "81702.12", "ETHUSDT": "2694.49", "SOLUSDT": "112.52", "XRPUSDT": "1.4297"}


def test_partial_cash_cannot_fire_failsafe():
    # cash-only is complete only if no other nonzero assets exist.
    # The false Ocean trip omitted assets. Incomplete = missing bids for present coins.
    omitted = build_canonical_nle(
        balances=_complete_balances(usdt="220.00"),
        bids={},
        as_of_epoch=1e12,
        now_epoch=1e12,
    )
    assert omitted["complete"] is False
    out = decide_account_failsafe(omitted, "228.54")
    assert out["tripped"] is False
    assert out["reason"] == "incomplete_nle_cannot_compare"


def test_stale_nle_cannot_fire():
    snap = build_canonical_nle(
        balances=_complete_balances(),
        bids=_bids(),
        as_of_epoch=100.0,
        now_epoch=400.0,
    )
    assert snap["stale"] is True
    out = decide_account_failsafe(snap, "228.54")
    assert out["tripped"] is False
    assert out["reason"] == "stale_nle_cannot_compare"


def test_complete_fresh_nle_can_trip_genuine_failsafe():
    snap = build_canonical_nle(
        balances=_complete_balances(usdt="160.00"),
        bids=_bids(),
        as_of_epoch=1000.0,
        now_epoch=1001.0,
    )
    assert snap["usable"] is True
    out = decide_account_failsafe(snap, "228.54")
    assert out["tripped"] is True
    assert float(out["nle"]) < 228.54 * 0.90


def test_complete_fresh_healthy_nle_does_not_trip():
    snap = build_canonical_nle(
        balances=_complete_balances(usdt="219.293"),
        bids=_bids(),
        as_of_epoch=1000.0,
        now_epoch=1001.0,
    )
    out = decide_account_failsafe(snap, "228.54")
    assert out["tripped"] is False
    assert float(out["nle"]) > 220.0
    assert snap["reservations_ignored"] == "0"


def test_reservations_do_not_reduce_equity():
    a = build_canonical_nle(balances=_complete_balances(), bids=_bids(), reservations=80, as_of_epoch=1, now_epoch=1)
    b = build_canonical_nle(balances=_complete_balances(), bids=_bids(), reservations=0, as_of_epoch=1, now_epoch=1)
    assert a["net_liquidatable_equity"] == b["net_liquidatable_equity"]
