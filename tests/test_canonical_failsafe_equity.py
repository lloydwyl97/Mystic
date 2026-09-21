"""Failsafe trips only on complete fresh NLE, never on partial cash."""

from __future__ import annotations

from backend.services.canonical_failsafe_equity import (
    balances_from_live_payload,
    build_canonical_nle,
    decide_account_failsafe,
)


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


def test_live_get_balance_payload_is_complete_nle_not_zero_cash():
    raw = {
        "status": "success",
        "balance": {
            "total": {"USDT": "221.27560086", "BTC": "0.00000907", "ETH": "0.00023132", "SOL": "0.013066", "XRP": "0.34804"},
            "free": {"USDT": "221.27560086", "BTC": "0.00000907", "ETH": "0.00023132", "SOL": "0.013066", "XRP": "0.34804"},
            "used": {"USDT": "0", "BTC": "0", "ETH": "0", "SOL": "0", "XRP": "0"},
        },
    }
    rows = balances_from_live_payload(raw)
    assert {r["asset"] for r in rows} == {"USDT", "BTC", "ETH", "SOL", "XRP"}
    snap = build_canonical_nle(balances=rows, bids=_bids(), as_of_epoch=1, now_epoch=1)
    assert snap["complete"] is True
    assert float(snap["cash_usdt"]) > 220.0
    out = decide_account_failsafe(snap, "228.07")
    assert out["tripped"] is False
    assert out["usable"] is True


def test_unparsed_live_payload_is_not_complete_zero_cash():
    rows = balances_from_live_payload({"status": "success", "balance": {"total": {}, "free": {}, "used": {}}})
    snap = build_canonical_nle(balances=rows, bids=_bids(), as_of_epoch=1, now_epoch=1)
    assert snap["complete"] is False
    assert decide_account_failsafe(snap, "228.07")["tripped"] is False
