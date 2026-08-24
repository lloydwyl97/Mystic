"""Local L2 book correctness — corrupt books must not stay authoritative."""

from __future__ import annotations

from backend.services.binance_scalp.l2_book import LocalL2Book, apply_partial_snapshot, reset_books


def _lvls(bid=100.0, ask=100.1, n=20, bsz=2.0, asz=2.0):
    bids = [[bid - i * 0.01, bsz] for i in range(n)]
    asks = [[ask + i * 0.01, asz] for i in range(n)]
    return bids, asks


def test_snapshot_bootstrap():
    b = LocalL2Book("BTC")
    bids, asks = _lvls()
    assert b.apply_snapshot(bids, asks, last_update_id=10, ts=1_000.0)
    assert b.is_authoritative(now=1_000.1)
    assert b.best_bid()[0] == 100.0
    assert b.best_ask()[0] == 100.1


def test_ordered_diff_updates():
    b = LocalL2Book("ETH")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=100, ts=1.0)
    assert b.apply_diff([[100.0, 5.0]], [[100.1, 1.0]], first_id=101, final_id=101, ts=1.1)
    assert b.bids[100.0] == 5.0
    assert b.last_update_id == 101


def test_duplicate_update_ignored():
    b = LocalL2Book("SOL")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=50, ts=1.0)
    assert not b.apply_snapshot(bids, asks, last_update_id=50, ts=1.1)
    assert b.duplicates >= 1


def test_missing_update_forces_rebuild():
    b = LocalL2Book("XRP")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=10, ts=1.0)
    assert not b.apply_diff([[100.0, 1.0]], [], first_id=20, final_id=20, ts=1.1)
    assert b.stale_reason == "sequence_gap"
    assert not b.is_authoritative(now=1.1)
    bids2, asks2 = _lvls(bid=99.5, ask=99.6)
    assert b.force_rebuild(bids2, asks2, last_update_id=21, ts=1.2)
    assert b.is_authoritative(now=1.2)
    assert b.rebuilds >= 1


def test_out_of_order_update_ignored():
    b = LocalL2Book("BTC")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=30, ts=1.0)
    assert not b.apply_diff([[100.0, 9.0]], [], first_id=20, final_id=20, ts=1.1)
    assert b.best_bid()[1] != 9.0 or b.duplicates >= 1


def test_stale_book_not_authoritative():
    b = LocalL2Book("ETH")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=1, ts=1.0)
    assert not b.is_authoritative(now=5.0)
    assert b.stale_reason == "stale"


def test_crossed_book_rejected():
    b = LocalL2Book("SOL")
    bids = [[101.0, 1.0]]
    asks = [[100.0, 1.0]]
    assert not b.apply_snapshot(bids, asks, last_update_id=1, ts=1.0)
    assert b.stale_reason == "crossed_book"
    assert not b.is_authoritative(now=1.0)


def test_empty_side_rejected():
    b = LocalL2Book("XRP")
    assert not b.apply_snapshot([[100.0, 1.0]], [], last_update_id=1, ts=1.0)
    assert b.stale_reason == "empty_side"


def test_reconnect_clears_authority():
    b = LocalL2Book("BTC")
    bids, asks = _lvls()
    b.apply_snapshot(bids, asks, last_update_id=5, ts=1.0)
    b.mark_reconnect()
    assert not b.is_authoritative(now=1.0)
    assert b.stale_reason == "reconnect"


def test_registry_reset_and_recovery():
    reset_books()
    bids, asks = _lvls()
    book = apply_partial_snapshot("BTCUSDT", bids, asks, last_update_id=7, ts=1_700.0)
    assert book.is_authoritative(now=1_700.1)
    reset_books()
    from backend.services.binance_scalp.l2_book import book_for

    assert not book_for("BTC").is_authoritative(now=1_700.1)
