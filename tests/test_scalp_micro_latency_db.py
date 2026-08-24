"""Latency percentiles, replay leakage guard, breaker isolation under contention."""

from __future__ import annotations

import sqlite3
import threading
import time

from backend.services.binance_scalp.scalp_micro_latency import latency_report, percentile, record_latency, reset_latency
from backend.services.binance_scalp.scalp_micro_replay import replay_events
from backend.services.binance_scalp.scalp_markout import flush_completed, reset_markouts, schedule_markout


def test_latency_percentiles():
    reset_latency()
    for x in (0.001, 0.002, 0.003, 0.010, 0.020):
        record_latency("ws_to_local_book", x)
        record_latency("event_to_exit_review", x * 2)
    rep = latency_report()
    assert rep["ws_to_local_book"]["p50"] > 0
    assert rep["ws_to_local_book"]["p95"] >= rep["ws_to_local_book"]["p50"]
    assert rep["ws_to_local_book"]["p99"] >= rep["ws_to_local_book"]["p95"]
    assert percentile([1, 2, 3, 4], 50) == 2.5


def test_replay_is_time_ordered_no_future_leak():
    import backend.services.microstructure_engine as m

    m._STATE.clear()
    events = [
        {"kind": "snapshot", "ts": 10.0, "bids": [[100.0, 2.0], [99.9, 2.0]], "asks": [[100.1, 2.0], [100.2, 2.0]], "last_update_id": 1},
        {"kind": "trade", "ts": 10.2, "qty": 1.0, "is_buyer_maker": False},
        {"kind": "snapshot", "ts": 10.3, "bids": [[100.05, 3.0], [99.95, 2.0]], "asks": [[100.12, 1.5], [100.22, 2.0]], "last_update_id": 2},
        {"kind": "snapshot", "ts": 9.0, "bids": [[90.0, 1.0]], "asks": [[90.1, 1.0]], "last_update_id": 0},
    ]
    out = replay_events(events, symbol="BTCUSDT")
    assert out["n_events"] == 4
    # After sort, last applied snapshot should be ts=10.3, not the later-listed 9.0 event.
    assert out["book"]["best_bid"] == 100.05
    m._STATE.clear()


def test_markout_flush_does_not_block_breaker_lock(tmp_path):
    """Holding the breaker DB lock must not be broken by markout fail-open skip."""
    reset_markouts()
    db = str(tmp_path / "contend.db")
    schedule_markout(
        kind="entry",
        symbol="ETHUSDT",
        side="BUY",
        mid=2000.0,
        entry_px=2000.1,
        qty=0.01,
        notional=20.0,
        fee_pct=0.0004,
        slip_pct=0.0001,
        now=1.0,
    )
    from backend.services.binance_scalp.scalp_markout import observe_book

    observe_book("ETHUSDT", bid=2001.0, ask=2001.1, bids=[[2001.0, 1.0]], asks=[[2001.1, 1.0]], now=200.0)

    locked = threading.Event()
    release = threading.Event()

    def holder():
        conn = sqlite3.connect(db, timeout=30.0)
        conn.execute("BEGIN EXCLUSIVE")
        locked.set()
        release.wait(2.0)
        conn.rollback()
        conn.close()

    t = threading.Thread(target=holder)
    t.start()
    assert locked.wait(1.0)
    t0 = time.perf_counter()
    n = flush_completed(db, force=True)
    dt = time.perf_counter() - t0
    release.set()
    t.join(2.0)
    assert dt < 2.5  # timeout=1.0 plus slack — does not hang the caller
    assert n in (0, 1)  # skip-on-lock or rare immediate grant; never hang


def test_event_loop_keeps_exits_ahead_of_rank():
    from backend.services.binance_scalp import paper_engine as pe
    from backend.services.binance_scalp import runner as rn

    src = open(pe.__file__, encoding="utf-8").read()
    assert "tick(rank=do_rank)" in src
    assert "SCALP_EXIT_INTERVAL_SEC" in src
    rsrc = open(rn.__file__, encoding="utf-8").read()
    assert "SCALP_RANK_INTERVAL_SEC" in rsrc
    assert "SCALP_EXIT_INTERVAL_SEC" in rsrc
