"""Tape-derived DAY order flow: aggressor sign, window slicing, bar alignment.

The proxy these columns replace was untrustworthy because a balanced tape and
an absent tape both read 0.0. Several tests below pin that distinction.
"""

from __future__ import annotations

import sqlite3
import time

from backend.services import day_order_flow_store as store
from backend.services import day_order_flow_tape as tape


class FakeRedis:
    """Minimal xrange/xrevrange over an in-memory stream."""

    def __init__(self, entries_by_key: dict[str, list[tuple[str, dict[str, str]]]]):
        self._entries = entries_by_key

    def xrange(self, key, start="-", end="+", count=None):
        rows = self._entries.get(key, [])
        lo = 0 if start in ("-", "") else int(str(start).split("-")[0])
        hi = float("inf") if end in ("+", "") else int(str(end).split("-")[0])
        out = [(sid, f) for sid, f in rows if lo <= int(sid.split("-")[0]) <= hi]
        return out[: int(count)] if count else out

    def xrevrange(self, key, end="+", start="-", count=None):
        out = list(reversed(self._entries.get(key, [])))
        return out[: int(count)] if count else out


def _print_entry(ts: float, price: float, qty: float, buyer_maker: bool) -> tuple[str, dict]:
    ms = int(ts * 1000)
    return (
        f"{ms}-0",
        {"s": "BTCUSDT", "a": str(ms), "p": str(price), "q": str(qty), "m": "1" if buyer_maker else "0", "T": str(ms)},
    )


def _client(prints) -> FakeRedis:
    return FakeRedis({f"{tape.TAPE_STREAM_PREFIX}BTCUSDT": list(prints)})


def test_aggressor_flag_maps_to_buy_and_sell_volume():
    """is_buyer_maker=True is an aggressive sell; False is an aggressive buy."""
    now = 1_700_000_000.0
    c = _client(
        [
            _print_entry(now - 10, 100.0, 2.0, False),  # aggressive buy
            _print_entry(now - 5, 100.0, 0.5, True),  # aggressive sell
        ]
    )
    w = tape.flow_windows("BTCUSDT", now=now, windows_sec=(60,), client=c)[60]
    assert w.trade_count == 2
    assert w.buy_qty == 2.0
    assert w.sell_qty == 0.5
    assert w.cvd_qty == 1.5
    assert w.imbalance == (2.0 - 0.5) / 2.5
    assert w.buy_notional == 200.0
    assert w.has_data


def test_empty_tape_is_distinguishable_from_balanced_tape():
    """Both read imbalance 0.0; only trade_count separates them."""
    now = 1_700_000_000.0
    empty = tape.flow_windows("BTCUSDT", now=now, windows_sec=(60,), client=_client([]))[60]
    balanced = tape.flow_windows(
        "BTCUSDT",
        now=now,
        windows_sec=(60,),
        client=_client(
            [
                _print_entry(now - 10, 100.0, 1.0, False),
                _print_entry(now - 9, 100.0, 1.0, True),
            ]
        ),
    )[60]
    assert empty.imbalance == 0.0
    assert balanced.imbalance == 0.0
    assert empty.trade_count == 0
    assert balanced.trade_count == 2
    assert not empty.has_data
    assert balanced.has_data


def test_windows_slice_by_age():
    now = 1_700_000_000.0
    c = _client(
        [
            _print_entry(now - 1800, 100.0, 5.0, True),  # only in 60m
            _print_entry(now - 600, 100.0, 3.0, False),  # in 15m and 60m
            _print_entry(now - 30, 100.0, 1.0, False),  # in all
        ]
    )
    wins = tape.flow_windows("BTCUSDT", now=now, windows_sec=(60, 900, 3600), client=c)
    assert wins[60].trade_count == 1
    assert wins[900].trade_count == 2
    assert wins[3600].trade_count == 3
    assert wins[900].cvd_qty == 4.0
    assert wins[3600].cvd_qty == -1.0


def test_prints_outside_requested_range_are_excluded():
    now = 1_700_000_000.0
    c = _client([_print_entry(now - 7200, 100.0, 9.0, False)])
    assert tape.read_tape("BTCUSDT", since_ts=now - 900, until_ts=now, client=c) == []


def test_heartbeat_fields_are_db_ready_and_flag_missing_tape():
    now = 1_700_000_000.0
    populated = tape.heartbeat_flow_fields("BTCUSDT", now=now, client=_client([_print_entry(now - 60, 100.0, 4.0, False)]))
    assert populated["flow_trades_15m"] == 1
    assert populated["flow_cvd_qty_15m"] == 4.0
    assert populated["flow_imbalance_15m"] == 1.0
    assert populated["flow_version"] == tape.FLOW_VERSION
    assert populated["flow_tape_stale_sec"] >= 0.0

    absent = tape.heartbeat_flow_fields("BTCUSDT", now=now, client=_client([]))
    assert absent["flow_version"] == 0
    assert absent["flow_trades_15m"] == 0


def test_bar_rows_align_to_the_15m_decision_grid():
    bar = 1_700_000_100 // 900 * 900
    c = _client(
        [
            _print_entry(bar + 10, 100.0, 1.0, False),
            _print_entry(bar + 800, 101.0, 1.0, False),
            _print_entry(bar + 900 + 10, 102.0, 2.0, True),  # next bar
        ]
    )
    rows = tape.bar_flow_rows("BTCUSDT", since_ts=bar - 10, until_ts=bar + 1800, client=c)
    assert len(rows) == 2
    assert rows[0]["bar_open_epoch"] == bar
    assert rows[0]["bar_open_epoch"] % 900 == 0
    assert rows[1]["bar_open_epoch"] == bar + 900
    assert rows[0]["trade_count"] == 2
    assert rows[0]["cvd_qty"] == 2.0
    assert rows[1]["cvd_qty"] == -2.0
    assert rows[0]["first_price"] == 100.0
    assert rows[0]["last_price"] == 101.0


def test_bars_without_prints_produce_no_row():
    """Absence of tape must stay absent rather than becoming a zero reading."""
    bar = 1_700_000_100 // 900 * 900
    rows = tape.bar_flow_rows("BTCUSDT", since_ts=bar, until_ts=bar + 900, client=_client([]))
    assert rows == []


def test_redis_failure_degrades_to_empty_not_exception():
    class Broken:
        def xrange(self, *a, **k):
            raise RuntimeError("redis down")

        def xrevrange(self, *a, **k):
            raise RuntimeError("redis down")

    assert tape.read_tape("BTCUSDT", since_ts=0, until_ts=time.time(), client=Broken()) == []
    fields = tape.heartbeat_flow_fields("BTCUSDT", client=Broken())
    assert fields["flow_version"] == 0


def test_upsert_is_idempotent_and_refreshes_forming_bars(tmp_path):
    db = str(tmp_path / "flow.db")
    store._tables_ready = False
    store.ensure_order_flow_tables(db)
    row = {
        "symbol": "BTCUSDT",
        "bar_open_epoch": 1_700_000_100 // 900 * 900,
        "bar_sec": 900,
        "trade_count": 1,
        "buy_qty": 1.0,
        "sell_qty": 0.0,
        "buy_notional": 100.0,
        "sell_notional": 0.0,
        "cvd_qty": 1.0,
        "cvd_notional": 100.0,
        "imbalance": 1.0,
        "notional_imbalance": 1.0,
        "vwap": 100.0,
        "first_price": 100.0,
        "last_price": 100.0,
        "first_print_epoch": 0.0,
        "last_print_epoch": 0.0,
        "coverage_sec": 0.0,
        "flow_version": 1,
    }
    assert store.upsert_bar_rows([row], db_path=db) == 1
    updated = dict(row, trade_count=5, cvd_qty=3.0)
    assert store.upsert_bar_rows([updated], db_path=db) == 1

    with sqlite3.connect(db) as conn:
        n, tc, cvd = conn.execute("SELECT COUNT(*), MAX(trade_count), MAX(cvd_qty) FROM day_order_flow_bars").fetchone()
    assert n == 1, "same bar must not duplicate"
    assert tc == 5
    assert cvd == 3.0

    summary = store.coverage_summary(db)
    assert summary[0]["symbol"] == "BTCUSDT"
    assert summary[0]["bars"] == 1
    store._tables_ready = False
