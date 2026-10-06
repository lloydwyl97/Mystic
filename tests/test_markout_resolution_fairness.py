"""Pending DAY markouts must not starve SCALP labels, and late resolves read the due minute."""

from __future__ import annotations

import sqlite3
import time

import pytest

from backend.services.adaptive_learning import ohlcv_quote, record_candidate, resolve_markouts

DAY = "DAY_V2"
SCALP = "SCALP_V2"
NOW = 1_791_000_000.0


def test_long_pending_day_rows_do_not_block_due_scalp_rows(tmp_path):
    db = str(tmp_path / "t.db")
    for i in range(260):
        record_candidate(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="neutral", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=NOW - 100.0 + i * 0.01)
    record_candidate(
        db,
        engine=SCALP,
        symbol="ETHUSDT",
        setup="VWAP_EMA_RECLAIM",
        regime="btcup_vollo",
        ref_price=100.0,
        roundtrip_cost=0.0006,
        signaled=True,
        evaluated_at=NOW - 2000.0,
        raw_expected_move=0.003,
    )
    learned = resolve_markouts(db, lambda _s, _t: 100.4, now=NOW)
    conn = sqlite3.connect(db)
    scalp = conn.execute("SELECT resolved, learned FROM adaptive_candidate_markouts WHERE engine_id=?", (SCALP,)).fetchone()
    day_pending = conn.execute("SELECT COUNT(*) FROM adaptive_candidate_markouts WHERE engine_id=? AND resolved=0", (DAY,)).fetchone()[0]
    conn.close()
    assert learned == 1
    assert scalp == (1, 1)
    assert day_pending == 260


def test_rows_owed_their_label_are_served_before_older_learned_rows(tmp_path):
    db = str(tmp_path / "t.db")
    for i in range(210):
        record_candidate(db, engine=DAY, symbol="BTCUSDT", setup="RANGE_BOUNCE", regime="neutral", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=NOW - 100.0 + i * 0.01)
    conn = sqlite3.connect(db)
    conn.execute("UPDATE adaptive_candidate_markouts SET learned=1")
    conn.commit()
    conn.close()
    record_candidate(db, engine=DAY, symbol="XRPUSDT", setup="RANGE_BOUNCE", regime="neutral", ref_price=100.0, roundtrip_cost=0.0006, signaled=True, evaluated_at=NOW - 400 * 60)
    assert resolve_markouts(db, lambda _s, _t: 103.0, now=NOW) == 1
    conn = sqlite3.connect(db)
    row = conn.execute("SELECT learned FROM adaptive_candidate_markouts WHERE symbol='XRPUSDT'").fetchone()
    conn.close()
    assert row == (1,)


def _bars(db: str, start: float, minutes: int) -> None:
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE IF NOT EXISTS feature_ohlcv (symbol TEXT, interval TEXT, ts TEXT, open REAL, high REAL, low REAL, close REAL)")
    for i in range(minutes):
        opened = start + i * 60
        ts = time.strftime("%Y-%m-%d %H:%M:%S.000000", time.gmtime(opened))
        conn.execute("INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)", ("BTC-USDT", "1m", ts, 100.0 + i, 100.0 + i, 100.0 + i, 100.0 + i))
    conn.execute(
        "INSERT INTO feature_ohlcv VALUES (?,?,?,?,?,?,?)",
        ("BTC-USDT", "15m", time.strftime("%Y-%m-%d %H:%M:%S.000000", time.gmtime(start)), 1.0, 1.0, 1.0, 999.0),
    )
    conn.commit()
    conn.close()


def test_late_quote_reads_the_minute_that_covered_the_due_time(tmp_path):
    db = str(tmp_path / "bars.db")
    start = 1_790_999_940.0
    _bars(db, start, 900)
    due = start + 5 * 60 + 30
    assert ohlcv_quote(db, "BTCUSDT", due) == pytest.approx(105.0)
    assert ohlcv_quote(db, "BTCUSDT", start + 600 * 60 + 10) == pytest.approx(700.0)


def test_missing_minute_is_not_filled_from_a_neighbour(tmp_path):
    db = str(tmp_path / "gap.db")
    start = 1_790_999_940.0
    _bars(db, start, 10)
    conn = sqlite3.connect(db)
    conn.execute("DELETE FROM feature_ohlcv WHERE interval='1m' AND close=103.0")
    conn.execute("DELETE FROM feature_ohlcv WHERE interval='15m'")
    conn.commit()
    conn.close()
    assert ohlcv_quote(db, "BTCUSDT", start + 3 * 60 + 30) is None
