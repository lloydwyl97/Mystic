"""Closed-bar loading must not admit a candle that has not closed."""

from __future__ import annotations

import sqlite3

from backend.services.candle_contract import INTERVAL_SEC, load_closed_bars

AS_OF = 1_700_000_000.0


def _db(tmp_path, interval: str, opens: list[float]) -> str:
    path = tmp_path / f"{interval}.db"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE feature_ohlcv (symbol TEXT, interval TEXT, ts REAL, open REAL, high REAL, low REAL, close REAL, volume REAL)")
    sec = INTERVAL_SEC[interval]
    for open_ts in opens:
        conn.execute(
            "INSERT INTO feature_ohlcv VALUES ('BTCUSDT', ?, ?, 1, 2, 0.5, 1.5, 10)",
            (interval, open_ts),
        )
    conn.commit()
    conn.close()
    assert sec > 0
    return str(path)


def test_forming_bar_is_excluded_and_exact_close_is_included(tmp_path) -> None:
    for interval in ("1m", "5m", "15m", "30m", "1h", "4h", "8h", "12h", "1d", "1w"):
        sec = INTERVAL_SEC[interval]
        closed_open = AS_OF - sec
        forming_open = AS_OF
        path = _db(tmp_path, interval, [closed_open, forming_open])
        bars = load_closed_bars(path, "BTCUSDT", interval, 10, as_of=AS_OF)
        epochs = [b["ts_epoch"] for b in bars]
        assert closed_open in epochs, interval
        assert forming_open not in epochs, interval


def test_tolerance_boundaries_stop_at_two_seconds(tmp_path) -> None:
    sec = INTERVAL_SEC["15m"]
    opens = {
        "minus_2": AS_OF - 2 - sec,
        "minus_1": AS_OF - 1 - sec,
        "exact": AS_OF - sec,
        "plus_1": AS_OF + 1 - sec,
        "plus_2": AS_OF + 2 - sec,
        "plus_3": AS_OF + 3 - sec,
        "next_candle": AS_OF,
    }
    path = _db(tmp_path, "15m", list(opens.values()))
    epochs = {b["ts_epoch"] for b in load_closed_bars(path, "BTCUSDT", "15m", 20, as_of=AS_OF)}
    for name in ("minus_2", "minus_1", "exact", "plus_1", "plus_2"):
        assert opens[name] in epochs, name
    assert opens["plus_3"] not in epochs
    assert opens["next_candle"] not in epochs


def test_two_second_close_tolerance_does_not_admit_the_next_bar(tmp_path) -> None:
    sec = INTERVAL_SEC["15m"]
    # Closes 1s after the decision: inside the 2s clock tolerance.
    early = AS_OF - sec + 1
    # Closes 3s after the decision: excluded.
    late = AS_OF - sec + 3
    path = _db(tmp_path, "15m", [early, late])
    epochs = [b["ts_epoch"] for b in load_closed_bars(path, "BTCUSDT", "15m", 10, as_of=AS_OF)]
    assert early in epochs
    assert late not in epochs
