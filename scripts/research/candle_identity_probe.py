"""Read-only: does feature_ohlcv carry exchange open_time, or persist-now timestamps?

Section 6 requires one canonical candle identity: symbol + interval + open_time.
feature_ohlcv exposes only `ts`. This probe decides what `ts` actually means and
whether (symbol, interval, ts) is unique and grid-aligned.
"""

from __future__ import annotations

import sqlite3
import sys
from collections import Counter

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"

STEP_SEC = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
    "1w": 604800,
}


def to_epoch(v: object) -> float | None:
    """feature_ohlcv.ts is TEXT; accept epoch-sec, epoch-ms and ISO strings."""
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        f = float(s)
    except ValueError:
        import datetime as _dt

        try:
            return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return f / 1000.0 if f > 2e10 else f


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== feature_ohlcv schema ===")
    cols = list(c.execute("PRAGMA table_info(feature_ohlcv)"))
    for x in cols:
        print(f"  {x[1]:<12} {x[2]:<8} notnull={x[3]} default={x[4]} pk={x[5]}")

    print("\n=== indexes / uniqueness on feature_ohlcv ===")
    idx = list(c.execute("PRAGMA index_list(feature_ohlcv)"))
    if not idx:
        print("  NO INDEXES AT ALL")
    for i in idx:
        info = [r[2] for r in c.execute(f"PRAGMA index_info({i[1]})")]
        print(f"  {i[1]:<40} unique={i[2]} cols={info}")

    print("\n=== stored ts type mix (typeof) ===")
    for r in c.execute("SELECT typeof(ts) t, COUNT(*) n FROM feature_ohlcv GROUP BY t"):
        print(f"  {r['t']:<10} {r['n']}")

    print("\n=== sample raw ts values ===")
    for r in c.execute("SELECT symbol, interval, ts FROM feature_ohlcv LIMIT 5"):
        print(f"  {r['symbol']:<10} {r['interval']:<4} ts={r['ts']!r}")

    print("\n=== grid alignment: is ts an exchange open time? ===")
    print("  (aligned = ts is an exact multiple of the interval step)")
    print(f"  {'symbol':<10} {'iv':<4} {'rows':>7} {'uniq_ts':>8} {'aligned':>8} {'dupes':>6}")
    q = "SELECT DISTINCT symbol, interval FROM feature_ohlcv ORDER BY symbol, interval"
    for sym, iv in c.execute(q):
        step = STEP_SEC.get(str(iv))
        rows = [to_epoch(x[0]) for x in c.execute("SELECT ts FROM feature_ohlcv WHERE symbol=? AND interval=?", (sym, iv))]
        good = [x for x in rows if x is not None]
        uniq = len(set(good))
        aligned = sum(1 for x in good if step and abs(x) % step == 0)
        dupes = len(good) - uniq
        pct = f"{(100.0 * aligned / len(good)):.1f}%" if good else "-"
        print(f"  {sym:<10} {iv:<4} {len(rows):>7} {uniq:>8} {pct:>8} {dupes:>6}")

    print("\n=== persist-now signature: ts values landing off-grid cluster near write time ===")
    off = Counter()
    for sym, iv in c.execute(q):
        step = STEP_SEC.get(str(iv))
        if not step:
            continue
        for x in c.execute("SELECT ts FROM feature_ohlcv WHERE symbol=? AND interval=?", (sym, iv)):
            e = to_epoch(x[0])
            if e is not None and abs(e) % step != 0:
                off[(str(sym), str(iv))] += 1
    if not off:
        print("  none — every row is grid aligned")
    for (sym, iv), n in off.most_common(20):
        print(f"  {sym:<10} {iv:<4} off_grid_rows={n}")


if __name__ == "__main__":
    main()
