"""Read-only: identify the exact off-grid feature_ohlcv rows and confirm 1w alignment.

The naive `epoch % step == 0` test misreports weekly bars: the Unix epoch is a
Thursday, so a Monday-aligned 1w open is never a multiple of 604800. Check 1w
against Monday 00:00 UTC instead, and dump the genuinely off-grid 1m rows.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
import sys

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
}


def parse(v: object) -> dt.datetime | None:
    s = str(v or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> None:
    c = sqlite3.connect(DB, timeout=20)
    c.row_factory = sqlite3.Row

    print("=== 1w bars: do they land on Monday 00:00 UTC? ===")
    for r in c.execute("SELECT symbol, ts FROM feature_ohlcv WHERE interval='1w' ORDER BY symbol, ts"):
        d = parse(r["ts"])
        if d is None:
            print(f"  {r['symbol']:<10} UNPARSEABLE {r['ts']!r}")
            continue
        day = d.strftime("%a")
        ok = day == "Mon" and (d.hour, d.minute, d.second) == (0, 0, 0)
        print(f"  {r['symbol']:<10} {d:%Y-%m-%d %H:%M} {day}  monday_aligned={ok}")

    print("\n=== genuinely off-grid rows (excluding 1w) ===")
    found = 0
    for iv, step in STEP_SEC.items():
        for r in c.execute(
            "SELECT id, symbol, interval, ts, open, high, low, close, volume FROM feature_ohlcv WHERE interval=? ORDER BY ts",
            (iv,),
        ):
            d = parse(r["ts"])
            if d is None:
                print(f"  UNPARSEABLE id={r['id']} {r['symbol']} {iv} ts={r['ts']!r}")
                found += 1
                continue
            if int(d.timestamp()) % step != 0:
                found += 1
                off = int(d.timestamp()) % step
                print(f"  id={r['id']:<8} {r['symbol']:<10} {iv:<4} ts={d:%Y-%m-%d %H:%M:%S} off_by={off}s  o={r['open']} h={r['high']} l={r['low']} c={r['close']} v={r['volume']}")
    if not found:
        print("  none")
    else:
        print(f"\n  total off-grid rows = {found}")

    print("\n=== symbol naming forms present in feature_ohlcv ===")
    for r in c.execute("SELECT DISTINCT symbol FROM feature_ohlcv ORDER BY symbol"):
        print(f"  {r['symbol']!r}")

    print("\n=== invalid OHLC rows (high<low, or close outside [low,high]) ===")
    bad = c.execute("SELECT COUNT(*) FROM feature_ohlcv WHERE high < low OR close > high OR close < low OR open > high OR open < low").fetchone()[0]
    print(f"  invalid_rows={bad}")

    print("\n=== zero/null volume rows by interval ===")
    for r in c.execute(
        "SELECT interval, SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) nulls, SUM(CASE WHEN volume=0 THEN 1 ELSE 0 END) zeros, COUNT(*) n FROM feature_ohlcv GROUP BY interval ORDER BY interval"
    ):
        print(f"  {r['interval']:<4} nulls={r['nulls']:<6} zeros={r['zeros']:<7} of {r['n']}")


if __name__ == "__main__":
    main()
