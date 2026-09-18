"""Read-only probe: is the DAY decision ledger actually being written?"""

from __future__ import annotations

import datetime
import sqlite3
import sys
import time

DB = sys.argv[1] if len(sys.argv) > 1 else "mystic_trading.db"


def norm(ts: float) -> float:
    return ts / 1000.0 if ts > 2e10 else float(ts)


def main() -> None:
    c = sqlite3.connect(DB, timeout=15)
    c.row_factory = sqlite3.Row

    print("=== day_decision_records: rows per day ===")
    q = "SELECT substr(created_at,1,10) AS d, COUNT(*) AS n, SUM(CASE WHEN final_decision='arm' THEN 1 ELSE 0 END) AS arms FROM day_decision_records GROUP BY d ORDER BY d DESC LIMIT 8"
    for r in c.execute(q):
        print(f"  {r['d']}  rows={r['n']:<6} arms={r['arms']}")

    cols = [x[1] for x in c.execute("PRAGMA table_info(day_decision_records)")]
    print("\n=== schema ===")
    print("  " + ", ".join(cols))

    print("\n=== newest 8 rows ===")
    order = "rowid"
    for r in c.execute(f"SELECT * FROM day_decision_records ORDER BY {order} DESC LIMIT 8"):
        d = dict(r)
        detail = str(d.pop("detail", ""))[:180]
        print("  " + " | ".join(f"{k}={v}" for k, v in d.items() if v not in (None, "", 0)))
        if detail:
            print(f"      detail={detail}")

    print("\n=== day_decision_* tables ===")
    tabs = [x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'day_decision%'")]
    for t in tabs:
        n = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        print(f"  {t:<34} rows={n}")

    print("\n=== newest 15m candle per symbol ===")
    ocols = [x[1] for x in c.execute("PRAGMA table_info(feature_ohlcv)")]
    tcol = next((x for x in ("open_time", "bar_time", "ts", "timestamp") if x in ocols), None)
    print(f"  (feature_ohlcv time column = {tcol}; cols = {', '.join(ocols)})")
    if tcol:
        now = time.time()
        for r in c.execute(f"SELECT symbol, MAX({tcol}) AS mx FROM feature_ohlcv WHERE interval='15m' GROUP BY symbol"):
            ts = norm(r["mx"])
            stamp = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%H:%M")
            print(f"  {r['symbol']:<10} open={stamp}Z  age={int(now - ts)}s")

    utc_now = datetime.datetime.now(tz=datetime.timezone.utc)
    print("\n  now = " + utc_now.strftime("%H:%M:%S") + "Z")


if __name__ == "__main__":
    main()
