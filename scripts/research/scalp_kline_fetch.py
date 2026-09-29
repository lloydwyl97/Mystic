"""Fetch closed Binance.US 1m klines (with taker-buy volume) for SCALP research.

Read-only against the venue. Output: one CSV per symbol under --out.
Columns: open_ms, open, high, low, close, volume, quote_volume, trades, taker_buy_base, taker_buy_quote.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

BASE = "https://api.binance.us/api/v3/klines"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
MINUTE_MS = 60_000


def _get(symbol: str, start_ms: int) -> list[list]:
    url = f"{BASE}?symbol={symbol}&interval=1m&startTime={start_ms}&limit=1000"
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read())
        except Exception:
            time.sleep(5.0 * (attempt + 1))
    raise RuntimeError(f"kline fetch failed {symbol} {start_ms}")


def fetch(symbol: str, start_ms: int, end_ms: int, out: Path) -> int:
    path = out / f"{symbol}_1m.csv"
    rows: dict[int, list] = {}
    if path.exists():
        with path.open() as fh:
            for r in csv.reader(fh):
                if r and r[0].isdigit():
                    rows[int(r[0])] = r
        if rows:
            start_ms = max(start_ms, max(rows) + MINUTE_MS)
    cursor = start_ms
    try:
        while cursor < end_ms:
            batch = _get(symbol, cursor)
            if not batch:
                break
            for k in batch:
                if int(k[6]) >= end_ms:
                    continue
                rows[int(k[0])] = [int(k[0]), k[1], k[2], k[3], k[4], k[5], k[7], k[8], k[9], k[10]]
            nxt = int(batch[-1][0]) + MINUTE_MS
            if nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.12)
    finally:
        _write(path, rows)
    return len(rows)


def _write(path: Path, rows: dict[int, list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["open_ms", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"])
        for key in sorted(rows):
            w.writerow(rows[key])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=180)
    ap.add_argument("--out", default="/tmp/scalp_research")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    end_ms = (int(time.time() * 1000) // MINUTE_MS) * MINUTE_MS
    start_ms = end_ms - args.days * 1440 * MINUTE_MS
    for sym in args.symbols.split(","):
        print(sym, fetch(sym, start_ms, end_ms, out), flush=True)


if __name__ == "__main__":
    main()
