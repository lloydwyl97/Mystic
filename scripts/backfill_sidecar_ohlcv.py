#!/usr/bin/env python3
"""Fetch Binance.US 1m klines into a sidecar SQLite. Does not touch live feature_ohlcv."""

from __future__ import annotations

import json
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
DEFAULT_DAYS = 90
DEFAULT_DB = ROOT / "data" / "sidecar_ohlcv_90d.db"


def _db_symbol(api_symbol: str) -> str:
    if api_symbol.endswith("USDT"):
        return f"{api_symbol[:-4]}-USDT"
    return api_symbol


def _fetch_page(symbol: str, start_ms: int, end_ms: int) -> list:
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={start_ms}&endTime={end_ms}&limit=1000"
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mystic-sidecar-ohlcv"})
            with urllib.request.urlopen(req, timeout=45) as resp:
                payload = json.loads(resp.read().decode())
            if isinstance(payload, list):
                return payload
            time.sleep(1.0 + attempt)
        except urllib.error.HTTPError as exc:
            if exc.code in (418, 429, 500, 502, 503, 504):
                time.sleep(2.0 * (attempt + 1))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep(1.0 + attempt)
    return []


def fetch_symbol(symbol: str, start_ms: int, end_ms: int) -> list[tuple]:
    rows: list[tuple] = []
    cursor = start_ms
    db_sym = _db_symbol(symbol)
    while cursor < end_ms:
        page = _fetch_page(symbol, cursor, end_ms)
        if not page:
            break
        for r in page:
            open_ms = int(r[0])
            ts = datetime.fromtimestamp(open_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")
            rows.append(
                (
                    db_sym,
                    "1m",
                    float(r[1]),
                    float(r[2]),
                    float(r[3]),
                    float(r[4]),
                    float(r[5]),
                    ts,
                )
            )
        last_ms = int(page[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + 60_000
        time.sleep(0.2)
    return rows


def main() -> int:
    days = DEFAULT_DAYS
    dest = DEFAULT_DB
    if len(sys.argv) > 1:
        days = int(sys.argv[1])
    if len(sys.argv) > 2:
        dest = Path(sys.argv[2])
    dest.parent.mkdir(parents=True, exist_ok=True)
    end = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    start = end - timedelta(days=days)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    if dest.exists():
        dest.unlink()
    conn = sqlite3.connect(str(dest))
    conn.execute(
        """
        CREATE TABLE feature_ohlcv (
            id INTEGER PRIMARY KEY,
            symbol TEXT NOT NULL,
            interval TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            ts TEXT NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX ix_sidecar_sym_ts ON feature_ohlcv(symbol, interval, ts)")
    total = 0
    for symbol in SYMBOLS:
        rows = fetch_symbol(symbol, start_ms, end_ms)
        conn.executemany(
            "INSERT INTO feature_ohlcv (symbol, interval, open, high, low, close, volume, ts) VALUES (?,?,?,?,?,?,?,?)",
            rows,
        )
        conn.commit()
        total += len(rows)
        print(json.dumps({"symbol": symbol, "bars": len(rows), "first": rows[0][-1] if rows else None, "last": rows[-1][-1] if rows else None}))
    conn.close()
    print(json.dumps({"db": str(dest), "days": days, "total_bars": total, "live_feature_ohlcv_untouched": True}))
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
