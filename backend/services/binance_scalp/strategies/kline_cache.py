"""Lightweight 1m kline cache for paper scalp strategies."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime, timezone, timedelta


def fetch_1m_bars(symbol: str, *, minutes: int = 30) -> list[dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(minutes=minutes)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    url = (
        f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "20", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows = json.loads(proc.stdout)
    return [
        {
            "ts_ms": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
            "volume": float(r[5]),
        }
        for r in rows
    ]


class KlineCache:
    def __init__(self, *, ttl_sec: float = 45.0) -> None:
        self._ttl = ttl_sec
        self._cache: dict[str, tuple[float, list[dict]]] = {}

    def get(self, symbol: str, *, minutes: int = 30) -> list[dict]:
        sym = symbol.upper()
        now = time.time()
        hit = self._cache.get(sym)
        if hit and (now - hit[0]) < self._ttl:
            return hit[1]
        bars = fetch_1m_bars(sym, minutes=minutes)
        self._cache[sym] = (now, bars)
        return bars
