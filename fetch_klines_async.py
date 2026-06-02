import argparse
import asyncio
import math
import time
from typing import Any

import aiohttp
import pandas as pd

BINANCE_US = "https://api.binance.us/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "BCHUSDT", "LTCUSDT", "AVAXUSDT", "LINKUSDT"]
INTERVAL_MS = {
    "1m": 60000,
    "3m": 180000,
    "5m": 300000,
    "15m": 900000,
    "30m": 1800000,
    "1h": 3600000,
    "2h": 7200000,
    "4h": 14400000,
    "6h": 21600000,
    "8h": 28800000,
    "12h": 43200000,
    "1d": 86400000,
    "3d": 259200000,
    "1w": 604800000,
    "1M": 2592000000,
}


class RateLimiter:
    def __init__(self, rate_per_sec: float):
        self.rate = rate_per_sec
        self.tokens = rate_per_sec
        self.updated = time.monotonic()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            self.tokens = min(self.rate, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens < 1:
                await asyncio.sleep((1 - self.tokens) / self.rate)
                self.tokens = 0
                self.updated = time.monotonic()
            else:
                self.tokens -= 1


async def fetch_chunk(session: aiohttp.ClientSession, rl: RateLimiter, symbol: str, interval: str, end_time_ms: int, limit: int = 1000, attempt: int = 1) -> list[list[Any]]:
    await rl.acquire()
    params = {"symbol": symbol, "interval": interval, "limit": str(limit), "endTime": str(end_time_ms)}
    try:
        async with session.get(BINANCE_US, params=params, timeout=aiohttp.ClientTimeout(total=30)) as r:
            if r.status in (418, 429):
                delay = min(5 * attempt, 30)
                await asyncio.sleep(delay)
                return await fetch_chunk(session, rl, symbol, interval, end_time_ms, limit, attempt + 1)
            r.raise_for_status()
            return await r.json()
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError):
        delay = min(2**attempt, 30)
        await asyncio.sleep(delay)
        return await fetch_chunk(session, rl, symbol, interval, end_time_ms, limit, attempt + 1)


def compute_end_times(total_bars: int, interval_ms: int, now_ms: int) -> list[int]:
    per = 1000
    chunks = math.ceil(total_bars / per)
    ends = []
    end = now_ms
    for _ in range(chunks):
        ends.append(end)
        end = end - per * interval_ms
        if end <= 0:
            break
    return list(reversed(ends))


def rows_to_df(rows: list[list[Any]], symbol: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume", "symbol"])
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"])
    df = df[["ts", "open", "high", "low", "close", "volume"]]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df["symbol"] = symbol
    return df.drop_duplicates(subset=["ts"]).sort_values("ts").reset_index(drop=True)


async def fetch_symbol(symbol: str, interval: str, total_bars: int, rl: RateLimiter, concurrency: int) -> pd.DataFrame:
    connector = aiohttp.TCPConnector(limit=0, ssl=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        now_ms = int(time.time() * 1000)
        interval_ms = INTERVAL_MS[interval]
        ends = compute_end_times(total_bars, interval_ms, now_ms)
        sem = asyncio.Semaphore(concurrency)

        async def task(end_ms: int) -> list[list[Any]]:
            async with sem:
                return await fetch_chunk(session, rl, symbol, interval, end_ms, 1000)

        tasks = [asyncio.create_task(task(e)) for e in ends]
        results: list[list[list[Any]]] = await asyncio.gather(*tasks)
        flat: list[list[Any]] = []
        for chunk in results:
            flat.extend(chunk)
        return rows_to_df(flat, symbol)


async def fetch_all(symbols: list[str], interval: str, total_bars: int, rps: float, per_symbol_concurrency: int) -> dict[str, pd.DataFrame]:
    rl = RateLimiter(rps)
    dfs: dict[str, pd.DataFrame] = {}

    async def run(sym: str):
        df = await fetch_symbol(sym, interval, total_bars, rl, per_symbol_concurrency)
        dfs[sym] = df

    await asyncio.gather(*(run(s) for s in symbols))
    return dfs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--timeframe", required=True)
    p.add_argument("--bars", type=int, required=True)
    p.add_argument("--rps", type=float, default=15.0)
    p.add_argument("--sym", nargs="*", default=SYMBOLS)
    p.add_argument("--concurrency", type=int, default=5)
    p.add_argument("--out", type=str, default="")
    args = p.parse_args()
    interval = args.timeframe
    if interval not in INTERVAL_MS:
        msg = f"Unsupported timeframe: {interval}"
        raise SystemExit(msg)
    dfs = asyncio.run(fetch_all(args.sym, interval, args.bars, args.rps, args.concurrency))
    if args.out:
        for s, df in dfs.items():
            path = f"{args.out.rstrip('/')}/{s}_{interval}_{args.bars}.parquet"
            df.to_parquet(path, index=False)
    else:
        totals = {s: len(df) for s, df in dfs.items()}
        print(totals)


if __name__ == "__main__":
    main()
