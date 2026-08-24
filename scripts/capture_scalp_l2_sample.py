"""Capture a bounded Binance.US depth20@100ms + aggTrade sample for replay tests."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "scalp_l2_sample.jsonl"
SYMBOLS = ("btcusdt", "ethusdt", "solusdt", "xrpusdt")
MAX_SEC = float(os.getenv("SCALP_L2_CAPTURE_SEC", "12"))
MAX_EVENTS = int(os.getenv("SCALP_L2_CAPTURE_EVENTS", "80"))


async def _run() -> int:
    import websockets

    streams = [f"{s}@depth20@100ms" for s in SYMBOLS] + [f"{s}@aggTrade" for s in SYMBOLS]
    url = "wss://stream.binance.us:9443/stream?streams=" + "/".join(streams)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    t0 = time.time()
    async with websockets.connect(url, open_timeout=20, ping_interval=20) as ws:
        with OUT.open("w", encoding="utf-8") as fh:
            async for raw in ws:
                msg = json.loads(raw)
                rec = {"ts": time.time(), "stream": msg.get("stream"), "data": msg.get("data") or {}}
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if n >= MAX_EVENTS or (time.time() - t0) >= MAX_SEC:
                    break
    print(f"wrote {n} events to {OUT}")
    return 0 if n else 1


def main() -> int:
    return asyncio.run(_run())


if __name__ == "__main__":
    raise SystemExit(main())
