#!/usr/bin/env python3
"""Sample live Binance.US REST depth spreads for scalp product symbols."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config  # noqa: E402
from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402

SAMPLES = 24
INTERVAL_SEC = 3.0


def depth_spread(symbol: str) -> dict | None:
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "12", f"https://api.binance.us/api/v3/depth?symbol={symbol}&limit=20"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    d = json.loads(proc.stdout)
    if not d.get("bids") or not d.get("asks"):
        return None
    bid = float(d["bids"][0][0])
    ask = float(d["asks"][0][0])
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return None
    return {
        "bid": bid,
        "ask": ask,
        "spread_mid_pct": (ask - bid) / mid,
        "spread_bid_pct": (ask - bid) / bid,
    }


def main() -> int:
    cfg = get_scalp_config()
    econ = ScalpEconomics.from_env()
    cap = econ.spread_cap_pct
    symbols = list(cfg.products)

    samples: dict[str, list[float]] = {s: [] for s in symbols}
    for _ in range(SAMPLES):
        for sym in symbols:
            row = depth_spread(sym)
            if row:
                samples[sym].append(row["spread_mid_pct"])
        time.sleep(INTERVAL_SEC)

    report: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "samples_per_symbol": SAMPLES,
        "interval_sec": INTERVAL_SEC,
        "spread_cap_pct": cap,
        "products": symbols,
        "by_symbol": {},
    }
    for sym in symbols:
        vals = samples[sym]
        if not vals:
            report["by_symbol"][sym] = {"error": "no_samples"}
            continue
        under = sum(1 for v in vals if v <= cap)
        report["by_symbol"][sym] = {
            "min_pct": round(min(vals) * 100, 4),
            "median_pct": round(statistics.median(vals) * 100, 4),
            "mean_pct": round(statistics.mean(vals) * 100, 4),
            "p75_pct": round(statistics.quantiles(vals, n=4)[2] * 100, 4)
            if len(vals) >= 4
            else round(max(vals) * 100, 4),
            "max_pct": round(max(vals) * 100, 4),
            "under_cap_pct": round(100 * under / len(vals), 1),
            "cap_realistic": under >= len(vals) * 0.5,
            "often_exceeds_cap": statistics.mean(vals) > cap,
        }

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
