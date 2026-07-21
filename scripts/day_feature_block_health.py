#!/usr/bin/env python3
"""DAY v5 audit — per-block health rollup."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_feature_audit import run_full_audit


async def main() -> int:
    report = await run_full_audit()
    blocks: dict[str, dict[str, Any]] = {}

    for sym, coin in (report.get("symbols") or {}).items():
        if coin.get("error"):
            continue
        for f in coin.get("features") or []:
            blk = f["block"]
            b = blocks.setdefault(blk, {"features": 0, "live": 0, "calculated": 0, "proxy": 0, "bad": 0, "symbols": set()})
            b["features"] += 1
            st = f["status"]
            if st == "LIVE":
                b["live"] += 1
            elif st == "CALCULATED":
                b["calculated"] += 1
            elif st == "CALCULATED_PROXY":
                b["proxy"] += 1
            if st in ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT") or (st == "WARMUP" and f.get("is_placeholder")):
                b["bad"] += 1
                b["symbols"].add(sym)

    print(json.dumps({"pass": report.get("pass"), "blocks": len(blocks)}, indent=2))
    for blk in sorted(blocks.keys()):
        b = blocks[blk]
        syms = sorted(b["symbols"]) if b["symbols"] else []
        health = round(100.0 * (b["live"] + b["calculated"] + b["proxy"]) / max(1, b["features"]), 1)
        print(f"{blk:24s} n={b['features']:4d} live={b['live']:4d} calc={b['calculated']:4d} proxy={b['proxy']:4d} bad={b['bad']:4d} health={health}% issues={syms}")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
