#!/usr/bin/env python3
"""Persist tape-derived DAY order flow into day_order_flow_bars.

The Redis tape retains roughly 60 hours of prints per symbol, so a single run
backfills that whole window and periodic runs only need to overlap it. Upserts
are keyed on (symbol, bar_open_epoch), which makes overlap harmless and lets a
later run complete a bar that was still forming when it was first written.

Capture only. Nothing here influences entry, exit, sizing, or the live model.

Usage:
    python3 scripts/day_order_flow_collect.py                # default 6h window
    python3 scripts/day_order_flow_collect.py --hours 72     # full retained tape
    python3 scripts/day_order_flow_collect.py --summary-only
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.day_order_flow_store import (
    collect_symbol,
    coverage_summary,
    ensure_order_flow_tables,
)

DAY_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _iso(epoch: int) -> str:
    if not epoch:
        return "-"
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hours", type=float, default=6.0, help="tape lookback window in hours")
    ap.add_argument("--symbols", default=",".join(DAY_SYMBOLS))
    ap.add_argument("--summary-only", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args()

    ensure_order_flow_tables()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    if not args.summary_only:
        lookback = max(900.0, args.hours * 3600.0)
        now = time.time()
        total = 0
        for sym in symbols:
            written = collect_symbol(sym, lookback_sec=lookback, now=now)
            total += written
            print(f"{sym:9s} bars_written={written}")
        print(f"total_bars_written={total} lookback_hours={args.hours:.1f}")

    print("\ncoverage:")
    rows = coverage_summary()
    if not rows:
        print("  no rows yet")
        return 0
    for r in rows:
        print(
            f"  {r['symbol']:9s} bars={r['bars']:5d} prints={r['prints']:7d} "
            f"with_prints={r['bars_with_prints']:5d} "
            f"avg_imbalance={r['avg_imbalance']:+.4f} "
            f"range={_iso(r['first_bar_epoch'])} -> {_iso(r['last_bar_epoch'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
