#!/usr/bin/env python3
"""DAY v5 audit — bad / unsafe features only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_feature_audit import run_full_audit


async def main() -> int:
    report = await run_full_audit()
    rows = report.get("all_bad_features") or []
    unsupported = []
    for sym, coin in (report.get("symbols") or {}).items():
        for name in coin.get("unsupported_for_spot") or []:
            unsupported.append({"symbol": sym, "name": name, "status": "UNSUPPORTED_FOR_SPOT"})
    print(json.dumps({"pass": report.get("pass"), "bad_count": len(rows), "unsupported_count": len(unsupported)}, indent=2))
    for bf in rows:
        print(
            f"{bf.get('symbol'):8s} idx={bf.get('index'):3d} {bf.get('name'):32s} "
            f"{bf.get('status'):20s} val={bf.get('value')} | {bf.get('repair_recommendation')}"
        )
    if unsupported:
        print("\n--- UNSUPPORTED_FOR_SPOT (excluded from learning) ---")
        for u in unsupported:
            print(f"  {u['symbol']} {u['name']}")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
