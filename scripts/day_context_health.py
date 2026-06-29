#!/usr/bin/env python3
"""DAY v5 audit — context dims 125-145 only."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_feature_audit import run_full_audit

CONTEXT_BLOCK = "context_125_145"


async def main() -> int:
    report = await run_full_audit()
    ok = True
    print(json.dumps({"feature_version": report.get("feature_version"), "context_dims": 21}, indent=2))
    for sym, coin in (report.get("symbols") or {}).items():
        ctx_feats = [f for f in (coin.get("features") or []) if f.get("block") == CONTEXT_BLOCK]
        bad = [f for f in ctx_feats if f["status"] in ("FALLBACK", "MISSING", "STALE", "ZERO_DEFAULT")]
        sym_pass = len(bad) == 0 and len(ctx_feats) == 21
        ok = ok and sym_pass
        print(f"\n=== {sym} context pass={sym_pass} bundle_age={coin.get('bundle_age_sec')} ctx_age={coin.get('ctx_age_sec')} ===")
        for f in ctx_feats:
            flag = "OK" if f["status"] in ("LIVE", "CALCULATED") else f["status"]
            print(
                f"  {f['index']:3d} {f['name']:28s} {f['value']:+.6f} {flag:18s} "
                f"trust={f['trust_score']:.2f} age={f.get('age_seconds')}"
            )
    print(f"\nCONTEXT PASS/FAIL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
