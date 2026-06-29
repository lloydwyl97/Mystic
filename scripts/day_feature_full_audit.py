#!/usr/bin/env python3
"""Full DAY v5 145-feature audit for BTC/ETH/SOL/XRP."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_feature_audit import format_coin_summary, run_full_audit


def _print_feature_table(coin: dict) -> None:
    feats = coin.get("features") or []
    if not feats:
        return
    print(f"\n--- {coin.get('symbol')} feature table (index name block BTC-val status source) ---")
    for f in feats:
        print(
            f"{f['index']:3d} {f['name'][:28]:28s} {f['block'][:22]:22s} "
            f"{f['value']:+.6f} {f['status']:22s} trust={f['trust_score']:.2f} "
            f"learn={f['learning_allowed']} | {f['source'][:50]}"
        )


async def main() -> int:
    report = await run_full_audit()
    print(json.dumps({"pass": report["pass"], "feature_version": report["feature_version"]}, indent=2))
    print("\n========== SUMMARY BY COIN ==========")
    for sym, coin in (report.get("symbols") or {}).items():
        print(format_coin_summary(coin))

    print("\n========== SUMMARY BY BLOCK (all coins) ==========")
    block_totals: dict[str, dict[str, int]] = {}
    for coin in (report.get("symbols") or {}).values():
        if coin.get("error"):
            continue
        for blk, stats in ((coin.get("summary") or {}).get("by_block") or {}).items():
            b = block_totals.setdefault(blk, {"total": 0, "good": 0, "bad": 0})
            b["total"] += stats.get("total", 0)
            b["good"] += stats.get("good", 0)
            b["bad"] += stats.get("bad", 0)
    for blk in (
        "basic_price",
        "technical_indicators",
        "volatility",
        "momentum",
        "trend",
        "volume_profile",
        "market_sentiment",
        "time_based",
        "advanced_ta",
        "advanced_volume",
        "microstructure",
        "context_125_145",
    ):
        st = block_totals.get(blk, {})
        print(f"  {blk:24s} total={st.get('total', 0):4d} good={st.get('good', 0):4d} bad={st.get('bad', 0):4d}")

    print("\n========== BAD FEATURE LIST ==========")
    for bf in report.get("all_bad_features") or []:
        print(
            f"  [{bf.get('symbol')}] idx={bf.get('index')} {bf.get('name')} block={bf.get('block')} "
            f"val={bf.get('value')} status={bf.get('status')} source={bf.get('source')} "
            f"repair={bf.get('repair_recommendation')}"
        )
    if not report.get("all_bad_features"):
        print("  (none — all features OK or explicitly marked UNSUPPORTED/CALCULATED_PROXY)")

    print(f"\n========== PASS/FAIL: {'PASS' if report.get('pass') else 'FAIL'} ==========")

    out_path = ROOT / "scripts" / "replay_baselines" / "day_feature_full_audit_latest.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0 if report.get("pass") else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
