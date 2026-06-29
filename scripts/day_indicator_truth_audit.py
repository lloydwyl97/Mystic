#!/usr/bin/env python3
"""Full DAY v5 indicator truth audit — 145 features x BTC/ETH/SOL/XRP."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.day_indicator_truth_audit import COIN_LABELS, run_indicator_truth_audit


def _print_table(title: str, rows: list[dict], cols: list[str], limit: int | None = None) -> None:
    print(f"\n{'=' * 20} {title} ({len(rows)} rows) {'=' * 20}")
    if not rows:
        print("  (none)")
        return
    show = rows if limit is None else rows[:limit]
    hdr = " | ".join(cols)
    print(hdr)
    print("-" * min(160, len(hdr)))
    for r in show:
        parts = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, float):
                parts.append(f"{v:.6f}" if abs(v) < 1000 else f"{v:.4g}")
            elif isinstance(v, bool):
                parts.append("Y" if v else "N")
            elif isinstance(v, dict):
                parts.append(json.dumps(v, separators=(",", ":"))[:40])
            else:
                parts.append(str(v)[:36] if v is not None else "")
        print(" | ".join(parts))
    if limit and len(rows) > limit:
        print(f"  ... {len(rows) - limit} more rows (see JSON artifact)")


def _print_full_table(rows: list[dict]) -> None:
    cols = [
        "index",
        "feature_name",
        "feature_block",
        "BTC",
        "ETH",
        "SOL",
        "XRP",
        "status",
        "trust_score",
        "learning_allowed",
        "used_by_rf_model",
        "used_by_ranker",
        "used_by_setup_score",
        "used_by_execution_score",
        "used_by_learning",
        "bounded",
        "rank_delta_cap",
        "can_affect_final_selection_score",
        "can_receive_positive_learning_credit",
        "needs_fix",
        "needs_adjustment",
    ]
    _print_table("FULL 145-FEATURE TABLE", rows, cols, limit=None)


async def main() -> int:
    report = await run_indicator_truth_audit()
    out_path = ROOT / "scripts" / "replay_baselines" / "day_indicator_truth_audit_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    slim = {k: v for k, v in report.items() if k != "per_coin_reports"}
    out_path.write_text(json.dumps(slim, indent=2), encoding="utf-8")

    print(json.dumps({"pass": report["pass"], "feature_version": report["feature_version"]}, indent=2))

    _print_full_table(report["features"])

    _print_table(
        "BAD / NEEDS-FIX",
        report["needs_fix"],
        ["index", "feature_name", "status", "trust_score", "learning_allowed", "reason"],
    )

    _print_table(
        "NEEDS-ADJUSTMENT",
        report["needs_adjustment"],
        ["index", "feature_name", "status", "trust_score", "reason"],
    )

    _print_table(
        "UNSUPPORTED-BUT-SAFE",
        report["unsupported_but_safe"],
        ["index", "feature_name", "status", "trust_score", "learning_allowed", "reason"],
    )

    _print_table(
        "RANK-IMPACT (direct or block)",
        report["rank_impact"],
        ["index", "feature_name", "feature_block", "status", "rank_delta_cap", "used_by_ranker"],
        limit=30,
    )
    print(f"  (total rank-impact features: {len(report['rank_impact'])})")

    _print_table(
        "LEARNING-IMPACT",
        report["learning_impact"],
        ["index", "feature_name", "status", "can_receive_positive_learning_credit", "learning_allowed"],
        limit=30,
    )
    print(f"  (total learning-impact features: {len(report['learning_impact'])})")

    _print_table(
        "EXECUTION-IMPACT",
        report["execution_impact"],
        ["index", "feature_name", "status", "freshness_age_seconds", "used_by_execution_score"],
    )

    print("\n========== FEATURE BLOCKS SUMMARY ==========")
    for blk, stats in sorted((report.get("block_summary") or {}).items()):
        print(
            f"  {blk:24s} total={stats['total']:3d} live={stats['live']:3d} calc={stats['calc']:3d} "
            f"proxy={stats['proxy']:3d} unsupported={stats['unsupported']:3d} bad={stats['bad']:3d} rank_used={stats['rank_used']:3d}"
        )

    print("\n========== PASS/FAIL SUMMARY ==========")
    for k, v in (report.get("pass_checks") or {}).items():
        print(f"  {k}: {v}")
    if report.get("fail_reasons"):
        print("  fail_reasons:")
        for fr in report["fail_reasons"]:
            print(f"    - {fr}")
    print(f"\nOVERALL: {'PASS' if report['pass'] else 'FAIL'}")
    print(f"Wrote {out_path}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
