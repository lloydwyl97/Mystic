#!/usr/bin/env python3
"""
Top-four capital scaling — execution replay only.

Scales notional on locked neutral-VWAP baseline (no new entries, no BNB).
Live floor unchanged until replay_promotion_gate passes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
OUT = BASELINE_DIR / "topfour_capital_scaling_latest.json"

from backend.services.replay_promotion_gate import (
    RESEARCH_EXHAUSTED,
    evaluate_day_promotion,
)
from scripts.run_day_execution_replay import _build_fee_profiles
from scripts.run_day_profit_growth_search import TARGET_MONTHLY
from scripts.run_frequency_first_rebuild import (
    _load_exec_bars,
    _run_day_candidate,
)

TARGET_500 = TARGET_MONTHLY["2pct"]

CANDIDATES = [
    ("locked_baseline_1_5x", {"notional_mult": 1.5}),
    ("capital_2_0x", {"notional_mult": 2.0}),
    ("capital_2_5x_full_account", {"notional_mult": 2.5}),
    ("capital_3_0x_cash_capped", {"notional_mult": 3.0}),
    ("capital_3_5x_cash_capped", {"notional_mult": 3.5}),
    ("capital_4_0x_cash_capped", {"notional_mult": 4.0}),
    ("maker_preferred_2_0x", {"notional_mult": 2.0, "execution_style": "maker_preferred"}),
    ("maker_preferred_2_5x", {"notional_mult": 2.5, "execution_style": "maker_preferred"}),
    ("maker_preferred_3_0x", {"notional_mult": 3.0, "execution_style": "maker_preferred"}),
]


def main() -> int:
    print("=== TOP-FOUR CAPITAL SCALING (execution replay) ===", flush=True)
    profiles = _build_fee_profiles()
    print("  loading bars...", flush=True)
    bars_1h, bars_exec = _load_exec_bars()

    results: list[dict] = []
    for label, spec in CANDIDATES:
        print(f"  replay {label}...", flush=True)
        row = _run_day_candidate(label, spec, bars_1h, bars_exec, profiles)
        row["execution_replay_verified"] = "error" not in row
        promoted, reasons = evaluate_day_promotion(
            row,
            stress_pass=bool(row.get("stress_all_pass")),
            walk_forward_test_pass=bool(row.get("all_pass")),
            walk_forward_val_pass=bool(row.get("all_pass")),
            execution_replay_verified=row["execution_replay_verified"],
        )
        row["promotion_accepted"] = promoted
        row["accept_or_reject_reason"] = reasons if reasons else "pass"
        results.append(row)

    passing = [r for r in results if r.get("all_pass")]
    best = max(results, key=lambda r: float(r.get("monthly_pnl_usd_on_25k") or -1e9))
    locked = next((r for r in results if r.get("label") == "locked_baseline_1_5x"), {})
    best_pass = max(passing, key=lambda r: float(r.get("monthly_pnl_usd_on_25k") or -1e9)) if passing else None

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_type": "topfour_capital_scaling",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "live_promoted": False,
        "target_monthly_usd": TARGET_500,
        "research_note": ("Linear notional scaling on locked neutral-VWAP only. Top-four caps concurrent slots at 4 symbols; scaling is per-slot notional_mult."),
        "research_exhausted": sorted(RESEARCH_EXHAUSTED),
        "excluded_from_research": ["BNB/USD", "BNB/USDT", "scalp_profit_contribution"],
        "candidates": results,
        "best_candidate": best,
        "best_all_pass_candidate": best_pass,
        "summary_table": [
            {
                "label": r.get("label"),
                "notional_mult": (r.get("spec") or {}).get("notional_mult"),
                "utilization_pct": (r.get("cash_safety") or {}).get("utilization_pct_of_25k"),
                "monthly_pnl_usd_on_25k": r.get("monthly_pnl_usd_on_25k"),
                "trades_per_month": r.get("trades_per_month"),
                "expectancy_per_trade_usd": r.get("expectancy_per_trade_usd"),
                "max_drawdown_pct": (r.get("metrics_90d") or {}).get("max_drawdown_pct"),
                "all_pass": r.get("all_pass"),
                "target_met_500": r.get("target_met_500", False),
            }
            for r in results
        ],
        "verdict": {
            "target_met_by_any": any(r.get("target_met_500") for r in results),
            "promotion_accepted_any": any(r.get("promotion_accepted") for r in results),
            "locked_baseline_monthly_usd": locked.get("monthly_pnl_usd_on_25k"),
            "best_monthly_usd": best.get("monthly_pnl_usd_on_25k"),
            "best_all_pass_label": best_pass.get("label") if best_pass else None,
            "best_all_pass_monthly_usd": best_pass.get("monthly_pnl_usd_on_25k") if best_pass else None,
            "gap_to_500_usd": round(TARGET_500 - float(best.get("monthly_pnl_usd_on_25k") or 0), 2),
        },
    }

    OUT.write_text(json.dumps(report, indent=2))
    print(f"  wrote {OUT}", flush=True)
    print(
        f"  locked={locked.get('monthly_pnl_usd_on_25k')} best={best.get('monthly_pnl_usd_on_25k')} target_met={report['verdict']['target_met_by_any']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
