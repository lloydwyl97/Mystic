#!/usr/bin/env python3
"""
Top-four profit rebuild — execution replay only.

Tests outcome-driven entry filters on locked neutral-VWAP baseline.
No BNB, no scalp profit, no controlled exits, no regime remaps, no LTF patterns.
Live floor unchanged until a candidate passes replay_promotion_gate.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
OUT = BASELINE_DIR / "topfour_profit_rebuild_latest.json"

from backend.services.day_trade_thesis import SETUP_VWAP_REVERSION
from backend.services.replay_promotion_gate import (
    RESEARCH_EXHAUSTED,
    evaluate_day_promotion,
)
from scripts.run_day_execution_replay import _build_fee_profiles
from scripts.run_day_profit_growth_search import (
    TARGET_MONTHLY,
)
from scripts.run_frequency_first_rebuild import (
    _load_exec_bars,
    _run_day_candidate,
)

TARGET_500 = TARGET_MONTHLY["2pct"]

CANDIDATES = [
    ("locked_live_baseline_1_5x", {"notional_mult": 1.5}),
    ("maker_preferred_1_5x", {"notional_mult": 1.5, "execution_style": "maker_preferred"}),
    ("outcome_neutral_relvol_1_5x", {
        "notional_mult": 1.5,
        "replay_entry_filter": "outcome_neutral_relvol",
        "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
    }),
    ("outcome_neutral_vwap_near_1_5x", {
        "notional_mult": 1.5,
        "replay_entry_filter": "outcome_neutral_vwap_near",
        "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
    }),
    ("outcome_neutral_adx_calm_1_5x", {
        "notional_mult": 1.5,
        "replay_entry_filter": "outcome_neutral_adx_calm",
        "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
    }),
    ("outcome_neutral_slope_mild_1_5x", {
        "notional_mult": 1.5,
        "replay_entry_filter": "outcome_neutral_slope_mild",
        "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
    }),
    ("outcome_combined_strict_1_5x", {
        "notional_mult": 1.5,
        "replay_entry_filter": "outcome_combined_strict",
        "replay_setup_allow": frozenset({SETUP_VWAP_REVERSION}),
    }),
]


def main() -> int:
    print("=== TOP-FOUR PROFIT REBUILD (execution replay) ===", flush=True)
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

    best = max(results, key=lambda r: float(r.get("monthly_pnl_usd_on_25k") or -1e9))
    locked = next((r for r in results if r.get("label") == "locked_live_baseline_1_5x"), {})

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_type": "topfour_profit_rebuild",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "live_promoted": False,
        "target_monthly_usd": TARGET_500,
        "research_exhausted": sorted(RESEARCH_EXHAUSTED),
        "excluded_from_research": ["BNB/USD", "BNB/USDT", "scalp_profit_contribution"],
        "candidates": results,
        "best_candidate": best,
        "summary_table": [
            {
                "row": "locked_top_four_live_floor",
                "monthly_pnl_usd_on_25k": locked.get("monthly_pnl_usd_on_25k", 87.0),
                "trades_per_month": locked.get("trades_per_month", 6.7),
                "all_pass": locked.get("all_pass"),
                "target_met_500": locked.get("target_met_500", False),
            },
            {
                "row": "best_rebuild_candidate",
                "label": best.get("label"),
                "monthly_pnl_usd_on_25k": best.get("monthly_pnl_usd_on_25k"),
                "trades_per_month": best.get("trades_per_month"),
                "expectancy_per_trade_usd": best.get("expectancy_per_trade_usd"),
                "win_rate_pct": best.get("win_rate_pct"),
                "profit_factor": best.get("profit_factor"),
                "max_drawdown_pct": best.get("max_drawdown_pct"),
                "longest_hold_hours": best.get("longest_hold_hours"),
                "all_pass": best.get("all_pass"),
                "target_met_500": best.get("target_met_500"),
                "promotion_accepted": best.get("promotion_accepted"),
                "accept_or_reject_reason": best.get("accept_or_reject_reason"),
            },
        ],
        "target_met_500": bool(best.get("target_met_500")),
        "any_promoted": any(r.get("promotion_accepted") for r in results),
        "conclusion": (
            "topfour_outcome_filters_exhausted_no_promotion"
            if not any(r.get("promotion_accepted") for r in results)
            else "candidate_requires_explicit_live_authorization"
        ),
    }

    OUT.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "best": best.get("label"),
        "monthly": best.get("monthly_pnl_usd_on_25k"),
        "target_met_500": report["target_met_500"],
        "out": str(OUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
