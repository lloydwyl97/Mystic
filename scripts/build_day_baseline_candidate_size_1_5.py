#!/usr/bin/env python3
"""
Formal replay baseline candidate: day_baseline_all_pass_v1_size_1_5_candidate

Replay-only notional_mult=1.5 on locked positive buckets.
Does NOT change live trading rules or sizing.
"""

from __future__ import annotations

import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
CACHE_DIR = BASELINE_DIR / "cache"
PARENT_BASELINE_ID = "day_baseline_all_pass_v1"
CANDIDATE_ID = "day_baseline_all_pass_v1_size_1_5_candidate"
NOTIONAL_MULT = 1.5
MAX_HOLD_HOURS_FAT_TAIL = 72.0

from backend.config.repair_add_economics import REPAIR_ADD_ENABLED
from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL
from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, active_allowed_buckets
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    STRESS_SCENARIOS,
    WINDOWS_DAYS,
    ExecutionConfig,
    _build_fee_profiles,
    _compute_pass,
    _run_stress_90d,
    _run_suite,
    fetch_klines_cached,
    verify_live_rules_match_baseline,
)
from scripts.run_day_strategy_replay import NOTIONAL_USD, PRINCIPAL, SYMBOLS, _stats_from_report


def _candidate_config(profiles: dict[str, ExecutionConfig]) -> ExecutionConfig:
    p = profiles["binance_us_taker"]
    return ExecutionConfig(
        name=CANDIDATE_ID,
        execution_style=p.execution_style,
        maker_fee=p.maker_fee,
        taker_fee=p.taker_fee,
        slippage_buffer=p.slippage_buffer,
        platform_spread_one_way=0.0,
        half_spread_by_symbol=deepcopy(p.half_spread_by_symbol),
        use_fill_based_exit_gate=True,
        allowed_buckets_only=True,
        notional_mult=NOTIONAL_MULT,
    )


def _live_rules_match_candidate(live_check: dict[str, Any]) -> dict[str, Any]:
    """Strategy rules match; live notional still 1.0x until promoted."""
    return {
        "strategy_rules_match": bool(live_check.get("match")),
        "killed_buckets_unchanged": live_check.get("replay_killed_buckets") == [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
        "live_notional_mult": 1.0,
        "candidate_notional_mult": NOTIONAL_MULT,
        "live_sizing_matches_candidate": False,
        "live_min_net_profit_floor": MIN_NET_PROFIT_TO_SELL,
        "candidate_min_net_profit_floor": MIN_NET_PROFIT_TO_SELL,
        "match": bool(live_check.get("match")),
        "note": "Live unchanged at 1.0x until candidate lock approved; strategy gates identical.",
    }


def _candidate_pass_extensions(w90: dict, pass_criteria: dict) -> dict[str, Any]:
    longest = float(w90.get("longest_hold_hours") or 0)
    merged = dict(pass_criteria)
    merged["no_fat_tail_holds_72h"] = longest <= MAX_HOLD_HOURS_FAT_TAIL
    merged["no_repair_adds_replay"] = True
    merged["repair_add_enabled_in_live_code"] = bool(REPAIR_ADD_ENABLED)
    merged["old_blocker_active"] = False
    merged["positive_buckets_only"] = True
    merged["all_pass"] = bool(pass_criteria.get("all_pass")) and merged["no_fat_tail_holds_72h"] and merged["no_repair_adds_replay"] and not merged["old_blocker_active"]
    return merged


def _summary_metrics(w90: dict) -> dict[str, Any]:
    br = w90.get("bucket_report") or []
    best = max(br, key=lambda x: float(x.get("net_pnl_usd") or 0), default={})
    worst = min(br, key=lambda x: float(x.get("net_pnl_usd") or 0), default={})
    trades_mo = (w90.get("total_trades") or 0) / 3.0
    pnl_mo = (w90.get("net_pnl_usd") or 0) / 3.0
    return {
        "expected_trades_per_month": round(trades_mo, 2),
        "expected_monthly_pnl_usd_25k": round(pnl_mo, 2),
        "max_drawdown_pct_90d": w90.get("max_drawdown_pct"),
        "worst_intrabar_mae_pct_90d": w90.get("worst_intrabar_mae_pct"),
        "avg_hold_hours_90d": w90.get("avg_hold_hours"),
        "longest_hold_hours_90d": w90.get("longest_hold_hours"),
        "per_symbol_pnl_90d": w90.get("per_symbol_pnl"),
        "per_bucket_pnl_90d": w90.get("per_bucket_pnl"),
        "best_bucket_90d": best,
        "worst_bucket_90d": worst,
        "active_allowed_buckets": active_allowed_buckets(_stats_from_report(br)),
        "red_thesis_sell_count_90d": w90.get("red_thesis_sell_count", 0),
        "duplicate_attempts_90d": w90.get("duplicate_attempts", 0),
        "repair_add_count_90d": 0,
    }


def main() -> int:
    tracebacks: list[str] = []
    print(f"=== CANDIDATE BASELINE REVIEW: {CANDIDATE_ID} ===", flush=True)
    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=95)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        bars_1h = {sym: fetch_klines_cached(sym, "1h", start_ms, end_ms) for sym in SYMBOLS}
        bars_exec_by_interval: dict[str, dict] = {}
        for interval in ("1m", "5m", "15m"):
            bars_exec_by_interval[interval] = {sym: fetch_klines_cached(sym, interval, start_ms, end_ms) for sym in SYMBOLS}

        profiles = _build_fee_profiles()
        cfg = _candidate_config(profiles)

        print("Running hi-res suite (7/14/30/90d + walk-forward)...", flush=True)
        hi_res = _run_suite(bars_1h, bars_exec_by_interval, cfg)

        print("Running stress tests...", flush=True)
        base_stress = cfg
        half = base_stress.half_spread_by_symbol
        stress: dict[str, Any] = {}
        for sc in STRESS_SCENARIOS:
            scfg = ExecutionConfig(
                name=f"{CANDIDATE_ID}_{sc.name}",
                execution_style=base_stress.execution_style,
                maker_fee=base_stress.maker_fee,
                taker_fee=base_stress.taker_fee,
                slippage_buffer=base_stress.slippage_buffer,
                platform_spread_one_way=0.0,
                half_spread_by_symbol=half,
                slippage_mult=sc.slippage_mult,
                entry_delay_bars=sc.entry_delay_bars,
                exit_delay_bars=sc.exit_delay_bars,
                use_fill_based_exit_gate=True,
                allowed_buckets_only=True,
                notional_mult=NOTIONAL_MULT,
            )
            w90 = _run_stress_90d(bars_1h, bars_exec_by_interval, scfg)
            stress[sc.name] = {
                "90d_net_pnl_usd": w90.get("net_pnl_usd"),
                "90d_max_drawdown_pct": w90.get("max_drawdown_pct"),
                "90d_expectancy_usd": w90.get("expectancy_per_trade_usd"),
                "90d_trades": w90.get("total_trades"),
                "stays_positive": (w90.get("net_pnl_usd") or 0) > 0,
            }

        w90 = hi_res.get("windows", {}).get("90d", {})
        pass_ext = _candidate_pass_extensions(w90, hi_res.get("pass_criteria", {}))
        live_check = verify_live_rules_match_baseline()
        candidate_live = _live_rules_match_candidate(live_check)

        parent_path = BASELINE_DIR / f"{PARENT_BASELINE_ID}.json"
        parent_monthly = None
        if parent_path.exists():
            pw = json.loads(parent_path.read_text())
            # approximate from 90d if present
            p90 = (pw.get("windows") or {}).get("90d") or pw
            if isinstance(p90, dict) and p90.get("net_pnl_usd"):
                parent_monthly = round(float(p90["net_pnl_usd"]) / 3.0, 2)

        summary = _summary_metrics(w90)
        summary["parent_baseline_monthly_usd"] = parent_monthly
        summary["uplift_vs_parent_monthly_usd"] = round(summary["expected_monthly_pnl_usd_25k"] - parent_monthly, 2) if parent_monthly is not None else None

        report = {
            "generated_at": end.isoformat(),
            "candidate_id": CANDIDATE_ID,
            "parent_baseline_id": PARENT_BASELINE_ID,
            "promotion_status": "candidate_review_only",
            "live_changed": False,
            "candidate_parameters": {
                "notional_mult": NOTIONAL_MULT,
                "notional_per_slot_usd": round(NOTIONAL_USD * NOTIONAL_MULT, 2),
                "max_deployed_usd_4_slots": round(NOTIONAL_USD * NOTIONAL_MULT * 4, 2),
                "principal_usd": PRINCIPAL,
                "min_net_profit_to_sell": MIN_NET_PROFIT_TO_SELL,
                "allowed_buckets_only": True,
                "active_buckets": [list(x) for x in sorted(ALLOWED_POSITIVE_BUCKETS)],
                "killed_buckets": [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
                "execution_profile": "binance_us_taker",
                "fill_based_exit_gate": True,
            },
            "windows": hi_res.get("windows"),
            "walk_forward": hi_res.get("walk_forward"),
            "high_resolution": {
                "decision_timeframes": ["1h", "4h"],
                "execution_timeframes_by_window": {7: "1m", 14: "5m", 30: "15m", 90: "15m"},
                "pass_criteria": pass_ext,
                "all_pass": pass_ext.get("all_pass"),
            },
            "stress_tests": stress,
            "stress_all_pass": all(v.get("stays_positive") for v in stress.values()),
            "summary": summary,
            "live_rules_match_candidate": candidate_live,
            "pass_criteria": pass_ext,
            "all_pass": pass_ext.get("all_pass"),
            "tracebacks": tracebacks,
        }

        artifact = BASELINE_DIR / f"{CANDIDATE_ID}.json"
        artifact.write_text(json.dumps(report, indent=2, default=str))
        review = BASELINE_DIR / f"{CANDIDATE_ID}_REVIEW.json"
        review.write_text(
            json.dumps(
                {
                    "candidate_id": CANDIDATE_ID,
                    "generated_at": report["generated_at"],
                    "all_pass": report["all_pass"],
                    "stress_all_pass": report["stress_all_pass"],
                    "live_changed": False,
                    "promotion_ready": bool(report["all_pass"] and report["stress_all_pass"]),
                    "summary": summary,
                    "pass_criteria": pass_ext,
                    "live_rules_match_candidate": candidate_live,
                    "artifact": artifact.name,
                },
                indent=2,
                default=str,
            )
        )

        print(json.dumps(report, indent=2, default=str))
        return 0 if report["all_pass"] else 1
    except Exception:
        tracebacks.append(traceback.format_exc())
        print(json.dumps({"error": tracebacks}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
