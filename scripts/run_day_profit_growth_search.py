#!/usr/bin/env python3
"""
DAY profit-growth search — replay-only. Does NOT modify live rules or promote variants.

1.5× baseline remains the locked safety floor; this script searches for replay-proven growth.
"""
from __future__ import annotations

import json
import sys
import traceback
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"

from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, bucket_key
from backend.services.day_regime_router import DAY_REGIME_BEAR, DAY_REGIME_BULL
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
)
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    ExecutionConfig,
    STRESS_SCENARIOS,
    _build_fee_profiles,
    _compute_pass,
    _run_stress_90d,
    _run_suite,
    extended_discovery,
    fetch_klines_cached,
)
from scripts.run_day_strategy_replay import NOTIONAL_USD, PRINCIPAL, SYMBOLS
from scripts.run_day_bucket_discovery import _scan_opportunities

MAX_HOLD_HOURS = 72.0
TARGET_MONTHLY = {"2pct": 500.0, "3pct": 750.0, "5pct": 1250.0}
MAX_DD_CAP = 8.0


def _metrics(w90: dict[str, Any]) -> dict[str, Any]:
    return {
        "net_pnl_usd": round(float(w90.get("net_pnl_usd") or 0), 2),
        "max_drawdown_pct": w90.get("max_drawdown_pct"),
        "worst_intrabar_mae_pct": w90.get("worst_intrabar_mae_pct"),
        "longest_hold_hours": round(float(w90.get("longest_hold_hours") or 0), 2),
        "total_trades": w90.get("total_trades"),
        "expectancy_per_trade_usd": round(float(w90.get("expectancy_per_trade_usd") or 0), 2),
        "red_thesis_sell_count": w90.get("red_thesis_sell_count", 0),
        "duplicate_attempts": w90.get("duplicate_attempts", 0),
    }


def _base_cfg(name: str, profiles: dict) -> ExecutionConfig:
    p = profiles["binance_us_taker"]
    return ExecutionConfig(
        name=name,
        execution_style=p.execution_style,
        maker_fee=p.maker_fee,
        taker_fee=p.taker_fee,
        slippage_buffer=p.slippage_buffer,
        platform_spread_one_way=0.0,
        half_spread_by_symbol=deepcopy(p.half_spread_by_symbol),
        use_fill_based_exit_gate=True,
        allowed_buckets_only=True,
    )


def _apply_variant(cfg: ExecutionConfig, spec: dict[str, Any]) -> ExecutionConfig:
    if "notional_mult" in spec:
        cfg.notional_mult = float(spec["notional_mult"])
    if "min_net_profit_floor" in spec:
        cfg.min_net_profit_floor = float(spec["min_net_profit_floor"])
    if "profit_capture_mode" in spec:
        cfg.profit_capture_mode = str(spec["profit_capture_mode"])
    if "execution_style" in spec:
        cfg.execution_style = str(spec["execution_style"])
    if spec.get("extra_allowed_buckets"):
        cfg.extra_allowed_buckets = frozenset(spec["extra_allowed_buckets"])
    return cfg


def _run_stress_battery(
    cfg: ExecutionConfig,
    bars_1h: dict,
    bars_exec_by_interval: dict,
) -> dict[str, Any]:
    base = deepcopy(cfg)
    half = base.half_spread_by_symbol
    out: dict[str, Any] = {}
    for sc in STRESS_SCENARIOS:
        scfg = ExecutionConfig(
            name=f"{cfg.name}_{sc.name}",
            execution_style=base.execution_style,
            maker_fee=base.maker_fee,
            taker_fee=base.taker_fee,
            slippage_buffer=base.slippage_buffer,
            platform_spread_one_way=base.platform_spread_one_way,
            half_spread_by_symbol=half,
            slippage_mult=sc.slippage_mult,
            entry_delay_bars=sc.entry_delay_bars,
            exit_delay_bars=sc.exit_delay_bars,
            fill_model=base.fill_model,
            use_fill_based_exit_gate=True,
            notional_mult=base.notional_mult,
            min_net_profit_floor=base.min_net_profit_floor,
            profit_capture_mode=base.profit_capture_mode,
            allowed_buckets_only=base.allowed_buckets_only,
            extra_allowed_buckets=base.extra_allowed_buckets,
        )
        try:
            w90 = _run_stress_90d(bars_1h, bars_exec_by_interval, scfg)
            out[sc.name] = {
                "90d_net_pnl_usd": w90.get("net_pnl_usd"),
                "90d_expectancy_usd": w90.get("expectancy_per_trade_usd"),
                "90d_max_drawdown_pct": w90.get("max_drawdown_pct"),
                "stays_positive": (w90.get("net_pnl_usd") or 0) > 0
                and (w90.get("expectancy_per_trade_usd") or 0) > 0,
            }
        except Exception:
            out[sc.name] = {"error": traceback.format_exc(), "stays_positive": False}
    return out


def _cash_safety(notional_mult: float) -> dict[str, Any]:
    per_slot = NOTIONAL_USD * notional_mult
    max_deploy = per_slot * 4
    return {
        "notional_mult": notional_mult,
        "per_slot_notional_usd": round(per_slot, 2),
        "max_deployed_usd_4_slots": round(max_deploy, 2),
        "principal_usd": PRINCIPAL,
        "cash_capped_at_principal": max_deploy > PRINCIPAL,
        "margin_safe_no_leverage": max_deploy <= PRINCIPAL,
        "utilization_pct_of_25k": round(100.0 * min(max_deploy, PRINCIPAL) / PRINCIPAL, 2),
    }


def _evaluate(label: str, spec: dict, suite: dict, stress: dict) -> dict[str, Any]:
    w90 = suite["windows"]["90d"]
    wf = suite["walk_forward"]
    pc = suite["pass_criteria"]
    m = _metrics(w90)
    net90 = float(m["net_pnl_usd"])
    monthly = round(net90 / 3.0, 2)
    pct_month = round(100.0 * monthly / PRINCIPAL, 4)
    trades_mo = round(float(m["total_trades"] or 0) / 3.0, 2)
    longest = float(m["longest_hold_hours"] or 0)
    mult = float(spec.get("notional_mult") or 1.0)
    stress_pass = all(
        v.get("stays_positive")
        for v in stress.values()
        if isinstance(v, dict) and "stays_positive" in v
    )
    fat_tail = longest > MAX_HOLD_HOURS
    dd_ok = (m.get("max_drawdown_pct") or 99) < MAX_DD_CAP
    all_pass = bool(pc.get("all_pass")) and stress_pass and not fat_tail and dd_ok
    meets = {k: monthly >= v for k, v in TARGET_MONTHLY.items()}

    reasons: list[str] = []
    if not pc.get("all_pass"):
        reasons.append("replay_or_walk_forward_failed")
    if not stress_pass:
        reasons.append("stress_test_failed")
    if fat_tail:
        reasons.append(f"fat_tail_hold_{longest}h")
    if not dd_ok:
        reasons.append("max_drawdown_over_cap")
    if monthly < TARGET_MONTHLY["2pct"]:
        reasons.append("below_2pct_month_target")
    if all_pass and monthly >= TARGET_MONTHLY["2pct"]:
        verdict = "accepted_growth_candidate"
    elif all_pass:
        verdict = "safe_pass_below_profit_target"
    else:
        verdict = "rejected"

    return {
        "label": label,
        "spec": spec,
        "cash_safety": _cash_safety(mult),
        "metrics_90d": m,
        "metrics_30d": _metrics(suite["windows"]["30d"]),
        "wf_validation_exp": round(float(wf["validation"].get("expectancy_per_trade_usd") or 0), 2),
        "wf_test_exp": round(float(wf["test"].get("expectancy_per_trade_usd") or 0), 2),
        "pass_criteria": pc,
        "stress_all_pass": stress_pass,
        "stress_detail": stress,
        "monthly_pnl_usd_on_25k": monthly,
        "pct_per_month_on_25k": pct_month,
        "trades_per_month": trades_mo,
        "meets_targets": meets,
        "all_pass": all_pass,
        "verdict": verdict,
        "accept_or_reject_reason": "; ".join(reasons) if reasons else "all_checks_pass",
    }


def _load_bars() -> tuple[dict, dict, dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=95)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bars_1h = {sym: fetch_klines_cached(sym, "1h", start_ms, end_ms) for sym in SYMBOLS}
    bars_exec_by_interval: dict[str, dict] = {}
    for interval in ("1m", "5m", "15m"):
        bars_exec_by_interval[interval] = {
            sym: fetch_klines_cached(sym, interval, start_ms, end_ms) for sym in SYMBOLS
        }
    ext_start_ms = int((end - timedelta(days=185)).timestamp() * 1000)
    bars_ext = {sym: fetch_klines_cached(sym, "1h", ext_start_ms, end_ms) for sym in SYMBOLS}
    return bars_1h, bars_exec_by_interval, bars_ext


def _bucket_candidates(bars_ext: dict) -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    scans = [
        ("bull_trend_pullback", DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK),
        ("bull_breakout_continuation", DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION),
        ("bear_exhaustion_vwap", DAY_REGIME_BEAR, SETUP_VWAP_REVERSION),
        ("bear_reversal_breakout", DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION),
        ("neutral_breakout", "neutral", SETUP_BREAKOUT_CONTINUATION),
        ("range_vwap_sol", "SOL/USDT", "range", SETUP_VWAP_REVERSION),
    ]
    for sym in SYMBOLS:
        for reg, thesis in (
            (DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK),
            (DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION),
            (DAY_REGIME_BEAR, SETUP_VWAP_REVERSION),
            (DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION),
        ):
            scan = _scan_opportunities(bars_ext, symbol=sym, regime=reg, thesis=thesis)
            bk = bucket_key(sym, reg, thesis)
            cands.append({
                "id": f"{sym}/{reg}/{thesis}",
                "bucket": bk,
                "scan_90d": scan,
                "killed": bk in REPLAY_KILLED_BUCKETS,
                "in_allowlist": bk in ALLOWED_POSITIVE_BUCKETS,
            })
    for seg_id, sym, reg, thesis in [
        ("sol_range_vwap", "SOL/USDT", "range", SETUP_VWAP_REVERSION),
    ]:
        scan = _scan_opportunities(bars_ext, symbol=sym, regime=reg, thesis=thesis)
        bk = bucket_key(sym, reg, thesis)
        cands.append({
            "id": seg_id,
            "bucket": bk,
            "scan_90d": scan,
            "killed": bk in REPLAY_KILLED_BUCKETS,
            "in_allowlist": bk in ALLOWED_POSITIVE_BUCKETS,
        })
    return cands


def _scalp_extended() -> dict[str, Any]:
    import os
    import subprocess

    hours = int(os.getenv("SCALP_REPLAY_HOURS", "72"))
    env = os.environ.copy()
    env["SCALP_REPLAY_HOURS"] = str(hours)
    proc = subprocess.run(
        [sys.executable, str(REPO / "scripts/run_scalp_strategy_replay.py")],
        capture_output=True,
        text=True,
        env=env,
        timeout=3600,
        cwd=str(REPO),
    )
    out: dict[str, Any] = {"replay_hours": hours, "exit_code": proc.returncode}
    try:
        rep = json.loads((REPO / "scripts/scalp_strategy_replay_report.json").read_text())
        total = float(rep.get("total_net_pnl_usd") or 0)
        n = int(rep.get("total_trades") or 0)
        wins = sum(v.get("wins", 0) for v in (rep.get("by_strategy") or {}).values())
        losses = sum(v.get("losses", 0) for v in (rep.get("by_strategy") or {}).values())
        wr = round(100.0 * wins / max(1, wins + losses), 2)
        hours_f = float(rep.get("replay_hours") or hours)
        trades_mo = round(n * (730.0 / max(1.0, hours_f)), 2)
        monthly = round(total * (730.0 / max(1.0, hours_f)) / 3.0, 2)
        exp = round(total / max(1, n), 4)
        out.update({
            "total_trades": n,
            "win_rate_pct": wr,
            "net_pnl_replay_usd": round(total, 4),
            "expectancy_per_trade_usd": exp,
            "extrapolated_trades_per_month": trades_mo,
            "extrapolated_monthly_pnl_usd": monthly,
            "replay_pass": rep.get("replay_pass"),
            "enabled_strategies": rep.get("enabled_strategies"),
            "disabled_strategies": rep.get("disabled_strategies"),
            "negative_enabled_strategies": rep.get("negative_enabled_strategies"),
            "positive_strategies": rep.get("positive_strategies"),
            "missed_profitable_windows": rep.get("missed_profitable_windows"),
            "by_strategy": rep.get("by_strategy"),
            "blocker_note": (
                "Scalp uses 1m replay; extrapolation assumes steady opportunity density. "
                "Check missed_profitable_windows and disabled_strategies for blockers."
            ),
        })
    except Exception:
        out["error"] = proc.stderr[-2000:] if proc.stderr else "no report"
    return out


def main() -> int:
    print("=== DAY PROFIT GROWTH SEARCH (replay-only) ===", flush=True)
    tracebacks: list[str] = []
    try:
        profiles = _build_fee_profiles()
        bars_1h, bars_exec, bars_ext = _load_bars()
        ext_discovery = extended_discovery(bars_ext)

        variants: list[tuple[str, dict]] = [
            ("locked_floor_1_5x", {"notional_mult": 1.5}),
            ("size_2_0x", {"notional_mult": 2.0}),
            ("size_2_5x_full_account", {"notional_mult": 2.5}),
            ("size_3_0x_cash_capped", {"notional_mult": 3.0}),
            ("combo_1_5x_floor_0_60", {"notional_mult": 1.5, "min_net_profit_floor": 0.006}),
            ("combo_2_0x_floor_0_60", {"notional_mult": 2.0, "min_net_profit_floor": 0.006}),
            ("combo_2_5x_floor_0_60", {"notional_mult": 2.5, "min_net_profit_floor": 0.006}),
            ("combo_1_5x_vwap_hold", {"notional_mult": 1.5, "profit_capture_mode": "vwap_continuation"}),
            ("combo_2_0x_vwap_hold", {"notional_mult": 2.0, "profit_capture_mode": "vwap_continuation"}),
            ("combo_1_5x_maker_entry", {"notional_mult": 1.5, "execution_style": "maker_entry_taker_exit"}),
            ("combo_2_0x_maker_entry", {"notional_mult": 2.0, "execution_style": "maker_entry_taker_exit"}),
            ("combo_2_5x_maker_entry", {"notional_mult": 2.5, "execution_style": "maker_entry_taker_exit"}),
        ]

        results: list[dict] = []
        for label, spec in variants:
            print(f"  full suite: {label}...", flush=True)
            try:
                cfg = _apply_variant(_base_cfg(label, profiles), spec)
                suite = _run_suite(bars_1h, bars_exec, cfg)
                print(f"  stress: {label}...", flush=True)
                stress = _run_stress_battery(cfg, bars_1h, bars_exec)
                results.append(_evaluate(label, spec, suite, stress))
            except Exception:
                tracebacks.append(traceback.format_exc())
                results.append({"label": label, "error": traceback.format_exc(), "all_pass": False})

        bucket_rows: list[dict] = []
        for cand in _bucket_candidates(bars_ext):
            bid = cand["id"]
            scan = cand["scan_90d"]
            we = int(scan.get("would_enter") or 0)
            if cand["killed"] and not cand["in_allowlist"]:
                bucket_rows.append({
                    "id": bid,
                    "verdict": "rejected",
                    "reason": "replay_negative_killed_bucket",
                    "would_enter_90d": we,
                })
                continue
            if cand["in_allowlist"]:
                bucket_rows.append({
                    "id": bid,
                    "verdict": "already_active",
                    "would_enter_90d": we,
                })
                continue
            if we < 3:
                bucket_rows.append({
                    "id": bid,
                    "verdict": "rejected",
                    "reason": f"insufficient_opportunities_{we}",
                    "would_enter_90d": we,
                })
                continue
            print(f"  bucket expansion exec replay: {bid}...", flush=True)
            try:
                spec = {"notional_mult": 1.5, "extra_allowed_buckets": [cand["bucket"]]}
                cfg = _apply_variant(_base_cfg(f"bucket_{bid}", profiles), spec)
                suite = _run_suite(bars_1h, bars_exec, cfg)
                stress = _run_stress_battery(cfg, bars_1h, bars_exec)
                row = _evaluate(f"bucket_{bid}", spec, suite, stress)
                row["bucket"] = list(cand["bucket"])
                row["would_enter_90d"] = we
                row["trades_added_vs_locked_floor"] = (
                    row["metrics_90d"]["total_trades"]
                    - next(
                        (r["metrics_90d"]["total_trades"] for r in results if r.get("label") == "locked_floor_1_5x"),
                        0,
                    )
                )
                bucket_rows.append(row)
            except Exception:
                tracebacks.append(traceback.format_exc())
                bucket_rows.append({"id": bid, "error": traceback.format_exc(), "all_pass": False})

        print("  scalp extended replay...", flush=True)
        scalp = _scalp_extended()

        locked = next((r for r in results if r.get("label") == "locked_floor_1_5x"), {})
        passing = [r for r in results if r.get("all_pass")]
        best_cap = max(
            (r for r in results if r.get("label", "").startswith("size_")),
            key=lambda x: x.get("monthly_pnl_usd_on_25k") or 0,
            default={},
        )
        best_floor = max(
            (r for r in results if "floor_0_60" in r.get("label", "")),
            key=lambda x: x.get("monthly_pnl_usd_on_25k") or 0,
            default={},
        )
        best_combo = max(passing, key=lambda x: x.get("monthly_pnl_usd_on_25k") or 0, default={})
        best_bucket = max(
            (b for b in bucket_rows if b.get("all_pass")),
            key=lambda x: x.get("monthly_pnl_usd_on_25k") or 0,
            default={},
        )

        scalp_mo = float(scalp.get("extrapolated_monthly_pnl_usd") or 0)
        best_day_mo = float(best_combo.get("monthly_pnl_usd_on_25k") or locked.get("monthly_pnl_usd_on_25k") or 0)

        growth_table = [
            {
                "row": "locked_safety_floor_1_5x",
                **{k: locked.get(k) for k in (
                    "monthly_pnl_usd_on_25k", "pct_per_month_on_25k", "trades_per_month",
                    "metrics_90d", "all_pass", "verdict", "accept_or_reject_reason",
                )},
                "note": "Live fallback floor — too small to call finished",
            },
            {
                "row": "best_full_capital_variant",
                **{k: best_cap.get(k) for k in (
                    "label", "monthly_pnl_usd_on_25k", "pct_per_month_on_25k", "trades_per_month",
                    "metrics_90d", "cash_safety", "all_pass", "verdict", "accept_or_reject_reason",
                )},
            },
            {
                "row": "best_profit_floor_variant",
                **{k: best_floor.get(k) for k in (
                    "label", "monthly_pnl_usd_on_25k", "pct_per_month_on_25k", "trades_per_month",
                    "metrics_90d", "all_pass", "verdict", "accept_or_reject_reason",
                )},
            },
            {
                "row": "best_combined_all_pass",
                **{k: best_combo.get(k) for k in (
                    "label", "monthly_pnl_usd_on_25k", "pct_per_month_on_25k", "trades_per_month",
                    "metrics_90d", "all_pass", "verdict", "accept_or_reject_reason",
                )},
            },
            {
                "row": "best_new_bucket_expansion",
                **(
                    {k: best_bucket.get(k) for k in (
                        "id", "monthly_pnl_usd_on_25k", "trades_per_month", "metrics_90d",
                        "all_pass", "verdict", "accept_or_reject_reason",
                    )}
                    if best_bucket
                    else {"note": "none passed"}
                ),
            },
            {
                "row": "scalp_contribution",
                "monthly_pnl_usd_on_25k": scalp_mo,
                "trades_per_month": scalp.get("extrapolated_trades_per_month"),
                "win_rate_pct": scalp.get("win_rate_pct"),
                "all_pass": scalp.get("replay_pass"),
                "note": scalp.get("blocker_note"),
            },
            {
                "row": "combined_day_plus_scalp",
                "monthly_pnl_usd_on_25k": round(best_day_mo + scalp_mo, 2),
                "pct_per_month_on_25k": round(100.0 * (best_day_mo + scalp_mo) / PRINCIPAL, 4),
                "day_component": best_combo.get("label") or "locked_floor_1_5x",
                "scalp_component_usd": scalp_mo,
            },
        ]

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "mystic_finished": False,
            "locked_floor_id": "day_baseline_all_pass_v1_size_1_5",
            "profit_targets_monthly_usd": TARGET_MONTHLY,
            "profit_targets_note": "2%=$500, 3%=$750, 5%=$1250 on $25k — none met by locked floor alone",
            "active_buckets": [list(x) for x in sorted(ALLOWED_POSITIVE_BUCKETS)],
            "killed_buckets": [list(k) for k in sorted(REPLAY_KILLED_BUCKETS)],
            "capital_utilization_variants": results,
            "combination_variants": [r for r in results if "combo_" in r.get("label", "")],
            "bucket_expansion": bucket_rows,
            "extended_discovery_scan": ext_discovery,
            "scalp": scalp,
            "growth_table": growth_table,
            "passing_variant_count": len(passing),
            "any_meets_2pct_target": any(r.get("meets_targets", {}).get("2pct") for r in results),
            "tracebacks": tracebacks,
        }

        out = BASELINE_DIR / "day_profit_growth_search_latest.json"
        out.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps({"growth_table": growth_table, "passing": len(passing), "out": str(out)}, indent=2))
        return 0
    except Exception:
        tracebacks.append(traceback.format_exc())
        print(json.dumps({"error": tracebacks}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
