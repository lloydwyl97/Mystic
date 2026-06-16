#!/usr/bin/env python3
"""
Controlled-risk replay search — EXHAUSTED (diagnostic only).

Set MYSTIC_FORCE_EXHAUSTED_RESEARCH=1 to re-run.
Active path: scripts/run_topfour_profit_rebuild.py
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

from backend.services.day_regime_router import DAY_REGIME_BULL
from backend.services.day_trade_thesis import SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK
from scripts.run_day_execution_replay import (
    ALLOWED_POSITIVE_BUCKETS,
    ExecutionConfig,
    STRESS_SCENARIOS,
    _build_fee_profiles,
    _run_stress_90d,
    _run_suite,
    fetch_klines_cached,
)
from scripts.run_day_profit_growth_search import _run_stress_battery
from scripts.run_day_strategy_replay import PRINCIPAL, SYMBOLS

TARGET_500 = 500.0
MAX_HOLD_H = 72.0
MAX_DD = 8.0

ATR_MULTS = (0.5, 0.75, 1.0, 1.25, 1.5)
TIME_STOPS_H = (6, 12, 24, 36, 48, 72)
PROFIT_FLOORS = (0.004, 0.006, 0.008, 0.010, 0.0125)


def _load_bars() -> tuple[dict, dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=95)
    ms0 = int(start.timestamp() * 1000)
    ms1 = int(end.timestamp() * 1000)
    bars_1h = {s: fetch_klines_cached(s, "1h", ms0, ms1) for s in SYMBOLS}
    bars_exec: dict[str, dict] = {}
    for iv in ("1m", "5m", "15m"):
        bars_exec[iv] = {s: fetch_klines_cached(s, iv, ms0, ms1) for s in SYMBOLS}
    return bars_1h, bars_exec


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


def _apply_bracket(cfg: ExecutionConfig, spec: dict) -> ExecutionConfig:
    if "notional_mult" in spec:
        cfg.notional_mult = float(spec["notional_mult"])
    if "min_net_profit_floor" in spec:
        cfg.min_net_profit_floor = float(spec["min_net_profit_floor"])
    if spec.get("controlled_exits_enabled"):
        cfg.controlled_exits_enabled = True
    if "atr_stop_mult" in spec:
        cfg.atr_stop_mult = float(spec["atr_stop_mult"])
    if "time_stop_hours" in spec:
        cfg.time_stop_hours = float(spec["time_stop_hours"])
    if "max_loss_pct" in spec:
        cfg.max_loss_pct = float(spec["max_loss_pct"])
    if spec.get("extra_allowed_buckets"):
        cfg.extra_allowed_buckets = frozenset(spec["extra_allowed_buckets"])
    if spec.get("regime_override_from_context"):
        cfg.regime_override_from_context = dict(spec["regime_override_from_context"])
    return cfg


def _profit_factor(w90: dict) -> float:
    aw = float(w90.get("average_win_usd") or 0)
    al = float(w90.get("average_loss_usd") or 0)
    wins = int(w90.get("wins") or 0)
    losses = int(w90.get("losses") or 0)
    gw = aw * wins
    gl = abs(al * losses)
    if gl < 1e-9:
        return 999.0 if gw > 0 else 0.0
    return round(gw / gl, 3)


def _evaluate_strategy(
    name: str,
    spec: dict,
    suite: dict,
    stress: dict,
) -> dict[str, Any]:
    w90 = suite["windows"]["90d"]
    wf = suite["walk_forward"]
    pc = suite["pass_criteria"]
    net90 = float(w90.get("net_pnl_usd") or 0)
    monthly = round(net90 / 3.0, 2)
    longest = float(w90.get("longest_hold_hours") or 0)
    dd = float(w90.get("max_drawdown_pct") or 99)
    stress_ok = all(
        v.get("stays_positive") for v in stress.values()
        if isinstance(v, dict) and "stays_positive" in v
    )
    fat_tail = longest > MAX_HOLD_H
    red_sells = int(w90.get("red_thesis_sell_count") or 0)
    pf = _profit_factor(w90)
    all_pass = bool(pc.get("all_pass")) and stress_ok and not fat_tail and dd < MAX_DD and red_sells == 0
    target_met = monthly >= TARGET_500

    reasons: list[str] = []
    if not pc.get("all_pass"):
        reasons.append("wf_or_window_fail")
    if not stress_ok:
        reasons.append("stress_fail")
    if fat_tail:
        reasons.append(f"fat_tail_{longest:.0f}h")
    if dd >= MAX_DD:
        reasons.append("max_dd")
    if red_sells:
        reasons.append("red_thesis_sells")
    if monthly < TARGET_500:
        reasons.append("below_500_mo")

    if all_pass and target_met:
        verdict = "accepted"
    elif all_pass:
        verdict = "safe_pass_below_target"
    else:
        verdict = "rejected"

    return {
        "strategy_name": name,
        "spec": spec,
        "trades_per_month": round(float(w90.get("total_trades") or 0) / 3.0, 2),
        "monthly_pnl_usd_on_25k": monthly,
        "pct_per_month_on_25k": round(100.0 * monthly / PRINCIPAL, 4),
        "win_rate_pct": round(float(w90.get("win_rate") or 0) * 100, 2),
        "avg_win_usd": round(float(w90.get("average_win_usd") or 0), 2),
        "avg_loss_usd": round(float(w90.get("average_loss_usd") or 0), 2),
        "profit_factor": pf,
        "expectancy_per_trade_usd": round(float(w90.get("expectancy_per_trade_usd") or 0), 2),
        "max_drawdown_pct": dd,
        "longest_hold_hours": longest,
        "worst_mae_pct": w90.get("worst_intrabar_mae_pct"),
        "wf_validation_exp": round(float(wf["validation"].get("expectancy_per_trade_usd") or 0), 2),
        "wf_test_exp": round(float(wf["test"].get("expectancy_per_trade_usd") or 0), 2),
        "all_pass": all_pass,
        "target_met_500": target_met,
        "verdict": verdict,
        "accept_or_reject_reason": "; ".join(reasons) if reasons else "pass",
        "stress_all_pass": stress_ok,
        "exit_counts_90d": {
            "net_profit": w90.get("net_profit_exit_count"),
            "extreme": w90.get("extreme_protection_count"),
        },
    }


def _quick_90(cfg: ExecutionConfig, bars_1h: dict, bars_exec: dict) -> dict:
    return _run_stress_90d(bars_1h, bars_exec, cfg)


def _run_scalp_controlled(name: str, strategy: str, profile: str, max_hold_min: int) -> dict[str, Any]:
    import os
    import subprocess

    env = os.environ.copy()
    env["SCALP_REPLAY_HOURS"] = "168"
    env["SCALP_CALIBRATION_PROFILE"] = profile
    env["SCALP_MAX_HOLD_SEC"] = str(max_hold_min * 60)
    env["SCALP_DISABLED_STRATEGIES"] = ",".join(
        s for s in ("orderbook_tape_scalp", "range_bounce_scalp", "vwap_ema_reclaim", "breakout_momentum")
        if s != strategy
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts/run_scalp_strategy_replay.py"),
            "--only-strategy",
            strategy,
            "--profile",
            profile,
            "--hours",
            "168",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
        cwd=str(REPO),
    )
    row: dict[str, Any] = {"strategy_name": name, "profile": profile, "max_hold_min": max_hold_min}
    try:
        rep = json.loads((REPO / "scripts/scalp_strategy_replay_report.json").read_text())
        n = int(rep.get("total_trades") or 0)
        net = float(rep.get("total_net_pnl_usd") or 0)
        hours = float(rep.get("replay_hours") or 168)
        wins = sum(v.get("wins", 0) for v in (rep.get("by_strategy") or {}).values())
        losses = sum(v.get("losses", 0) for v in (rep.get("by_strategy") or {}).values())
        wr = round(100.0 * wins / max(1, wins + losses), 2)
        mo_tr = round(n * (730.0 / hours), 2)
        mo_pnl = round(net * (730.0 / hours) / 3.0, 2)
        row.update({
            "trades_per_month": mo_tr,
            "monthly_pnl_usd_on_25k": mo_pnl,
            "pct_per_month_on_25k": round(100.0 * mo_pnl / PRINCIPAL, 4),
            "win_rate_pct": wr,
            "expectancy_per_trade_usd": round(net / max(1, n), 4),
            "all_pass": net >= 0 and n > 0,
            "target_met_500": mo_pnl >= TARGET_500,
            "verdict": "accepted" if net > 0 and n > 0 else "rejected",
            "accept_or_reject_reason": "negative_expectancy" if net < 0 and n > 0 else ("no_trades" if n == 0 else "pass"),
            "by_strategy": rep.get("by_strategy"),
        })
    except Exception:
        row["error"] = proc.stderr[-500:] if proc.stderr else "no report"
        row["all_pass"] = False
        row["target_met_500"] = False
        row["verdict"] = "rejected"
    return row


def main() -> int:
    from backend.services.replay_promotion_gate import block_exhausted_branch

    block_exhausted_branch("controlled_risk_bracket_exits")
    print("=== CONTROLLED-RISK REPLAY SEARCH ===", flush=True)
    profiles = _build_fee_profiles()
    bars_1h, bars_exec = _load_bars()
    results: list[dict] = []
    bull_all = frozenset(
        (s, DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK) for s in SYMBOLS
    ) | frozenset((s, DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION) for s in SYMBOLS)

    # Legacy baseline (profit-only)
    print("  legacy profit-only baseline...", flush=True)
    cfg0 = _base_cfg("legacy_profit_only_1_5x", profiles)
    cfg0.notional_mult = 1.5
    s0 = _run_suite(bars_1h, bars_exec, cfg0)
    st0 = _run_stress_battery(cfg0, bars_1h, bars_exec)
    results.append(_evaluate_strategy("legacy_profit_only_1_5x", {}, s0, st0))

    # ATR x time grid on neutral VWAP (quick 90d filter)
    print("  ATR x time sweep (90d quick)...", flush=True)
    quick_rows: list[tuple[float, dict]] = []
    for atr in ATR_MULTS:
        for th in TIME_STOPS_H:
            for floor in (0.004, 0.006, 0.008):
                spec = {
                    "controlled_exits_enabled": True,
                    "notional_mult": 1.5,
                    "atr_stop_mult": atr,
                    "time_stop_hours": th,
                    "min_net_profit_floor": floor,
                    "max_loss_pct": min(0.025, atr * 0.02 + 0.008),
                }
                label = f"vwap_bracket_atr{atr}_t{th}_f{floor}"
                cfg = _apply_bracket(_base_cfg(label, profiles), spec)
                w90 = _quick_90(cfg, bars_1h, bars_exec)
                net = float(w90.get("net_pnl_usd") or 0)
                longest = float(w90.get("longest_hold_hours") or 0)
                if longest <= MAX_HOLD_H and net > 0:
                    quick_rows.append((net, {"label": label, "spec": spec, "w90": w90}))

    quick_rows.sort(key=lambda x: x[0], reverse=True)
    top_brackets = quick_rows[:12]
    print(f"  full validation on top {len(top_brackets)} brackets...", flush=True)
    for _, item in top_brackets:
        cfg = _apply_bracket(_base_cfg(item["label"], profiles), item["spec"])
        suite = _run_suite(bars_1h, bars_exec, cfg)
        stress = _run_stress_battery(cfg, bars_1h, bars_exec)
        results.append(_evaluate_strategy(item["label"], item["spec"], suite, stress))

    # Strategy candidates B/C with mandatory 72h time stop
    candidates = [
        (
            "trending_up_pullback_bracket_72h",
            {
                "controlled_exits_enabled": True,
                "notional_mult": 1.5,
                "atr_stop_mult": 1.0,
                "time_stop_hours": 72.0,
                "min_net_profit_floor": 0.006,
                "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
                "extra_allowed_buckets": bull_all,
            },
        ),
        (
            "breakout_bracket_72h",
            {
                "controlled_exits_enabled": True,
                "notional_mult": 1.5,
                "atr_stop_mult": 0.75,
                "time_stop_hours": 48.0,
                "min_net_profit_floor": 0.008,
                "extra_allowed_buckets": frozenset(
                    (s, DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION) for s in SYMBOLS
                ),
                "regime_override_from_context": {"trending_up": DAY_REGIME_BULL},
            },
        ),
        (
            "neutral_vwap_bracket_best_floor",
            {
                "controlled_exits_enabled": True,
                "notional_mult": 1.5,
                "atr_stop_mult": 1.0,
                "time_stop_hours": 48.0,
                "min_net_profit_floor": 0.006,
            },
        ),
    ]
    for label, spec in candidates:
        print(f"  candidate {label}...", flush=True)
        cfg = _apply_bracket(_base_cfg(label, profiles), spec)
        suite = _run_suite(bars_1h, bars_exec, cfg)
        stress = _run_stress_battery(cfg, bars_1h, bars_exec)
        results.append(_evaluate_strategy(label, spec, suite, stress))

    # Scalp controlled-risk (single strategy enabled per run — breakout only first)
    print("  scalp controlled-risk samples...", flush=True)
    for strat in ("breakout_momentum",):
        for profile in ("strict", "moderate", "fast"):
            for mh in (30, 60, 90):
                row = _run_scalp_controlled(f"scalp_{strat}_{profile}_mh{mh}", strat, profile, mh)
                results.append(row)

    results.sort(key=lambda x: float(x.get("monthly_pnl_usd_on_25k") or 0), reverse=True)
    table = [
        {
            "strategy_name": r.get("strategy_name"),
            "trades_per_month": r.get("trades_per_month"),
            "monthly_pnl_usd_on_25k": r.get("monthly_pnl_usd_on_25k"),
            "pct_per_month_on_25k": r.get("pct_per_month_on_25k"),
            "win_rate_pct": r.get("win_rate_pct"),
            "avg_win_usd": r.get("avg_win_usd"),
            "avg_loss_usd": r.get("avg_loss_usd"),
            "profit_factor": r.get("profit_factor"),
            "expectancy_per_trade_usd": r.get("expectancy_per_trade_usd"),
            "max_drawdown_pct": r.get("max_drawdown_pct"),
            "longest_hold_hours": r.get("longest_hold_hours"),
            "worst_mae_pct": r.get("worst_mae_pct"),
            "all_pass": r.get("all_pass"),
            "target_met_500": r.get("target_met_500"),
            "accept_or_reject_reason": r.get("accept_or_reject_reason") or r.get("verdict"),
        }
        for r in results
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "master_rule": "profitability_first_controlled_risk",
        "live_applied": False,
        "target_monthly_usd": TARGET_500,
        "max_hold_hours_cap": MAX_HOLD_H,
        "active_buckets": [list(x) for x in sorted(ALLOWED_POSITIVE_BUCKETS)],
        "results_full": results,
        "results_table": table,
        "any_target_met_500": any(r.get("target_met_500") for r in results),
        "any_all_pass": any(r.get("all_pass") for r in results),
        "best_by_monthly_pnl": table[0] if table else None,
    }
    out = BASELINE_DIR / "controlled_risk_replay_search_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"best": table[:5], "any_target_met_500": report["any_target_met_500"], "out": str(out)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
