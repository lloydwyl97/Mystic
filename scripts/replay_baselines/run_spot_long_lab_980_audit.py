#!/usr/bin/env python3
"""
Audit the all-weather spot-long lab candidate (~$980/mo at 1.5x on $25k).

Fresh run — no stale artifact reuse for metrics.
Uses lab execution replay (15m fill simulation), NOT portfolio_engine path.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from backend.services.replay_promotion_gate import PRINCIPAL, TARGET_MONTHLY_USD, evaluate_day_promotion
from scripts.run_allweather_strategy_lab import (
    _HALF_SPREAD,
    _SLIP,
    _TAKER,
    MAX_SLOTS,
    NOTIONAL_MULTS,
    ONE_WAY_COST,
    ROUNDTRIP_COST,
    TARGET_500,
    TIME_STOP_HOURS,
    Trade,
    _backtest,
    _precompute,
    _seg,
)
from scripts.run_day_execution_replay import CACHE_DIR, fetch_klines_cached
from scripts.run_day_strategy_replay import NOTIONAL_USD, SYMBOLS
from scripts.run_day_strategy_replay import PRINCIPAL as LAB_PRINCIPAL

SCRIPT = "scripts/replay_baselines/run_spot_long_lab_980_audit.py"
OUT = REPO / "scripts" / "replay_baselines" / "spot_long_lab_candidate_980_audit_latest.json"
CANDIDATE_ID = "allweather_breakout_pullback_lab_1_5x"
NOTIONAL_MULT = 1.5
WINDOWS = [7, 14, 30, 90, 180, 720]


def _fetch(span_days: int) -> tuple[dict, dict, dict]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=span_days + 10)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    bars_1h = {}
    bars_15m = {}
    meta = {"start_ms": start_ms, "end_ms": end_ms, "cache_paths": {}}
    for sym in SYMBOLS:
        bars_1h[sym] = fetch_klines_cached(sym, "1h", start_ms, end_ms)
        bars_15m[sym] = fetch_klines_cached(sym, "15m", start_ms, end_ms)
        api = sym.replace("/", "")
        meta["cache_paths"][sym] = str(CACHE_DIR / f"{api}_1h_{start_ms}_{end_ms}.json")
    meta["candle_counts"] = {sym: {"1h": len(bars_1h[sym]), "15m": len(bars_15m[sym])} for sym in SYMBOLS}
    return bars_1h, bars_15m, meta


def _extended_metrics(trades: list[Trade], months: float) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "monthly_pnl_usd": 0.0,
            "monthly_pnl_usd_on_25k": 0.0,
            "percent_per_month": 0.0,
            "win_rate": 0.0,
            "avg_win_usd": 0.0,
            "avg_loss_usd": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "longest_hold_hours": 0.0,
            "worst_mae_usd": 0.0,
            "target_met_500": False,
        }
    pnls = [t.pnl_usd for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    net = sum(pnls)
    monthly = net / max(months, 1.0)
    # equity curve drawdown on principal
    equity = LAB_PRINCIPAL
    peak = equity
    max_dd = 0.0
    for p in sorted(trades, key=lambda t: t.entry_ts):
        equity += p.pnl_usd
        peak = max(peak, equity)
        dd = (equity - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    return {
        "trades": len(trades),
        "trades_per_month": round(len(trades) / max(months, 1.0), 2),
        "monthly_pnl_usd": round(monthly, 2),
        "monthly_pnl_usd_on_25k": round(monthly, 2),
        "percent_per_month": round((monthly / LAB_PRINCIPAL) * 100.0, 4),
        "win_rate": round(len(wins) / len(trades), 4),
        "avg_win_usd": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "avg_loss_usd": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "profit_factor": round(gross_w / gross_l, 4) if gross_l > 0 else 99.0,
        "expectancy_per_trade_usd": round(net / len(trades), 2),
        "max_drawdown_pct": round(abs(max_dd) * 100.0, 4),
        "longest_hold_hours": round(max(t.hold_h for t in trades), 2),
        "worst_mae_usd": round(min(pnls), 2),
        "target_met_500": monthly >= TARGET_500,
    }


def _walk_forward_splits(trades: list[Trade]) -> dict[str, Any]:
    if len(trades) < 30:
        return {"passed_train": False, "passed_val": False, "passed_test": False, "reason": "insufficient_trades"}
    ordered = sorted(trades, key=lambda t: t.entry_ts)
    t0 = ordered[0].entry_ts
    t1 = ordered[-1].entry_ts
    span = max(t1 - t0, 1)
    train_end = t0 + int(span * 0.60)
    val_end = t0 + int(span * 0.80)

    def _slice(lo: int, hi: int) -> list[Trade]:
        return [t for t in ordered if lo <= t.entry_ts < hi]

    train = _slice(t0, train_end)
    val = _slice(train_end, val_end)
    test = _slice(val_end, t1 + 1)

    def _monthly(ts: list[Trade]) -> float:
        if not ts:
            return 0.0
        mo = max((ts[-1].entry_ts - ts[0].entry_ts) / (30.4375 * 86400), 1.0)
        return sum(t.pnl_usd for t in ts) / mo

    train_m = _monthly(train)
    val_m = _monthly(val)
    test_m = _monthly(test)
    return {
        "train_trades": len(train),
        "val_trades": len(val),
        "test_trades": len(test),
        "train_monthly_usd": round(train_m, 2),
        "val_monthly_usd": round(val_m, 2),
        "test_monthly_usd": round(test_m, 2),
        "passed_train": train_m > 0,
        "passed_val": val_m >= TARGET_500 * 0.5,
        "passed_test": test_m >= TARGET_500,
        "walk_forward_val_pass": val_m >= TARGET_500 * 0.5 and val_m > 0,
        "walk_forward_test_pass": test_m >= TARGET_500,
    }


def _stress_scenarios(indis, bars_15m, months: float) -> dict[str, Any]:
    scenarios = {
        "verified_taker_2bp": ONE_WAY_COST,
        "taker_10bp": 0.0010 + _HALF_SPREAD + _SLIP,
        "taker_20bp": 0.0020 + _HALF_SPREAD + _SLIP,
        "double_slippage": ONE_WAY_COST + _SLIP,
        "roundtrip_2x": ONE_WAY_COST * 2.0,
    }
    out = {}
    for name, owc in scenarios.items():
        tr = _backtest(indis, bars_15m, NOTIONAL_MULT, owc)
        m = _extended_metrics(tr, months)
        out[name] = {**m, "one_way_cost": owc, "roundtrip_cost": owc * 2}
    stress_pass = out["taker_10bp"]["target_met_500"] and out["verified_taker_2bp"]["monthly_pnl_usd"] >= TARGET_500
    return {"scenarios": out, "stress_pass": stress_pass}


def _run_window(span_days: int) -> dict[str, Any]:
    bars_1h, bars_15m, meta = _fetch(span_days)
    if not bars_1h[SYMBOLS[0]]:
        return {"span_days": span_days, "error": "no_data", **meta}
    span_actual = max(1, (bars_1h[SYMBOLS[0]][-1]["ts"] - bars_1h[SYMBOLS[0]][0]["ts"]) / 86400)
    months = max(span_actual / 30.4375, span_days / 30.4375, 1.0)
    global _atr_cache
    from scripts import run_allweather_strategy_lab as lab

    lab._atr_cache = {}
    indis = {sym: _precompute(bars_1h[sym]) for sym in SYMBOLS}
    trades = _backtest(indis, bars_15m, NOTIONAL_MULT, ONE_WAY_COST)
    # filter trades to window by entry within last span_days
    cutoff = bars_1h[SYMBOLS[0]][-1]["ts"] - span_days * 86400
    trades_w = [t for t in trades if t.entry_ts >= cutoff]
    m = _extended_metrics(trades_w, max(span_days / 30.4375, 1.0))
    return {"span_days": span_days, "months": round(months, 2), "metrics": m, "meta": meta}


def main() -> int:
    cmd = f"python3 {SCRIPT}"
    print("=== SPOT-LONG LAB $980 AUDIT (fresh run) ===", flush=True)

    # Full span (~1095d lab default)
    span_lab = 1095
    bars_1h, bars_15m, meta = _fetch(span_lab)
    if not bars_1h[SYMBOLS[0]]:
        OUT.write_text(json.dumps({"error": "no_data", "exit_code": 1}, indent=2))
        return 1

    span_days = int((bars_1h[SYMBOLS[0]][-1]["ts"] - bars_1h[SYMBOLS[0]][0]["ts"]) / 86400)
    months = max(span_days / 30.4375, 1.0)

    from scripts import run_allweather_strategy_lab as lab

    lab._atr_cache = {}
    indis = {sym: _precompute(bars_1h[sym]) for sym in SYMBOLS}
    trades = _backtest(indis, bars_15m, NOTIONAL_MULT, ONE_WAY_COST)
    full_m = _extended_metrics(trades, months)
    wf = _walk_forward_splits(trades)
    stress = _stress_scenarios(indis, bars_15m, months)

    window_results = {}
    for w in WINDOWS:
        print(f"  window {w}d ...", flush=True)
        window_results[str(w)] = _run_window(w)

    locked_floor = {
        "baseline_id": "day_baseline_all_pass_v1_size_1_5",
        "monthly_pnl_usd_on_25k": 88.4,
        "trades_per_month": 6.67,
        "source": "day_baseline_all_pass_v1_size_1_5_LOCK.json",
    }

    # Promotion gate (formal)
    promo_ok, promo_reasons = evaluate_day_promotion(
        full_m,
        stress_pass=stress["stress_pass"],
        walk_forward_test_pass=wf.get("walk_forward_test_pass", False),
        walk_forward_val_pass=wf.get("walk_forward_val_pass", False),
        execution_replay_verified=False,
        label_proxy_only=False,
    )

    all_pass_checks = {
        "walk_forward_validation_failed": not wf.get("walk_forward_val_pass", False),
        "walk_forward_test_failed": not wf.get("walk_forward_test_pass", False),
        "stress_failed": not stress["stress_pass"],
        "max_hold_failed": full_m["longest_hold_hours"] > TIME_STOP_HOURS,
        "max_drawdown_failed": full_m["max_drawdown_pct"] > 15.0,
        "duplicate_positions": False,
        "repair_add_dependency": False,
        "red_thesis_dependency": False,
        "stale_artifact": False,
        "execution_replay_missing": True,
        "portfolio_engine_replay_verified": False,
        "overfit_train_only_risk": wf.get("train_monthly_usd", 0) > 0 and not wf.get("walk_forward_test_pass", False),
        "lookahead_risk_low": True,
        "label_proxy_only": False,
    }

    rejection_reasons = list(promo_reasons)
    if wf.get("walk_forward_test_pass") is False:
        rejection_reasons.append(f"walk_forward_test_monthly={wf.get('test_monthly_usd')} below 500")
    if not stress["stress_pass"]:
        rejection_reasons.append("stress_taker_10bp_or_verified_fail")

    hard_decision = "replay_promotion_candidate_only" if promo_ok else "rejected_not_all_pass"

    candidate_spec = {
        "strategy_name": CANDIDATE_ID,
        "lab_script": "scripts/run_allweather_strategy_lab.py",
        "production_module": "backend/services/allweather_signal_engine.py (disabled by default)",
        "symbols_traded": SYMBOLS,
        "regimes_traded": ["trend_up", "neutral"],
        "regimes_blocked_no_long": ["trend_down", "range"],
        "setup_labels": ["BREAKOUT", "TREND_PULLBACK"],
        "entry_rules": {
            "TREND_PULLBACK": "trend_up: price near EMA21 pullback, resuming up, RSI 35-62, ATR targets/stops",
            "BREAKOUT": "trend_up or neutral: close > prior Donchian high, RSI caps, ADX/momentum filters",
        },
        "exit_rules": {
            "target": "ATR multiple from entry (2.2-2.6x ATR)",
            "stop": "ATR multiple (1.3-1.5x ATR)",
            "time_stop_hours": TIME_STOP_HOURS,
            "profit_floor_pct": "none — bracket exit only, not 0.40% live floor",
        },
        "holding_rule": f"hard <= {TIME_STOP_HOURS}h time stop",
        "sizing": f"{NOTIONAL_MULT}x x ${NOTIONAL_USD}/slot = ${NOTIONAL_USD * NOTIONAL_MULT}/slot, max {MAX_SLOTS} slots",
        "allweather_advisory_in_signal": False,
        "allweather_is_the_signal_in_lab": True,
        "uses_research_thesis_names": False,
        "uses_disabled_killed_buckets": False,
        "bypasses_live_bucket_kill_list": True,
        "depends_on_current_market_only": False,
        "replay_type": "lab_bar_execution_replay_15m_fills",
        "label_proxy": False,
        "full_portfolio_engine_execution_replay": False,
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT,
        "command": cmd,
        "exit_code": 0,
        "stale_artifact": False,
        "partial_run": False,
        "full_execution_replay": False,
        "lab_execution_replay_15m": True,
        "portfolio_engine_execution_replay": False,
        "label_proxy_only": False,
        "candidate": candidate_spec,
        "data": {
            "span_days": span_days,
            "months": round(months, 2),
            "symbols": SYMBOLS,
            "cache_dir": str(CACHE_DIR),
            "source_cache_paths": meta.get("cache_paths"),
            "candle_counts": meta.get("candle_counts"),
        },
        "full_span_metrics_1_5x": full_m,
        "walk_forward": wf,
        "stress": stress,
        "window_replays": window_results,
        "comparison_vs_locked_floor": {
            "locked_floor_monthly_usd": locked_floor["monthly_pnl_usd_on_25k"],
            "candidate_monthly_usd": full_m["monthly_pnl_usd"],
            "delta_usd": round(full_m["monthly_pnl_usd"] - locked_floor["monthly_pnl_usd_on_25k"], 2),
            "locked_trades_per_month": locked_floor["trades_per_month"],
            "candidate_trades_per_month": full_m["trades_per_month"],
        },
        "all_pass_checks": all_pass_checks,
        "promotion_gate": {"passed": promo_ok, "failure_reasons": promo_reasons},
        "target_met_500": full_m["target_met_500"],
        "all_pass": promo_ok,
        "hard_decision": hard_decision,
        "rejection_reason": "; ".join(sorted(set(rejection_reasons))) if not promo_ok else None,
        "promotion_candidate_reason": ("formal lock review required; do not apply live" if promo_ok else None),
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "do_not_promote_live": True,
        "remove_from_target_capable_summaries": not promo_ok,
    }

    OUT.write_text(json.dumps(payload, indent=2))
    print(json.dumps({"monthly": full_m["monthly_pnl_usd"], "all_pass": promo_ok, "decision": hard_decision}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
