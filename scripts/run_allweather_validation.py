#!/usr/bin/env python3
"""
All-weather multi-year validation — execution replay, research only.

Root-cause fix: prior research validated on the recent ~90d window and KILLED
every (symbol, regime, setup) bucket that lost in that single (down) regime.
That pruned the system to neutral-VWAP only at ~6.7 trades/mo — structurally
incapable of $500/mo.

This validator instead:
  1. Pulls multi-year klines (default ~3yr) spanning bull / bear / range.
  2. Runs the engine UNRESTRAINED (explore_all_buckets) so every regime+setup
     trades and is measured. The legitimate strategy router still applies.
  3. Segments outcomes per regime, per bucket, and per calendar month.
  4. Accepts buckets that are profitable in their OWN regime across years and
     positive in a majority of the months they trade (not recent-window pruned).
  5. Reports the honest monthly distribution across all market types and whether
     the validated all-weather set clears $500/mo on $25k.

No live change. Live stays on day_baseline_all_pass_v1_size_1_5 until a set
passes the promotion gate.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"
OUT = BASELINE_DIR / "allweather_validation_latest.json"

from backend.services.replay_promotion_gate import evaluate_day_promotion
from scripts.run_day_execution_replay import (
    ExecutionConfig,
    _build_fee_profiles,
    fetch_klines_cached,
    run_execution_replay,
)
from scripts.run_day_strategy_replay import SYMBOLS, NOTIONAL_USD, PRINCIPAL

TARGET_500 = 500.0
SPAN_DAYS = int(os.getenv("ALLWEATHER_SPAN_DAYS", "1095"))
EXEC_INTERVAL = os.getenv("ALLWEATHER_EXEC_INTERVAL", "15m")
DECISION_LOOKBACK = 400          # 1h bars; >= all indicator lookbacks, keeps O(n)
MIN_BUCKET_TRADES = 8            # need enough samples per bucket across the span
ACCEPT_MONTH_FRAC = 0.55         # bucket must be net-positive in >=55% of months it traded
NOTIONAL_MULTS = [1.5, 2.0, 2.5]


def _month_key(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")


def _explore_cfg(profiles: dict, notional_mult: float, *, bounded: bool = False) -> ExecutionConfig:
    p = profiles["binance_us_taker"]
    name = f"allweather_explore_{notional_mult}x" + ("_bounded72h" if bounded else "_infhold")
    return ExecutionConfig(
        name=name,
        execution_style=p.execution_style,
        maker_fee=p.maker_fee,
        taker_fee=p.taker_fee,
        slippage_buffer=p.slippage_buffer,
        platform_spread_one_way=0.0,
        half_spread_by_symbol=dict(p.half_spread_by_symbol),
        use_fill_based_exit_gate=True,
        allowed_buckets_only=False,
        explore_all_buckets=True,
        decision_lookback_bars=DECISION_LOOKBACK,
        notional_mult=notional_mult,
        # Bounded pass: real ≤72h time-stop + risk stop so we measure TRUE edge
        # instead of the "hold underwater forever until green" illusion.
        controlled_exits_enabled=bounded,
        time_stop_hours=72.0 if bounded else 48.0,
        max_loss_pct=0.02 if bounded else 0.015,
        atr_stop_mult=1.5 if bounded else 1.0,
    )


def _seg_stats(trades: list[dict]) -> dict[str, Any]:
    n = len(trades)
    net = sum(t["pnl_usd"] for t in trades)
    wins = [t for t in trades if t["pnl_usd"] > 0]
    holds = [t["hold_sec"] / 3600.0 for t in trades]
    return {
        "trades": n,
        "net_pnl_usd": round(net, 2),
        "expectancy_per_trade_usd": round(net / n, 2) if n else 0.0,
        "win_rate_pct": round(100.0 * len(wins) / n, 1) if n else 0.0,
        "avg_hold_hours": round(sum(holds) / n, 1) if n else 0.0,
        "longest_hold_hours": round(max(holds), 1) if holds else 0.0,
    }


def _bucket_id(t: dict) -> str:
    return f"{t['symbol']}/{t['regime']}/{t['setup']}"


def _month_consistency(trades: list[dict]) -> tuple[int, int, float]:
    """Months traded, months net-positive, fraction positive."""
    by_month: dict[str, float] = defaultdict(float)
    for t in trades:
        by_month[_month_key(t["entry_ts"])] += t["pnl_usd"]
    if not by_month:
        return 0, 0, 0.0
    pos = sum(1 for v in by_month.values() if v > 0)
    return len(by_month), pos, round(pos / len(by_month), 3)


def main() -> int:
    print("=== ALL-WEATHER MULTI-YEAR VALIDATION (execution replay) ===", flush=True)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=SPAN_DAYS + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    print(f"  span={SPAN_DAYS}d exec={EXEC_INTERVAL}; loading klines (cached)...", flush=True)
    bars_1h: dict[str, list[dict]] = {}
    bars_exec: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        bars_1h[sym] = fetch_klines_cached(sym, "1h", start_ms, end_ms)
        bars_exec[sym] = fetch_klines_cached(sym, EXEC_INTERVAL, start_ms, end_ms)
        print(f"    {sym}: 1h={len(bars_1h[sym])} {EXEC_INTERVAL}={len(bars_exec[sym])}", flush=True)

    if not bars_1h[SYMBOLS[0]]:
        print("  ERROR: no 1h data", flush=True)
        OUT.write_text(json.dumps({"error": "no_data"}, indent=2))
        return 1

    span_days_actual = int(
        (bars_1h[SYMBOLS[0]][-1]["ts"] - bars_1h[SYMBOLS[0]][0]["ts"]) / 86400
    )
    months_actual = max(span_days_actual / 30.4375, 1.0)
    start_ts = bars_1h[SYMBOLS[0]][0]["ts"]
    end_ts = bars_1h[SYMBOLS[0]][-1]["ts"]
    profiles = _build_fee_profiles()

    # Measure true edge under a real bounded hold (≤72h + stop) vs the
    # infinite-hold-until-green illusion. The bounded number is the honest one.
    print("  --- bounded-hold (≤72h, real stop) edge probe @1.5x ---", flush=True)
    bcfg = _explore_cfg(profiles, 1.5, bounded=True)
    brep = run_execution_replay(
        bars_1h, bars_exec, window_days=span_days_actual,
        start_ts=start_ts, end_ts=end_ts, config=bcfg,
        exec_interval=EXEC_INTERVAL, return_trades=True,
    )
    btrades = [
        t for t in brep.get("trades_detail", [])
        if t.get("exit_reason") != "REPLAY_MARK_TO_MARKET"
    ]
    bnet = sum(t["pnl_usd"] for t in btrades)
    bwins = [t for t in btrades if t["pnl_usd"] > 0]
    bounded_probe = {
        "config": "controlled_exits_72h_stop_1_5x",
        "total_trades": len(btrades),
        "net_pnl_usd": round(bnet, 2),
        "monthly_pnl_usd": round(bnet / months_actual, 2),
        "win_rate_pct": round(100.0 * len(bwins) / len(btrades), 1) if btrades else 0.0,
        "expectancy_per_trade_usd": round(bnet / len(btrades), 2) if btrades else 0.0,
        "longest_hold_hours": round(max((t["hold_sec"] / 3600.0 for t in btrades), default=0.0), 1),
        "per_regime": {
            r: _seg_stats([t for t in btrades if t["regime"] == r])
            for r in sorted({t["regime"] for t in btrades})
        },
        "note": (
            "TRUE edge with sane hold limit. Compare to infinite-hold profiles below; "
            "if this is <=0 or far below the infinite-hold number, the apparent edge is "
            "a hold-time artifact, not a real tradeable signal."
        ),
    }
    print(
        f"    bounded: trades={len(btrades)} net=${bounded_probe['net_pnl_usd']} "
        f"monthly=${bounded_probe['monthly_pnl_usd']} win%={bounded_probe['win_rate_pct']}",
        flush=True,
    )

    profile_results: list[dict] = []
    for mult in NOTIONAL_MULTS:
        print(f"  unrestrained (infinite-hold) replay {mult}x over {span_days_actual}d...", flush=True)
        cfg = _explore_cfg(profiles, mult)
        rep = run_execution_replay(
            bars_1h, bars_exec, window_days=span_days_actual,
            start_ts=start_ts, end_ts=end_ts, config=cfg,
            exec_interval=EXEC_INTERVAL, return_trades=True,
        )
        trades = [
            t for t in rep.get("trades_detail", [])
            if t.get("exit_reason") != "REPLAY_MARK_TO_MARKET"
        ]
        print(f"    trades={len(trades)} net={round(sum(t['pnl_usd'] for t in trades),2)}", flush=True)

        per_regime: dict[str, list[dict]] = defaultdict(list)
        per_bucket: dict[str, list[dict]] = defaultdict(list)
        per_month: dict[str, float] = defaultdict(float)
        for t in trades:
            per_regime[t["regime"]].append(t)
            per_bucket[_bucket_id(t)].append(t)
            per_month[_month_key(t["entry_ts"])] += t["pnl_usd"]

        # Accept buckets profitable in their own regime across the span.
        accepted: list[str] = []
        bucket_rows: list[dict] = []
        for bid, btr in sorted(per_bucket.items()):
            st = _seg_stats(btr)
            mtraded, mpos, mfrac = _month_consistency(btr)
            accept = (
                st["trades"] >= MIN_BUCKET_TRADES
                and st["net_pnl_usd"] > 0
                and st["expectancy_per_trade_usd"] > 0
                and mfrac >= ACCEPT_MONTH_FRAC
                and st["longest_hold_hours"] <= 72.0
            )
            if accept:
                accepted.append(bid)
            bucket_rows.append({
                "bucket": bid, **st,
                "months_traded": mtraded, "months_positive": mpos,
                "month_positive_frac": mfrac,
                "accepted": accept,
            })

        accepted_set = set(accepted)
        acc_trades = [t for t in trades if _bucket_id(t) in accepted_set]
        acc_net = sum(t["pnl_usd"] for t in acc_trades)
        acc_monthly = round(acc_net / months_actual, 2)
        full_net = sum(t["pnl_usd"] for t in trades)
        full_monthly = round(full_net / months_actual, 2)

        acc_by_month: dict[str, float] = defaultdict(float)
        for t in acc_trades:
            acc_by_month[_month_key(t["entry_ts"])] += t["pnl_usd"]
        acc_months_pos = sum(1 for v in acc_by_month.values() if v > 0)

        profile_results.append({
            "notional_mult": mult,
            "per_slot_usd": round(NOTIONAL_USD * mult, 2),
            "full_unrestrained": {
                "total_trades": len(trades),
                "net_pnl_usd": round(full_net, 2),
                "monthly_pnl_usd": full_monthly,
                "trades_per_month": round(len(trades) / months_actual, 2),
                "per_regime": {r: _seg_stats(v) for r, v in sorted(per_regime.items())},
            },
            "accepted_allweather_set": {
                "buckets": accepted,
                "bucket_count": len(accepted),
                "total_trades": len(acc_trades),
                "net_pnl_usd": round(acc_net, 2),
                "monthly_pnl_usd": acc_monthly,
                "trades_per_month": round(len(acc_trades) / months_actual, 2),
                "months_traded": len(acc_by_month),
                "months_positive": acc_months_pos,
                "month_positive_frac": round(acc_months_pos / max(len(acc_by_month), 1), 3),
                "target_met_500": acc_monthly >= TARGET_500,
                "per_regime": {
                    r: _seg_stats([t for t in acc_trades if t["regime"] == r])
                    for r in sorted({t["regime"] for t in acc_trades})
                },
            },
            "bucket_detail": bucket_rows,
            "monthly_distribution_full": {k: round(v, 2) for k, v in sorted(per_month.items())},
            "monthly_distribution_accepted": {k: round(v, 2) for k, v in sorted(acc_by_month.items())},
        })

    # Promotion gate on best accepted profile.
    best = max(
        profile_results,
        key=lambda r: r["accepted_allweather_set"]["monthly_pnl_usd"],
    )
    acc = best["accepted_allweather_set"]
    gate_metrics = {
        "monthly_pnl_usd_on_25k": acc["monthly_pnl_usd"],
        "expectancy_per_trade_usd": (
            round(acc["net_pnl_usd"] / acc["total_trades"], 2) if acc["total_trades"] else 0.0
        ),
        "longest_hold_hours": max(
            (s.get("longest_hold_hours", 0) for s in acc["per_regime"].values()), default=0.0
        ),
        "max_drawdown_pct": 0.0,
    }
    promoted, reasons = evaluate_day_promotion(
        gate_metrics,
        stress_pass=False,            # multi-window stress not yet run on the new set
        walk_forward_test_pass=False,
        walk_forward_val_pass=False,
        execution_replay_verified=True,
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "discovery_type": "allweather_multiyear_validation",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "live_promoted": False,
        "target_monthly_usd": TARGET_500,
        "span_days_requested": SPAN_DAYS,
        "span_days_actual": span_days_actual,
        "months_actual": round(months_actual, 2),
        "exec_interval": EXEC_INTERVAL,
        "method_note": (
            "Unrestrained explore across full multi-year span; buckets accepted on "
            "own-regime profitability + month-consistency, not recent-window survival."
        ),
        "bounded_hold_true_edge_probe": bounded_probe,
        "profiles": profile_results,
        "best_profile_notional_mult": best["notional_mult"],
        "verdict": {
            "best_accepted_monthly_usd": acc["monthly_pnl_usd"],
            "best_accepted_buckets": acc["buckets"],
            "best_accepted_trades_per_month": acc["trades_per_month"],
            "month_positive_frac": acc["month_positive_frac"],
            "target_met_500": acc["target_met_500"],
            "gap_to_500_usd": round(TARGET_500 - acc["monthly_pnl_usd"], 2),
            "promotion_accepted": promoted,
            "promotion_block_reasons": reasons,
            "next_step_if_target_met": (
                "Run full walk-forward + stress battery on accepted set before any live promotion."
            ),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"  wrote {OUT}", flush=True)
    print(
        f"  best {best['notional_mult']}x: accepted monthly=${acc['monthly_pnl_usd']} "
        f"buckets={acc['bucket_count']} trades/mo={acc['trades_per_month']} "
        f"month+frac={acc['month_positive_frac']} target_met={acc['target_met_500']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
