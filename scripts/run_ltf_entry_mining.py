#!/usr/bin/env python3
"""
LTF entry mining — EXHAUSTED research branch (diagnostic only).

Set MYSTIC_FORCE_EXHAUSTED_RESEARCH=1 to re-run. Use run_topfour_profit_rebuild.py for active research.
"""
from __future__ import annotations

import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
BASELINE_DIR = REPO / "scripts" / "replay_baselines"

from backend.config.trading_economics import ORDERBOOK_HALF_SPREAD_ESTIMATE
from backend.config.binance_us_fee_schedule import verify_top_four_pairs
from backend.services.ltf_pattern_miner import (
    Economics,
    MinedTrade,
    PatternSpec,
    aggregate_metrics,
    make_pattern_catalog,
    mine_symbol_pattern,
    reject_candidate,
    regime_bucket_report,
    resample_bars,
    walk_forward_split,
)
from scripts.run_day_execution_replay import fetch_klines_cached
from scripts.run_day_strategy_replay import PRINCIPAL, SYMBOLS, SYMBOL_API

TARGET_MONTHLY = 500.0
MIN_PCT_MONTH = 2.0
DAYS = 90
SCALP_DAYS = 30
TRAIN_DAYS = 60


def _load_bars() -> tuple[int, int, int, dict[str, dict[str, list]]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=DAYS + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    start_ts = int((end - timedelta(days=DAYS)).timestamp())
    end_ts = int(end.timestamp())
    scalp_start_ts = int((end - timedelta(days=SCALP_DAYS)).timestamp())

    all_bars: dict[str, dict[str, list]] = {}
    for sym in SYMBOLS:
        sym_bars: dict[str, list] = {}
        for iv in ("1m", "5m", "15m", "1h"):
            sym_bars[iv] = fetch_klines_cached(sym, iv, start_ms, end_ms)
        sym_bars["30m"] = resample_bars(sym_bars["5m"], 30)
        sym_bars["4h"] = resample_bars(sym_bars["1h"], 240)
        all_bars[sym] = sym_bars
    return start_ts, end_ts, scalp_start_ts, all_bars


def _spread_stress_ok(trades: list[MinedTrade], spread_mult: float = 1.5) -> bool:
    if not trades:
        return False
    gross = sum(t.pnl_usd for t in trades)
    # approximate extra spread cost per round trip
    extra = sum(t.notional * 2 * ORDERBOOK_HALF_SPREAD_ESTIMATE * (spread_mult - 1.0) for t in trades)
    adj = gross - extra
    return adj > 0


def _combine_trades(candidates: list[dict[str, Any]], max_positions: int = 4) -> list[MinedTrade]:
    """Portfolio merge: max 4 concurrent, one per symbol."""
    all_t: list[MinedTrade] = []
    for c in candidates:
        all_t.extend(c.get("trades") or [])
    all_t.sort(key=lambda t: t.entry_ts)
    open_sym: dict[str, int] = {}
    merged: list[MinedTrade] = []
    for t in all_t:
        now_open = sum(1 for et in open_sym.values() if et > t.entry_ts)
        if t.symbol in open_sym and open_sym[t.symbol] > t.entry_ts:
            continue
        if now_open >= max_positions:
            continue
        merged.append(t)
        open_sym[t.symbol] = t.exit_ts
    return merged


def _evaluate_pattern(
    spec: PatternSpec,
    all_bars: dict[str, dict[str, list]],
    start_ts: int,
    end_ts: int,
    split_ts: int,
    half_spreads: dict[str, float],
    window_days: int = DAYS,
) -> dict[str, Any]:
    trades: list[MinedTrade] = []
    per_symbol: dict[str, list[MinedTrade]] = {}
    for sym in SYMBOLS:
        hs = half_spreads.get(sym, 0.00006)
        econ = Economics(half_spread=hs)
        sym_trades = mine_symbol_pattern(sym, spec, all_bars[sym], start_ts, end_ts, econ)
        per_symbol[sym] = sym_trades
        trades.extend(sym_trades)

    metrics = aggregate_metrics(trades, window_days)
    wf = walk_forward_split(trades, split_ts)
    regime = regime_bucket_report(trades, window_days)

    spread_ok = _spread_stress_ok(trades, 1.5)

    accepted, reasons = reject_candidate(metrics, wf, spread_ok=spread_ok)

    entry_cond = f"LTF {spec.timeframe_min}m pattern {spec.pattern_id}"
    exit_cond = (
        f"net>={spec.profit_target_pct:.2%} | stop={spec.stop_atr_mult}ATR | "
        f"time={spec.time_stop_hours}h | max72h"
    )

    return {
        "pattern_id": spec.pattern_id,
        "category": spec.category,
        "timeframe_min": spec.timeframe_min,
        "entry_condition": entry_cond,
        "exit_condition": exit_cond,
        "symbols_traded": {s: len(v) for s, v in per_symbol.items() if v},
        "metrics_90d": metrics,
        "walk_forward": wf,
        "regime_buckets": regime,
        "spread_stress_pass": spread_ok,
        "accepted": accepted,
        "reject_reasons": reasons,
        "trades": trades,
        "target_met_500": metrics.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY,
    }


def main() -> int:
    from backend.services.replay_promotion_gate import block_exhausted_branch

    block_exhausted_branch("ltf_hand_coded_patterns")
    print("=== LTF ENTRY MINING (pattern discovery) ===", flush=True)
    verified = verify_top_four_pairs()
    half_spreads = {k: float(v["orderbook_half_spread_pct"]) for k, v in verified["pairs"].items()}

    start_ts, end_ts, scalp_start_ts, all_bars = _load_bars()
    split_ts = start_ts + TRAIN_DAYS * 86400
    catalog = make_pattern_catalog()

    results: list[dict[str, Any]] = []
    for i, spec in enumerate(catalog):
        print(f"  [{i+1}/{len(catalog)}] {spec.pattern_id} ({spec.category})...", flush=True)
        try:
            mine_start = scalp_start_ts if spec.scalp else start_ts
            mine_days = SCALP_DAYS if spec.scalp else DAYS
            row = _evaluate_pattern(spec, all_bars, mine_start, end_ts, split_ts, half_spreads, mine_days)
            row.pop("trades", None)  # keep report lean; full trades in accepted only
            results.append(row)
        except Exception:
            results.append({
                "pattern_id": spec.pattern_id,
                "accepted": False,
                "reject_reasons": ["mining_error"],
                "error": traceback.format_exc(),
            })

    accepted = [r for r in results if r.get("accepted")]
    day_accepted = [r for r in accepted if r.get("category") == "day"]
    scalp_accepted = [r for r in accepted if r.get("category") == "scalp"]

    # Re-mine accepted for portfolio combine
    full_trades: list[dict] = []
    for spec in catalog:
        r = next((x for x in results if x["pattern_id"] == spec.pattern_id), None)
        if not r or not r.get("accepted"):
            continue
        for sym in SYMBOLS:
            hs = half_spreads.get(sym, 0.00006)
            ms = scalp_start_ts if spec.scalp else start_ts
            md = SCALP_DAYS if spec.scalp else DAYS
            ts = mine_symbol_pattern(sym, spec, all_bars[sym], ms, end_ts, Economics(half_spread=hs))
            if ts:
                full_trades.append({"pattern_id": spec.pattern_id, "category": spec.category, "trades": ts, "days": md})

    combined = _combine_trades(full_trades)
    combined_metrics = aggregate_metrics(combined, DAYS)
    combined_wf = walk_forward_split(combined, split_ts)
    combined_accepted, combined_reasons = reject_candidate(combined_metrics, combined_wf, spread_ok=True)
    combined_target = combined_metrics.get("monthly_pnl_usd", 0) >= TARGET_MONTHLY

    # Best singles
    by_monthly = sorted(results, key=lambda x: float((x.get("metrics_90d") or {}).get("monthly_pnl_usd") or -1e9), reverse=True)

    best_day = next((r for r in by_monthly if r.get("category") == "day"), by_monthly[0] if by_monthly else {})
    best_scalp = max(
        [r for r in results if r.get("category") == "scalp"],
        key=lambda x: float((x.get("metrics_90d") or {}).get("monthly_pnl_usd") or -1e9),
        default={},
    )
    bd = best_day.get("metrics_90d") or {}
    bs = best_scalp.get("metrics_90d") or {}

    summary_table = [
        {
            "row": "locked_live_DAY_floor",
            "monthly_pnl_usd": 87.0,
            "trades_per_month": 6.7,
            "note": "unchanged live baseline",
        },
        {
            "row": "best_mined_DAY_pattern",
            "pattern_id": best_day.get("pattern_id"),
            "trades_per_month": bd.get("trades_per_month"),
            "monthly_pnl_usd": bd.get("monthly_pnl_usd"),
            "pct_per_month": bd.get("pct_per_month"),
            "expectancy_per_trade": bd.get("expectancy_per_trade"),
            "win_rate_pct": bd.get("win_rate_pct"),
            "profit_factor": bd.get("profit_factor"),
            "max_drawdown_pct": bd.get("max_drawdown_pct"),
            "longest_hold_hours": bd.get("longest_hold_hours"),
            "accepted": best_day.get("accepted"),
            "reject_reasons": best_day.get("reject_reasons"),
        },
        {
            "row": "best_mined_scalp_pattern",
            "pattern_id": best_scalp.get("pattern_id"),
            "trades_per_month": bs.get("trades_per_month"),
            "monthly_pnl_usd": bs.get("monthly_pnl_usd"),
            "pct_per_month": bs.get("pct_per_month"),
            "expectancy_per_trade": bs.get("expectancy_per_trade"),
            "win_rate_pct": bs.get("win_rate_pct"),
            "profit_factor": bs.get("profit_factor"),
            "accepted": best_scalp.get("accepted"),
            "reject_reasons": best_scalp.get("reject_reasons"),
        },
        {
            "row": "combined_accepted_patterns",
            "trades_per_month": combined_metrics.get("trades_per_month"),
            "monthly_pnl_usd_on_25k": combined_metrics.get("monthly_pnl_usd"),
            "pct_per_month": combined_metrics.get("pct_per_month"),
            "expectancy_per_trade": combined_metrics.get("expectancy_per_trade"),
            "win_rate_pct": combined_metrics.get("win_rate_pct"),
            "profit_factor": combined_metrics.get("profit_factor"),
            "max_drawdown_pct": combined_metrics.get("max_drawdown_pct"),
            "longest_hold_hours": combined_metrics.get("longest_hold_hours"),
            "patterns_merged": len(full_trades),
            "accepted": combined_accepted,
            "target_met_500": combined_target,
            "reject_reasons": combined_reasons,
            "wf_test_positive": combined_wf.get("test_positive"),
        },
    ]

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mining_type": "ltf_pattern_discovery",
        "live_unchanged": "day_baseline_all_pass_v1_size_1_5",
        "target_monthly_usd": TARGET_MONTHLY,
        "window_days": DAYS,
        "patterns_tested": len(catalog),
        "patterns_accepted": len(accepted),
        "day_accepted": len(day_accepted),
        "scalp_accepted": len(scalp_accepted),
        "target_met_500": combined_target,
        "summary_table": summary_table,
        "top_15_by_monthly_pnl": [
            {
                "pattern_id": r.get("pattern_id"),
                "category": r.get("category"),
                "monthly_pnl_usd": (r.get("metrics_90d") or {}).get("monthly_pnl_usd"),
                "trades_per_month": (r.get("metrics_90d") or {}).get("trades_per_month"),
                "expectancy_per_trade": (r.get("metrics_90d") or {}).get("expectancy_per_trade"),
                "win_rate_pct": (r.get("metrics_90d") or {}).get("win_rate_pct"),
                "profit_factor": (r.get("metrics_90d") or {}).get("profit_factor"),
                "longest_hold_hours": (r.get("metrics_90d") or {}).get("longest_hold_hours"),
                "accepted": r.get("accepted"),
                "reject_reasons": r.get("reject_reasons"),
                "wf_test_positive": (r.get("walk_forward") or {}).get("test_positive"),
            }
            for r in by_monthly[:15]
        ],
        "accepted_patterns": [
            {k: v for k, v in r.items() if k != "trades"}
            for r in accepted
        ],
        "all_results": [
            {k: v for k, v in r.items() if k not in ("trades",)}
            for r in results
        ],
    }

    out = BASELINE_DIR / "ltf_entry_mining_latest.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({
        "accepted": len(accepted),
        "target_met_500": combined_target,
        "best_pattern": by_monthly[0].get("pattern_id") if by_monthly else None,
        "best_monthly": (by_monthly[0].get("metrics_90d") or {}).get("monthly_pnl_usd") if by_monthly else 0,
        "combined_monthly": combined_metrics.get("monthly_pnl_usd"),
        "out": str(out),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
