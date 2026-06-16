#!/usr/bin/env python3
"""
DAY bucket discovery — expansion testing on top of locked all_pass baseline.

Does NOT modify live rules. Tests candidate buckets additively via replay only.
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

from backend.services.day_bucket_quality import REPLAY_KILLED_BUCKETS, bucket_key, evaluate_bucket_entry
from backend.services.day_regime_router import (
    DAY_REGIME_BEAR,
    DAY_REGIME_BULL,
    DAY_REGIME_NEUTRAL,
    DAY_REGIME_RANGE,
    classify_day_regime,
    evaluate_day_entry_route,
)
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_NO_CLEAR_THESIS,
    SETUP_VWAP_REVERSION,
    apply_trade_thesis_to_candidate_fields,
)
from scripts.run_day_strategy_replay import (
    SYMBOLS,
    WINDOWS_DAYS,
    build_decision_data,
    fetch_klines_1h,
    run_replay,
    _atr_pct,
    _resample_4h,
    _stats_from_report,
)
from backend.services.day_bucket_quality import buckets_negative

PRINCIPAL = 25000.0


def _load_bars(max_days: int = 95) -> dict[str, list[dict]]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max_days + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return {sym: fetch_klines_1h(sym, start_ms, end_ms) for sym in SYMBOLS}


def _compute_pass_criteria(windows: dict, walk_forward: dict) -> dict[str, Any]:
    w90 = windows.get("90d", {})
    w30 = windows.get("30d", {})
    w14 = windows.get("14d", {})
    w7 = windows.get("7d", {})
    neutral_pnl = (w90.get("per_regime_pnl") or {}).get("neutral", 0)
    breakout_pnl = (w90.get("per_thesis_pnl") or {}).get("BREAKOUT_CONTINUATION", 0)
    htf_pnl = (w90.get("per_thesis_pnl") or {}).get("HTF_TREND_PULLBACK", 0)
    range_pnl = w90.get("range_vwap_pnl_usd", 0)
    wf_val = walk_forward.get("validation", {})
    wf_test = walk_forward.get("test", {})
    pc = {
        "7d_positive": bool(w7.get("expectancy_positive_after_fees")),
        "14d_improved": bool(w14.get("expectancy_positive_after_fees")) or (w14.get("net_pnl_usd", -999) > -400),
        "30d_improved": bool(w30.get("expectancy_positive_after_fees")) or (w30.get("net_pnl_usd", -999) > -1000),
        "90d_no_fat_tail": (w90.get("average_loss_usd") or 0) > -150 and (w90.get("max_drawdown_pct") or 99) < 8,
        "neutral_not_losing": neutral_pnl >= -50,
        "breakout_not_primary_loser": breakout_pnl >= -50,
        "htf_not_secondary_loser": htf_pnl >= -100,
        "range_vwap_not_losing": range_pnl >= -50,
        "walk_forward_val_positive": (wf_val.get("expectancy_per_trade_usd") or 0) > 0,
        "walk_forward_test_positive": (wf_test.get("expectancy_per_trade_usd") or 0) > 0,
    }
    pc["all_pass"] = all(pc.values())
    return pc


def _run_full_suite(all_bars: dict, discovery_allow: frozenset | None = None) -> dict[str, Any]:
    end_ts = all_bars[SYMBOLS[0]][-1]["ts"]
    start_ts_data = all_bars[SYMBOLS[0]][0]["ts"]
    windows = {f"{wd}d": run_replay(all_bars, wd, discovery_allow_buckets=discovery_allow) for wd in WINDOWS_DAYS}
    span = end_ts - start_ts_data
    t_end = start_ts_data + int(span * 0.50)
    v_end = start_ts_data + int(span * 0.75)
    train = run_replay(all_bars, start_ts=start_ts_data, end_ts=t_end, discovery_allow_buckets=discovery_allow)
    train_buckets = _stats_from_report(train.get("bucket_report", []))
    train_killed = buckets_negative(train_buckets, min_trades=3)
    val = run_replay(
        all_bars, start_ts=t_end, end_ts=v_end,
        extra_killed=train_killed, train_bucket_stats=train_buckets,
        discovery_allow_buckets=discovery_allow,
    )
    test = run_replay(
        all_bars, start_ts=v_end, end_ts=end_ts,
        extra_killed=train_killed, train_bucket_stats=train_buckets,
        discovery_allow_buckets=discovery_allow,
    )
    wf = {"train": train, "validation": val, "test": test, "train_killed_buckets": [list(k) for k in train_killed]}
    return {"windows": windows, "walk_forward": wf, "pass_criteria": _compute_pass_criteria(windows, wf)}


def _band(val: float, edges: list[float]) -> str:
    for i in range(len(edges) - 1):
        if edges[i] <= val < edges[i + 1]:
            return f"{edges[i]}-{edges[i+1]}"
    return f">={edges[-1]}"


def _segment_baseline_trades(trade_details: list[dict]) -> list[dict]:
    segments: dict[str, dict] = defaultdict(lambda: {"trades": 0, "net_pnl": 0.0, "holds": [], "mae": [], "mfe": []})
    for t in trade_details:
        tags = t.get("entry_tags") or {}
        adx = float(tags.get("adx") or 0)
        rsi = float(tags.get("rsi") or 50)
        bb = float(tags.get("bb_position") or 0.5)
        rv = float(tags.get("relative_volume") or 1)
        h4 = float(tags.get("htf_4h_align") or 0.5)
        keys = [
            f"adx_{_band(adx, [0, 12, 18, 22, 28, 50])}",
            f"rsi_{_band(rsi, [0, 32, 36, 40, 50, 100])}",
            f"bb_{_band(bb, [0, 0.15, 0.25, 0.35, 1.0])}",
            f"relvol_{_band(rv, [0, 0.8, 1.0, 1.3, 99])}",
            f"htf4h_{_band(h4, [0, 0.45, 0.55, 0.65, 1.0])}",
            f"{t['symbol']}/{t['regime']}/{t['setup']}",
        ]
        for k in keys:
            segments[k]["trades"] += 1
            segments[k]["net_pnl"] += float(t.get("pnl_usd") or 0)
            segments[k]["holds"].append(float(t.get("hold_hours") or 0))
            segments[k]["mae"].append(float(t.get("mae_pct") or 0))
            segments[k]["mfe"].append(float(t.get("mfe_pct") or 0))
    rows = []
    for k, v in sorted(segments.items()):
        n = v["trades"]
        rows.append({
            "segment": k,
            "trades": n,
            "net_pnl_usd": round(v["net_pnl"], 2),
            "expectancy_usd": round(v["net_pnl"] / n, 2) if n else 0,
            "avg_hold_hours": round(sum(v["holds"]) / n, 1) if n else 0,
            "avg_mae_pct": round(sum(v["mae"]) / n, 5) if n else 0,
            "avg_mfe_pct": round(sum(v["mfe"]) / n, 5) if n else 0,
        })
    return rows


def _scan_opportunities(all_bars: dict, *, symbol: str | None, regime: str | None, thesis: str | None) -> dict[str, int]:
    end_ts = all_bars[SYMBOLS[0]][-1]["ts"]
    start_ts = end_ts - 90 * 86400
    counts: dict[str, int] = defaultdict(int)
    idx_map = {s: 0 for s in SYMBOLS}
    for s in SYMBOLS:
        while idx_map[s] < len(all_bars[s]) and all_bars[s][idx_map[s]]["ts"] < start_ts:
            idx_map[s] += 1
    timeline = sorted(
        {all_bars[s][i]["ts"] for s in SYMBOLS for i in range(idx_map[s], len(all_bars[s])) if all_bars[s][i]["ts"] >= start_ts}
    )
    warmup = 80
    for bar_ts in timeline:
        for sym in SYMBOLS:
            if symbol and sym != symbol:
                continue
            bars = all_bars[sym]
            i = idx_map[sym]
            while i < len(bars) and bars[i]["ts"] < bar_ts:
                i += 1
            if i >= len(bars) or bars[i]["ts"] != bar_ts or i < warmup:
                continue
            slice_1h = bars[: i + 1]
            slice_4h = _resample_4h(slice_1h)
            dd = build_decision_data(sym, slice_1h, slice_4h)
            mark = dd["current_price"]
            atr = _atr_pct(slice_1h) * mark
            chop = 0.65 if dd["adx"] < 18 else 0.45
            ps = dd["price_structure_regime"]
            dd = apply_trade_thesis_to_candidate_fields(
                dd, symbol=sym, current_price=mark, atr=atr, strategy_id="day", price_structure_regime=ps,
            )
            reg = classify_day_regime(dd, context_payload=None, chop_score=chop, atr_ratio=_atr_pct(slice_1h), price_structure_regime=ps)
            setup = str(dd.get("setup_type") or SETUP_NO_CLEAR_THESIS)
            if regime and reg != regime:
                counts["regime_mismatch"] += 1
                continue
            if thesis and setup != thesis:
                counts["thesis_mismatch"] += 1
                continue
            if setup == SETUP_NO_CLEAR_THESIS:
                counts["no_clear_thesis"] += 1
                continue
            route = evaluate_day_entry_route(
                setup_type=setup, day_regime=reg, decision_data=dd,
                context_payload=None, current_price=mark, thesis_score=float(dd.get("thesis_score") or 0),
            )
            if not route.get("allowed"):
                counts[str(route.get("block_reason") or "route_block")] += 1
                continue
            bk = evaluate_bucket_entry(symbol=sym, regime=reg, setup=setup)
            if not bk.get("allowed"):
                counts[str(bk.get("block_reason") or "bucket_block")] += 1
                continue
            counts["would_enter"] += 1
    return dict(counts)


def _candidate_list() -> list[dict[str, Any]]:
    cands: list[dict[str, Any]] = []
    for seg in [
        "adx_12-18", "adx_18-22", "adx_22-28",
        "rsi_32-36", "bb_0-0.15", "bb_0.15-0.25",
        "relvol_0.8-1.0", "relvol_1.0-1.3", "htf4h_0.45-0.55",
    ]:
        cands.append({"id": f"baseline_segment/{seg}", "type": "segment", "segment": seg})

    cands.append({
        "id": "SOL/USDT/range/VWAP_REVERSION",
        "type": "expansion_bucket",
        "bucket": bucket_key("SOL/USDT", DAY_REGIME_RANGE, SETUP_VWAP_REVERSION),
    })

    for sym in ("BTC/USDT", "ETH/USDT", "XRP/USDT"):
        cands.append({
            "id": f"{sym}/range/VWAP_REVERSION",
            "type": "forbidden_revive",
            "bucket": bucket_key(sym, DAY_REGIME_RANGE, SETUP_VWAP_REVERSION),
        })

    for thesis in (SETUP_BREAKOUT_CONTINUATION, SETUP_HTF_TREND_PULLBACK):
        for reg in (DAY_REGIME_NEUTRAL, DAY_REGIME_RANGE):
            cands.append({"id": f"{reg}/{thesis}", "type": "forbidden_revive", "regime": reg, "thesis": thesis})

    for sym in SYMBOLS:
        for reg, thesis in (
            (DAY_REGIME_BULL, SETUP_HTF_TREND_PULLBACK),
            (DAY_REGIME_BULL, SETUP_BREAKOUT_CONTINUATION),
            (DAY_REGIME_BEAR, SETUP_VWAP_REVERSION),
            (DAY_REGIME_BEAR, SETUP_BREAKOUT_CONTINUATION),
        ):
            cands.append({"id": f"{sym}/{reg}/{thesis}", "type": "natural", "symbol": sym, "regime": reg, "thesis": thesis})
    return cands


def _reject_reason(pc: dict, val: dict, test: dict, scan: dict, baseline_trades: int, new_trades: int) -> str:
    if scan.get("would_enter", 0) < 3:
        return f"insufficient_opportunities_{scan.get('would_enter', 0)}"
    if new_trades <= baseline_trades:
        return "expansion_added_zero_trades"
    if not pc.get("all_pass"):
        return "combined_replay_all_pass_false"
    if (val.get("expectancy_per_trade_usd") or 0) <= 0:
        return "walk_forward_validation_negative"
    if (test.get("expectancy_per_trade_usd") or 0) <= 0:
        return "walk_forward_test_negative"
    return "fat_tail_or_expectancy"


def main() -> int:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    tracebacks: list[str] = []
    try:
        all_bars = _load_bars()
        print("Running locked baseline suite...", flush=True)
        baseline = _run_full_suite(all_bars, discovery_allow=None)
        baseline_trades_90 = baseline["windows"]["90d"].get("total_trades", 0)
        td = run_replay(all_bars, 90, return_trade_details=True)
        baseline["trade_details"] = td.get("trade_details", [])
        baseline["baseline_segments"] = _segment_baseline_trades(baseline["trade_details"])
        baseline["locked_at"] = datetime.now(timezone.utc).isoformat()
        baseline["baseline_id"] = "day_baseline_all_pass_v1"
        baseline_path = BASELINE_DIR / "day_baseline_all_pass_v1.json"
        baseline_path.write_text(json.dumps({k: v for k, v in baseline.items() if k != "trade_details"}, indent=2, default=str))

        evaluated: list[dict] = []
        for cand in _candidate_list():
            cid = cand["id"]
            try:
                if cand["type"] == "segment":
                    rows = [r for r in baseline["baseline_segments"] if cand["segment"] in r["segment"]]
                    evaluated.append({"id": cid, "status": "baseline_segment", "segments": rows})
                    continue

                if cand["type"] == "forbidden_revive":
                    evaluated.append({
                        "id": cid,
                        "status": "rejected",
                        "reason": "baseline_locked_replay_proven_negative",
                        "all_pass_after_add": False,
                    })
                    continue

                if cand["type"] == "expansion_bucket":
                    bucket = cand["bucket"]
                    scan = _scan_opportunities(all_bars, symbol=bucket[0], regime=bucket[1], thesis=bucket[2])
                    print(f"Testing expansion {cid} (scan would_enter={scan.get('would_enter', 0)})...", flush=True)
                    suite = _run_full_suite(all_bars, discovery_allow=frozenset({bucket}))
                    w90 = suite["windows"]["90d"]
                    wf = suite["walk_forward"]
                    val, test = wf["validation"], wf["test"]
                    pc = suite["pass_criteria"]
                    new_trades = w90.get("total_trades", 0)
                    accepted = (
                        pc["all_pass"]
                        and new_trades > baseline_trades_90
                        and (val.get("expectancy_per_trade_usd") or 0) > 0
                        and (test.get("expectancy_per_trade_usd") or 0) > 0
                        and scan.get("would_enter", 0) >= 3
                    )
                    evaluated.append({
                        "id": cid,
                        "status": "accepted" if accepted else "rejected",
                        "reason": "" if accepted else _reject_reason(pc, val, test, scan, baseline_trades_90, new_trades),
                        "opportunity_scan_90d": scan,
                        "all_pass_after_add": pc["all_pass"],
                        "90d_net_pnl_usd": w90.get("net_pnl_usd"),
                        "90d_expectancy_usd": w90.get("expectancy_per_trade_usd"),
                        "90d_max_drawdown_pct": w90.get("max_drawdown_pct"),
                        "90d_longest_hold_hours": w90.get("longest_hold_hours"),
                        "wf_validation_exp": val.get("expectancy_per_trade_usd"),
                        "wf_test_exp": test.get("expectancy_per_trade_usd"),
                        "expected_trades_per_month": round(new_trades / 3, 2),
                        "expected_monthly_pnl_usd_25k": round((w90.get("net_pnl_usd") or 0) / 3, 2),
                        "trades_added_vs_baseline": new_trades - baseline_trades_90,
                        "duplicate_attempts": w90.get("duplicate_attempts"),
                        "red_thesis_sells": w90.get("red_thesis_sell_count"),
                    })
                    continue

                # natural bull/bear
                scan = _scan_opportunities(
                    all_bars, symbol=cand["symbol"], regime=cand["regime"], thesis=cand["thesis"],
                )
                br = next(
                    (r for r in baseline["windows"]["90d"].get("bucket_report", [])
                     if r.get("symbol") == cand["symbol"] and r.get("regime") == cand["regime"] and r.get("thesis") == cand["thesis"]),
                    {},
                )
                trades = int(br.get("trades") or 0)
                exp = float(br.get("expectancy_usd") or 0)
                wf = baseline["walk_forward"]
                val, test = wf["validation"], wf["test"]
                pc = baseline["pass_criteria"]
                if scan.get("would_enter", 0) < 3:
                    reason = f"insufficient_opportunities_{scan.get('would_enter', 0)}"
                    accepted = False
                elif trades < 3:
                    reason = f"insufficient_trades_{trades}"
                    accepted = False
                elif exp <= 0:
                    reason = "negative_bucket_expectancy"
                    accepted = False
                elif not pc["all_pass"]:
                    reason = "baseline_context_only"
                    accepted = False
                else:
                    reason = ""
                    accepted = True
                evaluated.append({
                    "id": cid,
                    "status": "accepted" if accepted else "rejected",
                    "reason": reason if not accepted else "",
                    "opportunity_scan_90d": scan,
                    "bucket_stats_90d": br,
                    "all_pass_after_add": pc["all_pass"],
                    "expected_trades_per_month": round(trades / 3, 2) if trades else 0,
                    "expected_monthly_pnl_usd_25k": round(float(br.get("net_pnl_usd") or 0) / 3, 2),
                    "90d_max_drawdown_pct": baseline["windows"]["90d"].get("max_drawdown_pct"),
                    "90d_longest_hold_hours": baseline["windows"]["90d"].get("longest_hold_hours"),
                })
            except Exception:
                evaluated.append({"id": cid, "status": "error", "error": traceback.format_exc()})
                tracebacks.append(traceback.format_exc())

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "baseline_path": str(baseline_path),
            "baseline_pass_criteria": baseline["pass_criteria"],
            "baseline_summary": {
                w: {
                    "net_pnl_usd": baseline["windows"][w].get("net_pnl_usd"),
                    "expectancy_per_trade_usd": baseline["windows"][w].get("expectancy_per_trade_usd"),
                    "total_trades": baseline["windows"][w].get("total_trades"),
                    "win_rate": baseline["windows"][w].get("win_rate"),
                    "average_win_usd": baseline["windows"][w].get("average_win_usd"),
                    "average_loss_usd": baseline["windows"][w].get("average_loss_usd"),
                    "max_drawdown_pct": baseline["windows"][w].get("max_drawdown_pct"),
                    "avg_hold_hours": baseline["windows"][w].get("avg_hold_hours"),
                    "longest_hold_hours": baseline["windows"][w].get("longest_hold_hours"),
                    "per_symbol_pnl": baseline["windows"][w].get("per_symbol_pnl"),
                    "per_regime_pnl": baseline["windows"][w].get("per_regime_pnl"),
                    "per_thesis_pnl": baseline["windows"][w].get("per_thesis_pnl"),
                    "range_vwap_pnl_usd": baseline["windows"][w].get("range_vwap_pnl_usd"),
                }
                for w in ("7d", "14d", "30d", "90d")
            },
            "baseline_walk_forward": {
                ph: {
                    "net_pnl_usd": baseline["walk_forward"][ph].get("net_pnl_usd"),
                    "expectancy_per_trade_usd": baseline["walk_forward"][ph].get("expectancy_per_trade_usd"),
                    "total_trades": baseline["walk_forward"][ph].get("total_trades"),
                    "win_rate": baseline["walk_forward"][ph].get("win_rate"),
                }
                for ph in ("train", "validation", "test")
            },
            "baseline_active_buckets": [
                r for r in baseline["windows"]["90d"].get("bucket_report", [])
                if float(r.get("expectancy_usd") or 0) > 0
            ],
            "baseline_segments": baseline["baseline_segments"],
            "discovered_candidate_buckets": evaluated,
            "accepted_new_buckets": [e for e in evaluated if e.get("status") == "accepted"],
            "rejected_buckets": [e for e in evaluated if e.get("status") == "rejected"],
            "baseline_segments_only": [e for e in evaluated if e.get("status") == "baseline_segment"],
            "tracebacks": tracebacks,
        }
        out_path = BASELINE_DIR / "day_bucket_discovery_latest.json"
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(json.dumps(report, indent=2, default=str))
        return 0
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
