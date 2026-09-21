#!/usr/bin/env python3
"""Read-only verification of replay 4H as-of, hold units, and population.

Does not change live production. Prints JSON for the audit response.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("DAY_PATH_AWARE_EXIT", "true")
os.environ.setdefault("DAY_STALL_EXIT_ENABLED", "false")
os.environ.setdefault("DAY_GIVEBACK_EXIT_ENABLED", "false")
os.environ.setdefault("ESTIMATED_ROUNDTRIP_COST", "0.0006")

from backend.config.trading_economics import (
    COOLDOWN_SECONDS_AFTER_SELL,
    DAY_MAX_DEPLOYED_USD,
    DAY_MAX_OPEN_SLOTS,
    DAY_TARGET_NOTIONAL_PER_SLOT_USD,
)
from backend.services.day_controlled_exits import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    evaluate_engine_managed_exit,
    evaluate_pre_buy_exit_consistency,
)
from backend.services.day_production_lifecycle_replay import (
    BAR_DECISION_SEC,
    SYMBOLS,
    ReplayPos,
    _bar_index,
    decision_bars,
    executable_buy_price,
    fourh_bundle,
    load_1m_bars,
    load_inferences,
    production_exit_env,
    resample_4h,
    veto_current,
)
from backend.services.day_trade_thesis import (
    _bundle_tf_align,
    current_utc_4h_open_ms,
    day_4h_structure_snapshot,
    ohlcv_row_open_ms,
    resolve_day_4h_structure_bundle,
)
from backend.services.portfolio_engine import get_coin_profile

FOURH_SEC = 14400


def iso(ep: float) -> str:
    return datetime.fromtimestamp(float(ep), tz=timezone.utc).isoformat()


def completed_4h_close_epoch(open_sec: float) -> float:
    return float(open_sec) + FOURH_SEC


def asof_4h_rows(full_rows: list[list[float]], now_epoch: float) -> tuple[list[list[float]], dict]:
    """Completed bars only (close_ts <= now) plus forming bar if present."""
    completed = []
    forming = None
    for r in full_rows:
        open_sec = float(r[0])
        close_sec = completed_4h_close_epoch(open_sec)
        if close_sec <= now_epoch + 1e-9:
            completed.append(r)
        elif open_sec <= now_epoch + 1e-9:
            forming = r
    meta = {
        "n_completed": len(completed),
        "forming_present": forming is not None,
        "forming_open": float(forming[0]) if forming else None,
        "forming_close_ts": completed_4h_close_epoch(forming[0]) if forming else None,
        "last_completed_open": float(completed[-1][0]) if completed else None,
        "last_completed_close_ts": completed_4h_close_epoch(completed[-1][0]) if completed else None,
        "last_completed_fully_closed": (completed_4h_close_epoch(completed[-1][0]) <= now_epoch + 1e-9 if completed else None),
        "forming_fully_closed": False,
        "future_completed_used": False,
    }
    return completed + ([forming] if forming else []), meta


def broken_reason(bundle, price, now_epoch) -> dict:
    resolved = resolve_day_4h_structure_bundle(bundle, current_price=price, now_epoch=now_epoch)
    snap = day_4h_structure_snapshot(bundle, current_price=price, now_epoch=now_epoch)
    align = _bundle_tf_align(resolved, "4h")
    n4 = len((resolved or {}).get("4h") or [])
    return {
        "intact": bool(snap.get("htf_4h_rise_intact")),
        "broken": bool(snap.get("htf_4h_rise_broken")),
        "missing": bool(snap.get("4h_bundle_missing")),
        "prior_4h_low": snap.get("prior_4h_low"),
        "current_4h_close": snap.get("current_4h_close"),
        "forming_close_source": snap.get("forming_close_source"),
        "align": align,
        "n_4h_rows": n4,
        "current_4h_open_ms": current_utc_4h_open_ms(now_epoch),
    }


def main() -> None:
    db = os.path.join(os.path.dirname(__file__), "..", "mystic_trading.db")
    conn = sqlite3.connect(db)
    bars = load_1m_bars(conn)
    fourh_full = {s: resample_4h(bars[s]) for s in SYMBOLS}
    inferences = load_inferences(conn)
    events = decision_bars(inferences)
    inf_min = min((i["epoch"] for i in inferences), default=0)
    inf_max = max((i["epoch"] for i in inferences), default=0)

    report: dict = {
        "host": os.uname().nodename,
        "db": os.path.abspath(db),
        "sha_hint": "local_workspace",
        "inference_rows": len(inferences),
        "decision_bars": len(events),
        "inference_start": iso(inf_min) if inf_min else None,
        "inference_end": iso(inf_max) if inf_max else None,
        "sizing": {
            "DAY_TARGET_NOTIONAL_PER_SLOT_USD": DAY_TARGET_NOTIONAL_PER_SLOT_USD,
            "DAY_MAX_DEPLOYED_USD": DAY_MAX_DEPLOYED_USD,
            "DAY_MAX_OPEN_SLOTS": DAY_MAX_OPEN_SLOTS,
            "COOLDOWN_SECONDS_AFTER_SELL": COOLDOWN_SECONDS_AFTER_SELL,
        },
        "fourh_rows_per_symbol": {s: len(fourh_full[s]) for s in SYMBOLS},
        "replay_fourh_bundle_keeps": 8,
        "ema_align_needs_closes": 50,
    }

    # Walk Arm A the same way as diagnosis, but record 4H as-of at entry/exit.
    samples = []
    holds_sec = []
    flags = defaultdict(int)
    n_closed = 0
    n_structure = 0
    already_broken_at_entry = 0
    prebuy_would_block = 0
    same_snapshot = 0
    exit_before_new_4h_close = 0
    future_candle = 0
    partial_as_closed = 0
    near_boundary = defaultdict(int)
    by_sym = defaultdict(int)
    accepted = rejected = created = 0

    with production_exit_env():
        open_pos: dict[str, ReplayPos] = {}
        cooldown: dict[str, float] = defaultdict(float)
        last_adv: dict[str, int] = {}
        deployed = 0.0

        def flush(until: int) -> None:
            nonlocal n_closed, n_structure, already_broken_at_entry, prebuy_would_block
            nonlocal same_snapshot, exit_before_new_4h_close, future_candle, partial_as_closed, deployed
            for sym, pos in list(open_pos.items()):
                bset = bars.get(sym, [])
                start = int(last_adv.get(sym, pos.entry_time)) + 1
                idx = _bar_index(bset, start)
                if idx is None:
                    last_adv[sym] = until
                    continue
                profile = get_coin_profile(sym)
                for j in range(idx, len(bset)):
                    ep, _o, high, low, close = bset[j]
                    if ep > until:
                        break
                    pos.highest_price = max(pos.highest_price, high)
                    if pos.lowest_price <= 0 or low < pos.lowest_price:
                        pos.lowest_price = low
                    leaky = fourh_bundle(fourh_full[sym], float(ep))
                    asof_rows, asof_meta = asof_4h_rows(fourh_full[sym], float(ep))
                    leaky_last_open = leaky["4h"][-1][0] if leaky["4h"] else None
                    _asof_last_open = asof_rows[-1][0] if asof_rows else None
                    if leaky_last_open is not None and completed_4h_close_epoch(leaky_last_open) > ep + 1:
                        partial_as_closed += 1
                    if leaky_last_open is not None and leaky_last_open > ep:
                        future_candle += 1
                    hold_min = max(0.0, (ep - pos.entry_time) / 60.0)
                    net_pnl = (close - pos.entry_price) / pos.entry_price - 0.0006
                    decision = evaluate_engine_managed_exit(
                        position=pos,
                        current_price=close,
                        net_pnl_pct=net_pnl,
                        hold_minutes=hold_min,
                        coin_profile=profile,
                        bundle=leaky,
                        bar_low=low,
                        now_epoch=float(ep),
                    )
                    last_adv[sym] = ep
                    if str(decision.get("action") or "") != "sell":
                        continue
                    reason = str(decision.get("reason") or "")
                    n_closed += 1
                    by_sym[sym] += 1
                    hold_s = ep - pos.entry_time
                    holds_sec.append(hold_s)
                    entry_4h_open = getattr(pos, "_entry_4h_open", None)
                    next_legit_close = None
                    if entry_4h_open is not None:
                        cur_open = current_utc_4h_open_ms(pos.entry_time) / 1000.0
                        next_legit_close = cur_open + FOURH_SEC
                    if reason == EXIT_DAY_4H_STRUCTURE_BREAK:
                        n_structure += 1
                        if getattr(pos, "_broken_at_entry", False):
                            already_broken_at_entry += 1
                        if getattr(pos, "_prebuy_block", False):
                            prebuy_would_block += 1
                        if entry_4h_open is not None and leaky_last_open == entry_4h_open:
                            same_snapshot += 1
                        if next_legit_close is not None and ep < next_legit_close:
                            exit_before_new_4h_close += 1
                        if next_legit_close is not None:
                            # distance from entry to nearest 4H boundary (open or close)
                            b_open = current_utc_4h_open_ms(pos.entry_time) / 1000.0
                            d_open = pos.entry_time - b_open
                            d_close = next_legit_close - pos.entry_time
                            dmin = min(d_open, d_close)
                            for lim, key in ((60, "1m"), (300, "5m"), (900, "15m"), (1800, "30m")):
                                if dmin <= lim:
                                    near_boundary[key] += 1
                        if len(samples) < 28:
                            entry_snap = getattr(pos, "_entry_snap", {})
                            exit_snap = broken_reason(leaky, close, float(ep))
                            samples.append(
                                {
                                    "symbol": sym,
                                    "inference_or_decision_bar": iso(pos.entry_time),
                                    "entry_decision_epoch": pos.entry_time,
                                    "entry_fill_price": pos.entry_price,
                                    "last_closed_4h_at_entry_open": iso(getattr(pos, "_entry_last_closed_open", 0) or 0),
                                    "last_closed_4h_at_entry_close_ts": iso(getattr(pos, "_entry_last_closed_close", 0) or 0),
                                    "last_closed_4h_close_px": getattr(pos, "_entry_last_closed_px", None),
                                    "prior_4h_low_at_entry": entry_snap.get("prior_4h_low"),
                                    "broken_already_true_at_entry": getattr(pos, "_broken_at_entry", False),
                                    "prebuy_would_block": getattr(pos, "_prebuy_block", False),
                                    "prebuy_reason": getattr(pos, "_prebuy_reason", ""),
                                    "first_exit_eval_epoch": iso(ep),
                                    "exit_4h_open": iso(leaky_last_open) if leaky_last_open else None,
                                    "exit_4h_close_ts": iso(completed_4h_close_epoch(leaky_last_open)) if leaky_last_open else None,
                                    "exit_4h_fully_closed_at_eval": (completed_4h_close_epoch(leaky_last_open) <= ep + 1e-9 if leaky_last_open else None),
                                    "same_4h_open_as_entry": leaky_last_open == entry_4h_open,
                                    "exit_reason": reason,
                                    "hold_seconds": round(hold_s, 3),
                                    "hold_minutes": round(hold_s / 60.0, 4),
                                    "seconds_to_next_legit_4h_close": (round(next_legit_close - pos.entry_time, 1) if next_legit_close else None),
                                    "entry_align": entry_snap.get("align"),
                                    "entry_n_4h": entry_snap.get("n_4h_rows"),
                                    "exit_align": exit_snap.get("align"),
                                    "exit_n_4h": exit_snap.get("n_4h_rows"),
                                    "exit_prior_4h_low": exit_snap.get("prior_4h_low"),
                                    "exit_current_4h_close": exit_snap.get("current_4h_close"),
                                    "asof_meta_at_exit": asof_meta,
                                }
                            )
                    deployed = max(0.0, deployed - pos.notional)
                    cooldown[sym] = ep + COOLDOWN_SECONDS_AFTER_SELL
                    del open_pos[sym]
                    break

        for ev in events:
            bar = int(ev["bar_epoch"])
            flush(bar)
            ranked = sorted(ev["inferences"], key=lambda r: r["p_buy"], reverse=True)
            for inf in ranked:
                ok, _ = veto_current(inf)
                if not ok:
                    rejected += 1
                    continue
                accepted += 1
                sym = inf["symbol"]
                if sym in open_pos or bar < cooldown[sym] or len(open_pos) >= DAY_MAX_OPEN_SLOTS:
                    continue
                slot = float(DAY_TARGET_NOTIONAL_PER_SLOT_USD)
                if deployed + slot > float(DAY_MAX_DEPLOYED_USD) + 1e-9:
                    remain = float(DAY_MAX_DEPLOYED_USD) - deployed
                    if remain < slot * 0.25:
                        continue
                    slot = remain
                bset = bars.get(sym, [])
                bidx = _bar_index(bset, bar)
                if bidx is None:
                    continue
                mid = bset[bidx][4]
                if mid <= 0:
                    continue
                fill = executable_buy_price(mid, sym)
                leaky = fourh_bundle(fourh_full[sym], float(bar))
                _asof_rows, asof_meta = asof_4h_rows(fourh_full[sym], float(bar))
                snap = broken_reason(leaky, fill, float(bar))
                profile = get_coin_profile(sym)
                pre = evaluate_pre_buy_exit_consistency(
                    setup="path_net",
                    entry_price=fill,
                    stop_price=fill * (1.0 - float(profile["sl"])),
                    thesis_invalid_level=0.0,
                    thesis_target_level=fill * (1.0 + float(profile["tp"])),
                    entry_vwap=fill,
                    entry_ts=float(bar),
                    coin_profile=profile,
                    bundle=leaky,
                    bar_ts=float(bar),
                )
                pos = ReplayPos(
                    symbol=sym,
                    entry_price=fill,
                    entry_time=float(bar),
                    quantity=slot / fill,
                    notional=slot,
                    stop_price=fill * (1.0 - float(profile["sl"])),
                    take_profit_1_price=fill * (1.0 + float(profile["tp"])),
                    trail_pct=float(profile["trail"]),
                    highest_price=fill,
                    lowest_price=fill,
                    max_hold_min=int(profile["max_hold_min"]),
                    p_buy=inf["p_buy"],
                    setup="path_net",
                )
                pos._entry_snap = snap  # type: ignore[attr-defined]
                pos._broken_at_entry = bool(snap["broken"])  # type: ignore[attr-defined]
                pos._prebuy_block = not bool(pre.get("allowed"))  # type: ignore[attr-defined]
                pos._prebuy_reason = str(pre.get("block_reason") or "")  # type: ignore[attr-defined]
                pos._entry_4h_open = leaky["4h"][-1][0] if leaky["4h"] else None  # type: ignore[attr-defined]
                last_closed = None
                for r in fourh_full[sym]:
                    if completed_4h_close_epoch(r[0]) <= bar + 1e-9:
                        last_closed = r
                pos._entry_last_closed_open = last_closed[0] if last_closed else None  # type: ignore[attr-defined]
                pos._entry_last_closed_close = (  # type: ignore[attr-defined]
                    completed_4h_close_epoch(last_closed[0]) if last_closed else None
                )
                pos._entry_last_closed_px = last_closed[4] if last_closed else None  # type: ignore[attr-defined]
                pos._entry_asof = asof_meta  # type: ignore[attr-defined]
                open_pos[sym] = pos
                last_adv[sym] = bar
                deployed += slot
                created += 1
                flags["align_none_at_entry"] += int(snap["align"] is None)
                flags["n4_lt_50_at_entry"] += int(snap["n_4h_rows"] < 50)
                flags["broken_at_entry"] += int(snap["broken"])
                flags["prebuy_block"] += int(pos._prebuy_block)  # type: ignore[attr-defined]
                flags["forming_used_at_entry"] += int(
                    pos._entry_4h_open is not None  # type: ignore[attr-defined]
                    and completed_4h_close_epoch(pos._entry_4h_open) > bar + 1  # type: ignore[attr-defined]
                )
        if events:
            flush(int(events[-1]["bar_epoch"]) + 14 * 86400)

    holds_sorted = sorted(holds_sec)
    n = len(holds_sorted)

    def pct(x: int, den: int) -> float:
        return round(100.0 * x / den, 2) if den else 0.0

    report.update(
        {
            "accepted_candidates": accepted,
            "rejected_candidates": rejected,
            "created_positions": created,
            "completed_trades": n_closed,
            "symbol_counts": dict(by_sym),
            "structure_break_exits": n_structure,
            "entry_flags": dict(flags),
            "structure_break_aggregates": {
                "pct_already_true_at_entry": pct(already_broken_at_entry, n_structure),
                "pct_prebuy_would_block": pct(prebuy_would_block, n_structure),
                "pct_same_4h_snapshot_as_entry": pct(same_snapshot, n_structure),
                "pct_exit_before_new_4h_candle_completed": pct(exit_before_new_4h_close, n_structure),
                "pct_future_candle": pct(future_candle, max(n_structure, 1)),
                "near_4h_boundary_pct": {k: pct(v, n_structure) for k, v in near_boundary.items()},
                "counts": {
                    "already_true_at_entry": already_broken_at_entry,
                    "prebuy_would_block": prebuy_would_block,
                    "same_snapshot": same_snapshot,
                    "exit_before_new_4h_close": exit_before_new_4h_close,
                    "partial_forming_treated_as_last_bar": partial_as_closed,
                },
            },
            "hold_time": {
                "n": n,
                "median_seconds": holds_sorted[n // 2] if n else None,
                "median_minutes": (holds_sorted[n // 2] / 60.0) if n else None,
                "median_hours": (holds_sorted[n // 2] / 3600.0) if n else None,
                "p10_seconds": holds_sorted[int(n * 0.1)] if n else None,
                "p90_seconds": holds_sorted[int(n * 0.9)] if n else None,
                "pct_within_1m": pct(sum(1 for h in holds_sorted if h <= 60), n),
                "pct_within_5m": pct(sum(1 for h in holds_sorted if h <= 300), n),
                "pct_within_15m": pct(sum(1 for h in holds_sorted if h <= 900), n),
                "pct_within_30m": pct(sum(1 for h in holds_sorted if h <= 1800), n),
            },
            "samples": samples,
        }
    )
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
