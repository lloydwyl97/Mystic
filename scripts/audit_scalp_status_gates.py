#!/usr/bin/env python3
"""Read-only audit: spread, momentum warm, projection, gate order, parity."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.market_reader import ScalpMarketReader, fetch_depth_sync
from backend.services.binance_scalp.momentum_gross_estimate import compute_momentum_gross_estimate
from backend.services.binance_scalp.momentum_tracker import MomentumTracker
from backend.services.binance_scalp.orderbook_book import walk_buy_notional, walk_sell_qty
from backend.services.binance_scalp.protected_preflight import run_scalp_preflight
from backend.services.binance_scalp.status_snapshot import build_scalp_status
from scripts.watch_scalp_entry_opportunity import (
    evaluate_symbol,
    is_high_quality_near_pass,
    warm_momentum,
)


def _spread_audit(symbols: tuple[str, ...], config, econ) -> dict:
    reader = ScalpMarketReader(config)
    out = {}
    for sym in symbols:
        t0 = time.time()
        bids, asks = fetch_depth_sync(sym)
        fetch_ms = (time.time() - t0) * 1000.0
        snap = reader.read(sym)
        rest_bid = float(bids[0][0]) if bids else None
        rest_ask = float(asks[0][0]) if asks else None
        rest_mid = (rest_bid + rest_ask) / 2.0 if rest_bid and rest_ask else None
        rest_spread = (rest_ask - rest_bid) / rest_mid if rest_mid else None
        internal = None
        if snap:
            internal = {
                "best_bid": snap.best_bid,
                "best_ask": snap.best_ask,
                "spread_pct": snap.spread_pct,
                "spread_pct_display": snap.spread_pct * 100.0,
                "book_source": snap.book_source,
                "redis_spread_pct": snap.redis_spread_pct,
            }
        out[sym] = {
            "rest_depth": {
                "best_bid": rest_bid,
                "best_ask": rest_ask,
                "spread_pct_decimal": rest_spread,
                "spread_pct_display": (rest_spread * 100.0) if rest_spread else None,
                "fetch_ms": fetch_ms,
                "fetched_at_epoch": t0,
            },
            "internal_reader": internal,
            "bid_match": abs((rest_bid or 0) - (snap.best_bid if snap else 0)) < 0.02 if snap and rest_bid else None,
            "ask_match": abs((rest_ask or 0) - (snap.best_ask if snap else 0)) < 0.02 if snap and rest_ask else None,
            "spread_delta_pct": (abs(rest_spread - snap.spread_pct) if rest_spread and snap else None),
            "spread_over_cap": rest_spread > econ.spread_cap_pct if rest_spread else None,
            "spread_cap_decimal": econ.spread_cap_pct,
            "spread_cap_display_pct": econ.spread_cap_pct * 100.0,
        }
    return out


def _warm_compare(rounds_list: list[int]) -> dict:
    return {
        str(n): {
            "overall": build_scalp_status(warm_rounds=n)["overall_decision"],
            "symbols": {
                sym: {
                    k: row.get(k)
                    for k in (
                        "decision",
                        "reject_reason",
                        "momentum_confirmed",
                        "breakout_confirmed",
                        "projected_gross_pct",
                        "required_gross_pct",
                        "distance_to_pass",
                        "momentum_history_sec",
                        "momentum_samples",
                        "spread_pct",
                    )
                }
                for sym, row in build_scalp_status(warm_rounds=n)["symbols"].items()
            },
        }
        for n in rounds_list
    }


def _projection_audit(sym: str, reader, tracker, econ, config, warm: int) -> dict:
    warm_momentum(reader, tracker, (sym,), rounds=warm, interval_sec=5.0)
    snap = reader.read(sym)
    if not snap:
        return {"error": "NO_MARKET_DATA"}
    now = time.time()
    tracker.record(sym, now, snap.best_bid, snap.mid)
    mom = tracker.diagnostics(sym, now, snap.best_bid, snap.mid)
    notional = config.max_notional_paper
    buy_walk = walk_buy_notional(snap.asks, notional, snap.best_ask)
    walk_sell_qty(snap.bids, buy_walk.filled_qty or notional / snap.best_ask, snap.best_bid)
    estimate = compute_momentum_gross_estimate(snap, mom, econ)
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="BUY",
        notional_usd=notional,
        check_paper_enabled=False,
        momentum=mom,
        apply_entry_gate=True,
    )
    return {
        "spread_pct": snap.spread_pct,
        "gate_reject": pf.reject_reason,
        "gate_reachability": pf.reachability,
        "raw_estimate": estimate.as_dict(),
        "momentum": mom.as_dict(),
        "projection_zeroed_by_spread_gate": (pf.reject_reason == "SPREAD_TOO_WIDE" and not pf.reachability),
    }


def _parity(sym: str, reader, tracker, econ, config, warm: int) -> dict:
    warm_momentum(reader, tracker, config.products, rounds=warm, interval_sec=5.0)
    watch_row = evaluate_symbol(sym, reader, tracker, econ, config)
    build_scalp_status(warm_rounds=0)["symbols"].get(sym, {})
    # status warm=0 after shared warm above - rebuild single symbol eval
    tracker2 = MomentumTracker()
    warm_momentum(reader, tracker2, config.products, rounds=warm, interval_sec=5.0)
    snap = reader.read(sym)
    now = time.time()
    tracker2.record(sym, now, snap.best_bid, snap.mid)
    mom = tracker2.diagnostics(sym, now, snap.best_bid, snap.mid)
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="BUY",
        notional_usd=config.max_notional_paper,
        check_paper_enabled=False,
        momentum=mom,
        apply_entry_gate=True,
    )
    eval_row = {
        "symbol": sym,
        "spread_pct": snap.spread_pct,
        "buy_impact_pct": pf.buy_impact_pct,
        "sell_impact_pct": pf.sell_impact_pct,
        "projected_gross": float((pf.reachability or {}).get("projected_gross_move_pct") or 0),
        "required_gross": float((pf.reachability or {}).get("required_gross_move_pct") or 0),
        "projected_surplus": float((pf.reachability or {}).get("projected_surplus_pct") or 0),
        "momentum_confirmed": mom.momentum_confirmed,
        "breakout_confirmed": bool((pf.reachability or {}).get("breakout_confirmed")),
        "reject_reason": pf.reject_reason,
        "preflight_pass": pf.passed,
        "distance_to_pass": {"distance_to_pass_pct": 0},
    }
    from scripts.watch_scalp_entry_opportunity import _distance_to_pass

    eval_row["distance_to_pass"] = _distance_to_pass(
        projected_gross=eval_row["projected_gross"],
        required_gross=eval_row["required_gross"],
        projected_surplus=eval_row["projected_surplus"],
        min_surplus=econ.min_projected_surplus_pct,
    )
    hq = is_high_quality_near_pass(eval_row, econ)
    return {
        "watcher_preflight_pass": watch_row.get("preflight_pass"),
        "watcher_reject": watch_row.get("reject_reason"),
        "watcher_hq_arm": is_high_quality_near_pass(watch_row, econ),
        "parity_pf_pass": pf.passed,
        "parity_hq_arm": hq,
        "mismatch": watch_row.get("preflight_pass") != pf.passed,
    }


def main() -> int:
    config = get_scalp_config()
    econ = ScalpEconomics.from_env()
    symbols = config.products
    reader = ScalpMarketReader(config)

    report = {
        "decimal_env": {
            "spread_cap": econ.spread_cap_pct,
            "impact_cap": econ.impact_cap_pct,
            "net_profit_target": econ.net_profit_target_pct,
            "entry_edge_buffer": econ.entry_edge_buffer_pct,
            "min_projected_surplus": econ.min_projected_surplus_pct,
            "note": "all decimals: 0.0005=0.05%, 0.0025=0.25%",
        },
        "spread_audit": _spread_audit(symbols, config, econ),
        "warm_round_compare": _warm_compare([6, 12, 18]),
        "projection_audit": {sym: _projection_audit(sym, reader, MomentumTracker(), econ, config, 12) for sym in symbols},
        "gate_order_note": ("protected_preflight BUY: fee->paper->SPREAD(early return)->impact/depth->momentum estimate->entry_gate. SPREAD_TOO_WIDE returns before projection."),
        "parity_warm12": {sym: _parity(sym, reader, MomentumTracker(), econ, config, 12) for sym in symbols},
        "momentum_60s_requirement": {
            "rising_60_requires_sample_at_60s": True,
            "warm_6_history_sec_approx": "25-35s",
            "warm_12_history_sec_approx": "55-65s",
            "warm_18_history_sec_approx": "85-95s",
            "bug_risk": "warm_rounds=6 cannot satisfy 60s momentum_confirmed by design",
        },
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
