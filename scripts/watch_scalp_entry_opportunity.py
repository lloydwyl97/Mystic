#!/usr/bin/env python3
"""Read-only scalp entry opportunity watcher — no orders, SCALP_LIVE=false."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.config import get_scalp_config  # noqa: E402
from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from backend.services.binance_scalp.market_reader import ScalpMarketReader  # noqa: E402
from backend.services.binance_scalp.momentum_tracker import MomentumTracker  # noqa: E402
from backend.services.binance_scalp.protected_preflight import (  # noqa: E402
    DEPTH_INSUFFICIENT,
    PRICE_IMPACT_TOO_HIGH,
    SPREAD_TOO_WIDE,
    run_scalp_preflight,
)

NEAR_PASS_THRESHOLD = 0.0005  # 0.05% from passing
HQ_NEAR_PASS_DISALLOWED = frozenset(
    {SPREAD_TOO_WIDE, DEPTH_INSUFFICIENT, PRICE_IMPACT_TOO_HIGH}
)
LOG = logging.getLogger("scalp_opportunity_watch")


@dataclass
class WatchStats:
    ticks: int = 0
    near_pass_count: int = 0
    pass_count: int = 0
    hq_near_pass_arm_count: int = 0
    arm_events: list[dict] = field(default_factory=list)
    best_btc: dict | None = None
    best_eth: dict | None = None
    events: list[dict] = field(default_factory=list)


def is_high_quality_near_pass(row: dict, econ: ScalpEconomics) -> bool:
    """Conservative pre-arm candidate — does not bypass engine entry gate."""
    if row.get("error") or row.get("preflight_pass"):
        return False
    reject = row.get("reject_reason") or ""
    if reject in HQ_NEAR_PASS_DISALLOWED:
        return False
    def _pct(key: str, default: float = 999.0) -> float:
        val = row.get(key)
        return float(val) if val is not None else default

    spread = _pct("spread_pct")
    buy_i = _pct("buy_impact_pct")
    sell_i = _pct("sell_impact_pct")
    if spread > econ.spread_cap_pct:
        return False
    if buy_i > econ.impact_cap_pct or sell_i > econ.impact_cap_pct:
        return False
    if not econ.is_fee_model_verified():
        return False
    if not row.get("momentum_confirmed"):
        return False
    if not row.get("breakout_confirmed"):
        return False
    if float(row.get("projected_surplus") or -1.0) < 0.0:
        return False
    dist = float((row.get("distance_to_pass") or {}).get("distance_to_pass_pct") or 999.0)
    if dist > NEAR_PASS_THRESHOLD:
        return False
    return True


def build_arm_event(row: dict, *, arm_reason: str) -> dict:
    dist = row.get("distance_to_pass") or {}
    return {
        "event": "HIGH_QUALITY_NEAR_PASS_ARMED"
        if arm_reason == "HIGH_QUALITY_NEAR_PASS"
        else arm_reason,
        "arm_reason": arm_reason,
        "arm_symbol": row.get("symbol"),
        "arm_distance_to_pass": dist.get("distance_to_pass_pct"),
        "arm_projected_gross": row.get("projected_gross"),
        "arm_required_gross": row.get("required_gross"),
        "arm_surplus": row.get("projected_surplus"),
        "arm_ts": row.get("ts"),
        "row": row,
    }


def _distance_to_pass(
    *,
    projected_gross: float,
    required_gross: float,
    projected_surplus: float,
    min_surplus: float,
) -> dict:
    dist_gross = max(0.0, required_gross - projected_gross)
    dist_surplus = max(0.0, min_surplus - projected_surplus)
    combined = max(dist_gross, dist_surplus)
    return {
        "distance_gross_pct": dist_gross,
        "distance_surplus_pct": dist_surplus,
        "distance_to_pass_pct": combined,
    }


def evaluate_symbol(
    sym: str,
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    econ: ScalpEconomics,
    config,
) -> dict:
    snap = reader.read(sym)
    if snap is None:
        return {"symbol": sym, "error": "NO_MARKET_DATA", "ts": time.time()}

    now = time.time()
    tracker.record(sym, now, snap.best_bid, snap.mid)
    mom = tracker.diagnostics(sym, now, snap.best_bid, snap.mid)
    pf = run_scalp_preflight(
        snap,
        econ,
        config,
        side="BUY",
        check_paper_enabled=False,
        momentum=mom,
        apply_entry_gate=True,
    )
    reach = pf.reachability or {}
    projected = float(reach.get("projected_gross_move_pct") or 0.0)
    required = float(reach.get("required_gross_move_pct") or 0.0)
    surplus = float(reach.get("projected_surplus_pct") or 0.0)
    dist = _distance_to_pass(
        projected_gross=projected,
        required_gross=required,
        projected_surplus=surplus,
        min_surplus=econ.min_projected_surplus_pct,
    )

    estimate = reach
    opportunity = None
    if pf.passed:
        opportunity = "OPPORTUNITY_PASS"
    elif dist["distance_to_pass_pct"] <= NEAR_PASS_THRESHOLD:
        opportunity = "OPPORTUNITY_NEAR_PASS"

    buy_impact = float(pf.buy_impact_pct)
    sell_impact = float(pf.sell_impact_pct)

    return {
        "ts": now,
        "symbol": sym,
        "spread_pct": snap.spread_pct,
        "buy_impact_pct": buy_impact,
        "sell_impact_pct": sell_impact,
        "projected_gross": projected,
        "required_gross": required,
        "projected_surplus": surplus,
        "momentum_confirmed": mom.momentum_confirmed,
        "breakout_confirmed": bool(estimate.get("breakout_confirmed")),
        "reject_reason": pf.reject_reason or None,
        "preflight_pass": pf.passed,
        "distance_to_pass": dist,
        "opportunity": opportunity,
        "best_bid": snap.best_bid,
        "best_ask": snap.best_ask,
    }


def warm_momentum(
    reader: ScalpMarketReader,
    tracker: MomentumTracker,
    symbols: tuple[str, ...],
    *,
    rounds: int = 8,
    interval_sec: float = 5.0,
) -> None:
    for _ in range(rounds):
        now = time.time()
        for sym in symbols:
            snap = reader.read(sym)
            if snap:
                tracker.record(sym, now, snap.best_bid, snap.mid)
        time.sleep(interval_sec)


def _update_best(stats: WatchStats, row: dict) -> None:
    if row.get("error"):
        return
    sym = row["symbol"]
    key = "best_btc" if sym == "BTCUSDT" else "best_eth"
    current = getattr(stats, key)
    dist = float(row["distance_to_pass"]["distance_to_pass_pct"])
    if row.get("preflight_pass"):
        dist = -1.0
    if current is None:
        setattr(stats, key, row)
        return
    cur_dist = (
        -1.0
        if current.get("preflight_pass")
        else float(current["distance_to_pass"]["distance_to_pass_pct"])
    )
    if dist < cur_dist:
        setattr(stats, key, row)


def watch_loop(
    *,
    interval_sec: float = 5.0,
    max_sec: float = 7200.0,
    log_path: Path | None = None,
    on_pass: callable | None = None,
    on_arm: callable | None = None,
    arm_on_high_quality_near_pass: bool = False,
) -> WatchStats:
    config = get_scalp_config()
    if config.scalp_live:
        raise RuntimeError("SCALP_LIVE must be false for opportunity watcher")
    econ = ScalpEconomics.from_env()
    reader = ScalpMarketReader(config)
    tracker = MomentumTracker()
    symbols = config.products

    warm_momentum(reader, tracker, symbols)
    stats = WatchStats()
    start = time.time()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)

    log_file = log_path.open("a") if log_path else None
    try:
        while time.time() - start < max_sec:
            tick_rows = []
            arm_this_tick: dict | None = None
            for sym in symbols:
                row = evaluate_symbol(sym, reader, tracker, econ, config)
                tick_rows.append(row)
                _update_best(stats, row)
                opp = row.get("opportunity")
                if opp == "OPPORTUNITY_NEAR_PASS":
                    stats.near_pass_count += 1
                    event = {"event": opp, **row}
                    stats.events.append(event)
                    LOG.info(
                        "%s %s dist=%.6f reject=%s",
                        opp,
                        sym,
                        row["distance_to_pass"]["distance_to_pass_pct"],
                        row["reject_reason"],
                    )
                    if log_file:
                        log_file.write(json.dumps(event, default=str) + "\n")
                    if (
                        arm_on_high_quality_near_pass
                        and arm_this_tick is None
                        and is_high_quality_near_pass(row, econ)
                    ):
                        arm_this_tick = build_arm_event(row, arm_reason="HIGH_QUALITY_NEAR_PASS")
                        stats.hq_near_pass_arm_count += 1
                elif opp == "OPPORTUNITY_PASS":
                    stats.pass_count += 1
                    arm_this_tick = build_arm_event(row, arm_reason="OPPORTUNITY_PASS")
                    event = {"event": opp, **row}
                    stats.events.append(event)
                    LOG.info(
                        "%s %s projected=%.6f required=%.6f",
                        opp,
                        sym,
                        row["projected_gross"],
                        row["required_gross"],
                    )
                    if log_file:
                        log_file.write(json.dumps(event, default=str) + "\n")

            stats.ticks += 1
            if arm_this_tick is not None:
                stats.arm_events.append(arm_this_tick)
                LOG.info(
                    "HIGH_QUALITY_NEAR_PASS_ARMED reason=%s symbol=%s dist=%s gross=%s required=%s surplus=%s",
                    arm_this_tick["arm_reason"],
                    arm_this_tick["arm_symbol"],
                    arm_this_tick["arm_distance_to_pass"],
                    arm_this_tick["arm_projected_gross"],
                    arm_this_tick["arm_required_gross"],
                    arm_this_tick["arm_surplus"],
                )
                if log_file:
                    log_file.write(json.dumps(arm_this_tick, default=str) + "\n")
                if on_arm:
                    on_arm(tick_rows, arm_this_tick)
                elif on_pass:
                    on_pass(tick_rows)
                break
            time.sleep(interval_sec)
    finally:
        if log_file:
            log_file.close()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Scalp entry opportunity watcher")
    parser.add_argument("mode", choices=["once", "watch"])
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--max-sec", type=float, default=7200.0)
    parser.add_argument("--log", type=Path, default=None)
    args = parser.parse_args()

    config = get_scalp_config()
    econ = ScalpEconomics.from_env()
    reader = ScalpMarketReader(config)
    tracker = MomentumTracker()
    symbols = config.products
    warm_momentum(reader, tracker, symbols)

    if args.mode == "once":
        out = [evaluate_symbol(s, reader, tracker, econ, config) for s in symbols]
        print(json.dumps(out, indent=2, default=str))
        return 0

    stats = watch_loop(
        interval_sec=args.interval,
        max_sec=args.max_sec,
        log_path=args.log,
    )
    print(
        json.dumps(
            {
                "ticks": stats.ticks,
                "near_pass_count": stats.near_pass_count,
                "pass_count": stats.pass_count,
                "best_btc": stats.best_btc,
                "best_eth": stats.best_eth,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
