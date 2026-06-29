#!/usr/bin/env python3
"""Replay historical STALE_SCALP_TIMEOUT exits against new exit manager."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.calibration_profiles import apply_profile, economics_for_config
from backend.services.binance_scalp.config import ScalpConfig, get_scalp_config
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.exit_manager import (
    DECISION_SELL,
    EXIT_MOMENTUM_FAILED,
    EXIT_NET_PROFIT_TARGET,
    PositionTrack,
    evaluate_exit,
)
from backend.services.binance_scalp.momentum_tracker import MomentumTracker

NOTIONAL = 25.0
REVIEW_INTERVAL = 30


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={cursor}&endTime={end_ms}&limit=1000"
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "30", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        rows = json.loads(proc.stdout)
        if not isinstance(rows, list) or not rows:
            break
        for r in rows:
            bars.append(
                {
                    "ts_ms": int(r[0]),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                }
            )
        last_ms = int(rows[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + 60_000
        time.sleep(0.08)
    return bars


def parse_ts(ts: str) -> int:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return int(datetime.strptime(ts[:19], fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)
        except ValueError:
            continue
    return 0


class _Snap:
    def __init__(self, symbol: str, bid: float, ask: float, spread_pct: float) -> None:
        self.symbol = symbol
        self.best_bid = bid
        self.best_ask = ask
        self.mid = (bid + ask) / 2.0
        self.spread_pct = spread_pct


def simulate_new_manager(
    symbol: str,
    entry_px: float,
    entry_ms: int,
    bars: list[dict],
    econ: ScalpEconomics,
    config: ScalpConfig,
    stale_at_ms: int,
) -> dict:
    tracker = MomentumTracker()
    track = PositionTrack(entry_px, "OPEN", 0.0, 0.0, entry_px, 0, ())
    last_review_ms = 0
    stale_sec = econ.stale_scalp_timeout_sec
    hard_sec = max(stale_sec * 3, 900)

    decision_at_stale = "HOLD"
    state_at_stale = "STALE_REVIEW"

    for _i, bar in enumerate(bars):
        ts_ms = bar["ts_ms"]
        hold_sec = (ts_ms - entry_ms) / 1000.0
        bid = bar["close"] * 0.9998
        ask = bar["close"] * 1.0002
        sp = (ask - bid) / ((ask + bid) / 2)
        snap = _Snap(symbol, bid, ask, sp)
        epoch = ts_ms / 1000.0
        tracker.record(symbol, epoch, bid, snap.mid)
        mom = tracker.diagnostics(symbol, epoch, bid, snap.mid)

        rt = econ.roundtrip_cost_pct(sp, 0.0, 0.0)
        target_px = entry_px * (1.0 + econ.net_profit_target_pct + rt)
        profit_hit = bar["high"] >= target_px
        exit_px = target_px if profit_hit else bid
        gross = (exit_px - entry_px) / entry_px
        net_pct = gross - rt

        perform_review = hold_sec >= stale_sec and (track.stale_review_count == 0 or (ts_ms - last_review_ms) >= REVIEW_INTERVAL * 1000)
        if perform_review:
            last_review_ms = ts_ms

        review = evaluate_exit(
            track=track,
            snap=snap,
            mom=mom,
            econ=econ,
            config=config,
            trade_id="replay",
            hold_sec=hold_sec,
            executable_net_pct=net_pct,
            profit_hit=profit_hit,
            exit_spread_ok=True,
            perform_review=perform_review,
        )
        track = review.updated_track

        if abs(hold_sec - stale_sec) < 65 and perform_review:
            decision_at_stale = review.decision
            state_at_stale = review.state

        if review.decision == DECISION_SELL and review.exit_reason:
            pnl = NOTIONAL * net_pct
            return {
                "new_exit_reason": review.exit_reason,
                "new_hold_sec": hold_sec,
                "new_exit_price": exit_px,
                "new_pnl_usd": round(pnl, 4),
                "decision_at_stale": decision_at_stale,
                "state_at_stale": state_at_stale,
                "would_hold_at_stale": decision_at_stale == "HOLD",
            }

        if hold_sec >= hard_sec:
            pnl = NOTIONAL * net_pct
            return {
                "new_exit_reason": "MAX_HOLD_HARD_LIMIT",
                "new_hold_sec": hold_sec,
                "new_exit_price": bid,
                "new_pnl_usd": round(pnl, 4),
                "decision_at_stale": decision_at_stale,
                "state_at_stale": state_at_stale,
                "would_hold_at_stale": decision_at_stale == "HOLD",
            }

    bid = bars[-1]["close"] if bars else entry_px
    rt = econ.roundtrip_cost_pct(0.0003, 0.0, 0.0)
    net_pct = (bid - entry_px) / entry_px - rt
    return {
        "new_exit_reason": "REPLAY_END",
        "new_hold_sec": (bars[-1]["ts_ms"] - entry_ms) / 1000 if bars else 0,
        "new_exit_price": bid,
        "new_pnl_usd": round(NOTIONAL * net_pct, 4),
        "decision_at_stale": decision_at_stale,
        "state_at_stale": state_at_stale,
        "would_hold_at_stale": decision_at_stale == "HOLD",
    }


def main() -> int:
    db = REPO / "mystic_trading.db"
    config = get_scalp_config()
    econ = economics_for_config(config)
    econ = apply_profile(ScalpEconomics.from_env(), "moderate")
    econ = replace(econ, paper_spread_caps=__import__("backend.services.binance_scalp.paper_spread_caps", fromlist=["parse_paper_spread_caps_json"]).parse_paper_spread_caps_json())

    with sqlite3.connect(db) as conn:
        stale_trades = conn.execute(
            """
            SELECT t.trade_id, t.symbol, t.entry_price, t.price AS exit_price,
                   t.pnl_usd, t.created_at AS exit_ts, b.created_at AS entry_ts
            FROM scalp_paper_trades t
            JOIN scalp_paper_trades b ON b.trade_id = replace(t.trade_id, '_SELL', '')
            WHERE t.side='SELL' AND t.exit_reason='STALE_SCALP_TIMEOUT'
            ORDER BY t.created_at
            """
        ).fetchall()

    results: list[dict] = []
    by_sym: dict[str, dict] = defaultdict(lambda: {"n": 0, "old_pnl": 0.0, "new_pnl": 0.0, "hold_at_stale": 0, "momentum_fail": 0})

    for row in stale_trades:
        tid, sym, entry_px, _old_exit, old_pnl, exit_ts, entry_ts = row
        entry_ms = parse_ts(entry_ts)
        exit_ms = parse_ts(exit_ts)
        bars = fetch_klines(sym, entry_ms - 60_000, exit_ms + 900_000)
        sim = simulate_new_manager(sym, float(entry_px), entry_ms, bars, econ, config, exit_ms)
        improved = (sim["new_pnl_usd"] or 0) > (old_pnl or 0)
        rec = {
            "trade_id": tid,
            "symbol": sym,
            "entry_ts": entry_ts,
            "exit_ts": exit_ts,
            "old_pnl_usd": old_pnl,
            "old_exit_reason": "STALE_SCALP_TIMEOUT",
            **sim,
            "pnl_improved": improved,
        }
        results.append(rec)
        by_sym[sym]["n"] += 1
        by_sym[sym]["old_pnl"] += old_pnl or 0
        by_sym[sym]["new_pnl"] += sim["new_pnl_usd"]
        if sim["would_hold_at_stale"]:
            by_sym[sym]["hold_at_stale"] += 1
        if sim["new_exit_reason"] == EXIT_MOMENTUM_FAILED:
            by_sym[sym]["momentum_fail"] += 1

    old_total = sum(r["old_pnl_usd"] or 0 for r in results)
    new_total = sum(r["new_pnl_usd"] or 0 for r in results)
    hold_at_stale = sum(1 for r in results if r["would_hold_at_stale"])
    momentum_fail = sum(1 for r in results if r["new_exit_reason"] == EXIT_MOMENTUM_FAILED)
    profit_now = sum(1 for r in results if r["new_exit_reason"] == EXIT_NET_PROFIT_TARGET)

    out = {
        "stale_exits_replayed": len(results),
        "old_total_pnl_usd": round(old_total, 4),
        "new_total_pnl_usd": round(new_total, 4),
        "pnl_delta_usd": round(new_total - old_total, 4),
        "would_hold_at_stale_time": hold_at_stale,
        "would_momentum_failed_exit": momentum_fail,
        "would_profit_target_exit": profit_now,
        "by_symbol": dict(by_sym),
        "trades": results,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
