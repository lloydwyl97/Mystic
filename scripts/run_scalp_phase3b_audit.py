#!/usr/bin/env python3
"""Phase 3b tuning audit — read-only analysis of closed scalp trades."""

from __future__ import annotations

import json
import sqlite3
import statistics
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402

DB = REPO / "mystic_trading.db"
SOAK_SELL_IDS = {
    "scalp_paper_ETHUSDT_1780711418267_SELL",
    "scalp_paper_BTCUSDT_1780711822436_SELL",
    "scalp_paper_BTCUSDT_1780712217587_SELL",
    "scalp_paper_BTCUSDT_1780712599763_SELL",
    "scalp_paper_ETHUSDT_1780712917502_SELL",
}


def _fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    base = "https://api.binance.us/api/v3/klines"
    url = (
        f"{base}?symbol={symbol}&interval=1m"
        f"&startTime={start_ms}&endTime={end_ms}&limit=1000"
    )
    proc = subprocess.run(
        ["curl", "-s", "--max-time", "20", url],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return []
    rows = json.loads(proc.stdout)
    if isinstance(rows, dict) and rows.get("code"):
        return []
    out = []
    for r in rows:
        out.append(
            {
                "open_time_ms": int(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
            }
        )
    return out


def _parse_ts(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)


def _entry_diag(position_row: sqlite3.Row | None, buy_row: sqlite3.Row) -> dict:
    raw = position_row["diagnostics_json"] if position_row else buy_row["diagnostics_json"]
    if not raw:
        return {}
    data = json.loads(raw)
    pf = data.get("entry_preflight") or data.get("preflight") or data
    return pf


def _imbalance_from_entry(pf: dict, snap_spread: float) -> float | None:
    edge = pf.get("expected_net_edge_pct")
    spread = pf.get("spread_pct", snap_spread)
    if edge is None:
        return None
    econ = ScalpEconomics.from_env()
    costs = econ.roundtrip_cost_pct(
        spread,
        pf.get("buy_impact_pct", 0.0),
        pf.get("sell_impact_pct", 0.0),
    )
    projected = edge + costs
    imb_proxy = projected / (spread * 8.0) if spread > 0 else None
    return imb_proxy


def _profit_target_gross(entry: float, spread: float, buy_i: float, sell_i: float, econ: ScalpEconomics) -> float:
    costs = econ.roundtrip_cost_pct(spread, buy_i, sell_i)
    return entry * (costs + econ.net_profit_target_pct)


def _net_at_bid(entry: float, bid: float, qty: float, spread: float, sell_i: float, econ: ScalpEconomics) -> tuple[float, float]:
    gross = (bid - entry) / entry if entry > 0 else -1.0
    costs = econ.roundtrip_cost_pct(spread, 0.0, sell_i)
    net_pct = gross - costs
    fee = bid * qty * econ.taker_fee_pct * 2  # entry + exit approx
    slip = bid * qty * econ.slippage_buffer_pct
    net_usd = (bid - entry) * qty - fee - slip
    return net_pct, net_usd


def _hypothetical_exit(
    klines: list[dict],
    entry: float,
    qty: float,
    entry_ts: datetime,
    timeout_sec: int,
    econ: ScalpEconomics,
    spread_assumption: float,
) -> dict:
    end_ts = entry_ts.timestamp() + timeout_sec
    best_bid = entry
    best_net_pct = -999.0
    best_net_usd = -999.0
    exit_bid = entry
    hit_target = False
    exit_at = entry_ts.timestamp()

    for k in klines:
        t = k["open_time_ms"] / 1000.0
        if t < entry_ts.timestamp():
            continue
        if t > end_ts:
            break
        bid_proxy = k["low"]  # conservative bid proxy
        net_pct, net_usd = _net_at_bid(entry, bid_proxy, qty, spread_assumption, 0.0, econ)
        if bid_proxy > best_bid:
            best_bid = bid_proxy
        if net_pct > best_net_pct:
            best_net_pct = net_pct
            best_net_usd = net_usd
        if net_pct >= econ.net_profit_target_pct and not hit_target:
            hit_target = True
        exit_bid = k["close"]
        exit_at = t

    final_bid = exit_bid
    for k in klines:
        t = k["open_time_ms"] / 1000.0
        if entry_ts.timestamp() <= t <= end_ts:
            final_bid = k["close"]
            exit_at = t

    final_net_pct, final_net_usd = _net_at_bid(entry, final_bid, qty, spread_assumption, 0.0, econ)
    return {
        "timeout_sec": timeout_sec,
        "best_bid_proxy": best_bid,
        "best_net_pct": best_net_pct,
        "best_net_usd": best_net_usd,
        "final_bid": final_bid,
        "final_net_usd": final_net_usd,
        "hit_target_within_window": hit_target,
    }


def analyze_trade(conn: sqlite3.Connection, sell_row: sqlite3.Row, econ: ScalpEconomics) -> dict:
    sell_tid = sell_row["trade_id"]
    buy_tid = sell_tid.replace("_SELL", "")
    buy = conn.execute(
        "SELECT * FROM scalp_paper_trades WHERE trade_id=?", (buy_tid,)
    ).fetchone()
    pos = conn.execute(
        "SELECT * FROM scalp_paper_positions WHERE trade_id=?", (buy_tid,)
    ).fetchone()
    if not buy:
        raise RuntimeError(f"missing buy for {sell_tid}")

    entry_ts = _parse_ts(buy["created_at"])
    exit_ts = _parse_ts(sell_row["created_at"])
    hold = (exit_ts - entry_ts).total_seconds()
    entry = float(buy["price"])
    exit_p = float(sell_row["price"])
    qty = float(buy["quantity"])
    gross_move = (exit_p - entry) / entry
    net_usd = float(sell_row["pnl_usd"] or 0)

    entry_pf = _entry_diag(pos, buy)
    spread_e = float(entry_pf.get("spread_pct", 0))
    buy_i = float(entry_pf.get("buy_impact_pct", 0))
    sell_i = float(entry_pf.get("sell_impact_pct", 0))
    projected = float(entry_pf.get("expected_net_edge_pct", 0))

    sell_diag = json.loads(sell_row["diagnostics_json"] or "{}")
    sell_pf = sell_diag.get("preflight", {})

    start_ms = int(entry_ts.timestamp() * 1000) - 60_000
    end_ms = int(exit_ts.timestamp() * 1000) + 600_000
    klines = _fetch_klines(str(buy["symbol"]), start_ms, end_ms)

    max_fav = 0.0
    max_adv = 0.0
    best_bid = entry
    for k in klines:
        t = k["open_time_ms"] / 1000.0
        if t < entry_ts.timestamp() or t > exit_ts.timestamp():
            continue
        hi = k["high"]
        lo = k["low"]
        max_fav = max(max_fav, (hi - entry) / entry)
        max_adv = min(max_adv, (lo - entry) / entry)
        best_bid = max(best_bid, hi)

    target_bid = _profit_target_gross(entry, spread_e, buy_i, sell_i, econ)
    best_net_pct, _ = _net_at_bid(entry, best_bid, qty, spread_e, sell_i, econ)
    reached = best_net_pct >= econ.net_profit_target_pct
    gap_pct = econ.net_profit_target_pct - best_net_pct if not reached else 0.0

    post_klines = _fetch_klines(
        str(buy["symbol"]),
        int(exit_ts.timestamp() * 1000),
        int(exit_ts.timestamp() * 1000) + 600_000,
    )
    post_recovery = 0.0
    if post_klines:
        post_hi = max(k["high"] for k in post_klines)
        post_recovery = (post_hi - exit_p) / exit_p

    hypos = {
        str(t): _hypothetical_exit(klines, entry, qty, entry_ts, t, econ, spread_e)
        for t in (180, 300, 600)
    }

    immediate_underwater = spread_e + econ.entry_fee_pct() + buy_i

    return {
        "symbol": buy["symbol"],
        "trade_id": buy_tid,
        "entry_time": buy["created_at"],
        "exit_time": sell_row["created_at"],
        "hold_seconds": hold,
        "entry_price": entry,
        "exit_price": exit_p,
        "gross_move_pct": gross_move * 100,
        "net_pnl_usd": net_usd,
        "spread_at_entry_pct": spread_e * 100,
        "impact_at_entry_pct": buy_i * 100,
        "orderbook_imbalance_proxy": _imbalance_from_entry(entry_pf, spread_e),
        "projected_edge_at_entry_pct": projected * 100,
        "max_favorable_move_pct": max_fav * 100,
        "max_adverse_move_pct": max_adv * 100,
        "price_reached_profit_target": reached,
        "gap_from_target_at_best_pct": gap_pct * 100,
        "profit_target_bid_needed": target_bid,
        "best_bid_during_hold": best_bid,
        "exit_reason": sell_row["exit_reason"],
        "entry_levels_consumed": entry_pf.get("levels_consumed"),
        "entry_immediately_underwater_pct": immediate_underwater * 100,
        "post_exit_recovery_pct": post_recovery * 100,
        "hypothetical_timeouts": hypos,
        "exit_preflight_spread_pct": float(sell_pf.get("spread_pct", 0)) * 100,
        "exit_expected_net_pct": float(sell_pf.get("expected_net_edge_pct", 0)) * 100,
    }


def main() -> int:
    econ = ScalpEconomics.from_env()
    with sqlite3.connect(DB) as conn:
        conn.row_factory = sqlite3.Row
        open_rows = conn.execute(
            "SELECT id, symbol, status, entry_time, entry_price, trade_id FROM scalp_paper_positions WHERE status='OPEN'"
        ).fetchall()

        sells = conn.execute(
            """
            SELECT * FROM scalp_paper_trades
            WHERE side='SELL' AND exit_reason='STALE_SCALP_TIMEOUT'
            ORDER BY id DESC LIMIT 5
            """
        ).fetchall()
        sells = list(reversed(sells))

        trades = [analyze_trade(conn, s, econ) for s in sells]

    fav_moves = [t["max_favorable_move_pct"] for t in trades]
    req_gross = econ.net_profit_target_pct * 100
    avg_costs = statistics.mean(
        t["spread_at_entry_pct"] + t["impact_at_entry_pct"] * 2 + econ.roundtrip_fee_pct * 100
        for t in trades
    )

    report = {
        "open_positions_before_audit": [dict(r) for r in open_rows],
        "economics": econ.as_dict(),
        "required_gross_move_pct_estimate": req_gross + avg_costs,
        "net_profit_target_pct": econ.net_profit_target_pct * 100,
        "stale_timeout_sec": econ.stale_scalp_timeout_sec,
        "trades": trades,
        "aggregate": {
            "avg_max_favorable_move_pct": statistics.mean(fav_moves) if fav_moves else 0,
            "median_max_favorable_move_pct": statistics.median(fav_moves) if fav_moves else 0,
            "avg_hold_sec": statistics.mean([t["hold_seconds"] for t in trades]),
            "any_reached_target": any(t["price_reached_profit_target"] for t in trades),
            "hypo_180_total_usd": sum(
                t["hypothetical_timeouts"]["180"]["final_net_usd"] for t in trades
            ),
            "hypo_300_total_usd": sum(
                t["hypothetical_timeouts"]["300"]["final_net_usd"] for t in trades
            ),
            "hypo_600_total_usd": sum(
                t["hypothetical_timeouts"]["600"]["final_net_usd"] for t in trades
            ),
            "actual_total_usd": sum(t["net_pnl_usd"] for t in trades),
        },
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
