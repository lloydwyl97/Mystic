"""Ocean live DAY 66-fill production book. Analysis only.

Window: 2026-08-25 <= timestamp < 2026-09-02, strategy_id=day, mode=live, side=BUY.
First 53 fills are the briefing subset. Does not change production.
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from typing import Any

from backend.services.day_production_lifecycle_replay import parse_epoch

WINDOW_START = "2026-08-25"
WINDOW_END = "2026-09-02"
BRIEFING_N = 53
STRATEGY = "day"
MODE = "live"


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def load_live_buys(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    cols = [
        c
        for c in (
            "id",
            "timestamp",
            "symbol",
            "decision_id",
            "order_id",
            "trade_id",
            "quantity",
            "price",
            "remaining_position",
            "fees_paid",
            "slippage_cost",
            "spread_pct_used",
            "slippage_pct_used",
            "slippage_pct_implied",
            "status",
        )
        if c in present
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(cols)}
        FROM paper_trades
        WHERE strategy_id=? AND side='BUY' AND mode=? AND timestamp>=? AND timestamp<?
        ORDER BY timestamp
        """,
        (STRATEGY, MODE, WINDOW_START, WINDOW_END),
    ).fetchall()
    idx = {n: i for i, n in enumerate(cols)}
    out = []
    for row in rows:
        qty = float(row[idx["quantity"]] or 0)
        px = float(row[idx["price"]] or 0)
        out.append(
            {
                "id": row[idx["id"]] if "id" in idx else None,
                "timestamp": row[idx["timestamp"]],
                "epoch": parse_epoch(row[idx["timestamp"]]),
                "symbol": _api(row[idx["symbol"]]),
                "decision_id": str(row[idx["decision_id"]] or "") if "decision_id" in idx else "",
                "order_id": str(row[idx["order_id"]] or "") if "order_id" in idx else "",
                "trade_id": str(row[idx["trade_id"]] or "") if "trade_id" in idx else "",
                "quantity": qty,
                "price": px,
                "notional": qty * px,
                "fees_paid": float(row[idx["fees_paid"]] or 0) if "fees_paid" in idx else 0.0,
                "slippage_cost": float(row[idx["slippage_cost"]] or 0) if "slippage_cost" in idx else 0.0,
                "spread_pct_used": row[idx["spread_pct_used"]] if "spread_pct_used" in idx else None,
                "slippage_pct_used": row[idx["slippage_pct_used"]] if "slippage_pct_used" in idx else None,
            }
        )
    return out


def load_live_sells(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    present = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    cols = [
        c
        for c in (
            "id",
            "timestamp",
            "symbol",
            "decision_id",
            "quantity",
            "price",
            "pnl",
            "pnl_pct",
            "exit_reason",
            "hold_time_seconds",
            "fees_paid",
            "slippage_cost",
            "spread_pct_used",
            "slippage_pct_used",
        )
        if c in present
    ]
    rows = conn.execute(
        f"""
        SELECT {", ".join(cols)}
        FROM paper_trades
        WHERE strategy_id=? AND side='SELL' AND mode=? AND timestamp>=?
        ORDER BY timestamp
        """,
        (STRATEGY, MODE, WINDOW_START),
    ).fetchall()
    idx = {n: i for i, n in enumerate(cols)}
    out = []
    for row in rows:
        out.append(
            {
                "id": row[idx["id"]] if "id" in idx else None,
                "timestamp": row[idx["timestamp"]],
                "epoch": parse_epoch(row[idx["timestamp"]]),
                "symbol": _api(row[idx["symbol"]]),
                "decision_id": str(row[idx["decision_id"]] or "") if "decision_id" in idx else "",
                "quantity": float(row[idx["quantity"]] or 0),
                "price": float(row[idx["price"]] or 0),
                "pnl": float(row[idx["pnl"]] or 0) if "pnl" in idx else 0.0,
                "pnl_pct": float(row[idx["pnl_pct"]] or 0) if "pnl_pct" in idx else None,
                "exit_reason": str(row[idx["exit_reason"]] or "") if "exit_reason" in idx else "",
                "hold_sec": float(row[idx["hold_time_seconds"]] or 0) if "hold_time_seconds" in idx else 0.0,
                "fees_paid": float(row[idx["fees_paid"]] or 0) if "fees_paid" in idx else 0.0,
                "slippage_cost": float(row[idx["slippage_cost"]] or 0) if "slippage_cost" in idx else 0.0,
            }
        )
    return out


def pair_book(buys: list[dict[str, Any]], sells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_sym: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in sells:
        by_sym[s["symbol"]].append(s)
    used: set[Any] = set()
    pairs = []
    for i, b in enumerate(buys):
        match = None
        for s in by_sym[b["symbol"]]:
            if s["id"] in used:
                continue
            if str(s["timestamp"]) >= str(b["timestamp"]):
                match = s
                used.add(s["id"])
                break
        briefing = i < BRIEFING_N
        if match is None:
            pairs.append({**b, "paired": False, "briefing53": briefing, "counterfactual_authority": "unknown_open_or_unpaired"})
            continue
        rt_comm_bps = ((b["fees_paid"] + match["fees_paid"]) / b["notional"] * 1e4) if b["notional"] else None
        entry_slip_bps = (b["slippage_cost"] / b["notional"] * 1e4) if b["notional"] else None
        exit_slip_bps = (match["slippage_cost"] / b["notional"] * 1e4) if b["notional"] else None
        net_bps = (match["pnl"] / b["notional"] * 1e4) if b["notional"] else None
        pairs.append(
            {
                **b,
                "paired": True,
                "briefing53": briefing,
                "sell_id": match["id"],
                "sell_ts": match["timestamp"],
                "exit_reason": match["exit_reason"],
                "hold_sec": match["hold_sec"],
                "pnl_usd": match["pnl"],
                "net_bps": net_bps,
                "round_trip_commission_bps": rt_comm_bps,
                "entry_slippage_bps": entry_slip_bps,
                "exit_slippage_bps": exit_slip_bps,
                "commission_plus_slip_bps": (rt_comm_bps or 0) + (entry_slip_bps or 0) + (exit_slip_bps or 0),
                "counterfactual_authority": "unknown_without_group_pit",
            }
        )
    return pairs


def summarize_book(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    closed = [p for p in pairs if p.get("paired")]
    n = len(pairs)
    notion = sum(float(p["notional"]) for p in pairs)
    pnl = sum(float(p.get("pnl_usd") or 0) for p in closed)
    briefing = [p for p in pairs if p.get("briefing53")]
    by_sym: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "pnl": 0.0, "notional": 0.0})
    by_exit: dict[str, dict[str, Any]] = defaultdict(lambda: {"n": 0, "pnl": 0.0})
    for p in closed:
        by_sym[p["symbol"]]["n"] += 1
        by_sym[p["symbol"]]["pnl"] += float(p.get("pnl_usd") or 0)
        by_sym[p["symbol"]]["notional"] += float(p["notional"])
        by_exit[str(p.get("exit_reason") or "")]["n"] += 1
        by_exit[str(p.get("exit_reason") or "")]["pnl"] += float(p.get("pnl_usd") or 0)
    comm = sum(float(p.get("round_trip_commission_bps") or 0) for p in closed) / len(closed) if closed else None
    return {
        "host": "mystic-prod",
        "window": [WINDOW_START, WINDOW_END],
        "strategy": STRATEGY,
        "mode": MODE,
        "fills": n,
        "briefing_subset": BRIEFING_N,
        "closed": len(closed),
        "actual_pnl_usd": pnl,
        "actual_pnl_usd_briefing53": sum(float(p.get("pnl_usd") or 0) for p in briefing if p.get("paired")),
        "notional_usd": notion,
        "actual_net_bps": (pnl / notion * 1e4) if notion else None,
        "mean_rt_commission_bps": comm,
        "mean_entry_slip_bps": sum(float(p.get("entry_slippage_bps") or 0) for p in closed) / len(closed) if closed else None,
        "mean_exit_slip_bps": sum(float(p.get("exit_slippage_bps") or 0) for p in closed) / len(closed) if closed else None,
        "mean_commission_plus_slip_bps": sum(float(p.get("commission_plus_slip_bps") or 0) for p in closed) / len(closed) if closed else None,
        "by_symbol": dict(by_sym),
        "by_exit": dict(by_exit),
        "order_id_nonempty": sum(1 for p in pairs if p.get("order_id")),
        "decision_id_nonempty": sum(1 for p in pairs if p.get("decision_id")),
        "counterfactuals": "unknown_without_point_in_time_tape_and_quotes_on_this_extract",
        "oracle_not_used_as_edge": True,
    }


def report_ocean_66(conn: sqlite3.Connection) -> dict[str, Any]:
    buys = load_live_buys(conn)
    sells = load_live_sells(conn)
    pairs = pair_book(buys, sells)
    return summarize_book(pairs)
