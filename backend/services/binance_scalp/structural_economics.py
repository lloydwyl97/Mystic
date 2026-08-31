"""Structural LP economics. Fee numbers are simulation assumptions, not proven fills."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

MARKOUT_HORIZONS = (1, 3, 5, 10, 30, 60)


@dataclass(frozen=True)
class FeeAssumptions:
    maker_fee_pct: float
    taker_fee_pct: float
    timeout_slip_pct: float
    label: str = "simulation_assumption"

    def as_dict(self) -> dict[str, Any]:
        return {
            "maker_fee_pct": self.maker_fee_pct,
            "taker_fee_pct": self.taker_fee_pct,
            "timeout_slip_pct": self.timeout_slip_pct,
            "label": self.label,
            "note": "Binance.US fee env values used for paper accounting only. Not established live execution.",
        }


def min_quoteable_spread_bps(*, maker_fee_pct: float, min_net_edge_bps: float) -> float:
    """Need observable spread after assumed maker+maker fees before quoting."""
    fee_bps = float(maker_fee_pct) * 10_000.0 * 2.0
    return max(0.0, fee_bps + float(min_net_edge_bps))


def quote_blocked(
    *,
    spread_bps: float,
    maker_fee_pct: float,
    min_net_edge_bps: float,
    recent_range_bps: float,
    max_range_mult: float,
    adverse_1s_rate: float,
    max_adverse: float,
) -> str | None:
    need = min_quoteable_spread_bps(maker_fee_pct=maker_fee_pct, min_net_edge_bps=min_net_edge_bps)
    if spread_bps + 1e-9 < need:
        return "STRUCTURAL_MIN_NET_EDGE"
    if max_range_mult > 0 and recent_range_bps > spread_bps * max_range_mult:
        return "STRUCTURAL_VOLATILITY_GUARD"
    if adverse_1s_rate >= max_adverse > 0:
        return "STRUCTURAL_TOXIC_BOOK"
    return None


def roundtrip_pnl(
    *,
    entry: float,
    exit_px: float,
    qty: float,
    fees: FeeAssumptions,
    exit_maker: bool,
) -> dict[str, float]:
    notional_in = entry * qty
    notional_out = exit_px * qty
    entry_fee = notional_in * fees.maker_fee_pct
    exit_fee = notional_out * (fees.maker_fee_pct if exit_maker else fees.taker_fee_pct)
    slip = 0.0 if exit_maker else notional_out * fees.timeout_slip_pct
    gross_spread = (exit_px - entry) * qty
    net = gross_spread - entry_fee - exit_fee - slip
    return {
        "gross_spread_usd": gross_spread,
        "entry_fee_usd": entry_fee,
        "exit_fee_usd": exit_fee,
        "timeout_slip_usd": slip,
        "net_usd": net,
        "net_pct": net / notional_in if notional_in else 0.0,
    }


def ensure_markout_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_structural_markouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            fill_price REAL NOT NULL,
            fill_ts REAL NOT NULL,
            horizon_sec INTEGER NOT NULL,
            mark_bid REAL,
            mark_ask REAL,
            mark_mid REAL,
            markout_bps REAL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_struct_markouts_trade ON scalp_structural_markouts(trade_id)")


def record_markout(
    conn: sqlite3.Connection,
    *,
    trade_id: str,
    symbol: str,
    side: str,
    fill_price: float,
    fill_ts: float,
    horizon_sec: int,
    bid: float,
    ask: float,
) -> None:
    ensure_markout_table(conn)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    if fill_price <= 0 or mid <= 0:
        bps = None
    elif str(side).upper() == "BUY":
        bps = (mid - fill_price) / fill_price * 10_000.0
    else:
        bps = (fill_price - mid) / fill_price * 10_000.0
    conn.execute(
        """
        INSERT INTO scalp_structural_markouts
        (trade_id, symbol, side, fill_price, fill_ts, horizon_sec, mark_bid, mark_ask, mark_mid, markout_bps)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (trade_id, symbol, side, fill_price, fill_ts, int(horizon_sec), bid, ask, mid, bps),
    )


def structural_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    sells = list(
        conn.execute(
            """
            SELECT symbol, pnl_usd, exit_reason, quantity, price, entry_price, fee_usd, slippage_usd
            FROM scalp_paper_trades
            WHERE side='SELL' AND strategy_id='structural_lp'
              AND IFNULL(diagnostics_json,'') LIKE '%structural_event_queue_v1%'
            """
        )
    )
    by_sym: dict[str, dict[str, float]] = {}
    timeout = 0
    adverse = 0
    n = 0
    net = 0.0
    fees = 0.0
    slip = 0.0
    gross = 0.0
    for sym, pnl, reason, qty, px, entry, fee, sl in sells:
        n += 1
        pnl_f = float(pnl or 0.0)
        net += pnl_f
        fees += float(fee or 0.0)
        slip += float(sl or 0.0)
        if entry and qty:
            gross += (float(px) - float(entry)) * float(qty)
        rec = by_sym.setdefault(str(sym), {"sells": 0, "net": 0.0})
        rec["sells"] += 1
        rec["net"] += pnl_f
        if str(reason or "") == "STRUCTURAL_INVENTORY_TIMEOUT":
            timeout += 1
        if pnl_f < 0:
            adverse += 1
    m1 = conn.execute(
        """
        SELECT AVG(CASE WHEN markout_bps < 0 THEN 1.0 ELSE 0.0 END)
        FROM scalp_structural_markouts WHERE horizon_sec=1 AND side='BUY'
        """
    ).fetchone()
    return {
        "roundtrips": n,
        "net_usd": net,
        "gross_spread_usd": gross,
        "fees_usd": fees,
        "timeout_slip_usd": slip,
        "timeout_rate": (timeout / n) if n else 0.0,
        "adverse_selection_rate": (adverse / n) if n else 0.0,
        "adverse_1s_rate": float(m1[0] or 0.0) if m1 else 0.0,
        "fill_rate_note": "fill_rate is quotes_filled / quotes_posted in quote audit",
        "by_symbol": by_sym,
    }
