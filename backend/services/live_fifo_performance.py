"""Quantity-matched FIFO from exchange fills.

Unmatched inventory stays unmatched. Ghost and paper rows are excluded.
Historical rows are not deleted. Ghost SELLs stop counting toward realized.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class _Lot:
    qty: float
    price: float
    fee_quote: float
    order_id: str


@dataclass
class FifoReport:
    buy_qty: float = 0.0
    sell_qty: float = 0.0
    matched_qty: float = 0.0
    unmatched_buy_qty: float = 0.0
    unmatched_sell_qty: float = 0.0
    realized_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    fees_by_asset: dict[str, float] = field(default_factory=dict)
    manual_exits: int = 0
    automated_exits: int = 0
    ghost_rows: int = 0
    paper_rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        trades = self.wins + self.losses
        pf = (self.gross_win / abs(self.gross_loss)) if self.gross_loss else None
        return {
            "exchange_buy_qty": self.buy_qty,
            "exchange_sell_qty": self.sell_qty,
            "matched_qty": self.matched_qty,
            "unmatched_buy_qty": self.unmatched_buy_qty,
            "unmatched_sell_qty": self.unmatched_sell_qty,
            "realized_pnl": self.realized_pnl,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": (self.wins / trades) if trades else None,
            "profit_factor": pf,
            "fees_by_asset": dict(self.fees_by_asset),
            "manual_exits": self.manual_exits,
            "automated_exits": self.automated_exits,
            "ghost_rows": self.ghost_rows,
            "paper_rows": self.paper_rows,
        }


def _base(symbol: str) -> str:
    raw = str(symbol or "").upper().replace("-", "/").replace("USDT", "")
    return raw.strip("/")


def quarantine_ghost_rows(conn: sqlite3.Connection) -> int:
    """Stop NULL-order and non-live SELLs from driving realized P&L. Rows stay."""
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "counts_toward_realized" not in cols:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN counts_toward_realized INTEGER DEFAULT 1")
    cur = conn.execute(
        """
        UPDATE paper_trades
        SET counts_toward_realized=0
        WHERE side='SELL'
          AND COALESCE(counts_toward_realized, 1)=1
          AND (
            COALESCE(order_id, '')=''
            OR COALESCE(mode, '')!='live'
          )
        """
    )
    return int(cur.rowcount or 0)


def fifo_exchange_performance(db_path: str | Path) -> dict[str, Any]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    report = FifoReport()
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "paper_trades" in tables:
            report.ghost_rows = int(conn.execute("SELECT COUNT(*) FROM paper_trades WHERE side='SELL' AND (order_id IS NULL OR order_id='')").fetchone()[0])
            report.paper_rows = int(conn.execute("SELECT COUNT(*) FROM paper_trades WHERE COALESCE(mode,'')!='live'").fetchone()[0])
            if "exit_reason" in {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}:
                report.manual_exits = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM paper_trades WHERE side='SELL' AND COALESCE(mode,'')='live' AND COALESCE(order_id,'')!='' AND UPPER(COALESCE(exit_reason,'')) LIKE '%MANUAL%'"
                    ).fetchone()[0]
                )
        if "live_exchange_fills" not in tables:
            return report.as_dict()
        rows = conn.execute(
            """
            SELECT symbol, side, executed_qty, avg_fill_price, fee_amount, fee_asset,
                   exchange_order_id, COALESCE(event_ts_exchange, event_ts_recorded, '')
            FROM live_exchange_fills
            WHERE exchange_order_id IS NOT NULL AND exchange_order_id != ''
            ORDER BY event_ts_exchange, id
            """
        ).fetchall()
        lots: dict[str, list[_Lot]] = {}
        open_sell_pnl: dict[str, float] = {}
        for sym, side, qty, price, fee, fee_asset, order_id, _ts in rows:
            q = float(qty or 0.0)
            px = float(price or 0.0)
            fee_v = float(fee or 0.0)
            asset = str(fee_asset or "")
            if q <= 0 or px <= 0:
                continue
            report.fees_by_asset[asset or "UNKNOWN"] = report.fees_by_asset.get(asset or "UNKNOWN", 0.0) + fee_v
            key = str(sym)
            side_u = str(side or "").upper()
            base = _base(key)
            if side_u == "BUY":
                net_qty = q - fee_v if asset.upper() == base else q
                if net_qty <= 0:
                    continue
                report.buy_qty += net_qty
                quote_fee = fee_v if asset.upper() in {"USDT", "USD", ""} else 0.0
                lots.setdefault(key, []).append(_Lot(net_qty, px, quote_fee, str(order_id)))
            elif side_u == "SELL":
                sell_qty = q
                report.sell_qty += sell_qty
                proceeds_per = px
                if asset.upper() in {"USDT", "USD"}:
                    proceeds_per = px - (fee_v / q)
                remaining = sell_qty
                trade_pnl = 0.0
                matched = 0.0
                book = lots.setdefault(key, [])
                while remaining > 1e-12 and book:
                    lot = book[0]
                    take = min(remaining, lot.qty)
                    cost = lot.price * take
                    if lot.qty > 0 and lot.fee_quote:
                        portion = lot.fee_quote * (take / lot.qty)
                        cost += portion
                        lot.fee_quote -= portion
                    trade_pnl += proceeds_per * take - cost
                    lot.qty -= take
                    remaining -= take
                    matched += take
                    if lot.qty <= 1e-12:
                        book.pop(0)
                report.matched_qty += matched
                report.unmatched_sell_qty += remaining
                if matched > 0:
                    open_sell_pnl[str(order_id)] = open_sell_pnl.get(str(order_id), 0.0) + trade_pnl
                    report.realized_pnl += trade_pnl
        for book in lots.values():
            report.unmatched_buy_qty += sum(lot.qty for lot in book)
        for pnl in open_sell_pnl.values():
            if pnl > 0:
                report.wins += 1
                report.gross_win += pnl
            elif pnl < 0:
                report.losses += 1
                report.gross_loss += pnl
        if "paper_trades" in tables:
            report.automated_exits = int(
                conn.execute(
                    "SELECT COUNT(*) FROM paper_trades WHERE side='SELL' AND COALESCE(mode,'')='live' AND COALESCE(order_id,'')!='' AND UPPER(COALESCE(exit_reason,'')) NOT LIKE '%MANUAL%'"
                ).fetchone()[0]
                if "exit_reason" in {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
                else 0
            )
        return report.as_dict()
    finally:
        conn.close()
