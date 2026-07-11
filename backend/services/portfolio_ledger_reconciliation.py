"""
Deterministic paper-ledger cash reconciliation.

Purpose: derive *expected* cash purely from transaction evidence (non-synthetic
paper_trades BUY/SELL rows + fees), independent of whatever the persisted
`portfolio_engine_ledger` row currently says. This lets a caller detect and
report ledger drift (e.g. a race that reloaded pre-buy cash while a position
was already open, double-counting open notional) before ever mutating stored
state.

This module never writes to the database. It is read-only, diagnostic-only.
Any correction based on its output is a deliberate, separate action taken by
the caller (see docs/ops notes) — this module only proves the discrepancy.

Canonical accounting identity (must hold at all times):
    total_equity = cash_balance + positions_value(market)

A completed BUY reduces cash by (executed_qty * executed_price + entry_fee).
A completed SELL increases cash by (executed_qty * executed_price - exit_fee).
Realized P&L is a derived report of SELL proceeds vs. cost basis — it must
never be added to cash a second time on top of the SELL credit itself.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Any

DATABASE_PATH = "mystic_trading.db"


@dataclass
class ReconciliationResult:
    principal: float
    buy_debits: float
    sell_credits: float
    buy_count: int
    sell_count: int
    expected_cash: float
    stored_cash: float | None
    stored_positions_value: float | None
    stored_realized_pnl: float | None
    stored_total_equity: float | None
    open_position_cost_basis: float
    expected_equity_at_cost: float
    cash_discrepancy: float | None
    equity_discrepancy: float | None
    within_tolerance: bool
    tolerance_usd: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principal": self.principal,
            "buy_debits": round(self.buy_debits, 6),
            "sell_credits": round(self.sell_credits, 6),
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "expected_cash": round(self.expected_cash, 6),
            "stored_cash": self.stored_cash,
            "stored_positions_value": self.stored_positions_value,
            "stored_realized_pnl": self.stored_realized_pnl,
            "stored_total_equity": self.stored_total_equity,
            "open_position_cost_basis": round(self.open_position_cost_basis, 6),
            "expected_equity_at_cost": round(self.expected_equity_at_cost, 6),
            "cash_discrepancy": self.cash_discrepancy,
            "equity_discrepancy": self.equity_discrepancy,
            "within_tolerance": self.within_tolerance,
            "tolerance_usd": self.tolerance_usd,
            "notes": self.notes,
        }


def _row_fee(row: sqlite3.Row, *candidates: str) -> float:
    keys = row.keys()
    for c in candidates:
        if c not in keys:
            continue
        v = row[c]
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0


def compute_expected_cash(
    db_path: str = DATABASE_PATH,
    *,
    principal: float,
    paper_run_id: str | None = None,
) -> tuple[float, float, float, int, int]:
    """
    Sum all non-synthetic paper BUY/SELL rows to derive expected cash from
    trusted starting principal. Returns:
        (expected_cash, buy_debits, sell_credits, buy_count, sell_count)

    Tolerant of schema variants: only selects fee/synthetic columns that
    actually exist on this database's paper_trades table (test fixtures and
    older DBs may lack entry_fee_usd/exit_fee_usd/is_synthetic).
    """
    buy_debits = 0.0
    sell_credits = 0.0
    buy_count = 0
    sell_count = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        available_cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
        fee_cols = [c for c in ("entry_fee_usd", "exit_fee_usd", "fees_paid", "commission") if c in available_cols]
        select_cols = ["side", "quantity", "price"] + fee_cols
        if "mode" in available_cols:
            select_cols.append("mode")
        if "paper_run_id" in available_cols:
            select_cols.append("paper_run_id")
        query = f"SELECT {', '.join(select_cols)} FROM paper_trades WHERE 1=1"
        params: list[Any] = []
        if "mode" in available_cols:
            query += " AND mode = 'paper'"
        if "is_synthetic" in available_cols:
            query += " AND (is_synthetic IS NULL OR is_synthetic = 0)"
        if paper_run_id and "paper_run_id" in available_cols:
            query += " AND paper_run_id = ?"
            params.append(paper_run_id)
        query += " ORDER BY id ASC"
        for row in conn.execute(query, params):
            qty = float(row["quantity"] or 0.0)
            price = float(row["price"] or 0.0)
            notional = qty * price
            side = str(row["side"] or "").upper()
            if side == "BUY":
                fee = _row_fee(row, "entry_fee_usd", "fees_paid", "commission")
                buy_debits += notional + fee
                buy_count += 1
            elif side == "SELL":
                fee = _row_fee(row, "exit_fee_usd", "fees_paid", "commission")
                sell_credits += notional - fee
                sell_count += 1
        return (
            float(principal) - buy_debits + sell_credits,
            buy_debits,
            sell_credits,
            buy_count,
            sell_count,
        )


def _open_position_cost_basis(db_path: str) -> float:
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cols = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
            if "original_position_cost" in cols:
                rows = conn.execute(
                    "SELECT quantity, entry_price, original_position_cost FROM portfolio_engine_positions"
                ).fetchall()
                total = 0.0
                for r in rows:
                    cost = r["original_position_cost"]
                    if cost is not None and float(cost) > 0:
                        total += float(cost)
                    else:
                        total += float(r["quantity"] or 0.0) * float(r["entry_price"] or 0.0)
                return total
            rows = conn.execute("SELECT quantity, entry_price FROM portfolio_engine_positions").fetchall()
            return sum(float(r["quantity"] or 0.0) * float(r["entry_price"] or 0.0) for r in rows)
    except sqlite3.OperationalError:
        return 0.0


def reconcile_ledger_cash(
    db_path: str = DATABASE_PATH,
    *,
    tolerance_usd: float = 0.05,
    paper_run_id: str | None = None,
) -> ReconciliationResult:
    """
    Read-only reconciliation. Never mutates the database.

    Returns a ReconciliationResult describing whether stored cash/equity in
    portfolio_engine_ledger agrees with cash/equity derived purely from
    transaction evidence (principal +/- buy/sell notional and fees).
    """
    notes: list[str] = []
    stored_cash: float | None = None
    stored_positions_value: float | None = None
    stored_realized_pnl: float | None = None
    stored_total_equity: float | None = None
    stored_principal: float | None = None

    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT principal, cash_balance, positions_value, realized_pnl, total_equity "
                "FROM portfolio_engine_ledger WHERE id = 1"
            ).fetchone()
            if row:
                stored_principal = float(row["principal"])
                stored_cash = float(row["cash_balance"])
                stored_positions_value = float(row["positions_value"] or 0.0)
                stored_realized_pnl = float(row["realized_pnl"] or 0.0)
                stored_total_equity = float(row["total_equity"])
            else:
                notes.append("no portfolio_engine_ledger row id=1 found")
    except sqlite3.OperationalError as exc:
        notes.append(f"ledger read failed: {exc}")

    principal = stored_principal if stored_principal is not None else 25000.0
    expected_cash, buy_debits, sell_credits, buy_count, sell_count = compute_expected_cash(
        db_path, principal=principal, paper_run_id=paper_run_id
    )
    open_cost_basis = _open_position_cost_basis(db_path)
    expected_equity_at_cost = expected_cash + open_cost_basis

    cash_discrepancy = None
    equity_discrepancy = None
    within_tolerance = True
    if stored_cash is not None:
        cash_discrepancy = round(stored_cash - expected_cash, 6)
        within_tolerance = within_tolerance and abs(cash_discrepancy) <= tolerance_usd
    if stored_total_equity is not None:
        # Compare stored equity (which uses MTM positions_value, not cost basis) against
        # cash+MTM identity rather than cost-basis equity, since MTM can legitimately
        # differ from cost by unrealized P&L.
        mtm_equity_check = (stored_cash or 0.0) + (stored_positions_value or 0.0)
        equity_discrepancy = round(stored_total_equity - mtm_equity_check, 6)
        within_tolerance = within_tolerance and abs(equity_discrepancy) <= tolerance_usd

    if cash_discrepancy is not None and abs(cash_discrepancy) > tolerance_usd:
        notes.append(
            f"CASH_DRIFT: stored_cash={stored_cash:.2f} expected_cash={expected_cash:.2f} "
            f"diff={cash_discrepancy:.2f} (buy_debits={buy_debits:.2f} sell_credits={sell_credits:.2f} "
            f"buy_count={buy_count} sell_count={sell_count})"
        )
    if equity_discrepancy is not None and abs(equity_discrepancy) > tolerance_usd:
        notes.append(
            f"EQUITY_INVARIANT_BROKEN: stored_total_equity={stored_total_equity:.2f} != "
            f"cash+positions_value={(stored_cash or 0.0) + (stored_positions_value or 0.0):.2f}"
        )

    return ReconciliationResult(
        principal=principal,
        buy_debits=buy_debits,
        sell_credits=sell_credits,
        buy_count=buy_count,
        sell_count=sell_count,
        expected_cash=expected_cash,
        stored_cash=stored_cash,
        stored_positions_value=stored_positions_value,
        stored_realized_pnl=stored_realized_pnl,
        stored_total_equity=stored_total_equity,
        open_position_cost_basis=open_cost_basis,
        expected_equity_at_cost=expected_equity_at_cost,
        cash_discrepancy=cash_discrepancy,
        equity_discrepancy=equity_discrepancy,
        within_tolerance=within_tolerance,
        tolerance_usd=tolerance_usd,
        notes=notes,
    )


__all__ = [
    "ReconciliationResult",
    "compute_expected_cash",
    "reconcile_ledger_cash",
]
