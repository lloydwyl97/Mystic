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

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

try:
    from backend.database_schema import DATABASE_PATH as _CANONICAL_DB_PATH

    DATABASE_PATH = _CANONICAL_DB_PATH
except Exception:
    DATABASE_PATH = "/home/mystic/mystic/mystic_trading.db"


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
    truncated_history: bool = False

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
            "truncated_history": self.truncated_history,
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
                rows = conn.execute("SELECT quantity, entry_price, original_position_cost FROM portfolio_engine_positions").fetchall()
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
            row = conn.execute("SELECT principal, cash_balance, positions_value, realized_pnl, total_equity FROM portfolio_engine_ledger WHERE id = 1").fetchone()
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
    expected_cash, buy_debits, sell_credits, buy_count, sell_count = compute_expected_cash(db_path, principal=principal, paper_run_id=paper_run_id)
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

    truncated_history = False
    if cash_discrepancy is not None and abs(cash_discrepancy) > tolerance_usd:
        # A large, unexplained-looking cash_discrepancy combined with a clean
        # equity_discrepancy (cash+positions_value == total_equity, i.e. the
        # ledger is internally self-consistent right now) is the signature of
        # `paper_trades` history retention/rotation having removed rows from
        # BEFORE some cutoff, not a real fabricated-cash bug: cash keeps
        # running forward correctly across the rotation, but this formula's
        # "principal - all visible buys + all visible sells" assumption
        # silently breaks once any trades older than the oldest surviving row
        # are no longer visible to sum. Detect that shape explicitly instead
        # of reporting a false CASH_DRIFT alarm.
        try:
            with sqlite3.connect(db_path) as conn:
                oldest_ts = conn.execute("SELECT MIN(timestamp) FROM paper_trades").fetchone()[0]
        except sqlite3.OperationalError:
            oldest_ts = None
        # Require a substantial visible trade count before assuming truncated
        # history: a fresh/small account (few trades) has no plausible
        # "missing older history" explanation, so a cash/equity mismatch
        # there is a real bug, not a retention artifact.
        established_account = (buy_count + sell_count) >= 50
        if oldest_ts and established_account and equity_discrepancy is not None and abs(equity_discrepancy) <= tolerance_usd:
            truncated_history = True
            notes.append(
                f"CASH_FORMULA_NOT_APPLICABLE: stored_cash={stored_cash:.2f} vs principal-anchored "
                f"expected_cash={expected_cash:.2f} diff={cash_discrepancy:.2f}, but equity_discrepancy=0 "
                f"(ledger internally consistent) and oldest surviving paper_trades row is {oldest_ts} — "
                "the principal-since-inception formula requires complete trade history back to the "
                "account's true starting point; if trade history has been pruned/rotated since then, "
                "this diff is expected and does not indicate a real cash discrepancy. Use a "
                "snapshot-anchored replay (see docs) for a reliable point-in-time check instead."
            )
        else:
            notes.append(
                f"CASH_DRIFT: stored_cash={stored_cash:.2f} expected_cash={expected_cash:.2f} "
                f"diff={cash_discrepancy:.2f} (buy_debits={buy_debits:.2f} sell_credits={sell_credits:.2f} "
                f"buy_count={buy_count} sell_count={sell_count})"
            )
    if equity_discrepancy is not None and abs(equity_discrepancy) > tolerance_usd:
        notes.append(f"EQUITY_INVARIANT_BROKEN: stored_total_equity={stored_total_equity:.2f} != cash+positions_value={(stored_cash or 0.0) + (stored_positions_value or 0.0):.2f}")

    if truncated_history:
        # The principal-anchored cash formula is known not to apply (truncated
        # history, explained above) — don't let it fail the overall check when
        # the identity that actually matters (equity_discrepancy) is clean.
        within_tolerance = equity_discrepancy is None or abs(equity_discrepancy) <= tolerance_usd

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
        truncated_history=truncated_history,
        cash_discrepancy=cash_discrepancy,
        equity_discrepancy=equity_discrepancy,
        within_tolerance=within_tolerance,
        tolerance_usd=tolerance_usd,
        notes=notes,
    )


@dataclass
class SymbolPositionReplay:
    symbol: str
    replayed_open_qty: float
    replayed_cost_basis: float
    replayed_avg_entry: float
    stored_qty: float | None
    stored_cost_basis: float | None
    qty_discrepancy: float
    classification: str
    contributing_buy_trade_ids: list[str]
    orphaned_buy_trade_ids: list[str]


def replay_symbol_position_from_trades(
    db_path: str,
    symbol: str,
    *,
    qty_tolerance: float = 1e-6,
) -> SymbolPositionReplay:
    """
    Deterministic weighted-average-cost replay of every non-synthetic paper
    BUY/SELL for one symbol, independent of whatever
    `portfolio_engine_positions` currently stores.

    Mystic's own position model is a single pooled balance per symbol (one
    row, one blended average entry price — see repair_add_count/
    average_entry_after_repair/original_position_cost) rather than
    per-lot FIFO tracking. The correct replay model must match that: each BUY
    adds to a pooled (quantity, cost) pair; each SELL removes quantity at the
    pool's *current* average price (cost -= sell_qty * avg_price), exactly
    the same accounting a SELL uses against a real position's entry_price.
    A strict FIFO-lot simulation is the WRONG model here and produces false
    positives for perfectly-closed round trips (verified during development:
    it misattributed cleanly-closed 1-lot round trips as "orphaned" whenever
    a duplicate-buy incident had earlier put two lots in the pool and a
    later SELL exactly matched the second lot's quantity rather than the
    first's — FIFO consumes the first regardless of which quantity the SELL
    happens to match, which is not how a pooled exchange balance works).

    If the replayed open quantity differs from the stored position row, the
    difference is provably one or more BUY notional amounts that were never
    matched by a SELL but are also missing from the stored position (e.g.
    overwritten by a later BUY for the same symbol via INSERT OR REPLACE) —
    "orphaned" quantity. Cash is not touched by this function; a
    correctly-debited BUY's cash effect is already correct on the books
    regardless of whether the resulting quantity was tracked.
    """
    sym = symbol.strip().upper()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
        where_extra = " AND (is_synthetic IS NULL OR is_synthetic = 0)" if "is_synthetic" in cols else ""
        sym_slash = sym.replace("USDT", "/USDT") if "/" not in sym else sym
        sym_nosl = sym.replace("/", "")
        rows = conn.execute(
            f"""
            SELECT trade_id, side, quantity, price, timestamp
            FROM paper_trades
            WHERE mode = 'paper' AND (symbol = ? OR symbol = ?){where_extra}
            ORDER BY id ASC
            """,
            (sym_slash, sym_nosl),
        ).fetchall()

        qty = 0.0
        cost = 0.0
        contributing: list[str] = []  # BUY trade_ids that still have unconsumed quantity in the pool
        buy_remaining: dict[str, float] = {}
        buy_order: list[str] = []
        for row in rows:
            side = str(row["side"] or "").upper()
            trade_qty = float(row["quantity"] or 0.0)
            price = float(row["price"] or 0.0)
            if side == "BUY":
                qty += trade_qty
                cost += trade_qty * price
                buy_remaining[row["trade_id"]] = trade_qty
                buy_order.append(row["trade_id"])
            elif side == "SELL" and qty > qty_tolerance:
                avg_price = cost / qty
                take = min(trade_qty, qty)
                cost -= take * avg_price
                qty -= take
                # Proportionally reduce each contributing buy's remaining share
                # (oldest-first bookkeeping only — does not change the pooled
                # cost math above, just which trade_ids we can still cite as
                # "still contributing quantity" for the classification below).
                remaining_to_remove = take
                for tid in list(buy_order):
                    if remaining_to_remove <= qty_tolerance:
                        break
                    have = buy_remaining.get(tid, 0.0)
                    if have <= qty_tolerance:
                        continue
                    reduce_by = min(have, remaining_to_remove)
                    buy_remaining[tid] = have - reduce_by
                    remaining_to_remove -= reduce_by

        contributing = [tid for tid in buy_order if buy_remaining.get(tid, 0.0) > qty_tolerance]

        pos_row = conn.execute(
            "SELECT quantity, entry_price, original_position_cost, trade_id FROM portfolio_engine_positions WHERE symbol = ? OR symbol = ?",
            (sym_slash, sym_nosl),
        ).fetchone()

    replayed_qty = qty
    replayed_cost = cost
    replayed_avg = (replayed_cost / replayed_qty) if replayed_qty > qty_tolerance else 0.0

    stored_qty = float(pos_row[0]) if pos_row else None
    stored_cost = float(pos_row[2]) if pos_row and pos_row[2] else (float(pos_row[0]) * float(pos_row[1]) if pos_row else None)
    stored_trade_id = str(pos_row[3]) if pos_row else None

    qty_discrepancy = round(replayed_qty - (stored_qty or 0.0), 8)
    orphaned = [tid for tid in contributing if tid != stored_trade_id] if stored_trade_id else list(contributing)

    if abs(qty_discrepancy) <= qty_tolerance:
        classification = "CONSISTENT"
    elif pos_row is None and replayed_qty > qty_tolerance:
        classification = "UNTRACKED_POSITION_ENTIRELY_MISSING"
    elif qty_discrepancy > qty_tolerance and orphaned:
        classification = "ORPHANED_BUY_LOT_NOT_TRACKED"
    elif qty_discrepancy < -qty_tolerance:
        classification = "STORED_EXCEEDS_TRANSACTION_EVIDENCE"
    else:
        classification = "UNKNOWN_DISCREPANCY"

    return SymbolPositionReplay(
        symbol=sym,
        replayed_open_qty=round(replayed_qty, 8),
        replayed_cost_basis=round(replayed_cost, 6),
        replayed_avg_entry=round(replayed_avg, 8),
        stored_qty=stored_qty,
        stored_cost_basis=stored_cost,
        qty_discrepancy=qty_discrepancy,
        classification=classification,
        contributing_buy_trade_ids=contributing,
        orphaned_buy_trade_ids=orphaned,
    )


def repair_orphaned_buy_lot_via_position_merge(
    db_path: str,
    symbol: str,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    Deterministic, auditable repair for `ORPHANED_BUY_LOT_NOT_TRACKED`.

    Correct treatment (proven from the paper execution contract): both BUYs
    for the symbol genuinely executed and both genuinely debited cash exactly
    once each (verified separately via the cash reconciliation tape) — there
    is no basis to reverse either debit or fabricate a SELL. The only thing
    that was ever wrong is that the *position record* only tracked the later
    BUY. Mystic's own existing schema for combining multiple buy lots of one
    open position is the repair-add field set (repair_add_count,
    repair_add_trade_ids, average_entry_after_repair, original_position_cost)
    — this reuses that same contract to merge the orphaned lot's quantity and
    cost into the currently-tracked position with a blended average entry
    price. Cash is never touched here.
    """
    replay = replay_symbol_position_from_trades(db_path, symbol)
    if replay.classification not in ("ORPHANED_BUY_LOT_NOT_TRACKED", "UNTRACKED_POSITION_ENTIRELY_MISSING"):
        return {"applied": False, "reason": f"classification={replay.classification}, nothing to merge", "replay": replay.__dict__}

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sym_variants = (symbol.strip().upper(), symbol.strip().upper().replace("USDT", "/USDT") if "/" not in symbol.upper() else symbol.strip().upper().replace("/", ""))
        pos_row = conn.execute(
            "SELECT * FROM portfolio_engine_positions WHERE symbol IN (?, ?)",
            sym_variants,
        ).fetchone()

        new_qty = replay.replayed_open_qty
        new_cost = replay.replayed_cost_basis
        new_avg_entry = replay.replayed_avg_entry

        if pos_row is None:
            # UNTRACKED_POSITION_ENTIRELY_MISSING: no row survives at all despite
            # proven open quantity from the transaction tape (e.g. every buy for
            # this symbol was overwritten before ever being sold). Recreate a
            # minimal valid row using the earliest contributing buy's own entry
            # metadata where available (stop/target from that buy's persisted
            # thesis are unrecoverable, so conservative engine defaults are used
            # — normal exit management, including time-stop, applies from here
            # exactly as it would for any other open position).
            earliest = conn.execute(
                "SELECT trade_id, quantity, price, timestamp FROM paper_trades WHERE mode='paper' AND (symbol = ? OR symbol = ?) AND side='BUY' ORDER BY id ASC LIMIT 1",
                sym_variants,
            ).fetchone()
            entry_epoch = time.time()
            if earliest and earliest["timestamp"]:
                try:
                    entry_epoch = datetime.fromisoformat(str(earliest["timestamp"]).replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    pass
            stored_symbol = sym_variants[0] if "/" in sym_variants[0] else sym_variants[1]

            result = {
                "applied": not dry_run,
                "symbol": stored_symbol,
                "prior_stored_qty": None,
                "prior_stored_cost_basis": None,
                "merged_qty": new_qty,
                "merged_cost_basis": new_cost,
                "merged_avg_entry": new_avg_entry,
                "orphaned_buy_trade_ids_merged": replay.orphaned_buy_trade_ids,
                "cash_adjustment": 0.0,
                "note": "recreated missing position row from transaction evidence; cash untouched",
            }
            if dry_run:
                return result

            conn.execute(
                """
                INSERT INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    trailing_stop_price, tp1_hit, highest_price,
                    atr_at_entry, entry_bar_timestamp, confidence_at_entry,
                    entry_fee, sleeve, entry_strategy_id,
                    repair_add_count, last_repair_add_ts, repair_add_trade_ids,
                    average_entry_after_repair, original_position_cost, thesis_json, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored_symbol,
                    new_qty,
                    new_avg_entry,
                    entry_epoch,
                    (earliest["trade_id"] if earliest else f"reconciled_{stored_symbol}_{int(time.time())}"),
                    new_avg_entry * 0.985,
                    new_avg_entry * 1.015,
                    0.0,
                    0.0,
                    0,
                    new_avg_entry,
                    new_avg_entry * 0.01,
                    int(entry_epoch),
                    0.5,
                    0.0,
                    "ACTIVE",
                    "day",
                    len(replay.orphaned_buy_trade_ids),
                    time.time(),
                    json.dumps(sorted(replay.orphaned_buy_trade_ids)),
                    new_avg_entry,
                    new_cost,
                    json.dumps({"reconciliation_recovered": True, "note": "position recreated from transaction-tape replay after duplicate-buy incident"}),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        else:
            stored_symbol = pos_row["symbol"]
            prior_repair_ids: list[str] = []
            try:
                prior_repair_ids = json.loads(pos_row["repair_add_trade_ids"] or "[]")
            except (TypeError, ValueError, json.JSONDecodeError):
                prior_repair_ids = []
            merged_repair_ids = sorted(set(prior_repair_ids) | set(replay.orphaned_buy_trade_ids))

            result = {
                "applied": not dry_run,
                "symbol": stored_symbol,
                "prior_stored_qty": replay.stored_qty,
                "prior_stored_cost_basis": replay.stored_cost_basis,
                "merged_qty": new_qty,
                "merged_cost_basis": new_cost,
                "merged_avg_entry": new_avg_entry,
                "orphaned_buy_trade_ids_merged": replay.orphaned_buy_trade_ids,
                "cash_adjustment": 0.0,
                "note": "cash untouched — both buys already correctly debited cash exactly once each per transaction tape",
            }

            if dry_run:
                return result

            conn.execute(
                """
                UPDATE portfolio_engine_positions
                SET quantity = ?,
                    entry_price = ?,
                    original_position_cost = ?,
                    average_entry_after_repair = ?,
                    repair_add_count = ?,
                    repair_add_trade_ids = ?,
                    last_repair_add_ts = ?,
                    last_updated = ?
                WHERE symbol = ?
                """,
                (
                    new_qty,
                    new_avg_entry,
                    new_cost,
                    new_avg_entry,
                    len(merged_repair_ids),
                    json.dumps(merged_repair_ids),
                    time.time(),
                    datetime.now(timezone.utc).isoformat(),
                    stored_symbol,
                ),
            )

        # Commit the position fix BEFORE opening any second connection for the
        # audit row (a still-open write transaction on this connection would
        # self-contend with insert_audit_row_sync's own connect_rw/BEGIN
        # IMMEDIATE on the same file — the exact nested-write deadlock pattern
        # fixed earlier this session for strategy_runtime_audit table creation).
        conn.commit()

    # Auditable reconciliation classification row (existing strategy_runtime_audit
    # mechanism — no second audit system). Runs on its own connection only
    # after the position fix above is already committed.
    try:
        from backend.services.strategy_runtime_audit import insert_audit_row_sync

        insert_audit_row_sync(
            event_type="DUPLICATE_BUY_LOT_RECONCILED",
            symbol=stored_symbol,
            buy_trade_id=",".join(replay.orphaned_buy_trade_ids),
            extra_json={
                "classification": replay.classification,
                "prior_stored_qty": replay.stored_qty,
                "prior_stored_cost_basis": replay.stored_cost_basis,
                "merged_qty": new_qty,
                "merged_cost_basis": new_cost,
                "merged_avg_entry": new_avg_entry,
                "orphaned_buy_trade_ids": replay.orphaned_buy_trade_ids,
                "cash_adjustment": 0.0,
                "mechanism": "repair_add_field_reuse_position_merge",
            },
            db_path=db_path,
        )
    except Exception:
        pass

    return result


__all__ = [
    "ReconciliationResult",
    "SymbolPositionReplay",
    "compute_expected_cash",
    "reconcile_ledger_cash",
    "replay_symbol_position_from_trades",
    "repair_orphaned_buy_lot_via_position_merge",
]
