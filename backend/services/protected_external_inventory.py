"""Exchange inventory that is not a proven DAY V2 or SCALP V2 lot.

These quantities stay in account equity. They do not take a strategy slot,
do not enter strategy P&L, and cannot be sold by either engine.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

PROTECTED_EXTERNAL_INVENTORY = "PROTECTED_EXTERNAL_INVENTORY"
_STRATEGY_ENGINES = frozenset({"DAY_V2", "SCALP_V2"})
# decision_id -> exchange order id. Matched on Binance.US by the mystic
# clientOrderId prefix, the SCALP_V2 reservation timestamp, and the fill.
_VERIFIED_ENTRY_ORDERS = {
    "6f64eda8e6ace785": "917493916",
    "802320234d01fa1b": "1597036548",
    "23a69f82779d403c": "499715204",
}


def exit_route(engine_id: str) -> str:
    """DAY_V2 and SCALP_V2 keep their own exits. Everything else fails closed."""
    eid = str(engine_id or "")
    if eid == "DAY_V2":
        return "DAY_V2"
    if eid == "SCALP_V2":
        return "SCALP_V2"
    if eid == PROTECTED_EXTERNAL_INVENTORY:
        return "NONE"
    return "FAIL_CLOSED"


def strategy_sell_quantity(strategy_qty: float, requested: float) -> float:
    """A strategy sell may use only the quantity that strategy was credited."""
    owned = max(0.0, float(strategy_qty or 0.0))
    ask = max(0.0, float(requested or 0.0))
    return min(ask, owned)


def consumes_strategy_slot(position: Any) -> bool:
    if position is None:
        return False
    engine = str(getattr(position, "engine_id", "") or "")
    status = str(getattr(position, "status", "ACTIVE") or "ACTIVE")
    if PROTECTED_EXTERNAL_INVENTORY in {engine, status}:
        return False
    if status == "DUST_PENDING":
        return False
    return float(getattr(position, "quantity", 0) or 0) > 0


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS protected_external_inventory (
            symbol TEXT PRIMARY KEY,
            quantity REAL NOT NULL,
            cost_price REAL NOT NULL,
            source_trade_id TEXT NOT NULL DEFAULT '',
            entry_order_id TEXT NOT NULL DEFAULT '',
            classification TEXT NOT NULL DEFAULT 'PROTECTED_EXTERNAL_INVENTORY',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )


def _norm(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in raw and raw.endswith("USDT"):
        raw = raw[:-4] + "/USDT"
    return raw


def record_protected(
    conn: sqlite3.Connection,
    symbol: str,
    quantity: float,
    cost_price: float,
    *,
    source_trade_id: str = "",
    entry_order_id: str = "",
) -> None:
    ensure_schema(conn)
    now = time.time()
    sym = _norm(symbol)
    conn.execute(
        """
        INSERT INTO protected_external_inventory(
            symbol, quantity, cost_price, source_trade_id, entry_order_id,
            classification, created_at, updated_at
        ) VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(symbol) DO UPDATE SET
            quantity=excluded.quantity,
            cost_price=excluded.cost_price,
            source_trade_id=excluded.source_trade_id,
            entry_order_id=excluded.entry_order_id,
            classification=excluded.classification,
            updated_at=excluded.updated_at
        """,
        (sym, float(quantity), float(cost_price), str(source_trade_id or ""), str(entry_order_id or ""), PROTECTED_EXTERNAL_INVENTORY, now, now),
    )


def shrink_to_exchange(
    conn: sqlite3.Connection,
    total_balances: dict[str, float],
    strategy_qty: dict[str, float],
    min_notional: float = 1.0,
) -> list[tuple[str, float, float]]:
    """Cap each protected row at the exchange balance left after strategy lots.

    Protected quantity is subtracted from the exchange balance when strategy
    positions are reconciled, so a row that outlives its coins would make new
    strategy lots look vanished. Rows whose remainder is below ``min_notional``
    are deleted; dust retention owns that residue. Returns (symbol, old, new).
    """
    ensure_schema(conn)
    changed: list[tuple[str, float, float]] = []
    rows = conn.execute("SELECT symbol, quantity, cost_price FROM protected_external_inventory").fetchall()
    for symbol, qty, price in rows:
        sym = _norm(str(symbol))
        base = sym.split("/", maxsplit=1)[0]
        old = float(qty or 0.0)
        held = max(0.0, float(total_balances.get(base, 0.0) or 0.0) - max(0.0, float(strategy_qty.get(sym, 0.0) or 0.0)))
        new = min(old, held)
        if new * float(price or 0.0) < float(min_notional) or new <= 0.0:
            conn.execute("DELETE FROM protected_external_inventory WHERE symbol=?", (symbol,))
            changed.append((sym, old, 0.0))
        elif new < old:
            conn.execute(
                "UPDATE protected_external_inventory SET quantity=?, updated_at=? WHERE symbol=?",
                (new, time.time(), symbol),
            )
            changed.append((sym, old, new))
    return changed


def _reservation_owner(conn: sqlite3.Connection, symbol: str) -> tuple[str, str, str] | None:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "day_entry_reservations" not in tables:
        return None
    row = conn.execute(
        """
        SELECT COALESCE(sleeve, ''), COALESCE(decision_id, ''), COALESCE(reservation_id, '')
        FROM day_entry_reservations
        WHERE symbol=? AND sleeve IN ('SCALP_V2', 'DAY_V2')
        ORDER BY created_at DESC LIMIT 1
        """,
        (_norm(symbol),),
    ).fetchone()
    if row is None or not str(row[0] or ""):
        return None
    return str(row[0]), str(row[1] or ""), str(row[2] or "")


def _stamp_strategy_identity(conn: sqlite3.Connection, symbol: str, engine: str, decision_id: str, reservation_id: str) -> None:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
    assignments: list[str] = []
    params: list[str] = []
    order_id = _VERIFIED_ENTRY_ORDERS.get(decision_id, "")
    for column, value in (
        ("engine_id", engine),
        ("scalp_opportunity_id", decision_id),
        ("entry_decision_id", decision_id),
        ("entry_reservation_id", reservation_id),
        ("entry_order_id", order_id),
    ):
        if column == "entry_order_id" and not value:
            continue
        if column in cols:
            assignments.append(f"{column}=?")
            params.append(value)
    if not assignments:
        return
    params.append(symbol)
    conn.execute(f"UPDATE portfolio_engine_positions SET {', '.join(assignments)} WHERE symbol=?", params)


def reclassify_unowned_imports(conn: sqlite3.Connection) -> list[str]:
    """Move reconcile-import rows that have no strategy reservation out of positions.

    A SCALP_V2 or DAY_V2 reservation for the same symbol is exact local ownership.
    Those rows keep a strategy identity and are not protected inventory.
    """
    ensure_schema(conn)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "portfolio_engine_positions" not in tables:
        return []
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
    if "trade_id" not in cols:
        return []
    order_expr = "COALESCE(entry_order_id, '')" if "entry_order_id" in cols else "''"
    rows = conn.execute(
        f"""
        SELECT symbol, quantity, entry_price, COALESCE(trade_id, ''), {order_expr}
        FROM portfolio_engine_positions
        WHERE COALESCE(trade_id, '') LIKE 'reconcile_import_%'
        """
    ).fetchall()
    moved: list[str] = []
    for symbol, qty, price, trade_id, order_id in rows:
        if str(order_id or "").strip():
            continue
        owner = _reservation_owner(conn, str(symbol))
        if owner is not None:
            _stamp_strategy_identity(conn, str(symbol), owner[0], owner[1], owner[2])
            continue
        record_protected(
            conn,
            str(symbol),
            float(qty or 0.0),
            float(price or 0.0),
            source_trade_id=str(trade_id or ""),
            entry_order_id="",
        )
        conn.execute("DELETE FROM portfolio_engine_positions WHERE symbol=?", (symbol,))
        moved.append(_norm(str(symbol)))
    return moved


def protected_quantity(db_path: str | Path, symbol: str) -> float:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        ensure_schema(conn)
        row = conn.execute(
            "SELECT quantity FROM protected_external_inventory WHERE symbol=?",
            (_norm(symbol),),
        ).fetchone()
        return float(row[0] or 0.0) if row else 0.0
    finally:
        conn.close()


def list_protected(db_path: str | Path) -> list[dict[str, Any]]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        ensure_schema(conn)
        rows = conn.execute(
            """
            SELECT symbol, quantity, cost_price, source_trade_id, entry_order_id, classification
            FROM protected_external_inventory
            ORDER BY symbol
            """
        ).fetchall()
    finally:
        conn.close()
    return [
        {
            "symbol": r[0],
            "quantity": float(r[1] or 0.0),
            "cost_price": float(r[2] or 0.0),
            "source_trade_id": r[3],
            "entry_order_id": r[4],
            "classification": r[5],
        }
        for r in rows
    ]


def protected_equity(db_path: str | Path, prices: dict[str, float] | None = None) -> tuple[float, float]:
    """Return (market_value, cost) so total equity includes this inventory."""
    market = 0.0
    cost = 0.0
    marks = prices or {}
    for row in list_protected(db_path):
        qty = float(row["quantity"])
        basis = float(row["cost_price"])
        sym = str(row["symbol"])
        base = sym.split("/", maxsplit=1)[0]
        mark = marks.get(sym) or marks.get(base) or marks.get(sym.replace("/", "")) or basis
        px = float(mark) if mark and float(mark) > 0 else basis
        market += qty * px
        cost += qty * basis
    return market, cost


def unsold_scalp_v2_lot(conn: sqlite3.Connection, symbol: str) -> dict[str, Any] | None:
    """Latest live SCALP V2 BUY with a venue order id and unsold quantity.

    That row is Mystic's own fill record. Reconciliation must restore it as a
    strategy lot before any balance in the symbol is treated as external.
    """
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_trades" not in tables:
        return None
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)")}
    opp_expr = "COALESCE(scalp_opportunity_id, '')" if "scalp_opportunity_id" in cols else "''"
    decision_expr = "COALESCE(decision_id, '')" if "decision_id" in cols else "''"
    row = conn.execute(
        f"""
        SELECT trade_id, price, remaining_position, order_id, COALESCE(fees_paid, 0),
               COALESCE(atr_at_entry, 0), {opp_expr}, {decision_expr}, COALESCE(entry_timestamp, timestamp, '')
        FROM paper_trades
        WHERE symbol=? AND UPPER(side)='BUY' AND COALESCE(paper_run_id, '')='scalp_v2_live'
          AND COALESCE(mode, '')='live' AND COALESCE(order_id, '')!=''
          AND COALESCE(remaining_position, 0) > 0
        ORDER BY id DESC LIMIT 1
        """,
        (_norm(symbol),),
    ).fetchone()
    if row is None:
        return None
    entry_time = 0.0
    try:
        from datetime import datetime

        entry_time = datetime.fromisoformat(str(row[8])).timestamp() if row[8] else 0.0
    except ValueError:
        entry_time = 0.0
    return {
        "trade_id": str(row[0]),
        "price": float(row[1] or 0.0),
        "remaining": float(row[2] or 0.0),
        "order_id": str(row[3]),
        "fee": float(row[4] or 0.0),
        "atr": float(row[5] or 0.0),
        "opportunity_id": str(row[6] or ""),
        "decision_id": str(row[7] or ""),
        "entry_time": entry_time,
    }


def should_align_paper_remaining(trade_id: str, entry_order_id: str, paper_order_exists: bool) -> bool:
    """Do not paint an imported or unmatched venue order onto an older paper lot."""
    if str(trade_id or "").startswith("reconcile_import_"):
        return False
    return not (str(entry_order_id or "").strip() and not paper_order_exists)


def is_strategy_engine(engine_id: str) -> bool:
    return str(engine_id or "") in _STRATEGY_ENGINES
