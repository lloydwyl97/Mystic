"""One close identity for one unchanged dust lot.

A DUST_PENDING residual that the exchange snapshot does not show is still the
same lot on the next reconcile. Cleanup keeps a positive quantity, so that
pass is not a new close. The ledger and the learning table record the lot
once, keyed by engine, trade, symbol, and canonical quantity. A later
quantity or a later trade is a different event. Timestamps are not the key.
"""

from __future__ import annotations

import sqlite3

from backend.utils.symbols import normalize_symbol

_LEDGER = "position_close_ledger"
_LEARNING = "trade_learning_outcomes"
_LEDGER_INDEX = "ux_position_close_ledger_event_key"
_LEARNING_INDEX = "ux_trade_learning_outcomes_event_key"
_ENTRY_MATCH_SEC = 1e-3


def canonical_dust_qty(quantity: float) -> str:
    return format(float(quantity or 0.0), ".12g")


def dust_writeoff_event_key(*, engine_id: str, trade_id: str, symbol: str, quantity: float) -> str:
    engine = str(engine_id or "").strip() or "LEGACY_DAY_LIVE"
    trade = str(trade_id or "").strip()
    sym = normalize_symbol(symbol)
    return f"DUST_WRITEOFF|{engine}|{sym}|{trade}|{canonical_dust_qty(quantity)}"


def ensure_dust_event_schema(conn: sqlite3.Connection) -> None:
    _ensure_column(conn, _LEDGER, _LEDGER_INDEX)
    _ensure_column(conn, _LEARNING, _LEARNING_INDEX)


def dust_writeoff_books(
    conn: sqlite3.Connection,
    *,
    engine_id: str,
    trade_id: str,
    symbol: str,
    quantity: float,
    entry_time: float,
) -> tuple[bool, bool]:
    """Return (ledger_already_booked, learning_already_booked).

    A legacy learning row for this same lot, written before event keys
    existed, counts as both books. That stops a new ledger row from being
    added beside the historical duplicates.
    """
    ensure_dust_event_schema(conn)
    key = dust_writeoff_event_key(engine_id=engine_id, trade_id=trade_id, symbol=symbol, quantity=quantity)
    ledger = _has_event_key(conn, _LEDGER, key)
    learning = _has_event_key(conn, _LEARNING, key)
    if _legacy_learning_same_lot(conn, engine_id=engine_id, symbol=symbol, quantity=quantity, entry_time=entry_time):
        return True, True
    return ledger, learning


def _ensure_column(conn: sqlite3.Connection, table: str, index_name: str) -> None:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if row is None:
        return
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}
    if "event_key" not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN event_key TEXT")
    conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}(event_key) WHERE event_key IS NOT NULL AND event_key != ''")


def _has_event_key(conn: sqlite3.Connection, table: str, key: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    if row is None:
        return False
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({table})")}
    if "event_key" not in cols:
        return False
    found = conn.execute(f"SELECT 1 FROM {table} WHERE event_key=? LIMIT 1", (key,)).fetchone()
    return found is not None


def _symbol_forms(symbol: str) -> tuple[str, ...]:
    sym = normalize_symbol(symbol)
    plain = sym.replace("/", "")
    if plain == sym:
        return (sym,)
    return (sym, plain)


def _legacy_learning_same_lot(
    conn: sqlite3.Connection,
    *,
    engine_id: str,
    symbol: str,
    quantity: float,
    entry_time: float,
) -> bool:
    if float(entry_time or 0.0) <= 0.0:
        return False
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (_LEARNING,)).fetchone()
    if row is None:
        return False
    cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({_LEARNING})")}
    if "close_reason" not in cols or "entry_timestamp" not in cols:
        return False
    qty_key = canonical_dust_qty(quantity)
    engine = str(engine_id or "").strip()
    forms = _symbol_forms(symbol)
    placeholders = ",".join("?" for _ in forms)
    qty_col = "dust_remaining_qty" if "dust_remaining_qty" in cols else "quantity"
    engine_col = "engine_id" if "engine_id" in cols else "''"
    rows = conn.execute(
        f"""
        SELECT {engine_col}, entry_timestamp, {qty_col}
        FROM {_LEARNING}
        WHERE close_reason='DUST_WRITEOFF' AND symbol IN ({placeholders})
        """,
        forms,
    ).fetchall()
    for row_engine, row_entry, row_qty in rows:
        if row_entry is None:
            continue
        try:
            if abs(float(row_entry) - float(entry_time)) > _ENTRY_MATCH_SEC:
                continue
        except (TypeError, ValueError):
            continue
        if canonical_dust_qty(float(row_qty or 0.0)) != qty_key:
            continue
        stored_engine = str(row_engine or "").strip()
        if stored_engine and engine and stored_engine != engine:
            continue
        return True
    return False
