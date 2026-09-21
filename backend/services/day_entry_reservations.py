"""Persistent atomic DAY entry reservations — survive process restart."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

from backend.services.day_entry_spendable import money

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS day_entry_reservations (
    reservation_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    notional_usd REAL NOT NULL,
    risk_usd REAL NOT NULL DEFAULT 0,
    sleeve TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'ACTIVE',
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_day_res_decision_active
    ON day_entry_reservations(decision_id) WHERE status='ACTIVE';
CREATE INDEX IF NOT EXISTS idx_day_res_symbol_active
    ON day_entry_reservations(symbol) WHERE status='ACTIVE';
CREATE INDEX IF NOT EXISTS idx_day_res_expires ON day_entry_reservations(expires_at);
"""

DEFAULT_TTL_SEC = 120.0

# Terminal state for a reservation whose order actually filled. A reservation
# that became a real position is not "released" (the cash was spent, not handed
# back) and it is emphatically not "expired" or "timed out". Before this state
# existed, a filled reservation stayed ACTIVE until the staleness sweep relabelled
# it TIMEOUT, so the ledger claimed capital had been returned when it had in fact
# been deployed, and every filled entry looked like an abandoned reservation.
STATUS_CONSUMED = "CONSUMED"

# Statuses that must never be rewritten by the staleness sweep or a late release.
TERMINAL_STATUSES = frozenset({STATUS_CONSUMED})


def ensure_reservation_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(day_entry_reservations)")}
        if "notional_exact" not in cols:
            try:
                conn.execute("ALTER TABLE day_entry_reservations ADD COLUMN notional_exact TEXT")
            except sqlite3.OperationalError as exc:
                if "duplicate column name" not in str(exc).lower():
                    raise
        conn.commit()
    finally:
        conn.close()


def create_reservation(
    db_path: str | Path,
    *,
    decision_id: str,
    symbol: str,
    notional_usd: float,
    risk_usd: float = 0.0,
    sleeve: str = "",
    ttl_sec: float = DEFAULT_TTL_SEC,
    reservation_id: str | None = None,
) -> tuple[bool, str, str]:
    """Idempotent create by decision_id. Returns (ok, reason, reservation_id)."""
    ensure_reservation_schema(db_path)
    did = str(decision_id or "").strip()
    sym = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in sym and sym.endswith("USDT") and len(sym) > 4:
        sym = sym[:-4] + "/USDT"
    if not did:
        return False, "MISSING_DECISION_ID", ""
    if not sym:
        return False, "INVALID_SYMBOL", ""
    now = time.time()
    rid = reservation_id or f"res_{uuid.uuid4().hex[:16]}"
    expires = now + float(ttl_sec)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        # Expire stale first
        conn.execute(
            "UPDATE day_entry_reservations SET status='EXPIRED', updated_at=? WHERE status='ACTIVE' AND expires_at < ?",
            (now, now),
        )
        existing = conn.execute(
            "SELECT reservation_id, symbol, status FROM day_entry_reservations WHERE decision_id=? AND status='ACTIVE'",
            (did,),
        ).fetchone()
        if existing:
            conn.commit()
            return True, "IDEMPOTENT_EXISTING", str(existing[0])
        # Symbol occupancy
        sym_hit = conn.execute(
            "SELECT reservation_id FROM day_entry_reservations WHERE symbol=? AND status='ACTIVE' LIMIT 1",
            (sym,),
        ).fetchone()
        if sym_hit:
            conn.commit()
            return False, "SYMBOL_RESERVED", ""
        exact = str(money(notional_usd))
        conn.execute(
            """
            INSERT INTO day_entry_reservations(
                reservation_id, decision_id, symbol, notional_usd, risk_usd, sleeve,
                status, created_at, expires_at, updated_at, notional_exact
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?, ?)
            """,
            (rid, did, sym, float(exact), float(risk_usd or 0.0), str(sleeve or ""), now, expires, now, exact),
        )
        conn.commit()
        return True, "OK", rid
    except sqlite3.IntegrityError:
        conn.rollback()
        row = conn.execute(
            "SELECT reservation_id FROM day_entry_reservations WHERE decision_id=? AND status='ACTIVE'",
            (did,),
        ).fetchone()
        if row:
            return True, "IDEMPOTENT_EXISTING", str(row[0])
        return False, "RESERVATION_CONFLICT", ""
    except Exception as exc:
        conn.rollback()
        logger.warning("create_reservation failed: %s", exc)
        return False, f"RESERVATION_ERROR:{exc}", ""
    finally:
        conn.close()


def release_reservation(
    db_path: str | Path,
    *,
    reservation_id: str = "",
    decision_id: str = "",
    symbol: str = "",
    reason: str = "RELEASED",
) -> bool:
    """Idempotent release — safe to call twice."""
    ensure_reservation_schema(db_path)
    now = time.time()
    status = str(reason or "RELEASED")[:32]
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        if reservation_id:
            conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE reservation_id=? AND status='ACTIVE'",
                (status, now, reservation_id),
            )
        elif decision_id:
            conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE decision_id=? AND status='ACTIVE'",
                (status, now, decision_id),
            )
        elif symbol:
            sym = str(symbol).strip().upper().replace("-", "/")
            conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE symbol=? AND status='ACTIVE'",
                (status, now, sym),
            )
        else:
            conn.commit()
            return False
        conn.commit()
        return True
    finally:
        conn.close()


def consume_reservation(
    db_path: str | Path,
    *,
    reservation_id: str = "",
    decision_id: str = "",
    symbol: str = "",
) -> bool:
    """Mark a reservation CONSUMED because its order filled into a position.

    Exactly-once and idempotent: only an ACTIVE row transitions, so a second
    call (a retry, or both the submit path and the adoption path reaching the
    same fill) is a no-op and returns False. Returns True only for the single
    call that performed the transition.

    Resolution order is reservation_id, then decision_id, then symbol, because
    the intent row carries the authoritative reservation_id and in-memory engine
    metadata does not survive a restart.
    """
    ensure_reservation_schema(db_path)
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = None
        if reservation_id:
            cur = conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE reservation_id=? AND status='ACTIVE'",
                (STATUS_CONSUMED, now, str(reservation_id)),
            )
        if (cur is None or not cur.rowcount) and decision_id:
            cur = conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE decision_id=? AND status='ACTIVE'",
                (STATUS_CONSUMED, now, str(decision_id)),
            )
        if (cur is None or not cur.rowcount) and symbol:
            sym = str(symbol).strip().upper().replace("-", "/")
            cur = conn.execute(
                "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE symbol=? AND status='ACTIVE'",
                (STATUS_CONSUMED, now, sym),
            )
        changed = int(cur.rowcount or 0) if cur is not None else 0
        conn.commit()
        if changed:
            logger.info(
                "RESERVATION_CONSUMED reservation=%s decision=%s symbol=%s",
                reservation_id,
                decision_id,
                symbol,
            )
        return changed > 0
    finally:
        conn.close()


def correct_filled_reservation_to_consumed(
    db_path: str | Path,
    *,
    reservation_id: str,
) -> dict[str, Any]:
    """Canonicalize a filled BUY reservation to CONSUMED.

    A RELEASED or EXPIRED row that already funded an accepted fill is not a
    second reservation. This is the only transition that may move a non-ACTIVE
    row to CONSUMED.
    """
    rid = str(reservation_id or "").strip()
    if not rid:
        return {"changed": False, "previous_status": "", "status": ""}
    ensure_reservation_schema(db_path)
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status FROM day_entry_reservations WHERE reservation_id=?",
            (rid,),
        ).fetchone()
        previous = str(row[0]) if row else ""
        if not row:
            conn.commit()
            return {"changed": False, "previous_status": "", "status": ""}
        if previous == STATUS_CONSUMED:
            conn.commit()
            return {"changed": False, "previous_status": previous, "status": STATUS_CONSUMED}
        conn.execute(
            "UPDATE day_entry_reservations SET status=?, updated_at=? WHERE reservation_id=? AND status!=?",
            (STATUS_CONSUMED, now, rid, STATUS_CONSUMED),
        )
        conn.commit()
        logger.info("RESERVATION_CORRECTED_CONSUMED reservation=%s previous=%s", rid, previous)
        return {"changed": True, "previous_status": previous, "status": STATUS_CONSUMED}
    finally:
        conn.close()


def reservation_status(db_path: str | Path, reservation_id: str) -> str:
    """Current status of one reservation, or "" when it does not exist."""
    if not reservation_id:
        return ""
    ensure_reservation_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = conn.execute(
            "SELECT status FROM day_entry_reservations WHERE reservation_id=?",
            (str(reservation_id),),
        ).fetchone()
        return str(row[0]) if row else ""
    finally:
        conn.close()


def expire_stale(db_path: str | Path) -> int:
    """Sweep ACTIVE reservations past their TTL.

    CONSUMED rows are excluded by the ACTIVE predicate, so a filled reservation
    can never be relabelled EXPIRED after the fact.
    """
    ensure_reservation_schema(db_path)
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        cur = conn.execute(
            "UPDATE day_entry_reservations SET status='EXPIRED', updated_at=? WHERE status='ACTIVE' AND expires_at < ?",
            (now, now),
        )
        conn.commit()
        return int(cur.rowcount or 0)
    finally:
        conn.close()


def load_active_reservations(db_path: str | Path) -> list[dict[str, Any]]:
    ensure_reservation_schema(db_path)
    expire_stale(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT reservation_id, decision_id, symbol, notional_usd, risk_usd, sleeve,
                   status, created_at, expires_at, notional_exact
            FROM day_entry_reservations WHERE status='ACTIVE'
            """
        ).fetchall()
        out = []
        for r in rows:
            row = dict(r)
            row["notional_usd"] = money(row.get("notional_exact") or row.get("notional_usd") or 0)
            out.append(row)
        return out
    finally:
        conn.close()


def active_notional(db_path: str | Path, *, exclude_decision_id: str = ""):
    rows = load_active_reservations(db_path)
    n = 0.0
    for r in rows:
        if exclude_decision_id and str(r.get("decision_id")) == exclude_decision_id:
            continue
        n += money(r.get("notional_usd") or 0)
    return n


def active_symbols(db_path: str | Path) -> set[str]:
    return {str(r["symbol"]) for r in load_active_reservations(db_path)}


__all__ = [
    "STATUS_CONSUMED",
    "TERMINAL_STATUSES",
    "active_notional",
    "active_symbols",
    "consume_reservation",
    "correct_filled_reservation_to_consumed",
    "create_reservation",
    "ensure_reservation_schema",
    "expire_stale",
    "load_active_reservations",
    "release_reservation",
    "reservation_status",
]
