"""Persistent atomic DAY entry reservations — survive process restart."""

from __future__ import annotations

import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

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


def ensure_reservation_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
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
        conn.execute(
            """
            INSERT INTO day_entry_reservations(
                reservation_id, decision_id, symbol, notional_usd, risk_usd, sleeve,
                status, created_at, expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
            """,
            (rid, did, sym, float(notional_usd), float(risk_usd or 0.0), str(sleeve or ""), now, expires, now),
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


def expire_stale(db_path: str | Path) -> int:
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
                   status, created_at, expires_at
            FROM day_entry_reservations WHERE status='ACTIVE'
            """
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def active_notional(db_path: str | Path, *, exclude_decision_id: str = "") -> float:
    rows = load_active_reservations(db_path)
    n = 0.0
    for r in rows:
        if exclude_decision_id and str(r.get("decision_id")) == exclude_decision_id:
            continue
        n += float(r.get("notional_usd") or 0.0)
    return n


def active_symbols(db_path: str | Path) -> set[str]:
    return {str(r["symbol"]) for r in load_active_reservations(db_path)}


__all__ = [
    "active_notional",
    "active_symbols",
    "create_reservation",
    "ensure_reservation_schema",
    "expire_stale",
    "load_active_reservations",
    "release_reservation",
]
