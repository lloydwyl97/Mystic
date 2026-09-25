"""SCALP V2 opportunity identity.

The id is symbol + setup family + a price-zone anchor. The clock is not part
of the id. Closing a position keeps the row. Only one ARMED opportunity is
actionable per symbol. CLOSED and EXPIRED rows stay for audit and do not block.

Expired ARMED rows become EXPIRED. They are never deleted and never updated
back to ARMED. A later entry in the same zone inserts a new row and does not
inherit tracked_low.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SCALP_V2_ENGINE_ID = "SCALP_V2"
LEGACY_EXIT_ONLY = "LEGACY_EXIT_ONLY"
_ZONE = 1.005  # 0.5% price zone. Leaving it is the structural reset.

# ARMED opportunities older than this are stale. The live cycle reaps them
# even when no new arm is attempted. Default: 3600 s.
SCALP_V2_OPP_EXPIRY_SEC: float = float(os.getenv("SCALP_V2_OPP_EXPIRY_SEC", "3600"))

_EXTRA_COLUMNS = (
    ("version", "INTEGER NOT NULL DEFAULT 1"),
    ("reservation_id", "TEXT NOT NULL DEFAULT ''"),
    ("reservation_released", "INTEGER NOT NULL DEFAULT 0"),
    ("tracked_low", "REAL"),
)


@dataclasses.dataclass(frozen=True)
class ScalpOpportunityId:
    symbol: str
    setup_family: str
    structural_anchor: str
    entry_bar_15m: str = ""

    @property
    def canonical_id(self) -> str:
        raw = f"{self.symbol}:{self.setup_family}:{self.structural_anchor}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    @classmethod
    def from_intent(cls, symbol: str, setup: str, bar_timestamp: str, arm_price: float = 0.0) -> ScalpOpportunityId:
        _ = bar_timestamp  # retained for callers; the clock does not define the id
        setup_family = str(setup or "UNKNOWN").split("_", maxsplit=1)[0] or "UNKNOWN"
        return cls(
            symbol=str(symbol or "").upper(),
            setup_family=setup_family,
            structural_anchor=price_zone(arm_price),
        )


def _sym(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in raw and raw.endswith("USDT"):
        raw = raw[:-4] + "/USDT"
    return raw


def price_zone(price: float) -> str:
    px = float(price or 0.0)
    if px <= 0:
        return "NA"
    bucket = math.floor(math.log(px) / math.log(_ZONE))
    return f"pxb:{bucket}"


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_v2_opportunities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            opportunity_id TEXT NOT NULL,
            setup_family TEXT NOT NULL,
            structural_anchor TEXT NOT NULL,
            state TEXT NOT NULL,
            engine_id TEXT NOT NULL DEFAULT 'SCALP_V2',
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_v2_opp_symbol ON scalp_v2_opportunities(symbol, id)")
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(scalp_v2_opportunities)")}
    for name, decl in _EXTRA_COLUMNS:
        if name not in cols:
            conn.execute(f"ALTER TABLE scalp_v2_opportunities ADD COLUMN {name} {decl}")


def _queue_release(
    conn: sqlite3.Connection,
    row_id: int,
    reservation_id: str,
    released: int,
    symbol: str,
    queued: list[tuple[str, str]],
) -> None:
    """Mark the row released inside this transaction. Call the callback after commit."""
    if int(released or 0) == 1:
        return
    rid = str(reservation_id or "")
    conn.execute(
        "UPDATE scalp_v2_opportunities SET reservation_released=1, updated_at=? WHERE id=? AND reservation_released=0",
        (time.time(), row_id),
    )
    if rid:
        queued.append((rid, symbol))


def reap_expired_armed(
    db_path: str | Path,
    *,
    now: float | None = None,
    expiry_sec: float | None = None,
    release_reservation: Callable[[str, str], None] | None = None,
) -> dict[str, int]:
    """Expire every stale ARMED row. OPEN and CLOSED are never touched.

    Duplicate current ARMED rows for the same symbol and price-zone keep only
    the newest. Reservations on terminal rows are released at most once.
    """
    moment = float(now if now is not None else time.time())
    limit = float(SCALP_V2_OPP_EXPIRY_SEC if expiry_sec is None else expiry_sec)
    conn = sqlite3.connect(str(db_path), timeout=30)
    expired = 0
    collapsed = 0
    queued: list[tuple[str, str]] = []
    committed = False
    try:
        _ensure(conn)
        rows = conn.execute(
            """
            SELECT id, symbol, opportunity_id, created_at,
                   COALESCE(reservation_id, ''), COALESCE(reservation_released, 0), engine_id
            FROM scalp_v2_opportunities
            WHERE state='ARMED'
            ORDER BY id ASC
            """
        ).fetchall()
        for row in rows:
            age = moment - float(row[3] or 0.0)
            if age <= limit:
                continue
            conn.execute(
                "UPDATE scalp_v2_opportunities SET state='EXPIRED', updated_at=? WHERE id=? AND state='ARMED'",
                (moment, row[0]),
            )
            _queue_release(conn, int(row[0]), str(row[4]), int(row[5] or 0), str(row[1]), queued)
            expired += 1
        current = conn.execute(
            """
            SELECT id, symbol, opportunity_id, engine_id, created_at,
                   COALESCE(reservation_id, ''), COALESCE(reservation_released, 0)
            FROM scalp_v2_opportunities
            WHERE state='ARMED'
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        grouped: dict[tuple[str, str, str], list[tuple]] = {}
        for row in current:
            grouped.setdefault((str(row[1]), str(row[2]), str(row[3])), []).append(row)
        for group in grouped.values():
            if len(group) <= 1:
                continue
            for stale in group[:-1]:
                conn.execute(
                    "UPDATE scalp_v2_opportunities SET state='EXPIRED', updated_at=? WHERE id=? AND state='ARMED'",
                    (moment, stale[0]),
                )
                _queue_release(conn, int(stale[0]), str(stale[5]), int(stale[6] or 0), str(stale[1]), queued)
                collapsed += 1
        zoned = conn.execute(
            """
            SELECT id, symbol, structural_anchor, engine_id, created_at,
                   COALESCE(reservation_id, ''), COALESCE(reservation_released, 0)
            FROM scalp_v2_opportunities
            WHERE state='ARMED'
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        by_zone: dict[tuple[str, str, str], list[tuple]] = {}
        for row in zoned:
            by_zone.setdefault((str(row[1]), str(row[2]), str(row[3])), []).append(row)
        for group in by_zone.values():
            if len(group) <= 1:
                continue
            for stale in group[:-1]:
                conn.execute(
                    "UPDATE scalp_v2_opportunities SET state='EXPIRED', updated_at=? WHERE id=? AND state='ARMED'",
                    (moment, stale[0]),
                )
                _queue_release(conn, int(stale[0]), str(stale[5]), int(stale[6] or 0), str(stale[1]), queued)
                collapsed += 1
        by_symbol_rows = conn.execute(
            """
            SELECT id, symbol, created_at, COALESCE(reservation_id, ''), COALESCE(reservation_released, 0)
            FROM scalp_v2_opportunities
            WHERE state='ARMED'
            ORDER BY created_at ASC, id ASC
            """
        ).fetchall()
        by_symbol: dict[str, list[tuple]] = {}
        for row in by_symbol_rows:
            by_symbol.setdefault(str(row[1]), []).append(row)
        for group in by_symbol.values():
            if len(group) <= 1:
                continue
            for stale in group[:-1]:
                conn.execute(
                    "UPDATE scalp_v2_opportunities SET state='EXPIRED', updated_at=? WHERE id=? AND state='ARMED'",
                    (moment, stale[0]),
                )
                _queue_release(conn, int(stale[0]), str(stale[3]), int(stale[4] or 0), str(stale[1]), queued)
                collapsed += 1
        conn.commit()
        committed = True
    finally:
        conn.close()
    if committed and release_reservation is not None:
        for reservation_id, symbol in queued:
            try:
                release_reservation(reservation_id, symbol)
            except Exception:
                continue
    return {"expired": expired, "collapsed_duplicates": collapsed}


def bind_reservation(db_path: str | Path, symbol: str, opportunity_id: str, reservation_id: str) -> None:
    if not opportunity_id or not reservation_id:
        return
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        conn.execute(
            """
            UPDATE scalp_v2_opportunities
            SET reservation_id=?, reservation_released=0, updated_at=?
            WHERE id=(
                SELECT id FROM scalp_v2_opportunities
                WHERE symbol=? AND opportunity_id=? AND state='ARMED'
                ORDER BY id DESC LIMIT 1
            )
            """,
            (str(reservation_id), time.time(), _sym(symbol), str(opportunity_id)),
        )
        conn.commit()
    finally:
        conn.close()


def arm_opportunity(
    db_path: str | Path,
    symbol: str,
    setup: str,
    arm_price: float,
    engine_id: str = SCALP_V2_ENGINE_ID,
) -> tuple[str, bool]:
    """Return (opportunity_id, blocked).

    One fresh ARMED or OPEN row per symbol blocks a successor. An expired ARMED
    row is terminalized first and does not block. CLOSED history is left as-is.
    """
    opp = ScalpOpportunityId.from_intent(_sym(symbol), setup, "", arm_price=arm_price)
    sym = _sym(symbol)
    eid = str(engine_id or SCALP_V2_ENGINE_ID)
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        stale = conn.execute(
            """
            SELECT id, COALESCE(reservation_id, ''), COALESCE(reservation_released, 0)
            FROM scalp_v2_opportunities
            WHERE symbol=? AND engine_id=? AND state='ARMED'
              AND (? - created_at) > ?
            """,
            (sym, eid, now, SCALP_V2_OPP_EXPIRY_SEC),
        ).fetchall()
        for row_id, reservation_id, released in stale:
            conn.execute(
                "UPDATE scalp_v2_opportunities SET state='EXPIRED', updated_at=? WHERE id=? AND state='ARMED'",
                (now, row_id),
            )
            if str(reservation_id or "") and int(released or 0) == 0:
                conn.execute(
                    "UPDATE scalp_v2_opportunities SET reservation_released=1 WHERE id=? AND reservation_released=0",
                    (row_id,),
                )
        current = conn.execute(
            """
            SELECT id FROM scalp_v2_opportunities
            WHERE symbol=? AND engine_id=? AND state IN ('ARMED', 'OPEN')
            ORDER BY id DESC LIMIT 1
            """,
            (sym, eid),
        ).fetchone()
        if current:
            conn.commit()
            return opp.canonical_id, True
        version_row = conn.execute(
            """
            SELECT COALESCE(MAX(version), 0) FROM scalp_v2_opportunities
            WHERE symbol=? AND opportunity_id=? AND engine_id=?
            """,
            (sym, opp.canonical_id, eid),
        ).fetchone()
        version = int(version_row[0] or 0) + 1
        conn.execute(
            """
            INSERT INTO scalp_v2_opportunities(
                symbol, opportunity_id, setup_family, structural_anchor,
                state, engine_id, created_at, updated_at, version,
                reservation_id, reservation_released, tracked_low
            ) VALUES (?,?,?,?, 'ARMED', ?, ?, ?, ?, '', 0, NULL)
            """,
            (sym, opp.canonical_id, opp.setup_family, opp.structural_anchor, eid, now, now, version),
        )
        conn.commit()
        return opp.canonical_id, False
    finally:
        conn.close()


def mark_opportunity_on(conn: sqlite3.Connection, symbol: str, opportunity_id: str, state: str) -> None:
    if not opportunity_id:
        return
    _ensure(conn)
    conn.execute(
        """
        UPDATE scalp_v2_opportunities
        SET state=?, updated_at=?
        WHERE id=(
            SELECT id FROM scalp_v2_opportunities
            WHERE symbol=? AND opportunity_id=? AND state NOT IN ('RESET', 'EXPIRED')
            ORDER BY id DESC LIMIT 1
        )
        """,
        (state, time.time(), _sym(symbol), opportunity_id),
    )


def mark_opportunity(db_path: str | Path, symbol: str, opportunity_id: str, state: str) -> None:
    if not opportunity_id:
        return
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        mark_opportunity_on(conn, symbol, opportunity_id, state)
        conn.commit()
    finally:
        conn.close()


def opportunity_inventory(db_path: str | Path, *, now: float | None = None) -> dict[str, Any]:
    """Current working state. Stale ARMED rows are reported as expired, not current."""
    moment = float(now if now is not None else time.time())
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        rows = conn.execute(
            """
            SELECT symbol, state, created_at, opportunity_id,
                   COALESCE(reservation_id, ''), COALESCE(reservation_released, 0),
                   structural_anchor
            FROM scalp_v2_opportunities
            """
        ).fetchall()
    finally:
        conn.close()
    armed_current = 0
    armed_expired = 0
    counts = {"RESET": 0, "OPEN": 0, "CLOSED": 0, "EXPIRED": 0}
    by_symbol: dict[str, dict[str, int]] = {}
    oldest_active_age = None
    reservations: list[dict[str, Any]] = []
    for sym, state, created, opp_id, rid, released, anchor in rows:
        bucket = by_symbol.setdefault(str(sym), {"ARMED": 0, "EXPIRED": 0, "OPEN": 0, "CLOSED": 0, "RESET": 0})
        age = moment - float(created or 0.0)
        if state == "ARMED" and age > SCALP_V2_OPP_EXPIRY_SEC:
            armed_expired += 1
            bucket["EXPIRED"] += 1
        elif state == "ARMED":
            armed_current += 1
            bucket["ARMED"] += 1
            oldest_active_age = age if oldest_active_age is None else max(oldest_active_age, age)
            if rid and int(released or 0) == 0:
                reservations.append({"symbol": sym, "opportunity_id": opp_id, "reservation_id": rid, "anchor": anchor})
        else:
            counts[state] = counts.get(state, 0) + 1
            key = "EXPIRED" if state == "EXPIRED" else state
            if key in bucket:
                bucket[key] += 1
            if state == "OPEN":
                oldest_active_age = age if oldest_active_age is None else max(oldest_active_age, age)
                if rid and int(released or 0) == 0:
                    reservations.append({"symbol": sym, "opportunity_id": opp_id, "reservation_id": rid, "anchor": anchor})
    return {
        "armed_current": armed_current,
        "armed_expired": armed_expired,
        "reset": counts.get("RESET", 0),
        "open": counts.get("OPEN", 0),
        "closed": counts.get("CLOSED", 0),
        "expired": counts.get("EXPIRED", 0) + armed_expired,
        "by_symbol": by_symbol,
        "oldest_active_age_sec": oldest_active_age,
        "active_reservations": reservations,
    }
