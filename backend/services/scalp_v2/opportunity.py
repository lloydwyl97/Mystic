"""SCALP V2 opportunity identity.

The id is symbol + setup family + a price-zone anchor. The clock is not part
of the id. Closing a position keeps the row. A new entry in the same zone is
blocked until price leaves that zone, which marks the closed opportunity reset.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import sqlite3
import time
from pathlib import Path

SCALP_V2_ENGINE_ID = "SCALP_V2"
LEGACY_EXIT_ONLY = "LEGACY_EXIT_ONLY"
_ZONE = 1.005  # 0.5% price zone. Leaving it is the structural reset.


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


def arm_opportunity(db_path: str | Path, symbol: str, setup: str, arm_price: float) -> tuple[str, bool]:
    """Return (opportunity_id, blocked).

    Blocked means this arm is the same price zone as an opportunity that has
    not been structurally reset. A different zone resets closed rows.
    """
    opp = ScalpOpportunityId.from_intent(_sym(symbol), setup, "", arm_price=arm_price)
    sym = _sym(symbol)
    now = time.time()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        row = conn.execute(
            """
            SELECT id, opportunity_id, state FROM scalp_v2_opportunities
            WHERE symbol=? ORDER BY id DESC LIMIT 1
            """,
            (sym,),
        ).fetchone()
        if row and row[1] == opp.canonical_id and row[2] in ("ARMED", "OPEN", "CLOSED"):
            conn.commit()
            return opp.canonical_id, True
        if row and row[1] != opp.canonical_id:
            conn.execute(
                """
                UPDATE scalp_v2_opportunities
                SET state='RESET', updated_at=?
                WHERE symbol=? AND state='CLOSED'
                """,
                (now, sym),
            )
        existing = conn.execute(
            """
            SELECT id, state FROM scalp_v2_opportunities
            WHERE symbol=? AND opportunity_id=? ORDER BY id DESC LIMIT 1
            """,
            (sym, opp.canonical_id),
        ).fetchone()
        if existing and existing[1] == "RESET":
            conn.execute(
                "UPDATE scalp_v2_opportunities SET state='ARMED', updated_at=? WHERE id=?",
                (now, existing[0]),
            )
        elif not existing or existing[1] not in ("ARMED", "OPEN", "CLOSED"):
            conn.execute(
                """
                INSERT INTO scalp_v2_opportunities(
                    symbol, opportunity_id, setup_family, structural_anchor,
                    state, engine_id, created_at, updated_at
                ) VALUES (?,?,?,?, 'ARMED', ?, ?, ?)
                """,
                (sym, opp.canonical_id, opp.setup_family, opp.structural_anchor, SCALP_V2_ENGINE_ID, now, now),
            )
        else:
            conn.commit()
            return opp.canonical_id, True
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
        WHERE symbol=? AND opportunity_id=? AND state!='RESET'
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
