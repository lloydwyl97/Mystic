"""One live symbol claim shared by SCALP V2 and DAY V2.

Dust does not consume a slot. A second engine asking for a claimed symbol
records SYMBOL_OCCUPIED_BY_OTHER_ENGINE. The reservation row is the lock.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.services.day_entry_reservations import create_reservation, release_reservation

COMBINED_POSITION_CAP = 4
SYMBOL_OCCUPIED_BY_OTHER_ENGINE = "SYMBOL_OCCUPIED_BY_OTHER_ENGINE"


def _norm(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in raw and raw.endswith("USDT"):
        raw = raw[:-4] + "/USDT"
    return raw


def held_slot_count(positions: dict, *, dust_status: str = "DUST_PENDING") -> int:
    n = 0
    for pos in (positions or {}).values():
        status = str(getattr(pos, "status", "ACTIVE") or "ACTIVE")
        qty = float(getattr(pos, "quantity", 0) or 0)
        if qty > 0 and status != dust_status:
            n += 1
    return n


def symbol_owner(positions: dict, symbol: str, *, dust_status: str = "DUST_PENDING") -> str:
    pos = None
    want = _norm(symbol).replace("/", "")
    for key, row in (positions or {}).items():
        if str(key).upper().replace("-", "").replace("/", "") == want:
            pos = row
            break
    if pos is None:
        return ""
    status = str(getattr(pos, "status", "ACTIVE") or "ACTIVE")
    qty = float(getattr(pos, "quantity", 0) or 0)
    if qty <= 0 or status == dust_status:
        return ""
    return str(getattr(pos, "engine_id", "") or "")


def claim_symbol(
    db_path: str | Path,
    symbol: str,
    engine_id: str,
    decision_id: str,
    notional_usd: float,
    *,
    positions: dict | None = None,
    max_positions: int = COMBINED_POSITION_CAP,
) -> tuple[bool, str, str]:
    """Return (ok, reason, reservation_id). Exactly one engine can hold the symbol."""
    owner = symbol_owner(positions or {}, symbol)
    eid = str(engine_id or "")
    if owner and owner != eid:
        return False, SYMBOL_OCCUPIED_BY_OTHER_ENGINE, ""
    if owner and owner == eid:
        return False, "SYMBOL_OCCUPIED", ""
    if held_slot_count(positions or {}) >= int(max_positions):
        return False, "MAX_COMBINED_POSITIONS", ""
    ok, reason, rid = create_reservation(
        db_path,
        decision_id=str(decision_id),
        symbol=_norm(symbol),
        notional_usd=float(notional_usd),
        sleeve=eid,
        ttl_sec=float(3600),
    )
    if ok:
        return True, reason, rid
    if reason == "SYMBOL_RESERVED":
        other = _reservation_sleeve(db_path, _norm(symbol))
        if other and other != eid:
            return False, SYMBOL_OCCUPIED_BY_OTHER_ENGINE, ""
        return False, "SYMBOL_OCCUPIED", ""
    return False, reason, ""


def release_claim(db_path: str | Path, *, reservation_id: str = "", decision_id: str = "", symbol: str = "") -> bool:
    return release_reservation(
        db_path,
        reservation_id=reservation_id,
        decision_id=decision_id,
        symbol=_norm(symbol) if symbol else "",
        reason="RELEASED",
    )


def _reservation_sleeve(db_path: str | Path, symbol: str) -> str:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        row = conn.execute(
            "SELECT sleeve FROM day_entry_reservations WHERE symbol=? AND status='ACTIVE' LIMIT 1",
            (symbol,),
        ).fetchone()
    except sqlite3.OperationalError:
        return ""
    finally:
        conn.close()
    return str(row[0] or "") if row else ""
