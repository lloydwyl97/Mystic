"""Two-engine symbol claims for SCALP V2 and DAY V2.

Two-engine contract: each engine owns up to 4 slots (combined hard max 8) and
position identity is (engine_id, symbol). Cross-engine same-symbol coexistence
is allowed, so a claim by one engine never blocks the other engine:

- same engine + same symbol held      -> SYMBOL_OCCUPIED (duplicate protection)
- engine at its own cap               -> ENGINE_MAX_POSITIONS
- combined lots at 8                  -> MAX_COMBINED_POSITIONS
- duplicate ACTIVE reservation within the SAME engine+symbol -> SYMBOL_OCCUPIED

Dust does not consume a slot. The reservation row remains a short-lived
execution mutex; it is never held for the lifetime of a position.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from backend.services.day_entry_reservations import create_reservation, release_reservation

DAY_MAX_OPEN_POSITIONS = 4
SCALP_MAX_OPEN_POSITIONS = 4
COMBINED_ENGINE_MAX_POSITIONS = 8
# Legacy alias: previously the shared 4-slot cap. Kept for import compat.
COMBINED_POSITION_CAP = 4
SYMBOL_OCCUPIED_BY_OTHER_ENGINE = "SYMBOL_OCCUPIED_BY_OTHER_ENGINE"
SYMBOL_OCCUPIED = "SYMBOL_OCCUPIED"
ENGINE_MAX_POSITIONS = "ENGINE_MAX_POSITIONS"
MAX_COMBINED_POSITIONS = "MAX_COMBINED_POSITIONS"


def engine_cap(engine_id: str) -> int:
    return SCALP_MAX_OPEN_POSITIONS if str(engine_id or "") == "SCALP_V2" else DAY_MAX_OPEN_POSITIONS


def _norm(symbol: str) -> str:
    raw = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in raw and raw.endswith("USDT"):
        raw = raw[:-4] + "/USDT"
    return raw


def _pos_engine(pos: object) -> str:
    return str(getattr(pos, "engine_id", "") or "")


def _slots_of(positions: dict, engine_id: str) -> list:
    """Slot-consuming lots owned by one engine (dust excluded)."""
    from backend.services.protected_external_inventory import consumes_strategy_slot

    eid = str(engine_id or "")
    return [pos for pos in (positions or {}).values() if consumes_strategy_slot(pos) and str(getattr(pos, "status", "ACTIVE") or "ACTIVE") != "DUST_PENDING" and _pos_engine(pos) == eid]


def held_slot_count(positions: dict, *, dust_status: str = "DUST_PENDING", engine_id: str = "") -> int:
    """Slot-consuming lots. Engine-scoped when engine_id given, else combined."""
    if engine_id:
        return len(_slots_of(positions, engine_id))
    from backend.services.protected_external_inventory import consumes_strategy_slot

    return sum(1 for pos in (positions or {}).values() if consumes_strategy_slot(pos) and str(getattr(pos, "status", "ACTIVE") or "ACTIVE") != dust_status)


def symbol_owner(positions: dict, symbol: str, *, dust_status: str = "DUST_PENDING", engine_id: str = "") -> str:
    """Owning engine of a held lot on symbol (optionally restricted to engine_id)."""
    from backend.services.protected_external_inventory import consumes_strategy_slot

    want = _norm(symbol).replace("/", "")
    for key, row in (positions or {}).items():
        key_sym = str(key).upper().replace("-", "").replace("/", "")
        if key_sym != want and _norm(str(getattr(row, "symbol", "") or "")).replace("/", "") != want:
            continue
        status = str(getattr(row, "status", "ACTIVE") or "ACTIVE")
        if not consumes_strategy_slot(row) or status == dust_status:
            continue
        owner = _pos_engine(row)
        if engine_id and owner != str(engine_id):
            continue
        if owner:
            return owner
    return ""


def claim_symbol(
    db_path: str | Path,
    symbol: str,
    engine_id: str,
    decision_id: str,
    notional_usd: float,
    *,
    positions: dict | None = None,
    max_positions: int = COMBINED_ENGINE_MAX_POSITIONS,
    engine_max_positions: int = 0,
) -> tuple[bool, str, str]:
    """Return (ok, reason, reservation_id) under the two-engine contract.

    Cross-engine same-symbol claims are allowed. Same-engine duplicates,
    per-engine caps, and the combined 8-cap still block.
    """
    eid = str(engine_id or "")
    ecap = int(engine_max_positions) if engine_max_positions else engine_cap(eid)
    if symbol_owner(positions or {}, symbol, engine_id=eid):
        return False, SYMBOL_OCCUPIED, ""
    if len(_slots_of(positions or {}, eid)) >= ecap:
        return False, ENGINE_MAX_POSITIONS, ""
    if held_slot_count(positions or {}) >= int(max_positions):
        return False, MAX_COMBINED_POSITIONS, ""
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
        # With sleeve-aware reservations this only fires on same-engine
        # duplicates (cross-engine reservations coexist).
        return False, SYMBOL_OCCUPIED, ""
    return False, reason, ""


def release_claim(
    db_path: str | Path,
    *,
    reservation_id: str = "",
    decision_id: str = "",
    symbol: str = "",
    engine_id: str = "",
) -> bool:
    return release_reservation(
        db_path,
        reservation_id=reservation_id,
        decision_id=decision_id,
        symbol=_norm(symbol) if symbol else "",
        reason="RELEASED",
        sleeve=str(engine_id or ""),
    )


def _reservation_sleeves(db_path: str | Path, symbol: str) -> set[str]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        rows = conn.execute(
            "SELECT sleeve FROM day_entry_reservations WHERE symbol=? AND status='ACTIVE'",
            (symbol,),
        ).fetchall()
    except sqlite3.OperationalError:
        return set()
    finally:
        conn.close()
    return {str(r[0] or "") for r in rows}


def _reservation_sleeve(db_path: str | Path, symbol: str) -> str:
    sleeves = _reservation_sleeves(db_path, symbol)
    return next(iter(sleeves)) if sleeves else ""
