"""Engine-owned strategy dust that outlives its lot row.

``portfolio_engine_positions`` holds one row per (engine_id, symbol), so a
same-engine re-entry would overwrite a DUST_PENDING residual. The residual is
still that engine's inventory: it moves here, keeps its fill provenance, holds
no slot, and is never protected external inventory.

A held dust lot leaves this store only through a non-strategy balance event
(``external_balance_events``): no P&L, no learning, no cooldown, no exit.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

DUST_TABLE = "engine_strategy_dust"
EVENT_TABLE = "external_balance_events"
STATUS_HELD = "HELD"
STATUS_RETIRED = "RETIRED"
EVENT_CONVERSION = "EXTERNAL_BALANCE_CONVERSION"
EVENT_UNATTRIBUTED = "EXTERNAL_BALANCE_REMOVAL_UNATTRIBUTED"
EXTERNAL_EVENT_CLASSES = frozenset({EVENT_CONVERSION, EVENT_UNATTRIBUTED})

_SCHEMA = (
    f"""
    CREATE TABLE IF NOT EXISTS {DUST_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        engine_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        source_trade_id TEXT NOT NULL UNIQUE,
        quantity REAL NOT NULL,
        quantity_exact TEXT,
        entry_price REAL,
        provenance_json TEXT NOT NULL DEFAULT '{{}}',
        status TEXT NOT NULL DEFAULT '{STATUS_HELD}',
        retired_class TEXT,
        retired_ref TEXT,
        created_at TEXT NOT NULL,
        retired_at TEXT
    )
    """,
    f"CREATE INDEX IF NOT EXISTS ix_engine_dust_symbol ON {DUST_TABLE}(symbol, status)",
    f"""
    CREATE TABLE IF NOT EXISTS {EVENT_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detected_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        asset TEXT NOT NULL,
        engine_id TEXT,
        source_trade_id TEXT,
        quantity REAL NOT NULL,
        event_class TEXT NOT NULL,
        venue_ref TEXT,
        venue_time_utc TEXT,
        venue_evidence_json TEXT NOT NULL DEFAULT '{{}}',
        source TEXT NOT NULL,
        UNIQUE(source_trade_id, event_class)
    )
    """,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA:
        conn.execute(stmt)


def _norm(symbol: str) -> str:
    s = str(symbol or "").strip().upper()
    if "/" not in s and s.endswith("USDT"):
        s = f"{s[:-4]}/USDT"
    return s


def preserve_overwritten_dust(conn: sqlite3.Connection, *, engine_id: str, symbol: str, new_trade_id: str) -> dict[str, Any] | None:
    """Move this engine's DUST_PENDING row aside before a new lot takes its key.

    Runs inside the caller's position-write transaction so the dust and the
    new lot can never both be lost or both be the row.
    """
    ensure_schema(conn)
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
    want = [c for c in ("trade_id", "quantity", "entry_price", "status", "entry_time", "entry_order_id", "entry_fill_ids_json", "scalp_opportunity_id", "dust_detected_at") if c in cols]
    row = conn.execute(
        f"SELECT {', '.join(want)} FROM portfolio_engine_positions WHERE engine_id=? AND symbol=?",
        (str(engine_id or ""), _norm(symbol)),
    ).fetchone()
    if row is None:
        return None
    rec = dict(zip(want, row, strict=False))
    old_tid = str(rec.get("trade_id") or "")
    qty = float(rec.get("quantity") or 0.0)
    if str(rec.get("status") or "").upper() != "DUST_PENDING" or not old_tid or old_tid == str(new_trade_id or "") or qty <= 0:
        return None
    provenance = {
        "entry_time": rec.get("entry_time"),
        "entry_order_id": rec.get("entry_order_id"),
        "entry_fill_ids_json": rec.get("entry_fill_ids_json"),
        "scalp_opportunity_id": rec.get("scalp_opportunity_id"),
        "dust_detected_at": rec.get("dust_detected_at"),
        "superseded_by_trade_id": str(new_trade_id or ""),
    }
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {DUST_TABLE}
        (engine_id, symbol, source_trade_id, quantity, quantity_exact, entry_price, provenance_json, status, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            str(engine_id or ""),
            _norm(symbol),
            old_tid,
            qty,
            format(qty, ".12g"),
            float(rec.get("entry_price") or 0.0),
            json.dumps(provenance, default=str),
            STATUS_HELD,
            _now(),
        ),
    )
    return {"engine_id": str(engine_id or ""), "symbol": _norm(symbol), "source_trade_id": old_tid, "quantity": qty}


def held_lots(db_path_or_conn: str | sqlite3.Connection, symbol: str | None = None) -> list[dict[str, Any]]:
    def _q(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        ensure_schema(conn)
        sql = f"SELECT engine_id, symbol, source_trade_id, quantity, entry_price, provenance_json, created_at FROM {DUST_TABLE} WHERE status=?"
        params: list[Any] = [STATUS_HELD]
        if symbol:
            sql += " AND symbol=?"
            params.append(_norm(symbol))
        keys = ("engine_id", "symbol", "source_trade_id", "quantity", "entry_price", "provenance_json", "created_at")
        return [dict(zip(keys, r, strict=False)) for r in conn.execute(sql + " ORDER BY id", params).fetchall()]

    if isinstance(db_path_or_conn, sqlite3.Connection):
        return _q(db_path_or_conn)
    try:
        with sqlite3.connect(str(db_path_or_conn), timeout=10) as conn:
            return _q(conn)
    except sqlite3.Error:
        return []


def held_quantity(db_path_or_conn: str | sqlite3.Connection, symbol: str) -> float:
    return float(sum(float(r["quantity"] or 0.0) for r in held_lots(db_path_or_conn, symbol)))


def attribute_unclaimed_strategy_dust(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    max_quantity: float,
) -> list[dict[str, Any]]:
    """Claim fill-proven strategy dust not yet tracked in open lots or engine_strategy_dust.

    Prevents reconciliation from collapsing SCALP_V2 or DAY_V2 sub-step residuals
    into LEGACY_DAY_LIVE / dust_exchange:* if a lot closed out of open_positions
    while leaving fill-proven base-asset dust on the venue.
    """
    from decimal import Decimal

    from backend.services.engine_lot_ownership import STRATEGY_ENGINES, fill_owned_quantity

    ensure_schema(conn)
    sym = _norm(symbol)
    budget = Decimal(str(max(0.0, float(max_quantity or 0.0))))
    if budget <= Decimal("0"):
        return []

    tables = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
    if "live_exchange_fills" not in tables:
        return []

    open_tids: set[str] = set()
    if "portfolio_engine_positions" in tables:
        open_tids = {str(r[0] or "") for r in conn.execute("SELECT trade_id FROM portfolio_engine_positions WHERE symbol=?", (sym,)).fetchall() if r[0]}
    known_dust_tids = {str(r[0] or "") for r in conn.execute(f"SELECT source_trade_id FROM {DUST_TABLE} WHERE symbol=?", (sym,)).fetchall() if r[0]}
    retired_event_tids = {str(r[0] or "") for r in conn.execute(f"SELECT source_trade_id FROM {EVENT_TABLE} WHERE symbol=?", (sym,)).fetchall() if r[0]}

    cand_rows = conn.execute(
        """
        SELECT mystic_trade_id, MAX(avg_fill_price), MIN(id)
        FROM live_exchange_fills
        WHERE UPPER(side)='BUY' AND (symbol=? OR REPLACE(UPPER(symbol), '/', '')=?)
          AND mystic_trade_id IS NOT NULL AND mystic_trade_id != ''
        GROUP BY mystic_trade_id
        ORDER BY MIN(id) DESC
        """,
        (sym, sym.replace("/", "")),
    ).fetchall()

    claimed: list[dict[str, Any]] = []
    for raw_tid, avg_px, _min_id in cand_rows:
        if budget <= Decimal("1e-12"):
            break
        tid = str(raw_tid or "").strip()
        if not tid or tid in open_tids or tid in known_dust_tids or tid in retired_event_tids:
            continue

        owned = fill_owned_quantity(conn, tid, sym)
        if owned is None or owned <= Decimal("1e-12"):
            continue

        eng = ""
        entry_px = float(avg_px or 0.0)
        entry_time = None
        if "paper_trades" in tables:
            pt_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()}
            sel_eng = "engine_id" if "engine_id" in pt_cols else "''"
            sel_strat = "strategy_id" if "strategy_id" in pt_cols else "''"
            pt = conn.execute(
                f"SELECT {sel_eng}, {sel_strat}, price, timestamp FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY' LIMIT 1",
                (tid,),
            ).fetchone()
            if pt is not None:
                raw_eng = str(pt[0] or "").strip().upper()
                raw_strat = str(pt[1] or "").strip().lower()
                if raw_eng in STRATEGY_ENGINES:
                    eng = raw_eng
                elif "scalp" in raw_strat:
                    eng = "SCALP_V2"
                if float(pt[2] or 0.0) > 0:
                    entry_px = float(pt[2])
                if pt[3]:
                    try:
                        entry_time = datetime.fromisoformat(str(pt[3]).replace("Z", "+00:00")).timestamp()
                    except ValueError:
                        entry_time = None
        if not eng:
            if tid.lower().startswith("scalp_v2_"):
                eng = "SCALP_V2"
            elif tid.startswith("mystic_"):
                eng = "DAY_V2"
        if eng not in STRATEGY_ENGINES:
            continue

        claim_dec = min(owned, budget)
        if claim_dec <= Decimal("1e-12"):
            continue
        claim_qty = float(claim_dec)
        prov = {
            "entry_time": entry_time,
            "reconciled_from": "live_exchange_fills",
            "fill_owned_qty": format(owned.normalize(), "f"),
        }
        cur = conn.execute(
            f"""
            INSERT OR IGNORE INTO {DUST_TABLE}
            (engine_id, symbol, source_trade_id, quantity, quantity_exact, entry_price, provenance_json, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                eng,
                sym,
                tid,
                claim_qty,
                format(claim_dec.normalize(), "f"),
                entry_px,
                json.dumps(prov, default=str),
                STATUS_HELD,
                _now(),
            ),
        )
        if cur.rowcount:
            budget -= claim_dec
            claimed.append(
                {
                    "engine_id": eng,
                    "symbol": sym,
                    "source_trade_id": tid,
                    "quantity": claim_qty,
                }
            )
    return claimed


def held_inventory_equity(
    db_path: str,
    prices: dict[str, float] | None,
    open_trade_ids: set[str],
    mark: Callable[[str], float] | None = None,
) -> tuple[float, float]:
    """Market value and cost of HELD dust not already marked as an open lot.

    These coins are owned. They are not an active strategy position, so the
    caller adds them to account equity and not to a sleeve's realized P&L.
    ``mark(symbol)`` prices a symbol missing from ``prices``; cost basis is the
    last resort.
    """
    market = 0.0
    cost = 0.0
    marks = prices or {}
    for lot in held_lots(db_path):
        if str(lot.get("source_trade_id") or "") in open_trade_ids:
            continue
        qty = float(lot.get("quantity") or 0.0)
        basis = float(lot.get("entry_price") or 0.0)
        if qty <= 0 or basis < 0:
            continue
        sym = str(lot.get("symbol") or "")
        base = sym.split("/", maxsplit=1)[0]
        found = marks.get(sym) or marks.get(base) or marks.get(sym.replace("/", "")) or (mark(sym) if mark else 0.0)
        px = float(found) if found and float(found) > 0 else basis
        market += qty * px
        cost += qty * basis
    return market, cost


def record_external_balance_event(
    conn: sqlite3.Connection,
    *,
    symbol: str,
    quantity: float,
    event_class: str,
    source: str,
    engine_id: str = "",
    source_trade_id: str = "",
    venue_ref: str = "",
    venue_time_utc: str = "",
    evidence: dict[str, Any] | None = None,
) -> bool:
    if event_class not in EXTERNAL_EVENT_CLASSES:
        raise ValueError(f"not an external balance event class: {event_class}")
    ensure_schema(conn)
    sym = _norm(symbol)
    cur = conn.execute(
        f"""
        INSERT OR IGNORE INTO {EVENT_TABLE}
        (detected_at, symbol, asset, engine_id, source_trade_id, quantity, event_class, venue_ref, venue_time_utc, venue_evidence_json, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            _now(),
            sym,
            sym.split("/", 1)[0],
            str(engine_id or ""),
            str(source_trade_id or ""),
            float(quantity or 0.0),
            event_class,
            str(venue_ref or ""),
            str(venue_time_utc or ""),
            json.dumps(evidence or {}, default=str),
            str(source or ""),
        ),
    )
    return bool(cur.rowcount)


def zero_lot_remaining(conn: sqlite3.Connection, source_trade_id: str) -> int:
    """The venue no longer holds this lot's residue; its BUY row has nothing left."""
    try:
        cur = conn.execute(
            "UPDATE paper_trades SET remaining_position=0 WHERE trade_id=? AND UPPER(side)='BUY' AND COALESCE(remaining_position,0)>0",
            (str(source_trade_id or ""),),
        )
    except sqlite3.OperationalError:
        return 0
    return int(cur.rowcount or 0)


def retire_held_dust(conn: sqlite3.Connection, source_trade_id: str, *, event_class: str, venue_ref: str = "") -> bool:
    if event_class not in EXTERNAL_EVENT_CLASSES:
        raise ValueError(f"not an external balance event class: {event_class}")
    ensure_schema(conn)
    cur = conn.execute(
        f"UPDATE {DUST_TABLE} SET status=?, retired_class=?, retired_ref=?, retired_at=? WHERE source_trade_id=? AND status=?",
        (STATUS_RETIRED, event_class, str(venue_ref or ""), _now(), str(source_trade_id or ""), STATUS_HELD),
    )
    if cur.rowcount:
        zero_lot_remaining(conn, source_trade_id)
    return bool(cur.rowcount)


def match_dust_conversion(
    venue_rows: list[dict[str, Any]],
    *,
    asset: str,
    after_epoch: float,
) -> dict[str, Any] | None:
    """Latest venue dust conversion of ``asset`` at or after ``after_epoch``.

    ``venue_rows`` is Binance.US ``asset/query/dust-logs`` userDustConvertHistory.
    Only an exchange record proves a conversion; no match means unattributed.
    """
    best: dict[str, Any] | None = None
    want = str(asset or "").upper()
    for conv in venue_rows or []:
        if not isinstance(conv, dict):
            continue
        op_ms = float(conv.get("operateTime") or 0.0)
        if op_ms / 1000.0 + 1e-6 < float(after_epoch or 0.0):
            continue
        for d in conv.get("userAssetDribbletDetails") or []:
            if str(d.get("fromAsset") or "").upper() != want:
                continue
            cand = {
                "tran_id": str(d.get("tranId") or conv.get("tranId") or ""),
                "operate_time_ms": int(op_ms),
                "operate_time_utc": datetime.fromtimestamp(op_ms / 1000.0, tz=timezone.utc).isoformat(),
                "from_asset": want,
                "amount": float(d.get("amount") or 0.0),
                "to_asset": str(d.get("toAsset") or conv.get("toAsset") or "USDT"),
                "transferred_amount": float(d.get("transferedAmount") or d.get("transferredAmount") or 0.0),
                "service_charge": float(d.get("serviceChargeAmount") or 0.0),
            }
            if best is None or cand["operate_time_ms"] > best["operate_time_ms"]:
                best = cand
    return best


def symbol_inventory_identity(
    *,
    exchange_qty: float,
    active: dict[str, float],
    lot_dust: dict[str, float],
    held_dust: dict[str, float],
    protected: float,
) -> dict[str, Any]:
    """exchange = sum(active) + sum(lot dust) + sum(held dust) + protected + residue."""
    accounted = sum(active.values()) + sum(lot_dust.values()) + sum(held_dust.values()) + float(protected or 0.0)
    return {
        "exchange": float(exchange_qty or 0.0),
        "active": dict(active),
        "lot_dust": dict(lot_dust),
        "held_dust": dict(held_dust),
        "protected": float(protected or 0.0),
        "accounted": accounted,
        "residue": float(exchange_qty or 0.0) - accounted,
        "computed_at": time.time(),
    }
