"""Durable DAY trailing-buy intents. Atomic transitions. One active intent per symbol."""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WAIT_DIP = "WAIT_DIP"
TRAIL_LOW = "TRAIL_LOW"
SUBMITTING = "SUBMITTING"
ORDER_OPEN = "ORDER_OPEN"
PARTIALLY_FILLED = "PARTIALLY_FILLED"
FILLED = "FILLED"
EXPIRED = "EXPIRED"
CANCELED = "CANCELED"
FAILED = "FAILED"
RETRYABLE = "RETRYABLE"

IN_FLIGHT_STATES = frozenset({SUBMITTING, ORDER_OPEN, PARTIALLY_FILLED})
ACTIVE_STATES = frozenset({WAIT_DIP, TRAIL_LOW}) | IN_FLIGHT_STATES
TERMINAL_STATES = frozenset({FILLED, EXPIRED, CANCELED, FAILED})
_ACTIVE_SQL = "'WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED'"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS day_trailing_buy_intents (
    intent_id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    inference_id TEXT NOT NULL DEFAULT '',
    symbol TEXT NOT NULL,
    setup TEXT NOT NULL DEFAULT '',
    arm_ts REAL NOT NULL,
    arm_bid REAL NOT NULL,
    arm_ask REAL NOT NULL,
    arm_midpoint REAL NOT NULL,
    decision_score REAL NOT NULL DEFAULT 0,
    predicted_ev REAL NOT NULL DEFAULT 0,
    round_trip_cost_bps REAL NOT NULL,
    spread_bps REAL NOT NULL,
    required_improvement_bps REAL NOT NULL,
    rebound_bps REAL NOT NULL,
    min_dip_bps REAL NOT NULL,
    lowest_ask REAL NOT NULL DEFAULT 0,
    lowest_ask_ts REAL NOT NULL DEFAULT 0,
    current_ask REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    expires_at REAL NOT NULL,
    cancel_reason TEXT NOT NULL DEFAULT '',
    order_id TEXT NOT NULL DEFAULT '',
    client_order_id TEXT NOT NULL,
    fill_id TEXT NOT NULL DEFAULT '',
    trade_id TEXT NOT NULL DEFAULT '',
    reservation_id TEXT NOT NULL DEFAULT '',
    order_accepted INTEGER NOT NULL DEFAULT 0,
    quantity REAL NOT NULL DEFAULT 0,
    stop_price REAL NOT NULL DEFAULT 0,
    atr REAL NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    bar_timestamp INTEGER NOT NULL DEFAULT 0,
    sleeve TEXT NOT NULL DEFAULT '',
    notional_usd REAL NOT NULL DEFAULT 0,
    thesis_invalid_level REAL NOT NULL DEFAULT 0,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_day_tb_symbol_active
    ON day_trailing_buy_intents(symbol)
    WHERE status IN ('WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED');
CREATE UNIQUE INDEX IF NOT EXISTS idx_day_tb_client_order
    ON day_trailing_buy_intents(client_order_id);
CREATE INDEX IF NOT EXISTS idx_day_tb_decision
    ON day_trailing_buy_intents(decision_id);
CREATE INDEX IF NOT EXISTS idx_day_tb_status
    ON day_trailing_buy_intents(status);
"""


def _now() -> float:
    return time.time()


def _slash_symbol(symbol: str) -> str:
    sym = str(symbol or "").strip().upper().replace("-", "/")
    if "/" not in sym and sym.endswith("USDT") and len(sym) > 4:
        sym = sym[:-4] + "/USDT"
    return sym


def ensure_trailing_buy_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        cols = {r[1] for r in conn.execute("PRAGMA table_info(day_trailing_buy_intents)")}
        if "engine_id" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN engine_id TEXT DEFAULT 'LEGACY_DAY_LIVE'")
        if "scalp_opportunity_id" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN scalp_opportunity_id TEXT DEFAULT ''")
        # DAY_STRUCTURAL_PULLBACK_V1 columns (added idempotently)
        if "policy_version" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN policy_version TEXT DEFAULT 'LEGACY'")
        if "structural_entry_level" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN structural_entry_level REAL DEFAULT 0")
        if "pullback_reached" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN pullback_reached INTEGER DEFAULT 0")
        if "red_5m_seen" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN red_5m_seen INTEGER DEFAULT 0")
        if "reversal_candle_ts" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN reversal_candle_ts REAL DEFAULT 0")
        if "reversal_level" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN reversal_level REAL DEFAULT 0")
        if "freq_limit_state" not in cols:
            conn.execute("ALTER TABLE day_trailing_buy_intents ADD COLUMN freq_limit_state TEXT DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    out = dict(row)
    raw = out.get("payload_json") or "{}"
    try:
        out["payload"] = json.loads(raw) if isinstance(raw, str) else {}
    except (TypeError, json.JSONDecodeError):
        out["payload"] = {}
    out["order_accepted"] = bool(int(out.get("order_accepted") or 0))
    return out


def _fetch(conn: sqlite3.Connection, intent_id: str) -> dict[str, Any] | None:
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM day_trailing_buy_intents WHERE intent_id=?", (intent_id,)).fetchone()
    return _row_to_dict(row) if row else None


def new_intent_id() -> str:
    return f"tb{uuid.uuid4().hex[:16]}"


def client_order_key(intent_id: str) -> str:
    iid = str(intent_id or "").replace("-", "")
    return (iid if iid.startswith("tb") else f"tb{iid}")[:36]


def create_intent(
    db_path: str | Path,
    *,
    fields: dict[str, Any],
    supersede_reason: str = "SUPERSEDED_BY_NEWER_DECISION",
) -> tuple[bool, str, dict[str, Any] | None]:
    """Insert WAIT_DIP. One active intent per symbol. Never resets an active watch."""
    _ = supersede_reason
    ensure_trailing_buy_schema(db_path)
    symbol = _slash_symbol(str(fields.get("symbol") or ""))
    decision_id = str(fields.get("decision_id") or "").strip()
    if not symbol:
        return False, "INVALID_SYMBOL", None
    if not decision_id:
        return False, "MISSING_DECISION_ID", None
    now = _now()
    intent_id = str(fields.get("intent_id") or new_intent_id())
    cid = str(fields.get("client_order_id") or client_order_key(intent_id))
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        existing = conn.execute(
            """
            SELECT * FROM day_trailing_buy_intents
            WHERE symbol=? AND status IN ('WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED')
            LIMIT 1
            """,
            (symbol,),
        ).fetchone()
        if existing:
            cur = _row_to_dict(existing)
            if str(cur.get("decision_id") or "") == decision_id:
                conn.commit()
                return True, "IDEMPOTENT_EXISTING", cur
            if str(cur.get("status") or "") in IN_FLIGHT_STATES:
                conn.commit()
                return False, "SYMBOL_SUBMITTING", cur
            # Later decision cycles must keep the armed ask and tracked low.
            conn.commit()
            return True, "PRESERVED_EXISTING", cur
        payload = fields.get("payload") if isinstance(fields.get("payload"), dict) else {}
        # Check which optional v2 columns exist (PRAGMA, safe for any schema version)
        _cols_present = {r[1] for r in conn.execute("PRAGMA table_info(day_trailing_buy_intents)").fetchall()}
        _has_policy = "policy_version" in _cols_present
        _has_struct = "structural_entry_level" in _cols_present
        _extra_cols = ""
        _extra_vals: list[Any] = []
        if _has_policy:
            _extra_cols += ", policy_version"
            _extra_vals.append(str(fields.get("policy_version") or "LEGACY"))
        if _has_struct:
            _extra_cols += ", structural_entry_level"
            _extra_vals.append(float(fields.get("structural_entry_level") or 0.0))
        _n_extra = len(_extra_vals)
        _placeholders_extra = ("," + ",".join(["?"] * _n_extra)) if _n_extra else ""
        conn.execute(
            f"""
            INSERT INTO day_trailing_buy_intents(
                intent_id, decision_id, inference_id, symbol, setup, arm_ts,
                arm_bid, arm_ask, arm_midpoint, decision_score, predicted_ev,
                round_trip_cost_bps, spread_bps, required_improvement_bps,
                rebound_bps, min_dip_bps, lowest_ask, lowest_ask_ts, current_ask,
                status, expires_at, cancel_reason, order_id, client_order_id,
                fill_id, trade_id, reservation_id, order_accepted, quantity,
                stop_price, atr, confidence, bar_timestamp, sleeve, notional_usd,
                thesis_invalid_level, payload_json, created_at, updated_at,
                engine_id, scalp_opportunity_id{_extra_cols}
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?{_placeholders_extra})
            """,
            (
                intent_id,
                decision_id,
                str(fields.get("inference_id") or ""),
                symbol,
                str(fields.get("setup") or ""),
                float(fields.get("arm_ts") or now),
                float(fields.get("arm_bid") or 0.0),
                float(fields.get("arm_ask") or 0.0),
                float(fields.get("arm_midpoint") or 0.0),
                float(fields.get("decision_score") or 0.0),
                float(fields.get("predicted_ev") or 0.0),
                float(fields.get("round_trip_cost_bps") or 0.0),
                float(fields.get("spread_bps") or 0.0),
                float(fields.get("required_improvement_bps") or 0.0),
                float(fields.get("rebound_bps") or 0.0),
                float(fields.get("min_dip_bps") or 0.0),
                0.0,
                0.0,
                float(fields.get("arm_ask") or 0.0),
                WAIT_DIP,
                float(fields.get("expires_at") or now),
                "",
                "",
                cid,
                "",
                "",
                str(fields.get("reservation_id") or ""),
                0,
                float(fields.get("quantity") or 0.0),
                float(fields.get("stop_price") or 0.0),
                float(fields.get("atr") or 0.0),
                float(fields.get("confidence") or 0.0),
                int(fields.get("bar_timestamp") or 0),
                str(fields.get("sleeve") or ""),
                float(fields.get("notional_usd") or 0.0),
                float(fields.get("thesis_invalid_level") or 0.0),
                json.dumps(payload, default=str),
                now,
                now,
                str(fields.get("engine_id") or "LEGACY_DAY_LIVE"),
                str(fields.get("scalp_opportunity_id") or ""),
                *_extra_vals,
            ),
        )
        conn.commit()
        return True, "OK", _fetch(conn, intent_id)
    except sqlite3.IntegrityError as exc:
        conn.rollback()
        logger.warning("create_intent conflict: %s", exc)
        return False, "INTENT_CONFLICT", None
    except Exception as exc:
        conn.rollback()
        logger.warning("create_intent failed: %s", exc)
        return False, f"INTENT_ERROR:{exc}", None
    finally:
        conn.close()


def load_intent(db_path: str | Path, intent_id: str) -> dict[str, Any] | None:
    ensure_trailing_buy_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        return _fetch(conn, intent_id)
    finally:
        conn.close()


def load_active_intents(db_path: str | Path) -> list[dict[str, Any]]:
    ensure_trailing_buy_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT * FROM day_trailing_buy_intents
            WHERE status IN ('WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED')
            ORDER BY created_at ASC
            """
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def load_intent_by_symbol(db_path: str | Path, symbol: str) -> dict[str, Any] | None:
    ensure_trailing_buy_schema(db_path)
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            """
            SELECT * FROM day_trailing_buy_intents
            WHERE symbol=? AND status IN ('WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED')
            LIMIT 1
            """,
            (_slash_symbol(symbol),),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def update_watch(
    db_path: str | Path,
    intent_id: str,
    *,
    status: str | None = None,
    lowest_ask: float | None = None,
    lowest_ask_ts: float | None = None,
    current_ask: float | None = None,
    cancel_reason: str | None = None,
) -> dict[str, Any] | None:
    ensure_trailing_buy_schema(db_path)
    now = _now()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = _fetch(conn, intent_id)
        if not cur or str(cur.get("status") or "") not in ACTIVE_STATES:
            conn.commit()
            return cur
        if str(cur.get("status") or "") == SUBMITTING and status not in (None, SUBMITTING):
            conn.commit()
            return cur
        sets = ["updated_at=?"]
        args: list[Any] = [now]
        if status is not None:
            sets.append("status=?")
            args.append(status)
        if lowest_ask is not None:
            sets.append("lowest_ask=?")
            args.append(float(lowest_ask))
        if lowest_ask_ts is not None:
            sets.append("lowest_ask_ts=?")
            args.append(float(lowest_ask_ts))
        if current_ask is not None:
            sets.append("current_ask=?")
            args.append(float(current_ask))
        if cancel_reason is not None:
            sets.append("cancel_reason=?")
            args.append(str(cancel_reason)[:120])
        args.append(intent_id)
        conn.execute(f"UPDATE day_trailing_buy_intents SET {', '.join(sets)} WHERE intent_id=?", args)
        conn.commit()
        return _fetch(conn, intent_id)
    finally:
        conn.close()


def claim_submitting(db_path: str | Path, intent_id: str) -> tuple[bool, dict[str, Any] | None]:
    """Atomic TRAIL_LOW → SUBMITTING. Exactly one worker wins."""
    ensure_trailing_buy_schema(db_path)
    now = _now()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            UPDATE day_trailing_buy_intents
            SET status=?, updated_at=?
            WHERE intent_id=? AND status=?
            """,
            (SUBMITTING, now, intent_id, TRAIL_LOW),
        )
        ok = int(cur.rowcount or 0) == 1
        conn.commit()
        return ok, _fetch(conn, intent_id)
    finally:
        conn.close()


def mark_terminal(
    db_path: str | Path,
    intent_id: str,
    status: str,
    *,
    reason: str = "",
    order_id: str = "",
    fill_id: str = "",
    trade_id: str = "",
    order_accepted: bool | None = None,
    current_ask: float | None = None,
) -> dict[str, Any] | None:
    if status not in TERMINAL_STATES:
        raise ValueError(f"not terminal: {status}")
    ensure_trailing_buy_schema(db_path)
    now = _now()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        sets = ["status=?", "cancel_reason=?", "updated_at=?"]
        args: list[Any] = [status, str(reason)[:120], now]
        if order_id:
            sets.append("order_id=?")
            args.append(order_id)
        if fill_id:
            sets.append("fill_id=?")
            args.append(fill_id)
        if trade_id:
            sets.append("trade_id=?")
            args.append(trade_id)
        if order_accepted is not None:
            sets.append("order_accepted=?")
            args.append(1 if order_accepted else 0)
        if current_ask is not None:
            sets.append("current_ask=?")
            args.append(float(current_ask))
        args.extend([intent_id])
        conn.execute(
            f"UPDATE day_trailing_buy_intents SET {', '.join(sets)} WHERE intent_id=? AND status IN ('WAIT_DIP','TRAIL_LOW','SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED')",
            args,
        )
        conn.commit()
        return _fetch(conn, intent_id)
    finally:
        conn.close()


def mark_order_accepted(
    db_path: str | Path,
    intent_id: str,
    *,
    order_id: str = "",
    fill_id: str = "",
    trade_id: str = "",
) -> dict[str, Any] | None:
    """Stamp the accepted order's identifiers. Never blanks a known identifier.

    This used to write all three columns unconditionally, so any later partial
    stamp erased identity that an earlier call had already proven. The recovery
    path calls it with order_id alone while an order is merely open, which wiped
    trade_id and fill_id; that is how FILLED intents ended up carrying an
    exchange order id and nothing else. Only non-empty values are now written.
    """
    ensure_trailing_buy_schema(db_path)
    now = _now()
    sets = ["order_accepted=1", "updated_at=?"]
    args: list[Any] = [now]
    for col, val in (("order_id", order_id), ("fill_id", fill_id), ("trade_id", trade_id)):
        if str(val or "").strip():
            sets.append(f"{col}=?")
            args.append(str(val))
    args.append(intent_id)
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"UPDATE day_trailing_buy_intents SET {', '.join(sets)} WHERE intent_id=?",
            args,
        )
        conn.commit()
        return _fetch(conn, intent_id)
    finally:
        conn.close()


def mark_in_flight(
    db_path: str | Path,
    intent_id: str,
    status: str,
    *,
    order_id: str = "",
    fill_id: str = "",
    trade_id: str = "",
    reason: str = "",
) -> dict[str, Any] | None:
    """SUBMITTING → ORDER_OPEN / PARTIALLY_FILLED. Never a new submit."""
    if status not in {ORDER_OPEN, PARTIALLY_FILLED}:
        raise ValueError(f"not in-flight resolve: {status}")
    ensure_trailing_buy_schema(db_path)
    now = _now()
    sets = ["status=?", "updated_at=?", "order_accepted=1"]
    args: list[Any] = [status, now]
    if reason:
        sets.append("cancel_reason=?")
        args.append(str(reason)[:120])
    for col, val in (("order_id", order_id), ("fill_id", fill_id), ("trade_id", trade_id)):
        if str(val or "").strip():
            sets.append(f"{col}=?")
            args.append(str(val))
    args.extend([intent_id])
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute(
            f"UPDATE day_trailing_buy_intents SET {', '.join(sets)} WHERE intent_id=? AND status IN ('SUBMITTING','ORDER_OPEN','PARTIALLY_FILLED')",
            args,
        )
        conn.commit()
        return _fetch(conn, intent_id)
    finally:
        conn.close()


def release_submitting_for_retry(db_path: str | Path, intent_id: str) -> bool:
    """Only when it is proven no venue order was accepted."""
    ensure_trailing_buy_schema(db_path)
    now = _now()
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            UPDATE day_trailing_buy_intents
            SET status=?, updated_at=?
            WHERE intent_id=? AND status=? AND order_accepted=0
            """,
            (TRAIL_LOW, now, intent_id, SUBMITTING),
        )
        conn.commit()
        return int(cur.rowcount or 0) == 1
    finally:
        conn.close()


def active_symbols(db_path: str | Path) -> set[str]:
    return {str(r["symbol"]) for r in load_active_intents(db_path)}


def active_notional(db_path: str | Path, *, exclude_intent_id: str = "") -> float:
    total = 0.0
    for row in load_active_intents(db_path):
        if exclude_intent_id and str(row.get("intent_id")) == exclude_intent_id:
            continue
        total += float(row.get("notional_usd") or 0.0)
    return total


__all__ = [
    "ACTIVE_STATES",
    "CANCELED",
    "EXPIRED",
    "FAILED",
    "FILLED",
    "IN_FLIGHT_STATES",
    "ORDER_OPEN",
    "PARTIALLY_FILLED",
    "RETRYABLE",
    "SUBMITTING",
    "TERMINAL_STATES",
    "TRAIL_LOW",
    "WAIT_DIP",
    "active_notional",
    "active_symbols",
    "claim_submitting",
    "client_order_key",
    "create_intent",
    "ensure_trailing_buy_schema",
    "load_active_intents",
    "load_intent",
    "load_intent_by_symbol",
    "mark_in_flight",
    "mark_order_accepted",
    "mark_terminal",
    "new_intent_id",
    "release_submitting_for_retry",
    "update_watch",
]
