"""Structured DAY HOLD / action records. Telemetry only — no order authority."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

MODEL_HOLD_TELEMETRY = "MODEL_HOLD_TELEMETRY"
NO_RANKED_CANDIDATE = "NO_RANKED_CANDIDATE"
WAITING_FOR_DIP = "WAITING_FOR_DIP"
TRAILING_LOW = "TRAILING_LOW"
WAITING_FOR_REBOUND = "WAITING_FOR_REBOUND"
OPEN_POSITION_HOLD = "OPEN_POSITION_HOLD"
HARD_SAFETY_BLOCK = "HARD_SAFETY_BLOCK"
CAPITAL_OR_SLOT_BLOCK = "CAPITAL_OR_SLOT_BLOCK"
ORDER_PENDING = "ORDER_PENDING"
DATA_REPAIR_REQUIRED = "DATA_REPAIR_REQUIRED"
OPERATOR_CONTROL_BLOCK = "OPERATOR_CONTROL_BLOCK"
COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
TRAILING_BUY_TERMINAL = "TRAILING_BUY_TERMINAL"

HOLD_CATEGORIES = (
    MODEL_HOLD_TELEMETRY,
    NO_RANKED_CANDIDATE,
    WAITING_FOR_DIP,
    TRAILING_LOW,
    WAITING_FOR_REBOUND,
    OPEN_POSITION_HOLD,
    HARD_SAFETY_BLOCK,
    CAPITAL_OR_SLOT_BLOCK,
    ORDER_PENDING,
    DATA_REPAIR_REQUIRED,
    OPERATOR_CONTROL_BLOCK,
    COOLDOWN_ACTIVE,
    TRAILING_BUY_TERMINAL,
)

STATE_KEY = "day_decision_holds"

# Snapshot key for universe-level model telemetry. Deliberately not a tradable
# symbol so it can never collide with, or overwrite, a per-symbol trailing state.
PATH_EV_TELEMETRY_KEY = "DAY_PATH_EV"

# Append-only episode ledger. The operational_state blob above is a latest-per-symbol
# snapshot that is overwritten every cycle, so it cannot answer "what did DAY decide
# over the last 24h". This table keeps one row per contiguous state episode per symbol
# with an observation counter, which stays compact while making dips, lows, rebounds
# and decision gaps countable after the fact.
EPISODE_TABLE = "day_decision_hold_episodes"

_EPISODE_DDL = (
    f"""
    CREATE TABLE IF NOT EXISTS {EPISODE_TABLE} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        category TEXT NOT NULL,
        exact_reason TEXT NOT NULL,
        controlling_authority TEXT,
        decision_id TEXT,
        intent_id TEXT,
        blocks_live_execution INTEGER NOT NULL DEFAULT 0,
        first_seen_ts REAL NOT NULL,
        last_seen_ts REAL NOT NULL,
        observation_count INTEGER NOT NULL DEFAULT 1,
        expires_at REAL,
        next_reevaluation REAL,
        first_observed_json TEXT,
        last_observed_json TEXT,
        required_json TEXT
    )
    """,
    f"CREATE INDEX IF NOT EXISTS ix_{EPISODE_TABLE}_symbol_last ON {EPISODE_TABLE}(symbol, last_seen_ts)",
    f"CREATE INDEX IF NOT EXISTS ix_{EPISODE_TABLE}_last ON {EPISODE_TABLE}(last_seen_ts)",
    f"CREATE INDEX IF NOT EXISTS ix_{EPISODE_TABLE}_category ON {EPISODE_TABLE}(category, last_seen_ts)",
)

_episode_ready: set[str] = set()

_BLOCKS_LIVE = {
    HARD_SAFETY_BLOCK,
    CAPITAL_OR_SLOT_BLOCK,
    ORDER_PENDING,
    DATA_REPAIR_REQUIRED,
    OPERATOR_CONTROL_BLOCK,
    COOLDOWN_ACTIVE,
}

_CAPITAL_MARKERS = (
    "INSUFFICIENT_CASH",
    "INSUFFICIENT_EXECUTABLE",
    "MAX_POSITIONS",
    "ACCOUNT_OVERALLOCATED",
    "ENTRY_RESERVED",
    "BELOW_MIN_NOTIONAL",
    "NO_REMAINING_SLOT_CASH",
)
_ORDER_MARKERS = ("PENDING_BUY", "ORDER_ACCEPTED", "SUBMITTING")
_OPERATOR_MARKERS = ("KILL", "TRADING_PAUSED", "PAUSE", "FAILSAFE", "CIRCUIT")
_COOLDOWN_MARKERS = ("COOLDOWN",)
_DATA_MARKERS = ("STALE_MARKET", "STALE_OR_MISSING_BOOK", "EXIT_MARK_STALE", "NO_CANONICAL", "DATA_REPAIR")
# Watch TTL / retained-improvement expiry. These end an intent; they are not
# hard safety and must not block the next ranked cycle.
_TRAILING_TERMINAL_REASONS = frozenset({"TIMEOUT", "IMPROVEMENT_LOST", "INTENT_EXPIRED"})


def classify_hold_category(
    *,
    trailing_status: str = "",
    observe_action: str = "",
    observe_reason: str = "",
    model_side: str = "",
    reject_reason: str = "",
    open_position: bool = False,
    ranked: bool | None = None,
) -> str:
    reason = str(reject_reason or observe_reason or "").upper()
    status = str(trailing_status or "").upper()
    if any(m in reason for m in _OPERATOR_MARKERS):
        return OPERATOR_CONTROL_BLOCK
    if any(m in reason for m in _COOLDOWN_MARKERS):
        return COOLDOWN_ACTIVE
    if any(m in reason for m in _DATA_MARKERS) or (observe_action == "cancel" and "STALE" in reason):
        return DATA_REPAIR_REQUIRED
    if any(m in reason for m in _ORDER_MARKERS) or status == "SUBMITTING":
        return ORDER_PENDING
    if any(m in reason for m in _CAPITAL_MARKERS):
        return CAPITAL_OR_SLOT_BLOCK
    if reason in _TRAILING_TERMINAL_REASONS or reason.startswith("INTENT_EXPIRED"):
        return TRAILING_BUY_TERMINAL
    if reason and status in {"", "CANCELED", "FAILED", "EXPIRED"}:
        return HARD_SAFETY_BLOCK
    if status == "WAIT_DIP" or (observe_action == "watch" and status == "WAIT_DIP"):
        return WAITING_FOR_DIP
    if status == "TRAIL_LOW" and observe_reason == "NEW_LOW":
        return TRAILING_LOW
    if status == "TRAIL_LOW":
        return WAITING_FOR_REBOUND
    if open_position:
        return OPEN_POSITION_HOLD
    if ranked is False:
        return NO_RANKED_CANDIDATE
    side = str(model_side or "").upper()
    if side in {"HOLD", "SELL"}:
        return MODEL_HOLD_TELEMETRY
    return MODEL_HOLD_TELEMETRY


def hold_blocks_live_execution(category: str) -> bool:
    return str(category or "") in _BLOCKS_LIVE


def build_hold_record(
    *,
    symbol: str,
    category: str,
    reason: str,
    authority: str,
    decision_id: str = "",
    intent_id: str = "",
    observed: dict[str, Any] | None = None,
    required: dict[str, Any] | None = None,
    expires_at: float | None = None,
    next_reeval_sec: float = 15.0,
    now: float | None = None,
) -> dict[str, Any]:
    ts = float(now if now is not None else time.time())
    return {
        "symbol": str(symbol or ""),
        "category": str(category or MODEL_HOLD_TELEMETRY),
        "decision_id": str(decision_id or ""),
        "intent_id": str(intent_id or ""),
        "timestamp": ts,
        "controlling_authority": str(authority or ""),
        "exact_reason": str(reason or ""),
        "observed": observed or {},
        "required": required or {},
        "blocks_live_execution": hold_blocks_live_execution(category),
        "expires_at": expires_at,
        "next_reevaluation": ts + float(next_reeval_sec),
    }


def _ensure_episode_table(conn: sqlite3.Connection, db_path: str) -> None:
    if db_path in _episode_ready:
        return
    for stmt in _EPISODE_DDL:
        conn.execute(stmt)
    _episode_ready.add(db_path)


def _append_episode(conn: sqlite3.Connection, record: dict[str, Any]) -> None:
    """Extend the current episode, or open a new one when the state changes."""
    symbol = str(record.get("symbol") or "")
    category = str(record.get("category") or "")
    reason = str(record.get("exact_reason") or "")
    ts = float(record.get("timestamp") or time.time())
    observed = json.dumps(record.get("observed") or {}, default=str)

    last = conn.execute(
        f"SELECT id, category, exact_reason FROM {EPISODE_TABLE} WHERE symbol=? ORDER BY id DESC LIMIT 1",
        (symbol,),
    ).fetchone()

    if last and last[1] == category and last[2] == reason:
        conn.execute(
            f"UPDATE {EPISODE_TABLE} SET last_seen_ts=?, observation_count=observation_count+1, last_observed_json=?, next_reevaluation=?, expires_at=?, decision_id=?, intent_id=? WHERE id=?",
            (
                ts,
                observed,
                record.get("next_reevaluation"),
                record.get("expires_at"),
                str(record.get("decision_id") or ""),
                str(record.get("intent_id") or ""),
                last[0],
            ),
        )
        return

    conn.execute(
        f"INSERT INTO {EPISODE_TABLE}("
        "symbol, category, exact_reason, controlling_authority, decision_id, intent_id, "
        "blocks_live_execution, first_seen_ts, last_seen_ts, observation_count, "
        "expires_at, next_reevaluation, first_observed_json, last_observed_json, required_json"
        ") VALUES(?,?,?,?,?,?,?,?,?,1,?,?,?,?,?)",
        (
            symbol,
            category,
            reason,
            str(record.get("controlling_authority") or ""),
            str(record.get("decision_id") or ""),
            str(record.get("intent_id") or ""),
            1 if record.get("blocks_live_execution") else 0,
            ts,
            ts,
            record.get("expires_at"),
            record.get("next_reevaluation"),
            observed,
            observed,
            json.dumps(record.get("required") or {}, default=str),
        ),
    )


def persist_hold_record(db_path: str, record: dict[str, Any]) -> None:
    if not db_path:
        return
    symbol = str(record.get("symbol") or "")
    if not symbol:
        return
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=8)
        conn.execute("PRAGMA busy_timeout=8000")
        conn.execute("CREATE TABLE IF NOT EXISTS operational_state (key TEXT PRIMARY KEY, value_json TEXT, updated_ts INTEGER)")
        _ensure_episode_table(conn, db_path)
        # BEGIN IMMEDIATE: the snapshot is a read-modify-write of one shared JSON blob
        # keyed by symbol. Without an exclusive write transaction, two symbols updating
        # concurrently each read the pre-image and the second write drops the first.
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT value_json FROM operational_state WHERE key=?", (STATE_KEY,)).fetchone()
        current: dict[str, Any] = {}
        if row and row[0]:
            loaded = json.loads(row[0])
            if isinstance(loaded, dict):
                current = loaded
        current[symbol] = record
        payload = json.dumps(current, default=str)
        conn.execute(
            "INSERT INTO operational_state(key, value_json, updated_ts) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, updated_ts=excluded.updated_ts",
            (STATE_KEY, payload, int(time.time())),
        )
        _append_episode(conn, record)
        conn.commit()
    except Exception:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.rollback()
        logger.debug("persist_hold_record failed", exc_info=True)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def load_hold_episodes(db_path: str, *, since_ts: float | None = None, symbol: str = "", limit: int = 500) -> list[dict[str, Any]]:
    """Read the append-only episode ledger for runtime decision audits."""
    if not db_path:
        return []
    sql = (
        f"SELECT symbol, category, exact_reason, controlling_authority, decision_id, intent_id, "
        f"blocks_live_execution, first_seen_ts, last_seen_ts, observation_count, expires_at, "
        f"next_reevaluation, first_observed_json, last_observed_json, required_json "
        f"FROM {EPISODE_TABLE} WHERE 1=1"
    )
    args: list[Any] = []
    if since_ts is not None:
        sql += " AND last_seen_ts >= ?"
        args.append(float(since_ts))
    if symbol:
        sql += " AND symbol = ?"
        args.append(symbol)
    sql += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, tuple(args)).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        logger.debug("load_hold_episodes failed", exc_info=True)
        return []


def load_hold_records(db_path: str) -> list[dict[str, Any]]:
    if not db_path:
        return []
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("PRAGMA busy_timeout=5000")
        row = conn.execute("SELECT value_json FROM operational_state WHERE key=?", (STATE_KEY,)).fetchone()
        conn.close()
        if not row or not row[0]:
            return []
        data = json.loads(row[0])
        if isinstance(data, dict):
            return list(data.values())
    except Exception:
        logger.debug("load_hold_records failed", exc_info=True)
    return []


def record_trailing_observe(db_path: str, intent: dict[str, Any], decision: Any, *, ask: float) -> dict[str, Any]:
    action = str(getattr(decision, "action", "") or "")
    status = str(getattr(decision, "status", "") or intent.get("status") or "")
    reason = str(getattr(decision, "reason", "") or "")
    category = classify_hold_category(
        trailing_status=status,
        observe_action=action,
        observe_reason=reason,
    )
    record = build_hold_record(
        symbol=str(intent.get("symbol") or ""),
        category=category,
        reason=reason or status,
        authority="day_trailing_buy.observe_book",
        decision_id=str(intent.get("decision_id") or ""),
        intent_id=str(intent.get("intent_id") or ""),
        observed={"ask": ask, "action": action, "status": status},
        required={
            "min_dip_bps": intent.get("min_dip_bps"),
            "rebound_bps": intent.get("rebound_bps"),
        },
        expires_at=float(intent.get("expires_at") or 0.0) or None,
    )
    persist_hold_record(db_path, record)
    return record
