"""Structured DAY HOLD / action records. Telemetry only — no order authority."""

from __future__ import annotations

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
)

STATE_KEY = "day_decision_holds"

_BLOCKS_LIVE = {
    HARD_SAFETY_BLOCK,
    CAPITAL_OR_SLOT_BLOCK,
    ORDER_PENDING,
    DATA_REPAIR_REQUIRED,
    OPERATOR_CONTROL_BLOCK,
    COOLDOWN_ACTIVE,
}

_CAPITAL_MARKERS = ("INSUFFICIENT_CASH", "INSUFFICIENT_EXECUTABLE", "MAX_POSITIONS", "ACCOUNT_OVERALLOCATED", "ENTRY_RESERVED", "BELOW_MIN_NOTIONAL")
_ORDER_MARKERS = ("PENDING_BUY", "ORDER_ACCEPTED", "SUBMITTING")
_OPERATOR_MARKERS = ("KILL", "TRADING_PAUSED", "PAUSE", "FAILSAFE", "CIRCUIT")
_COOLDOWN_MARKERS = ("COOLDOWN",)
_DATA_MARKERS = ("STALE_MARKET", "EXIT_MARK_STALE", "NO_CANONICAL", "DATA_REPAIR")


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


def persist_hold_record(db_path: str, record: dict[str, Any]) -> None:
    if not db_path:
        return
    symbol = str(record.get("symbol") or "")
    if not symbol:
        return
    try:
        conn = sqlite3.connect(db_path, timeout=8)
        conn.execute("PRAGMA busy_timeout=8000")
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
        conn.commit()
        conn.close()
    except Exception:
        logger.debug("persist_hold_record failed", exc_info=True)


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
