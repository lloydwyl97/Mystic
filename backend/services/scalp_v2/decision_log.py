"""One terminal result per SCALP evaluation. Waiting is not a silent miss."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

SCALP_RESULTS = frozenset(
    {
        "ARMED",
        "WAITING_FOR_PULLBACK",
        "WAITING_FOR_REBOUND",
        "SUBMITTING",
        "FILLED",
        "EXPIRED",
        "CANCELED",
        "FAILED",
    }
)

SCALP_REJECT_REASONS = frozenset(
    {
        "NO_CANDIDATE",
        "PRICE_ZONE_ALREADY_ACTIVE",
        "SYMBOL_OPPORTUNITY_ALREADY_ACTIVE",
        "OPPORTUNITY_EXPIRED",
        "ENTRY_NOT_ELIGIBLE",
        "SPREAD_TOO_WIDE",
        "BOOK_STALE",
        "CANDLE_STALE",
        "INSUFFICIENT_HISTORY",
        "NO_PULLBACK",
        "NO_REBOUND",
        "GREEN_CHASE_BLOCKED",
        "SYMBOL_OCCUPIED",
        "MAX_COMBINED_POSITIONS",
        "INSUFFICIENT_EXECUTABLE_CASH",
        "KILL_OR_PAUSE",
        "EXCHANGE_REJECTED",
        "DUPLICATE_CLIENT_ORDER",
        "UNKNOWN_ENGINE",
    }
)


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_v2_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            cycle_ts REAL NOT NULL,
            result TEXT NOT NULL,
            reason TEXT NOT NULL,
            detail TEXT NOT NULL DEFAULT ''
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scalp_v2_decisions_cycle ON scalp_v2_decisions(cycle_ts, symbol)")


def record_scalp_decision(
    db_path: str | Path,
    symbol: str,
    result: str,
    reason: str,
    *,
    cycle_ts: float | None = None,
    detail: str = "",
) -> None:
    moment = float(cycle_ts if cycle_ts is not None else time.time())
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        conn.execute(
            "INSERT INTO scalp_v2_decisions(symbol, cycle_ts, result, reason, detail) VALUES (?,?,?,?,?)",
            (str(symbol), moment, str(result), str(reason), str(detail or "")[:500]),
        )
        conn.commit()
    finally:
        conn.close()


def reason_counts(db_path: str | Path, *, since_ts: float) -> dict[str, dict[str, int]]:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        _ensure(conn)
        rows = conn.execute(
            "SELECT symbol, result, reason, COUNT(*) FROM scalp_v2_decisions WHERE cycle_ts>=? GROUP BY symbol, result, reason",
            (float(since_ts),),
        ).fetchall()
    finally:
        conn.close()
    out: dict[str, dict[str, int]] = {}
    for sym, result, reason, n in rows:
        label = str(result)
        if reason and label.startswith("REJECTED") and reason not in label:
            label = f"REJECTED:{reason}"
        bucket = out.setdefault(str(sym), {})
        bucket[label] = bucket.get(label, 0) + int(n)
    return out


def classify_scalp_candidate(row: dict | None) -> tuple[str, str]:
    """Map a router row to exactly one terminal result and a reason code."""
    if not row:
        return "REJECTED:NO_CANDIDATE", "NO_CANDIDATE"
    hard = str(row.get("hard_block") or "")
    soft = str(row.get("soft_reason") or "")
    blob = f"{hard} {soft}".upper()
    mapping = (
        ("SPREAD", "SPREAD_TOO_WIDE"),
        ("BOOK_STALE", "BOOK_STALE"),
        ("STALE_DATA", "CANDLE_STALE"),
        ("CANDLE", "CANDLE_STALE"),
        ("INSUFFICIENT", "INSUFFICIENT_HISTORY"),
        ("NO_PULLBACK", "NO_PULLBACK"),
        ("PULLBACK", "NO_PULLBACK"),
        ("NO_REBOUND", "NO_REBOUND"),
        ("REBOUND", "NO_REBOUND"),
        ("REJECTION_WICK", "NO_REBOUND"),
        ("GREEN", "GREEN_CHASE_BLOCKED"),
        ("CHASE", "GREEN_CHASE_BLOCKED"),
        ("UNKNOWN_ENGINE", "UNKNOWN_ENGINE"),
        ("DUPLICATE", "DUPLICATE_CLIENT_ORDER"),
    )
    for needle, code in mapping:
        if needle in blob:
            if code == "NO_PULLBACK":
                return "WAITING_FOR_PULLBACK", code
            if code == "NO_REBOUND":
                return "WAITING_FOR_REBOUND", code
            return f"REJECTED:{code}", code
    if row.get("snap") is None and not row.get("entry_eligible") and not blob.strip():
        return "REJECTED:BOOK_STALE", "BOOK_STALE"
    if not row.get("entry_eligible"):
        return "REJECTED:ENTRY_NOT_ELIGIBLE", "ENTRY_NOT_ELIGIBLE"
    return "ARMED", "ARMED"
