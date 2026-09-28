"""SCALP_V2 live loss breaker — halts new SCALP entries, never touches exits.

Restores the contract the retired paper engine enforced (removed in 87ef12e
together with the paper runner) against the live SCALP_V2 fills:

1. Daily loss: today's (UTC) closed SCALP_V2 P&L <= -SCALP_DAILY_LOSS_LIMIT_PCT
   * SCALP principal (ledger principal * SCALP_CAPITAL_SHARE).
2. Consecutive losses: the last SCALP_MAX_CONSECUTIVE_LOSSES closed trades after
   the evaluation floor are all <= 0. The trip persists a cooldown of
   SCALP_BREAKER_RECOVERY_SEC from the newest loss; afterwards a fresh window
   starts, so the same historical streak cannot re-trip. A restart reads the
   persisted cooldown. SCALP_CIRCUIT_BREAKER_EPOCH is an operator floor.

State unavailable -> halt (fail-closed), as before.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

_CLOSED_SELLS = """
    FROM paper_trades
    WHERE engine_id = 'SCALP_V2' AND upper(side) = 'SELL' AND pnl IS NOT NULL
      AND COALESCE(counts_toward_realized, 1) != 0
      AND COALESCE(is_synthetic, 0) = 0
"""


@dataclass(frozen=True)
class BreakerResult:
    halt: bool
    reason: str = ""
    recovery_until: str = ""
    detail: str = ""


def _parse(ts: str) -> datetime | None:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _ensure(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_v2_breaker_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            tripped_at TEXT,
            recovery_until TEXT,
            eval_after TEXT,
            reason TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute("INSERT OR IGNORE INTO scalp_v2_breaker_state (id) VALUES (1)")


def _save(conn: sqlite3.Connection, *, tripped_at: str, recovery_until: str, eval_after: str, reason: str) -> None:
    conn.execute(
        "UPDATE scalp_v2_breaker_state SET tripped_at=?, recovery_until=?, eval_after=?, reason=?, updated_at=datetime('now') WHERE id=1",
        (tripped_at or None, recovery_until or None, eval_after or None, reason or None),
    )


def _evaluate(conn: sqlite3.Connection, *, principal: float, daily_loss_limit_pct: float, max_consec: int, recovery_sec: int, epoch: str, now: datetime) -> BreakerResult:
    _ensure(conn)
    today = now.strftime("%Y-%m-%d")
    epoch_dt = _parse(epoch)
    epoch_iso = epoch_dt.isoformat() if epoch_dt else ""

    today_pnl = float(
        conn.execute(
            f"SELECT COALESCE(SUM(pnl), 0) {_CLOSED_SELLS} AND date(timestamp) = ? AND (? = '' OR julianday(timestamp) >= julianday(?))",
            (today, epoch_iso, epoch_iso),
        ).fetchone()[0]
    )
    daily_limit = float(daily_loss_limit_pct) * float(principal)
    if daily_limit > 0 and today_pnl <= -daily_limit:
        return BreakerResult(True, "DAILY_LOSS_LIMIT", detail=f"today_pnl={today_pnl:.4f} limit=-{daily_limit:.4f}")

    row = conn.execute("SELECT tripped_at, recovery_until, eval_after FROM scalp_v2_breaker_state WHERE id=1").fetchone()
    tripped_at, recovery_raw, eval_after = (str(v or "") for v in row)
    recovery_dt = _parse(recovery_raw)
    trip_dt = _parse(tripped_at)
    if recovery_dt and now < recovery_dt and epoch_dt and trip_dt and epoch_dt > trip_dt:
        _save(conn, tripped_at="", recovery_until="", eval_after=eval_after, reason="")
        recovery_dt = None
    if recovery_dt and now < recovery_dt:
        return BreakerResult(True, "CONSECUTIVE_LOSSES_COOLDOWN", recovery_raw)
    if recovery_dt and now >= recovery_dt:
        eval_after = recovery_raw
        _save(conn, tripped_at="", recovery_until="", eval_after=eval_after, reason="")

    floors = [d for d in (epoch_dt, _parse(eval_after)) if d is not None]
    floor = max(floors).isoformat() if floors else ""
    if max_consec <= 0:
        return BreakerResult(False)
    recent = conn.execute(
        f"SELECT pnl, timestamp {_CLOSED_SELLS} AND (? = '' OR julianday(timestamp) > julianday(?)) ORDER BY julianday(timestamp) DESC, id DESC LIMIT ?",
        (floor, floor, int(max_consec)),
    ).fetchall()
    if len(recent) >= max_consec and all(float(r[0]) <= 0.0 for r in recent):
        newest = _parse(str(recent[0][1])) or now
        until = newest + timedelta(seconds=max(0, int(recovery_sec)))
        until_iso = until.isoformat()
        if now < until:
            _save(conn, tripped_at=newest.isoformat(), recovery_until=until_iso, eval_after=eval_after, reason="CONSECUTIVE_LOSSES")
            return BreakerResult(True, "CONSECUTIVE_LOSSES_COOLDOWN", until_iso, detail=f"{max_consec} consecutive losses")
        _save(conn, tripped_at="", recovery_until="", eval_after=until_iso, reason="")
    return BreakerResult(False)


def check_scalp_loss_breaker(db_path: str, config, *, principal: float, now: datetime | None = None) -> BreakerResult:
    """Return halt=True when new SCALP_V2 entries must stop."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            result = _evaluate(
                conn,
                principal=principal,
                daily_loss_limit_pct=float(getattr(config, "daily_loss_limit_pct", 0.0) or 0.0),
                max_consec=int(getattr(config, "max_consecutive_losses", 0) or 0),
                recovery_sec=int(getattr(config, "breaker_recovery_sec", 14400) or 0),
                epoch=str(getattr(config, "circuit_breaker_epoch", "") or ""),
                now=now,
            )
            conn.commit()
            return result
        finally:
            conn.close()
    except Exception as exc:
        logger.error("[SCALP_V2_BREAKER] state unavailable fail-closed: %s", exc)
        return BreakerResult(True, "SCALP_BREAKER_STATE_UNAVAILABLE", detail=str(exc)[:200])
