"""Structural-LP circuit breaker. Isolated from the retired ranking consec breaker."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.structural_mode import FILL_MODEL_VERSION

BREAKER_ID = 1


@dataclass
class StructuralBreakerState:
    open: bool
    reason: str
    tripped_at: str
    recovery_until: str
    stats: dict[str, Any]
    thresholds: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "open": self.open,
            "status": "OPEN" if self.open else "CLOSED",
            "reason": self.reason,
            "tripped_at": self.tripped_at,
            "recovery_until": self.recovery_until,
            "stats": self.stats,
            "thresholds": self.thresholds,
            "next_eligible_recovery": self.recovery_until or "auto_after_window_or_fresh_tape",
            "book": "structural_lp",
        }


def default_thresholds(*, consec: int, daily_loss_usd: float, timeout_rate: float, adverse_rate: float, recovery_sec: int) -> dict[str, Any]:
    return {
        "max_consecutive_losses": int(consec),
        "daily_net_loss_usd": float(daily_loss_usd),
        "rolling_expectancy_n": 20,
        "max_timeout_rate": float(timeout_rate),
        "max_adverse_1s_rate": float(adverse_rate),
        "recovery_sec": int(recovery_sec),
        "stale_tape_opens": True,
        "fill_model": FILL_MODEL_VERSION,
    }


def ensure_structural_breaker_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_structural_breaker (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_open INTEGER NOT NULL DEFAULT 0,
            reason TEXT,
            tripped_at TEXT,
            recovery_until TEXT,
            stats_json TEXT,
            thresholds_json TEXT,
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    if conn.execute("SELECT 1 FROM scalp_structural_breaker WHERE id=1").fetchone() is None:
        conn.execute("INSERT INTO scalp_structural_breaker (id, is_open) VALUES (1, 0)")


def load_state(conn: sqlite3.Connection, thresholds: dict[str, Any]) -> StructuralBreakerState:
    ensure_structural_breaker_table(conn)
    row = conn.execute("SELECT is_open, reason, tripped_at, recovery_until, stats_json, thresholds_json FROM scalp_structural_breaker WHERE id=1").fetchone()
    stats = {}
    stored_th = dict(thresholds)
    if row and row[4]:
        try:
            stats = json.loads(row[4])
        except json.JSONDecodeError:
            stats = {}
    if row and row[5]:
        try:
            stored_th = {**thresholds, **json.loads(row[5])}
        except json.JSONDecodeError:
            stored_th = dict(thresholds)
    if str(stored_th.get("fill_model") or "") != FILL_MODEL_VERSION:
        return StructuralBreakerState(open=False, reason="", tripped_at="", recovery_until="", stats=stats, thresholds=thresholds)
    return StructuralBreakerState(
        open=bool(row[0]) if row else False,
        reason=str(row[1] or "") if row else "",
        tripped_at=str(row[2] or "") if row else "",
        recovery_until=str(row[3] or "") if row else "",
        stats=stats,
        thresholds=stored_th,
    )


def save_state(conn: sqlite3.Connection, state: StructuralBreakerState) -> None:
    ensure_structural_breaker_table(conn)
    conn.execute(
        """
        INSERT INTO scalp_structural_breaker
        (id, is_open, reason, tripped_at, recovery_until, stats_json, thresholds_json, updated_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          is_open=excluded.is_open,
          reason=excluded.reason,
          tripped_at=excluded.tripped_at,
          recovery_until=excluded.recovery_until,
          stats_json=excluded.stats_json,
          thresholds_json=excluded.thresholds_json,
          updated_at=datetime('now')
        """,
        (
            1 if state.open else 0,
            state.reason,
            state.tripped_at,
            state.recovery_until,
            json.dumps(state.stats),
            json.dumps(state.thresholds),
        ),
    )


def evaluate(
    *,
    consec_losses: int,
    daily_pnl: float,
    rolling_pnl: list[float],
    timeout_rate: float,
    adverse_1s_rate: float,
    tape_stale: bool,
    now: datetime,
    prior: StructuralBreakerState,
) -> StructuralBreakerState:
    th = prior.thresholds
    reasons: list[str] = []
    if tape_stale and th.get("stale_tape_opens", True):
        reasons.append("STALE_TRADE_STREAM")
    if consec_losses >= int(th["max_consecutive_losses"]):
        reasons.append("CONSECUTIVE_LOSSES")
    if daily_pnl <= -abs(float(th["daily_net_loss_usd"])):
        reasons.append("DAILY_NET_LOSS")
    n = int(th.get("rolling_expectancy_n") or 20)
    window = rolling_pnl[-n:] if rolling_pnl else []
    if len(window) >= max(5, n // 2) and sum(window) < 0:
        reasons.append("NEGATIVE_ROLLING_EXPECTANCY")
    if timeout_rate >= float(th["max_timeout_rate"]) and len(window) >= 5:
        reasons.append("EXCESSIVE_TIMEOUT_RATE")
    if adverse_1s_rate >= float(th["max_adverse_1s_rate"]) and len(window) >= 5:
        reasons.append("EXTREME_ADVERSE_SELECTION")

    stats = {
        "consec_losses": consec_losses,
        "daily_pnl": daily_pnl,
        "rolling_sum": sum(window),
        "rolling_n": len(window),
        "timeout_rate": timeout_rate,
        "adverse_1s_rate": adverse_1s_rate,
        "tape_stale": tape_stale,
    }
    iso = now.astimezone(timezone.utc).isoformat()
    if reasons:
        only_stale = reasons == ["STALE_TRADE_STREAM"]
        until_iso = ""
        if not only_stale:
            until = now.astimezone(timezone.utc).timestamp() + float(th["recovery_sec"])
            until_iso = datetime.fromtimestamp(until, tz=timezone.utc).isoformat()
        return StructuralBreakerState(
            open=True,
            reason="+".join(reasons),
            tripped_at=prior.tripped_at or iso,
            recovery_until=until_iso or "when_trade_stream_fresh",
            stats=stats,
            thresholds=th,
        )
    if prior.open and prior.recovery_until:
        try:
            rec = datetime.fromisoformat(prior.recovery_until.replace("Z", "+00:00"))
            if now.astimezone(timezone.utc) < rec.astimezone(timezone.utc):
                return StructuralBreakerState(
                    open=True,
                    reason=prior.reason,
                    tripped_at=prior.tripped_at,
                    recovery_until=prior.recovery_until,
                    stats=stats,
                    thresholds=th,
                )
        except ValueError:
            pass
    return StructuralBreakerState(open=False, reason="", tripped_at="", recovery_until="", stats=stats, thresholds=th)
