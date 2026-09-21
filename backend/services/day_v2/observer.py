"""DAY V2 shadow observer.

A lightweight async loop that runs alongside the main engine as a read-only
observer. It never calls execute_buy_fifo or any order placement function.

Design rules (IMMUTABLE):
- NEVER calls execute_buy_fifo, submit_order, or any order placement function.
- NEVER modifies portfolio_engine_positions, paper_trades, portfolio_engine_ledger.
- NEVER calls stop_mystic.sh / start_mystic.sh or modifies service state.
- Reads ai_inference_log, feature_ohlcv, and open positions for observation only.
- Writes ONLY to day_v2_shadow_observations.
- Starts only if DAY_V2_ENABLED=true in the environment.

This observer accumulates data for future evaluation. It has NO trading authority.
DAY_V2_ENABLED is False by default — no data accumulates unless explicitly enabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import time
from typing import Any

logger = logging.getLogger(__name__)

# Polling interval: check for new 15m bars every 60 seconds.
_OBSERVER_POLL_SECONDS = 60

# Table name written by this observer (read-only to everything else).
SHADOW_TABLE = "day_v2_shadow_observations"

_CREATE_SHADOW_TABLE = f"""
CREATE TABLE IF NOT EXISTS {SHADOW_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    opportunity_id TEXT,
    state TEXT NOT NULL,
    exit_role_evaluated TEXT,
    exit_should_fire INTEGER,
    net_pnl_pct REAL,
    hold_minutes REAL,
    bar_ts TEXT,
    recorded_at TEXT DEFAULT (datetime('now'))
)
"""


def _observer_enabled() -> bool:
    """Return True iff DAY_V2_ENABLED=true is set in the environment."""
    raw = os.getenv("DAY_V2_ENABLED", "false")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _ensure_shadow_table(db_path: str) -> None:
    """Create the shadow observations table if it does not exist."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(_CREATE_SHADOW_TABLE)
            conn.commit()
    except Exception as exc:
        logger.warning("day_v2_observer: could not create shadow table: %s", exc)


def _write_observation(db_path: str, row: dict[str, Any]) -> None:
    """Write one observation row to day_v2_shadow_observations. Read-only to everything else."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                f"""
                INSERT INTO {SHADOW_TABLE}
                    (symbol, opportunity_id, state, exit_role_evaluated,
                     exit_should_fire, net_pnl_pct, hold_minutes, bar_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("symbol", "")),
                    str(row.get("opportunity_id", "") or ""),
                    str(row.get("state", "SHADOW_OBSERVED")),
                    str(row.get("exit_role_evaluated", "") or ""),
                    1 if row.get("exit_should_fire") else 0,
                    float(row.get("net_pnl_pct") or 0.0),
                    float(row.get("hold_minutes") or 0.0),
                    str(row.get("bar_ts", "") or ""),
                ),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("day_v2_observer: write_observation failed: %s", exc)


def _read_open_positions(db_path: str) -> list[dict[str, Any]]:
    """Read current open positions. No writes."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT symbol, entry_price, entry_time, highest_price, trailing_stop_price,
                       stop_price, take_profit_1_price, engine_id, scalp_opportunity_id
                FROM portfolio_engine_positions
                WHERE quantity > 0
                """
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _read_latest_inference(db_path: str, symbol: str, limit: int = 5) -> list[dict[str, Any]]:
    """Read the most recent inference log entries for a symbol. No writes."""
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT bar_timestamp, p_buy, p_sell, buy_margin, signal_passed
                FROM ai_inference_log
                WHERE symbol = ?
                ORDER BY bar_timestamp DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _observe_positions(db_path: str) -> None:
    """For each open position, evaluate what a DAY V2 exit role would say (shadow only)."""
    try:
        from backend.services.day_v2.exit_roles import evaluate_all_roles
    except ImportError:
        logger.debug("day_v2_observer: exit_roles not available yet")
        return

    positions = _read_open_positions(db_path)
    now_ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for pos in positions:
        symbol = str(pos.get("symbol", ""))
        entry_price = float(pos.get("entry_price") or 0.0)
        entry_time = float(pos.get("entry_time") or 0.0)
        highest_price = float(pos.get("highest_price") or entry_price)
        engine_id = str(pos.get("engine_id") or "LEGACY_DAY_LIVE")
        scalp_opp_id = str(pos.get("scalp_opportunity_id") or "")

        if entry_price <= 0 or entry_time <= 0:
            continue

        hold_minutes = (time.time() - entry_time) / 60.0
        # Use entry price as current price proxy (observer cannot fetch live ticks)
        # This is intentionally conservative — real evaluation needs live price.
        net_pnl_pct = 0.0  # unknown without live price

        try:
            roles = evaluate_all_roles(
                symbol=symbol,
                entry_price=entry_price,
                current_price=entry_price,  # conservative proxy
                highest_price=highest_price,
                net_pnl_pct=net_pnl_pct,
                hold_minutes=hold_minutes,
                engine_id=engine_id,
            )
        except Exception as exc:
            logger.debug("day_v2_observer: evaluate_all_roles failed for %s: %s", symbol, exc)
            roles = []

        for role_result in roles:
            _write_observation(
                db_path,
                {
                    "symbol": symbol,
                    "opportunity_id": scalp_opp_id,
                    "state": "SHADOW_POSITION_OBSERVED",
                    "exit_role_evaluated": str(role_result.get("role", "")),
                    "exit_should_fire": role_result.get("should_fire", False),
                    "net_pnl_pct": net_pnl_pct,
                    "hold_minutes": hold_minutes,
                    "bar_ts": now_ts,
                },
            )


async def run_day_v2_observer(db_path: str) -> None:
    """Main observer loop. Starts only if DAY_V2_ENABLED=true.

    This coroutine never returns under normal operation. It polls every
    _OBSERVER_POLL_SECONDS seconds and writes to day_v2_shadow_observations.

    NEVER places orders. NEVER modifies positions or cash. Read-only.
    """
    if not _observer_enabled():
        logger.info("day_v2_observer: DAY_V2_ENABLED is false — observer not started")
        return

    logger.info("day_v2_observer: starting (DAY_V2_ENABLED=true, db=%s)", db_path)
    _ensure_shadow_table(db_path)

    while True:
        try:
            _observe_positions(db_path)
        except Exception as exc:
            logger.warning("day_v2_observer: observation cycle error: %s", exc)
        await asyncio.sleep(_OBSERVER_POLL_SECONDS)
