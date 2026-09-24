"""SCALP V2 live checkpoint management.

Pre-cutover context
───────────────────
34 SCALP_V2 SELL rows in paper_trades existed at live promotion (2026-09-24).
These rows are labelled SCALP_V2_PAPER_PRE_LIVE and excluded from the live
checkpoint counter.  Their historical PnL remains visible for audit.

Live checkpoint definition
──────────────────────────
A qualifying live SCALP V2 round trip requires ALL of:
  • engine_id = 'SCALP_V2'
  • side      = 'SELL'
  • mode      = 'live'
  • scalp_checkpoint_phase IS NULL  (i.e. not pre-live)
  • is_synthetic IS NOT 1
  • pnl_usd_net IS NOT NULL
  • trade_id NOT IN (SELECT trade_id FROM day_trailing_buy_intents)

The last condition excludes DAY V2 trades mislabelled SCALP_V2 by the
commit-628c47f bug (corrected by commit f4ecb36).

Usage
──────
from backend.services.scalp_v2.checkpoint import (
    mark_pre_live_rows,
    live_checkpoint_count,
    CUTOVER_TIMESTAMP,
)
"""

from __future__ import annotations

import logging
import sqlite3
import time

logger = logging.getLogger(__name__)

# Timestamp when the live promotion occurred.
# Set at module load from the scalp_v2_monitor_state table if present;
# otherwise uses the canonical cutover time written by mark_pre_live_rows().
CUTOVER_KEY = "live_cutover_ts"
PHASE_PRE_LIVE = "SCALP_V2_PAPER_PRE_LIVE"
PHASE_LIVE = "LIVE"  # NULL in the column = live; this constant is for code clarity


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Add scalp_checkpoint_phase column if absent (idempotent)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
    if "scalp_checkpoint_phase" not in cols:
        conn.execute("ALTER TABLE paper_trades ADD COLUMN scalp_checkpoint_phase TEXT DEFAULT NULL")
    # State table for cutover timestamp
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scalp_v2_checkpoint_state (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )


def mark_pre_live_rows(db_path: str, cutover_ts: float | None = None) -> int:
    """Label all SCALP_V2 SELL rows that existed before the live promotion.

    Idempotent — once the cutover timestamp is stored in scalp_v2_checkpoint_state
    this function immediately returns 0 so that post-cutover live rows are never
    mislabelled as pre-live.

    Returns the count of rows labelled (0 if already done).
    """
    ts = cutover_ts or time.time()
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        _ensure_schema(conn)
        # True idempotency: if we already stored a cutover timestamp, we are done.
        existing = conn.execute(
            "SELECT value FROM scalp_v2_checkpoint_state WHERE key=?",
            (CUTOVER_KEY,),
        ).fetchone()
        if existing is not None:
            logger.info(
                "SCALP_V2_CHECKPOINT_PRE_LIVE_ALREADY_DONE stored_ts=%s",
                existing[0],
            )
            return 0

        # First (and only) run — label all current SCALP_V2 SELL rows.
        cur = conn.execute(
            """
            UPDATE paper_trades
            SET    scalp_checkpoint_phase = ?
            WHERE  engine_id = 'SCALP_V2'
              AND  side      = 'SELL'
              AND  scalp_checkpoint_phase IS NULL
            """,
            (PHASE_PRE_LIVE,),
        )
        labelled = cur.rowcount
        # Persist cutover timestamp — subsequent calls exit early above.
        conn.execute(
            "INSERT OR REPLACE INTO scalp_v2_checkpoint_state (key, value) VALUES (?, ?)",
            (CUTOVER_KEY, str(ts)),
        )
        conn.commit()
        logger.warning(
            "SCALP_V2_CHECKPOINT_PRE_LIVE_LABELLED count=%d cutover_ts=%.0f",
            labelled,
            ts,
        )
        return labelled
    finally:
        conn.close()


def live_checkpoint_count(db_path: str) -> int:
    """Return the count of genuine live SCALP V2 closed round trips.

    Excludes:
      • All rows labelled SCALP_V2_PAPER_PRE_LIVE
      • Synthetic fills
      • Rows without pnl_usd_net
      • DAY V2 rows mislabelled SCALP_V2 (appear in day_trailing_buy_intents)
    """
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        try:
            _ensure_schema(conn)
            # Check whether day_trailing_buy_intents exists (it does on Ocean)
            has_intents = bool(conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='day_trailing_buy_intents'").fetchone())
            if has_intents:
                row = conn.execute(
                    """
                    SELECT count(*) FROM paper_trades
                    WHERE  engine_id               = 'SCALP_V2'
                      AND  side                    = 'SELL'
                      AND  mode                    = 'live'
                      AND  (scalp_checkpoint_phase IS NULL OR scalp_checkpoint_phase = 'LIVE')
                      AND  (is_synthetic IS NULL OR is_synthetic != 1)
                      AND  pnl_usd_net             IS NOT NULL
                      AND  trade_id NOT IN (
                               SELECT trade_id FROM day_trailing_buy_intents
                               WHERE trade_id IS NOT NULL AND trade_id != ''
                           )
                    """
                ).fetchone()
            else:
                row = conn.execute(
                    """
                    SELECT count(*) FROM paper_trades
                    WHERE  engine_id               = 'SCALP_V2'
                      AND  side                    = 'SELL'
                      AND  mode                    = 'live'
                      AND  (scalp_checkpoint_phase IS NULL OR scalp_checkpoint_phase = 'LIVE')
                      AND  (is_synthetic IS NULL OR is_synthetic != 1)
                      AND  pnl_usd_net             IS NOT NULL
                    """
                ).fetchone()
            return int(row[0]) if row else 0
        finally:
            conn.close()
    except Exception:
        logger.warning("SCALP_V2_CHECKPOINT_COUNT_FAILED", exc_info=True)
        return 0


def get_cutover_ts(db_path: str) -> float | None:
    """Return the stored live-promotion timestamp, or None if not set."""
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        try:
            _ensure_schema(conn)
            row = conn.execute(
                "SELECT value FROM scalp_v2_checkpoint_state WHERE key=?",
                (CUTOVER_KEY,),
            ).fetchone()
            return float(row[0]) if row else None
        finally:
            conn.close()
    except Exception:
        return None
