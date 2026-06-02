import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.database_schema import DATABASE_PATH

logger = logging.getLogger(__name__)


class TradePerformanceTracker:
    """Tracks trade performance for API reporting"""

    _initialized: bool = False

    @classmethod
    def _ensure_initialized(cls) -> None:
        if cls._initialized:
            return
        # Attempt to create table and indexes once per process
        if cls.initialize_trade_performance_table():
            cls._initialized = True

    @staticmethod
    async def log_trade_performance(
        trade_id: int,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        pnl: float,
        pnl_pct: float,
        is_win: int | None,  # 1=win, 0=loss, None=breakeven
        hold_time_seconds: int | None = None,
        strategy: str | None = None,
        confidence: float | None = None,
        mode: str | None = None,
        regime: str | None = None,  # OPTION 3: 'trend', 'range', 'normal'
        tp_sl_ratio: float | None = None,  # OPTION 3: TP/SL risk-reward ratio
    ) -> bool:
        """Log trade performance to database"""
        conn = None
        try:
            TradePerformanceTracker._ensure_initialized()
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO trade_performance
                (trade_id, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,
                 is_win, hold_time_seconds, strategy, confidence, mode, regime, tp_sl_ratio, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade_id,
                    symbol,
                    side,
                    entry_price,
                    exit_price,
                    quantity,
                    pnl,
                    pnl_pct,
                    is_win,
                    hold_time_seconds,
                    strategy,
                    confidence,
                    mode,
                    regime,
                    tp_sl_ratio,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

            conn.commit()
            logger.debug(f"Logged trade performance: {symbol} {side} PnL=${pnl:.2f}")

        except Exception as e:
            logger.exception(f"Failed to log trade performance: {e}")
            return False
        else:
            return True
        finally:
            if conn:
                conn.close()

    @staticmethod
    def get_trade_performance_summary(since_timestamp: str | None = None) -> dict[str, Any]:
        """Get summary statistics for API endpoints, optionally filtered by timestamp"""
        conn = None
        try:
            TradePerformanceTracker._ensure_initialized()
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            # Build query with optional timestamp filter
            # Include all trades (paper and live) for unified reporting
            query = """
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN is_win = 1 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN is_win = 0 THEN 1 ELSE 0 END) as losing_trades,
                    AVG(pnl) as avg_pnl,
                    SUM(pnl) as total_pnl,
                    AVG(pnl_pct) as avg_pnl_pct,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trade_performance
            """

            params = []
            if since_timestamp:
                query += " WHERE timestamp >= ?"
                params.append(since_timestamp)

            cursor.execute(query, params)
            result = cursor.fetchone()

            total_trades = result[0] or 0
            winning_trades = result[1] or 0
            losing_trades = result[2] or 0

            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0.0

            return {
                "total_trades": total_trades,
                "winning_trades": winning_trades,
                "losing_trades": losing_trades,
                "win_rate": win_rate,
                "avg_pnl": result[3] or 0.0,
                "total_pnl": result[4] or 0.0,
                "avg_pnl_pct": result[5] or 0.0,
                "best_trade": result[6] or 0.0,
                "worst_trade": result[7] or 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filtered_since": since_timestamp,
            }

        except Exception as e:
            logger.exception(f"Failed to get trade performance summary: {e}")
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "total_pnl": 0.0,
                "avg_pnl_pct": 0.0,
                "best_trade": 0.0,
                "worst_trade": 0.0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "filtered_since": since_timestamp,
            }
        finally:
            if conn:
                conn.close()

    @staticmethod
    def archive_old_trades(before_timestamp: str) -> dict[str, int]:
        """Move trades before timestamp to archive table and return counts"""
        conn = None
        try:
            TradePerformanceTracker._ensure_initialized()
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            # Create archive table if it doesn't exist
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_performance_archive (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    is_win INTEGER,
                    hold_time_seconds INTEGER,
                    strategy TEXT,
                    confidence REAL,
                    timestamp TEXT NOT NULL,
                    archived_at TEXT NOT NULL
                )
            """)

            # Count trades to be archived
            cursor.execute("SELECT COUNT(*) FROM trade_performance WHERE timestamp < ?", (before_timestamp,))
            count_to_archive = cursor.fetchone()[0]

            # Move old trades to archive
            cursor.execute(
                """
                INSERT INTO trade_performance_archive
                SELECT *, ? FROM trade_performance WHERE timestamp < ?
            """,
                (datetime.now(timezone.utc).isoformat(), before_timestamp),
            )

            # Delete from main table
            cursor.execute("DELETE FROM trade_performance WHERE timestamp < ?", (before_timestamp,))

            conn.commit()

            archived_count = cursor.rowcount

            logger.info(f"Archived {archived_count} trades before {before_timestamp}")

        except Exception as e:
            logger.exception(f"Failed to archive old trades: {e}")
            return {"archived_count": 0, "total_found": 0}
        else:
            return {"archived_count": archived_count, "total_found": count_to_archive}
        finally:
            if conn:
                conn.close()

    @staticmethod
    def initialize_trade_performance_table() -> bool:
        """Create trade_performance table if it doesn't exist"""
        conn = None
        try:
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    pnl REAL NOT NULL,
                    pnl_pct REAL NOT NULL,
                    is_win INTEGER,
                    hold_time_seconds INTEGER,
                    strategy TEXT,
                    confidence REAL,
                    mode TEXT,
                    regime TEXT,              -- OPTION 3: Market regime ('trend', 'range', 'normal')
                    tp_sl_ratio REAL,         -- OPTION 3: Risk-reward ratio (TP/SL)
                    timestamp TEXT NOT NULL
                )
            """)

            # Add missing columns if they don't exist (for existing tables)
            columns_to_add = [
                ("mode", "TEXT"),
                ("regime", "TEXT"),
                ("tp_sl_ratio", "REAL"),
            ]
            for col_name, col_type in columns_to_add:
                try:
                    cursor.execute(f"ALTER TABLE trade_performance ADD COLUMN {col_name} {col_type}")
                    logger.info(f"Added {col_name} column to existing trade_performance table")
                except sqlite3.OperationalError:
                    # Column already exists
                    pass

            # Create index for faster queries
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_performance_timestamp ON trade_performance(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_performance_symbol ON trade_performance(symbol)")

            conn.commit()
            logger.info("Trade performance table initialized successfully")

        except Exception as e:
            logger.exception(f"Failed to initialize trade performance table: {e}")
            return False
        else:
            return True
        finally:
            if conn:
                conn.close()

    @staticmethod
    def backfill_from_paper_trades(db_path: str | None = None) -> dict[str, Any]:
        """Insert trade_performance rows from canonical paper_trades SELLs (idempotent by paper row id)."""
        path = db_path or DATABASE_PATH
        TradePerformanceTracker._ensure_initialized()
        inserted = 0
        skipped = 0
        conn = None
        try:
            conn = sqlite3.connect(path)
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT id, trade_id, symbol, quantity, price, entry_price, pnl, pnl_pct,
                       hold_time_seconds, strategy, strategy_id, confidence, mode, timestamp
                FROM paper_trades
                WHERE UPPER(side) = 'SELL' AND status = 'executed'
                  AND COALESCE(exit_type, '') NOT IN ('ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR')
                ORDER BY id ASC
                """
            )
            rows = cursor.fetchall()
            for row in rows:
                paper_id = int(row[0])
                cursor.execute("SELECT 1 FROM trade_performance WHERE trade_id = ? LIMIT 1", (paper_id,))
                if cursor.fetchone():
                    skipped += 1
                    continue
                pnl = float(row[6] or 0)
                entry_px = float(row[5] or 0)
                exit_px = float(row[4] or 0)
                qty = float(row[3] or 0)
                pnl_pct = float(row[7] or 0)
                if pnl_pct == 0 and entry_px > 0 and qty > 0:
                    pnl_pct = (pnl / (entry_px * qty)) * 100.0
                is_win = 1 if pnl > 0 else (0 if pnl < 0 else None)
                hold = row[8]
                hold_sec = int(hold) if hold is not None and int(hold) > 0 else None
                strat = str(row[10] or row[9] or "day").strip() or "day"
                mode = str(row[12] or "paper").strip() or "paper"
                ts = str(row[13] or datetime.now(timezone.utc).isoformat())
                cursor.execute(
                    """
                    INSERT INTO trade_performance
                    (trade_id, symbol, side, entry_price, exit_price, quantity, pnl, pnl_pct,
                     is_win, hold_time_seconds, strategy, confidence, mode, regime, tp_sl_ratio, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        paper_id,
                        str(row[2] or ""),
                        "sell",
                        entry_px,
                        exit_px,
                        qty,
                        pnl,
                        pnl_pct,
                        is_win,
                        hold_sec,
                        strat,
                        float(row[11]) if row[11] is not None else None,
                        mode,
                        None,
                        None,
                        ts,
                    ),
                )
                inserted += 1
            conn.commit()
        except Exception as e:
            logger.exception("backfill_from_paper_trades failed: %s", e)
            return {"inserted": inserted, "skipped": skipped, "error": str(e)}
        finally:
            if conn:
                conn.close()
        return {"inserted": inserted, "skipped": skipped, "total_sells": len(rows) if "rows" in locals() else 0}


# Convenience functions for easy importing
async def log_trade_performance(*args, **kwargs):
    return await TradePerformanceTracker.log_trade_performance(*args, **kwargs)


def backfill_trade_performance_from_paper_trades(db_path: str | None = None) -> dict[str, Any]:
    return TradePerformanceTracker.backfill_from_paper_trades(db_path)


def get_trade_performance_summary(since_timestamp: str | None = None):
    return TradePerformanceTracker.get_trade_performance_summary(since_timestamp)


def initialize_trade_performance_table():
    return TradePerformanceTracker.initialize_trade_performance_table()
