import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Import from single source of truth
try:
    from backend.config.trading_universe import EXCHANGE_ID
except (ImportError, ModuleNotFoundError, AttributeError, ValueError, TypeError, RuntimeError) as e:
    msg = f"Failed to import EXCHANGE_ID from trading_universe: {e}"
    raise RuntimeError(msg) from e

DB_FILE = os.getenv("TRADE_LOG_DB", "trades.db")

logger = logging.getLogger(__name__)


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    # Ensure directory for DB file exists when using a file path (not :memory:)
    if DB_FILE and DB_FILE != ":memory:":
        db_dir = str(Path(DB_FILE).parent)
        if db_dir:
            try:
                Path(db_dir).mkdir(parents=True, exist_ok=True)
            except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
                logger.warning(f"Could not create DB directory {db_dir}: {e}")
    conn = sqlite3.connect(DB_FILE, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Failed to set PRAGMAs: {e}")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    # Quote the table name safely for PRAGMA table_info
    safe_table = table.replace('"', '""')
    cur = conn.execute(f'PRAGMA table_info("{safe_table}");')
    cols = [row[1] for row in cur.fetchall()]
    return column in cols


def _ensure_trades_table(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                profit_usd REAL,
                timestamp TEXT
            )
            """
        )
        if not _column_exists(conn, "trades", "strategy"):
            conn.execute("ALTER TABLE trades ADD COLUMN strategy TEXT;")
        if not _column_exists(conn, "trades", "profit_usd"):
            conn.execute("ALTER TABLE trades ADD COLUMN profit_usd REAL DEFAULT 0.0;")
        if not _column_exists(conn, "trades", "timestamp"):
            conn.execute("ALTER TABLE trades ADD COLUMN timestamp TEXT DEFAULT '1970-01-01T00:00:00+00:00';")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);")


def _ensure_mutation_table(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS strategy_mutations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_strategy TEXT NOT NULL,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                notes TEXT
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_mutations_created_at ON strategy_mutations(created_at);")


def _ensure_schema() -> None:
    try:
        with _connect() as conn:
            _ensure_trades_table(conn)
            _ensure_mutation_table(conn)
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"Schema ensure failed: {e}")


def fetch_recent_strategy_stats(hours_back: int = 24) -> list:
    try:
        _ensure_schema()
        since_time = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        with _connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT strategy,
                       COUNT(*),
                       AVG(COALESCE(profit_usd, 0.0)),
                       SUM(COALESCE(profit_usd, 0.0)),
                       SUM(CASE WHEN COALESCE(profit_usd, 0.0) > 0 THEN 1 ELSE 0 END)
                FROM trades
                WHERE timestamp > ?
                GROUP BY strategy
                """,
                (since_time,),
            )
            rows = cur.fetchall()

        stats = []
        for row in rows:
            trade_count = int(row[1] or 0)
            win_count = int(row[4] or 0)
            win_rate = (win_count / trade_count) if trade_count else 0.0
            stats.append(
                {
                    "strategy": "" if row[0] is None else str(row[0]),
                    "trade_count": trade_count,
                    "avg_profit": float(row[2] or 0.0),
                    "total_profit": float(row[3] or 0.0),
                    "win_count": win_count,
                    "win_rate": float(win_rate),
                }
            )
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"fetch_recent_strategy_stats failed: {e}")
        return []
    else:
        return stats


def promote_strategies() -> list:
    stats = fetch_recent_strategy_stats(hours_back=24)
    promoted = [s for s in stats if s["win_rate"] > 0.6 and s["total_profit"] > 0.0]
    for s in promoted:
        logger.info(f"[PROMOTE] {s['strategy']} | P: ${s['total_profit']:.2f} | WR: {s['win_rate']:.2%}")
    return promoted


def mutate_from_template(strategy_name: str) -> None:
    try:
        _ensure_schema()
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO strategy_mutations (source_strategy, created_at, status, notes)
                VALUES (?, ?, ?, ?)
                """,
                (
                    strategy_name,
                    _now_utc_iso(),
                    "pending",
                    "auto-generated from promotion",
                ),
            )
        logger.info(f"[MUTATE] Cloning {strategy_name} -> new version queued")
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, RuntimeError) as e:
        logger.exception(f"mutate_from_template failed for {strategy_name}: {e}")


def evolve_strategies() -> list:
    winners = promote_strategies()
    for strat in winners:
        mutate_from_template(strat["strategy"])
    return winners


def get_mutation_candidates() -> list:
    stats = fetch_recent_strategy_stats(hours_back=24)
    candidates = []
    for s in stats:
        if s["trade_count"] >= 5:
            if s["win_rate"] > 0.5 and s["total_profit"] > 0.0:
                candidates.append(
                    {
                        "strategy": s["strategy"],
                        "score": float(s["win_rate"] * s["total_profit"]),
                        "type": "promote",
                    }
                )
            elif s["win_rate"] < 0.3 and s["total_profit"] < 0.0:
                candidates.append(
                    {
                        "strategy": s["strategy"],
                        "score": float(abs(s["total_profit"])),
                        "type": "retire",
                    }
                )
    return candidates


def run_mutation_cycle() -> None:
    logger.info("[MUTATION] Starting mutation cycle")
    candidates = get_mutation_candidates()
    for candidate in candidates:
        if candidate["type"] == "promote":
            logger.info(f"[MUTATION] Promoting {candidate['strategy']} (score: {candidate['score']:.2f})")
            mutate_from_template(candidate["strategy"])
        elif candidate["type"] == "retire":
            logger.info(f"[MUTATION] Retiring {candidate['strategy']} (loss: {candidate['score']:.2f})")
    logger.info(f"[MUTATION] Cycle complete. Processed {len(candidates)} candidates.")


if __name__ == "__main__":
    _ensure_schema()
    run_mutation_cycle()
