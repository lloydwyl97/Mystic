import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_FILE = os.getenv("TRADE_LOG_DB", "trades.db")
LOCK_FILE = "strategy_locks.txt"


def get_strategy_leaderboard(hours_back=24):
    """Get strategy leaderboard sorted by total profit"""
    conn = None
    try:
        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()

        since = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        c.execute(
            """
            SELECT strategy, COUNT(*), AVG(profit_usd), SUM(profit_usd),
                SUM(CASE WHEN profit_usd > 0 THEN 1 ELSE 0 END)
            FROM trades
            WHERE timestamp > ?
            GROUP BY strategy
            ORDER BY SUM(profit_usd) DESC
        """,
            (since,),
        )
        rows = c.fetchall()
    except sqlite3.Error:
        logger.exception("Failed to query strategy leaderboard from DB")
        return []
    finally:
        if conn:
            conn.close()

    leaderboard = []
    for row in rows:
        # Normalize possible NULLs from SQL to sensible Python defaults
        strategy = row[0]
        trades = int(row[1]) if row[1] is not None else 0
        avg_profit = float(row[2]) if row[2] is not None else 0.0
        total_profit = float(row[3]) if row[3] is not None else 0.0
        wins = int(row[4]) if row[4] is not None else 0
        win_rate = wins / trades if trades else 0.0
        leaderboard.append(
            {
                "strategy": strategy,
                "trades": trades,
                "avg_profit": avg_profit,
                "total_profit": total_profit,
                "win_rate": win_rate,
            }
        )
    return leaderboard


def load_locked_strategies():
    """Load list of locked strategies"""
    lock_file_path = Path(LOCK_FILE)
    if not lock_file_path.exists():
        return set()
    with lock_file_path.open(encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def lock_strategy(name):
    """Lock a strategy to prevent further mutations"""
    # Append the strategy name to the lock file (allow duplicates)
    lock_file_path = Path(LOCK_FILE)
    lock_file_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_file_path.open("a", encoding="utf-8") as f:
        f.write(f"{name}\n")
    logger.info(f"[LEADERBOARD] Locked strategy: {name}")


def auto_evolve():
    """Automatically evolve strategies based on performance"""
    leaderboard = get_strategy_leaderboard(hours_back=24)
    locked = load_locked_strategies()

    for strat in leaderboard:
        name = strat["strategy"]
        if name in locked:
            continue

        if strat["win_rate"] > 0.6 and strat["total_profit"] > 0:
            logger.info(f"[PROMOTE] {name} | ${strat['total_profit']:.2f} | WR: {strat['win_rate']:.2%}")
            lock_strategy(name)
            clone_and_mutate(name)

        if strat["win_rate"] < 0.3 and strat["total_profit"] < 0:
            logger.info(f"[RETIRE] {name} | Losses: ${strat['total_profit']:.2f}")
            retire_strategy(name)


def clone_and_mutate(base_strategy):
    """Clone and mutate a winning strategy"""
    new_name = base_strategy + "_mutant_" + datetime.now(timezone.utc).strftime("%H%M%S")
    logger.info(f"[MUTATE] {base_strategy} → {new_name}")


def retire_strategy(strategy_name):
    """Retire a losing strategy"""
    logger.info(f"[ARCHIVE] Disabling {strategy_name}")


def get_top_strategies(limit=5):
    """Get top performing strategies"""
    leaderboard = get_strategy_leaderboard()
    return leaderboard[:limit]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("=== STRATEGY LEADERBOARD ===")
    top = get_top_strategies()
    for i, strat in enumerate(top, 1):
        logger.info(f"{i}. {strat['strategy']}: ${strat['total_profit']:.2f} | WR: {strat['win_rate']:.2%}")
