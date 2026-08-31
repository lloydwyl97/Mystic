#!/usr/bin/env python3
"""
Purge pre-day-trade-regime closed trades and poisoned learning tables.

Keeps:
  - portfolio_engine_ledger (cash / equity)
  - open portfolio_engine_positions
  - paper_trades BUY rows whose trade_id is still an open position

Deletes closed DAY/SCALP trade history and derived learning artifacts so AI
learns only from the corrected day-trade loop going forward.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[1] / "mystic_trading.db"

LEARNING_TABLES = (
    "trade_learning_outcomes",
    "day_outcome_attribution",
    "ai_outcome_training_rows",
    "ai_signal_outcome_rollups",
    "ai_trade_memory_scores",
    "ai_post_trade_feature_reviews",
    "market_role_trade_outcomes",
    "market_role_outcome_stats",
    "trade_performance",
    "portfolio_engine_scoreboard_daily",
    "coin_performance",
    "strategy_performance",
    "portfolio_engine_rejects",
    "portfolio_engine_audit",
    "portfolio_engine_orders",
    "scalp_learning_outcomes",
    "scalp_outcome_attribution",
    "scalp_strategy_score_weights",
    "scalp_scoreboard_daily",
    "scalp_paper_trades",
    "ai_peer_shadow_outcomes",
)


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (name,),
    ).fetchone()
    return bool(row)


def purge(db_path: str, *, dry_run: bool = False) -> dict[str, int]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    stats: dict[str, int] = {}
    try:
        keep_ids = []
        if _table_exists(conn, "portfolio_engine_positions"):
            keep_ids = [
                str(r[0])
                for r in conn.execute(
                    """
                    SELECT trade_id FROM portfolio_engine_positions
                    WHERE trade_id IS NOT NULL AND TRIM(trade_id) != ''
                    """
                )
            ]
        stats["keep_open_trade_ids"] = len(keep_ids)

        if _table_exists(conn, "paper_trades"):
            if keep_ids:
                placeholders = ",".join("?" * len(keep_ids))
                to_delete = conn.execute(
                    f"SELECT COUNT(*) FROM paper_trades WHERE trade_id NOT IN ({placeholders})",
                    keep_ids,
                ).fetchone()[0]
                stats["paper_trades_delete"] = int(to_delete)
                if not dry_run:
                    conn.execute(
                        f"DELETE FROM paper_trades WHERE trade_id NOT IN ({placeholders})",
                        keep_ids,
                    )
            else:
                n = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
                stats["paper_trades_delete"] = int(n)
                if not dry_run:
                    conn.execute("DELETE FROM paper_trades")

        for table in LEARNING_TABLES:
            if not _table_exists(conn, table):
                continue
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            stats[f"{table}_delete"] = int(n)
            if not dry_run and n:
                conn.execute(f"DELETE FROM {table}")

        if _table_exists(conn, "scalp_paper_positions") and not dry_run:
            n = conn.execute("SELECT COUNT(*) FROM scalp_paper_positions").fetchone()[0]
            stats["scalp_paper_positions_delete"] = int(n)
            if n:
                conn.execute("DELETE FROM scalp_paper_positions")

        if not dry_run:
            conn.commit()
            conn.execute("VACUUM")
            conn.commit()
        return stats
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    stats = purge(args.db, dry_run=args.dry_run)
    mode = "DRY_RUN" if args.dry_run else "APPLIED"
    print(f"PURGE_INVALID_TRADE_LEARNING {mode} db={args.db}")
    for k in sorted(stats):
        print(f"  {k}={stats[k]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
