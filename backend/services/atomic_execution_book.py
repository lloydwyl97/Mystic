"""Atomic OPEN/CLOSE contract, orphan detection, and SCALP money-DB isolation.

Authoritative money state is SQLite. A BUY or SELL either commits
trade + position + ledger + audit together, or none of them commit.

Redis and in-memory objects are projections of that commit, never a
competing ledger. Disagreement is logged CRITICAL and restored only from
an already-committed trade row — cash is never invented.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCALP_MONEY_TABLES = (
    "scalp_meta",
    "scalp_paper_ledger",
    "scalp_paper_positions",
    "scalp_paper_trades",
    "scalp_rejects",
    "scalp_scoreboard_daily",
    "scalp_trade_audit",
    "scalp_position_reviews",
    "scalp_outcome_attribution",
    "scalp_post_trade_feature_reviews",
    "scalp_strategy_score_weights",
    "scalp_gate_counters",
    "scalp_shadow_rejects",
)


def _connect(path: str | Path, *, readonly: bool = False, timeout: float = 2.0) -> sqlite3.Connection:
    target = str(path)
    if readonly:
        conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True, timeout=timeout)
    else:
        conn = sqlite3.connect(target, timeout=timeout)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(timeout * 1000)}")
    return conn


def find_orphaned_day_buys(db_path: str | Path) -> list[dict[str, Any]]:
    """BUY rows with remaining qty and no matching portfolio_engine_positions row."""
    path = str(db_path)
    conn = _connect(path, readonly=True, timeout=2.0)
    try:
        rows = conn.execute(
            """
            SELECT t.id, t.trade_id, t.symbol, t.quantity, t.remaining_position, t.price, t.created_at
            FROM paper_trades t
            WHERE upper(t.side) = 'BUY'
              AND IFNULL(t.remaining_position, 0) > 1e-12
              AND NOT EXISTS (
                  SELECT 1 FROM portfolio_engine_positions p
                  WHERE replace(replace(upper(p.symbol), '/', ''), '-', '')
                      = replace(replace(upper(t.symbol), '/', ''), '-', '')
                    AND IFNULL(p.quantity, 0) > 1e-12
              )
            ORDER BY t.id
            """
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()


def find_cash_position_disagreement(db_path: str | Path) -> dict[str, Any]:
    """Compare ledger cash+positions_value to equity and leftover BUY remainings."""
    path = str(db_path)
    conn = _connect(path, readonly=True, timeout=2.0)
    out: dict[str, Any] = {
        "ok": True,
        "orphans": [],
        "ledger": None,
        "implied_orphan_notional": 0.0,
        "identity_diff": 0.0,
    }
    try:
        led = conn.execute(
            """
            SELECT cash_balance, positions_value, realized_pnl, unrealized_pnl, total_equity, principal
            FROM portfolio_engine_ledger WHERE id=1
            """
        ).fetchone()
        if led is None:
            out["ok"] = False
            return out
        out["ledger"] = dict(led)
        identity = float(led["cash_balance"] or 0) + float(led["positions_value"] or 0)
        out["identity_diff"] = abs(identity - float(led["total_equity"] or 0))
        orphans = find_orphaned_day_buys(path)
        out["orphans"] = orphans
        out["implied_orphan_notional"] = sum(
            float(o.get("remaining_position") or 0) * float(o.get("price") or 0) for o in orphans
        )
        if orphans or out["identity_diff"] > 0.05:
            out["ok"] = False
        return out
    except sqlite3.OperationalError as exc:
        out["ok"] = False
        out["error"] = str(exc)
        return out
    finally:
        conn.close()


def restore_orphaned_day_buys(db_path: str | Path) -> list[dict[str, Any]]:
    """Restore missing position rows from already-committed BUY trades.

    Cash is not invented. The BUY already debited cash; this writes the
    missing inventory so cash + marked positions can equal equity again.
    Logs CRITICAL for every restore. Returns restored rows.
    """
    path = str(db_path)
    orphans = find_orphaned_day_buys(path)
    if not orphans:
        return []

    restored: list[dict[str, Any]] = []
    conn = _connect(path, timeout=15.0)
    try:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.cursor()
        for orphan in orphans:
            symbol = str(orphan.get("symbol") or "")
            qty = float(orphan.get("remaining_position") or orphan.get("quantity") or 0)
            price = float(orphan.get("price") or 0)
            trade_id = str(orphan.get("trade_id") or "")
            if qty <= 0 or price <= 0 or not symbol or not trade_id:
                logger.critical(
                    "ACCOUNTING_ORPHAN_SKIPPED id=%s symbol=%s qty=%s price=%s trade_id=%s",
                    orphan.get("id"),
                    symbol,
                    qty,
                    price,
                    trade_id,
                )
                continue
            existing = cur.execute(
                """
                SELECT quantity FROM portfolio_engine_positions
                WHERE replace(replace(upper(symbol), '/', ''), '-', '')
                    = replace(replace(upper(?), '/', ''), '-', '')
                """,
                (symbol,),
            ).fetchone()
            if existing and float(existing[0] or 0) > 1e-12:
                continue
            cur.execute(
                """
                INSERT OR REPLACE INTO portfolio_engine_positions (
                    symbol, quantity, entry_price, entry_time, trade_id,
                    stop_price, take_profit_1_price, take_profit_2_price,
                    trailing_stop_price, tp1_hit, highest_price, lowest_price,
                    atr_at_entry, entry_bar_timestamp, confidence_at_entry,
                    entry_fee, last_updated
                ) VALUES (?, ?, ?, strftime('%s','now'), ?, ?, ?, ?, 0, 0, ?, ?, 0, 0, 0.5, 0, datetime('now'))
                """,
                (
                    symbol,
                    qty,
                    price,
                    trade_id,
                    price * 0.97,
                    price * 1.02,
                    price * 1.05,
                    price,
                    price,
                ),
            )
            notional = qty * price
            logger.critical(
                "ACCOUNTING_ORPHAN_RESTORED id=%s trade_id=%s symbol=%s qty=%.8f price=%.8f notional=%.4f "
                "(cash unchanged; inventory restored from committed BUY row)",
                orphan.get("id"),
                trade_id,
                symbol,
                qty,
                price,
                notional,
            )
            restored.append({**orphan, "restored_notional": notional})

        if restored:
            pos_val = float(
                cur.execute(
                    "SELECT COALESCE(SUM(quantity * entry_price), 0) FROM portfolio_engine_positions WHERE IFNULL(quantity,0) > 0"
                ).fetchone()[0]
                or 0
            )
            led = cur.execute(
                "SELECT cash_balance, realized_pnl, principal FROM portfolio_engine_ledger WHERE id=1"
            ).fetchone()
            if led is not None:
                cash = float(led[0] or 0)
                realized = float(led[1] or 0)
                equity = cash + pos_val
                cur.execute(
                    """
                    UPDATE portfolio_engine_ledger
                    SET positions_value = ?, unrealized_pnl = 0, total_equity = ?, last_updated = datetime('now')
                    WHERE id = 1
                    """,
                    (pos_val, equity),
                )
                logger.critical(
                    "ACCOUNTING_ORPHAN_LEDGER_REMARK cash=%.4f positions_value=%.4f equity=%.4f realized=%.4f restored_n=%d",
                    cash,
                    pos_val,
                    equity,
                    realized,
                    len(restored),
                )
        conn.commit()
        return restored
    except Exception:
        conn.rollback()
        logger.exception("ACCOUNTING_ORPHAN_RESTORE_FAILED")
        raise
    finally:
        conn.close()


def resolve_scalp_database_path(repo_root: str, env_get) -> str:
    """SCALP money tables live in their own file so they cannot lock DAY writers."""
    explicit = (env_get("SCALP_DATABASE_PATH") or "").strip()
    if explicit:
        return explicit
    return str(Path(repo_root) / "mystic_scalp.db")


def migrate_scalp_money_database(day_db: str, scalp_db: str) -> dict[str, Any]:
    """Copy scalp_* money tables from the shared DAY file into mystic_scalp.db.

    Idempotent: if the destination already has scalp_paper_trades rows, skip.
    Learning tables (trade_learning_outcomes) stay on the DAY database.
    """
    from backend.services.binance_scalp.schema import init_scalp_schema

    src_path = str(day_db)
    dst_path = str(scalp_db)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    init_scalp_schema(dst_path)

    result: dict[str, Any] = {
        "src": src_path,
        "dst": dst_path,
        "migrated": False,
        "copied_tables": {},
        "reason": "",
    }
    if not Path(src_path).exists():
        result["reason"] = "day_db_missing"
        return result

    dst = _connect(dst_path, timeout=15.0)
    src = _connect(src_path, readonly=True, timeout=15.0)
    try:
        try:
            existing = int(dst.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0] or 0)
        except sqlite3.OperationalError:
            existing = 0
        if existing > 0:
            result["reason"] = "already_populated"
            result["existing_trades"] = existing
            return result

        src_tables = {str(r[0]) for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        dst.execute("BEGIN IMMEDIATE")
        copied = 0
        for table in SCALP_MONEY_TABLES:
            if table not in src_tables:
                continue
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
            if not rows:
                result["copied_tables"][table] = 0
                continue
            cols = [d[0] for d in src.execute(f"SELECT * FROM {table} LIMIT 0").description]
            placeholders = ",".join("?" * len(cols))
            col_sql = ",".join(cols)
            dst.execute(f"DELETE FROM {table}")
            dst.executemany(
                f"INSERT INTO {table} ({col_sql}) VALUES ({placeholders})",
                [tuple(row[c] for c in cols) for row in rows],
            )
            result["copied_tables"][table] = len(rows)
            copied += len(rows)
        dst.commit()
        result["migrated"] = copied > 0
        result["reason"] = "copied" if copied else "source_empty"
        logger.critical(
            "SCALP_MONEY_DB_MIGRATED src=%s dst=%s copied_rows=%d tables=%s",
            src_path,
            dst_path,
            copied,
            result["copied_tables"],
        )
        return result
    except Exception:
        dst.rollback()
        logger.exception("SCALP_MONEY_DB_MIGRATE_FAILED")
        raise
    finally:
        src.close()
        dst.close()


def assert_cash_plus_marks_equals_equity(db_path: str | Path, *, tolerance: float = 0.05) -> dict[str, Any]:
    """Prove cash + marked positions = equity. Used by crash-injection tests."""
    path = str(db_path)
    conn = _connect(path, readonly=True, timeout=2.0)
    try:
        led = conn.execute(
            "SELECT cash_balance, positions_value, total_equity FROM portfolio_engine_ledger WHERE id=1"
        ).fetchone()
        if led is None:
            return {"ok": False, "error": "no_ledger"}
        cash = float(led[0] or 0)
        pos = float(led[1] or 0)
        equity = float(led[2] or 0)
        identity = cash + pos
        orphans = find_orphaned_day_buys(path)
        ok = abs(identity - equity) <= tolerance and not orphans
        return {
            "ok": ok,
            "cash": cash,
            "positions_value": pos,
            "equity": equity,
            "identity": identity,
            "diff": abs(identity - equity),
            "orphans": orphans,
        }
    finally:
        conn.close()
