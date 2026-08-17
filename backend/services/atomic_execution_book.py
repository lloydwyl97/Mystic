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


def _norm_sym(symbol: str) -> str:
    return str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()


def _buy_closed_by_trade_id(conn: sqlite3.Connection, trade_id: str, symbol: str, buy_id: int) -> bool:
    """True only when THIS buy lot was closed — not when a later same-symbol sell exists."""
    tid = str(trade_id or "").strip()
    if not tid:
        return False
    like = f"%{tid}%"
    try:
        row = conn.execute(
            """
            SELECT 1 FROM paper_trades
            WHERE upper(side) = 'SELL'
              AND id > ?
              AND (
                    IFNULL(explainability_json, '') LIKE ?
                 OR IFNULL(diagnostics_json, '') LIKE ?
                 OR IFNULL(context_snapshot_json, '') LIKE ?
              )
            LIMIT 1
            """,
            (int(buy_id), like, like, like),
        ).fetchone()
        if row:
            return True
    except sqlite3.OperationalError:
        pass
    try:
        row = conn.execute(
            """
            SELECT 1 FROM position_close_ledger
            WHERE IFNULL(detail, '') LIKE ?
            LIMIT 1
            """,
            (f"%buy_trade_id={tid}%",),
        ).fetchone()
        if row:
            return True
    except sqlite3.OperationalError:
        pass
    try:
        row = conn.execute(
            """
            SELECT 1 FROM operational_state
            WHERE key = 'ledger_orphan_buy_cash_restore'
              AND IFNULL(value_json, '') LIKE ?
            LIMIT 1
            """,
            (f"%\"trade_id\": \"{tid}\"%",),
        ).fetchone()
        if row:
            return True
    except sqlite3.OperationalError:
        pass
    _ = symbol
    return False


def find_orphaned_day_buys(db_path: str | Path) -> list[dict[str, Any]]:
    """BUY rows whose cash was spent and inventory is gone, with no matching close.

    A later SELL of the same symbol does not close this lot unless it references
    this buy's trade_id. ``created_at`` is an alias of ``timestamp``.
    """
    path = str(db_path)
    conn = _connect(path, readonly=True, timeout=2.0)
    try:
        try:
            buys = conn.execute(
                """
                SELECT t.id, t.trade_id, t.symbol, t.quantity,
                       IFNULL(t.remaining_position, 0) AS remaining_raw,
                       CASE WHEN IFNULL(t.remaining_position, 0) > 1e-12
                            THEN t.remaining_position ELSE t.quantity END AS remaining_position,
                       t.price, t.timestamp AS created_at,
                       IFNULL(t.diagnostics_json, '') AS diagnostics_json
                FROM paper_trades t
                WHERE upper(t.side) = 'BUY'
                  AND IFNULL(t.quantity, 0) > 1e-12
                ORDER BY t.id
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        open_tids: set[str] = set()
        open_syms: set[str] = set()
        try:
            for r in conn.execute(
                """
                SELECT trade_id, symbol FROM portfolio_engine_positions
                WHERE IFNULL(quantity, 0) > 1e-12
                """
            ):
                if r["trade_id"]:
                    open_tids.add(str(r["trade_id"]))
                open_syms.add(_norm_sym(r["symbol"]))
        except sqlite3.OperationalError:
            pass
        later_sell_syms: set[str] = set()
        try:
            for r in conn.execute(
                """
                SELECT b.id AS buy_id, b.symbol
                FROM paper_trades b
                WHERE upper(b.side) = 'BUY'
                  AND EXISTS (
                      SELECT 1 FROM paper_trades s
                      WHERE upper(s.side) = 'SELL'
                        AND replace(replace(upper(s.symbol), '/', ''), '-', '')
                            = replace(replace(upper(b.symbol), '/', ''), '-', '')
                        AND s.id > b.id
                  )
                """
            ):
                later_sell_syms.add(f"{int(r['buy_id'])}")
        except sqlite3.OperationalError:
            later_sell_syms = set()
        out: list[dict[str, Any]] = []
        for r in buys:
            d = dict(r)
            if "ORPHAN_CASH_RESTORED" in str(d.get("diagnostics_json") or ""):
                continue
            tid = str(d.get("trade_id") or "")
            if tid and tid in open_tids:
                continue
            if not tid and _norm_sym(d.get("symbol") or "") in open_syms:
                continue
            if _buy_closed_by_trade_id(conn, tid, str(d.get("symbol") or ""), int(d.get("id") or 0)):
                continue
            remaining_raw = float(d.get("remaining_raw") or 0.0)
            has_later_same_symbol_sell = str(int(d.get("id") or 0)) in later_sell_syms
            # remaining=0 + a later same-symbol SELL is normal FIFO history, not an orphan.
            # remaining=0 with no later same-symbol SELL is a vanished lot (cash spent, inventory gone).
            if remaining_raw <= 1e-12 and has_later_same_symbol_sell:
                continue
            d.pop("diagnostics_json", None)
            d.pop("remaining_raw", None)
            out.append(d)
        return out
    except sqlite3.DatabaseError:
        return []
    finally:
        conn.close()


def find_unclosed_buy_cash_debits(db_path: str | Path) -> list[dict[str, Any]]:
    """BUY lots that spent cash and were never closed by trade_id.

    Unlike ``find_orphaned_day_buys``, a later same-symbol SELL of a *different*
    lot does not hide this debit. Used for ledger identity repair.
    """
    path = str(db_path)
    conn = _connect(path, readonly=True, timeout=2.0)
    try:
        try:
            buys = conn.execute(
                """
                SELECT t.id, t.trade_id, t.symbol, t.quantity,
                       CASE WHEN IFNULL(t.remaining_position, 0) > 1e-12
                            THEN t.remaining_position ELSE t.quantity END AS remaining_position,
                       t.price, t.timestamp AS created_at,
                       IFNULL(t.diagnostics_json, '') AS diagnostics_json
                FROM paper_trades t
                WHERE upper(t.side) = 'BUY'
                  AND IFNULL(t.quantity, 0) > 1e-12
                ORDER BY t.id
                """
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        open_tids: set[str] = set()
        try:
            for r in conn.execute(
                "SELECT trade_id FROM portfolio_engine_positions WHERE IFNULL(quantity, 0) > 1e-12"
            ):
                if r["trade_id"]:
                    open_tids.add(str(r["trade_id"]))
        except sqlite3.OperationalError:
            pass
        out: list[dict[str, Any]] = []
        for r in buys:
            d = dict(r)
            if "ORPHAN_CASH_RESTORED" in str(d.get("diagnostics_json") or ""):
                continue
            tid = str(d.get("trade_id") or "")
            if tid and tid in open_tids:
                continue
            if _buy_closed_by_trade_id(conn, tid, str(d.get("symbol") or ""), int(d.get("id") or 0)):
                continue
            d.pop("diagnostics_json", None)
            out.append(d)
        return out
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
                "UPDATE paper_trades SET remaining_position = ? WHERE id = ?",
                (qty, orphan.get("id")),
            )
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
    """Ensure mystic_scalp.db exists. Never import leftover DAY-file scalp_* history.

    Isolation is complete. Historical scalp rows on mystic_trading.db are
    analysis-only. Copying them into the money DB contaminates clean
    acceptance and trips the consecutive-loss breaker on pre-cutoff losses.
    Learning tables stay on the DAY database.
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
    try:
        try:
            existing = int(dst.execute("SELECT COUNT(*) FROM scalp_paper_trades").fetchone()[0] or 0)
        except sqlite3.OperationalError:
            existing = 0
        if existing > 0:
            result["reason"] = "already_populated"
            result["existing_trades"] = existing
            return result
        result["reason"] = "isolation_complete_no_import"
        result["existing_trades"] = existing
        return result
    finally:
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
