"""
Canonical packet writer for live positions closed on-exchange outside Mystic protected sell.

Used by periodic_reconcile / vanished-position handling and one-off backfills.
All rows are idempotent and sourced from recovered exchange fill data only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.utils.sqlite_runtime import connect_rw, run_locked_retry

logger = logging.getLogger(__name__)

RECOVERED_SELL_EXIT_TYPE = "EXCHANGE_RECONCILE_CLOSE"
RECOVERED_CLOSE_REASON = "EXCHANGE_RECONCILE_CLOSE"


@dataclass(frozen=True)
class RecoveredCloseFill:
    buy_trade_id: str
    symbol: str  # CCXT form e.g. BTC/USDT
    quantity: float
    entry_price: float
    exit_price: float
    exchange_sell_order_id: str
    closed_at_iso: str
    closed_at_epoch: float
    source: str
    fill_recovered: bool
    realized_profit_usd: float | None
    fee_usd: float | None = None
    entry_time_epoch: float | None = None
    close_ledger_id: int | None = None
    paper_run_id: str | None = None
    sleeve: str = "ACTIVE"
    strategy_id: str = "day"
    confidence: float | None = None
    mode: str = "live"
    venue_trade_ids: str = ""


def recovered_sell_trade_id(symbol: str, exchange_sell_order_id: str) -> str:
    sym = symbol.replace("/", "_")
    return f"mystic_recovered_sell_{sym}_{exchange_sell_order_id}"


def _ledger_tag(detail: str | None, key: str, value: str) -> str:
    base = str(detail or "").strip()
    token = f"{key}={value}"
    if token in base:
        return base
    return f"{base};{token}" if base else token


def _find_existing_sell(conn: sqlite3.Connection, fill: RecoveredCloseFill) -> sqlite3.Row | None:
    sell_tid = recovered_sell_trade_id(fill.symbol, fill.exchange_sell_order_id)
    row = conn.execute(
        "SELECT id, trade_id FROM paper_trades WHERE trade_id = ? AND side = 'SELL' LIMIT 1",
        (sell_tid,),
    ).fetchone()
    if row:
        return row
    return conn.execute(
        """
        SELECT id, trade_id FROM paper_trades
        WHERE side = 'SELL' AND mode = 'live' AND symbol = ?
          AND diagnostics_json LIKE ?
        LIMIT 1
        """,
        (fill.symbol, f'%"exchange_sell_order_id":"{fill.exchange_sell_order_id}"%'),
    ).fetchone()


def _learning_exists(conn: sqlite3.Connection, fill: RecoveredCloseFill) -> bool:
    needle = fill.buy_trade_id
    row = conn.execute(
        f"""
        SELECT id FROM trade_learning_outcomes
        WHERE extra_json LIKE ? AND close_reason IN ('HUMAN_MANUAL_SELL', '{RECOVERED_CLOSE_REASON}')
        LIMIT 1
        """,
        (f"%{needle}%",),
    ).fetchone()
    return row is not None


def _audit_sell_exists(conn: sqlite3.Connection, sell_trade_id: str) -> bool:
    row = conn.execute(
        "SELECT id FROM portfolio_engine_audit WHERE trade_id = ? AND action = 'SELL' LIMIT 1",
        (sell_trade_id,),
    ).fetchone()
    return row is not None


def _performance_exists(conn: sqlite3.Connection, exchange_order_id: str) -> bool:
    try:
        oid = int(exchange_order_id)
    except (TypeError, ValueError):
        return False
    row = conn.execute(
        "SELECT id FROM trade_performance WHERE trade_id = ? AND side = 'sell' LIMIT 1",
        (oid,),
    ).fetchone()
    return row is not None


def _load_buy_context(conn: sqlite3.Connection, buy_trade_id: str) -> dict[str, Any]:
    row = conn.execute(
        """
        SELECT paper_run_id, explainability_json, diagnostics_json, sleeve, strategy_id,
               confidence, timestamp, price, quantity
        FROM paper_trades WHERE trade_id = ? AND side = 'BUY' LIMIT 1
        """,
        (buy_trade_id,),
    ).fetchone()
    if not row:
        return {}
    out: dict[str, Any] = {
        "paper_run_id": row[0],
        "sleeve": row[3] or "ACTIVE",
        "strategy_id": row[4] or "day",
        "confidence": row[5],
        "buy_timestamp": row[6],
    }
    if row[1]:
        with suppress(Exception):
            out["explainability"] = json.loads(row[1])
    if row[2]:
        with suppress(Exception):
            out["buy_diagnostics"] = json.loads(row[2])
    return out


def persist_recovered_close(
    fill: RecoveredCloseFill,
    *,
    db_path: str = DATABASE_PATH,
    write_trade_performance: bool = True,
) -> dict[str, Any]:
    """
    Idempotently write canonical recovered-close rows.
    Returns dict of table -> id created or existing.
    """
    result: dict[str, Any] = {"created": {}, "existing": {}, "errors": [], "economic_sell_written": False}
    sell_trade_id = recovered_sell_trade_id(fill.symbol, fill.exchange_sell_order_id)
    qty = float(fill.quantity)
    entry = float(fill.entry_price)
    exit_px = float(fill.exit_price)
    if qty <= 0 or entry <= 0 or exit_px <= 0:
        result["errors"].append("invalid_qty_or_prices")
        return result
    from backend.services.canonical_failsafe_equity import is_real_binance_order_id
    from backend.services.live_close_integrity import persist_pending_close_event

    venue_trades = str(fill.venue_trade_ids or "").strip()
    oid = str(fill.exchange_sell_order_id or "").strip()
    if not is_real_binance_order_id(oid) or not venue_trades:
        persist_pending_close_event(
            db_path,
            symbol=fill.symbol,
            event_type="EXCHANGE_RECONCILE_AUDIT",
            exit_trigger=RECOVERED_CLOSE_REASON,
            quantity=qty,
            price_snapshot=exit_px,
            exchange_order_id=oid,
            detail="missing_real_venue_sell_identity; no economic SELL written",
        )
        result["existing"]["audit_only_invalid_identity"] = True
        result["errors"].append("missing_real_venue_sell_identity")
        return result

    gross_pnl = (exit_px - entry) * qty
    fee = float(fill.fee_usd or 0.0)
    if fill.realized_profit_usd is not None:
        realized = float(fill.realized_profit_usd)
    else:
        realized = gross_pnl - fee

    def _op() -> None:
        nonlocal sell_trade_id
        with connect_rw(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            buy_ctx = _load_buy_context(conn, fill.buy_trade_id)
            sleeve = fill.sleeve or buy_ctx.get("sleeve") or "ACTIVE"

            existing_sell = _find_existing_sell(conn, fill)
            oid = str(fill.exchange_sell_order_id or "").strip()
            already_venue = None
            if oid:
                already_venue = conn.execute(
                    """
                    SELECT id, trade_id FROM paper_trades
                    WHERE side = 'SELL' AND TRIM(COALESCE(order_id, '')) = ?
                    LIMIT 1
                    """,
                    (oid,),
                ).fetchone()
            from backend.services.live_close_integrity import ECONOMIC_TABLE, ensure_live_close_tables

            ensure_live_close_tables(conn)
            claimed_new = False
            if oid:
                claim_cur = conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {ECONOMIC_TABLE}
                    (exchange_order_id, side, mystic_trade_id, symbol, created_at)
                    VALUES (?, 'SELL', ?, ?, ?)
                    """,
                    (oid, sell_trade_id, fill.symbol, fill.closed_at_iso),
                )
                claimed_new = int(claim_cur.rowcount or 0) == 1
            if already_venue and not existing_sell:
                result["existing"]["paper_trades_sell"] = already_venue[0]
                result["existing"]["venue_order_already_attributed"] = True
                sell_trade_id = str(already_venue[1])
            elif existing_sell:
                result["existing"]["paper_trades_sell"] = existing_sell[0]
                sell_trade_id = str(existing_sell[1])
            elif oid and not claimed_new:
                result["existing"]["economic_close_already_claimed"] = True
            result["existing"]["economic_sell_suppressed"] = True
            result["economic_sell_written"] = False

            cur.execute(
                """
                UPDATE paper_trades SET remaining_position = 0
                WHERE trade_id = ? AND side = 'BUY'
                """,
                (fill.buy_trade_id,),
            )

            if not _audit_sell_exists(conn, sell_trade_id):
                buy_audit = conn.execute(
                    """
                    SELECT post_ledger_json FROM portfolio_engine_audit
                    WHERE trade_id = ? AND action = 'BUY' ORDER BY id DESC LIMIT 1
                    """,
                    (fill.buy_trade_id,),
                ).fetchone()
                pre_ledger = {"cash_balance": 0.0, "positions_value": 0.0, "total_equity": 0.0}
                if buy_audit and buy_audit[0]:
                    with suppress(Exception):
                        pre_ledger = json.loads(buy_audit[0])
                proceeds = qty * exit_px - fee
                post_ledger = {
                    "cash_balance": float(pre_ledger.get("cash_balance", 0.0)) + proceeds,
                    "positions_value": 0.0,
                    "total_equity": float(pre_ledger.get("cash_balance", 0.0)) + proceeds,
                }
                pos_digest = hashlib.md5(b"{}").hexdigest()[:16]
                cur.execute(
                    """
                    INSERT INTO portfolio_engine_audit (
                        ts, action, symbol, qty, price, fees, slippage,
                        decision_id, trade_id, ranked_candidates_json,
                        pre_ledger_json, post_ledger_json,
                        pre_positions_digest, post_positions_digest,
                        invariant_ok, invariant_diff, entry_reason, exit_reason, sleeve
                    ) VALUES (?, 'SELL', ?, ?, ?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, 1, ?, ?, ?, ?)
                    """,
                    (
                        fill.closed_at_iso,
                        fill.symbol,
                        qty,
                        exit_px,
                        fee,
                        0.0,
                        None,
                        sell_trade_id,
                        json.dumps(pre_ledger),
                        json.dumps(post_ledger),
                        pos_digest,
                        pos_digest,
                        realized,
                        f"{RECOVERED_CLOSE_REASON};source={fill.source};buy_trade_id={fill.buy_trade_id}",
                        RECOVERED_CLOSE_REASON,
                        sleeve,
                    ),
                )
                result["created"]["portfolio_engine_audit_sell"] = int(cur.lastrowid)
            else:
                row = conn.execute(
                    "SELECT id FROM portfolio_engine_audit WHERE trade_id = ? AND action = 'SELL' LIMIT 1",
                    (sell_trade_id,),
                ).fetchone()
                if row:
                    result["existing"]["portfolio_engine_audit_sell"] = row[0]

            if fill.close_ledger_id:
                row = conn.execute(
                    "SELECT detail FROM position_close_ledger WHERE id = ?",
                    (fill.close_ledger_id,),
                ).fetchone()
                detail = _ledger_tag(row[0] if row else None, "canonical_sell_trade_id", sell_trade_id)
                detail = _ledger_tag(detail, "buy_trade_id", fill.buy_trade_id)
                cur.execute(
                    "UPDATE position_close_ledger SET detail = ? WHERE id = ?",
                    (detail, fill.close_ledger_id),
                )

            conn.commit()

    run_locked_retry(_op)

    conn_check = sqlite3.connect(db_path)
    try:
        learning_exists = _learning_exists(conn_check, fill)
        perf_exists = _performance_exists(conn_check, fill.exchange_sell_order_id)
    finally:
        conn_check.close()

    persist_pending_close_event(
        db_path,
        symbol=fill.symbol,
        event_type="EXCHANGE_RECONCILE_AUDIT",
        exit_trigger=RECOVERED_CLOSE_REASON,
        quantity=qty,
        price_snapshot=exit_px,
        exchange_order_id=oid,
        detail=f"audit_only source={fill.source} venue_trades={venue_trades}",
    )
    result["created"]["reconcile_audit"] = True
    result["existing"]["trade_learning_outcomes"] = bool(learning_exists)
    result["existing"]["trade_performance"] = fill.exchange_sell_order_id if perf_exists else None
    _ = write_trade_performance
    result["sell_trade_id"] = sell_trade_id
    logger.info(
        "RECOVERED_CLOSE_CANONICAL_OK buy=%s sell=%s exchange_order=%s created=%s existing=%s",
        fill.buy_trade_id,
        sell_trade_id,
        fill.exchange_sell_order_id,
        result.get("created"),
        result.get("existing"),
    )
    return result
