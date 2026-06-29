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
from datetime import datetime
from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST
from backend.config.trading_mode import TradingMode
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
    result: dict[str, Any] = {"created": {}, "existing": {}, "errors": []}
    sell_trade_id = recovered_sell_trade_id(fill.symbol, fill.exchange_sell_order_id)
    qty = float(fill.quantity)
    entry = float(fill.entry_price)
    exit_px = float(fill.exit_price)
    if qty <= 0 or entry <= 0 or exit_px <= 0:
        result["errors"].append("invalid_qty_or_prices")
        return result

    gross_pnl = (exit_px - entry) * qty
    fee = float(fill.fee_usd or 0.0)
    if fill.realized_profit_usd is not None:
        realized = float(fill.realized_profit_usd)
    else:
        realized = gross_pnl - fee
    pnl_pct = (exit_px - entry) / entry if entry > 0 else 0.0
    net_pct = pnl_pct - ESTIMATED_ROUNDTRIP_COST

    entry_epoch = float(fill.entry_time_epoch or 0.0)
    if entry_epoch <= 0:
        try:
            entry_epoch = datetime.fromisoformat(str(fill.closed_at_iso).replace("Z", "+00:00")).timestamp() - 1161.0
        except Exception:
            entry_epoch = float(fill.closed_at_epoch) - 1161.0
    hold_seconds = max(0, int(float(fill.closed_at_epoch) - entry_epoch))
    buy_ctx_pre = _load_buy_context(sqlite3.connect(db_path), fill.buy_trade_id)
    explain_for_review = dict(buy_ctx_pre.get("explainability") or {})
    strategy_id = fill.strategy_id or buy_ctx_pre.get("strategy_id") or "day"
    confidence = fill.confidence if fill.confidence is not None else buy_ctx_pre.get("confidence")

    def _op() -> None:
        nonlocal sell_trade_id
        with connect_rw(db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            cur = conn.cursor()
            buy_ctx = _load_buy_context(conn, fill.buy_trade_id)
            paper_run_id = fill.paper_run_id or buy_ctx.get("paper_run_id") or "recovered-close"
            sleeve = fill.sleeve or buy_ctx.get("sleeve") or "ACTIVE"
            strategy_id = fill.strategy_id or buy_ctx.get("strategy_id") or "day"
            confidence = fill.confidence if fill.confidence is not None else buy_ctx.get("confidence")
            explain = dict(buy_ctx.get("explainability") or {})

            diagnostics = {
                "exchange_sell_order_id": str(fill.exchange_sell_order_id),
                "source": str(fill.source),
                "fill_recovered": bool(fill.fill_recovered),
                "not_ai_protected_sell": True,
                "not_duplicate": True,
                "live_mode": True,
                "buy_trade_id": fill.buy_trade_id,
                "close_ledger_id": fill.close_ledger_id,
                "recovered_close": True,
            }

            explain.update(
                {
                    "trade_id": sell_trade_id,
                    "symbol": fill.symbol,
                    "side": "SELL",
                    "buy_trade_id": fill.buy_trade_id,
                    "exchange_sell_order_id": str(fill.exchange_sell_order_id),
                    "close_reason": RECOVERED_CLOSE_REASON,
                    "source": fill.source,
                    "fill_recovered": fill.fill_recovered,
                    "not_ai_protected_sell": True,
                    "live_ai_strategy": strategy_id,
                }
            )

            existing_sell = _find_existing_sell(conn, fill)
            if existing_sell:
                result["existing"]["paper_trades_sell"] = existing_sell[0]
                sell_trade_id = str(existing_sell[1])
            else:
                cur.execute(
                    """
                    INSERT INTO paper_trades (
                        trade_id, paper_run_id, mode, symbol, side, quantity, price,
                        entry_price, pnl, pnl_pct, remaining_position, hold_time_seconds,
                        fees_paid, slippage_cost, exit_type, timestamp, status,
                        explainability_json, diagnostics_json, sleeve, exit_reason,
                        entry_timestamp, decision_id, strategy_id, confidence
                    ) VALUES (?, ?, ?, ?, 'SELL', ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sell_trade_id,
                        paper_run_id,
                        fill.mode,
                        fill.symbol,
                        qty,
                        exit_px,
                        entry,
                        realized,
                        pnl_pct,
                        hold_seconds,
                        fee,
                        0.0,
                        RECOVERED_SELL_EXIT_TYPE,
                        fill.closed_at_iso,
                        "executed",
                        json.dumps(explain, separators=(",", ":"), default=str),
                        json.dumps(diagnostics, separators=(",", ":")),
                        sleeve,
                        RECOVERED_CLOSE_REASON,
                        buy_ctx.get("buy_timestamp"),
                        explain.get("decision_id"),
                        strategy_id,
                        confidence,
                    ),
                )
                result["created"]["paper_trades_sell"] = int(cur.lastrowid)

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

    if not learning_exists:
        try:
            from backend.services.trade_learning_writer import TradeLearningRecord, record_trade_outcome

            record = TradeLearningRecord(
                symbol=fill.symbol.replace("/", ""),
                entry_timestamp=entry_epoch,
                exit_timestamp=float(fill.closed_at_epoch),
                entry_price=entry,
                exit_price=exit_px,
                quantity=qty,
                fees_paid=fee,
                net_profit_usd=realized,
                net_profit_pct=net_pct,
                hold_seconds=float(hold_seconds),
                decision_reason=f"engine_close:{RECOVERED_CLOSE_REASON}:{fill.source}",
                confidence=float(confidence) if confidence is not None else None,
                manual_sell_flag=True,
                close_reason=RECOVERED_CLOSE_REASON,
                realized_profit_unknown=not fill.fill_recovered,
                extra={
                    "source": fill.source,
                    "fill_recovered": fill.fill_recovered,
                    "not_ai_protected_sell": True,
                    "buy_trade_id": fill.buy_trade_id,
                    "exchange_sell_order_id": fill.exchange_sell_order_id,
                    "canonical_sell_trade_id": sell_trade_id,
                    "close_ledger_id": fill.close_ledger_id,
                    "lesson": "exchange_reconcile_close_not_engine_sell",
                },
            )
            if record_trade_outcome(record, db_path=db_path, mode_override=TradingMode.LIVE):
                result["created"]["trade_learning_outcomes"] = True
        except Exception as exc:
            result["errors"].append(f"learning:{exc}")
            logger.warning("RECOVERED_CLOSE_LEARNING_FAILED buy=%s err=%s", fill.buy_trade_id, exc)
    else:
        result["existing"]["trade_learning_outcomes"] = True

    if write_trade_performance and not perf_exists:
        try:
            from backend.services.trade_performance_tracker import TradePerformanceTracker

            TradePerformanceTracker._ensure_initialized()
            oid = int(fill.exchange_sell_order_id)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO trade_performance (
                        trade_id, symbol, side, entry_price, exit_price, quantity,
                        pnl, pnl_pct, is_win, hold_time_seconds, strategy, confidence,
                        mode, timestamp
                    ) VALUES (?, ?, 'sell', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        oid,
                        fill.symbol.replace("/", ""),
                        entry,
                        exit_px,
                        qty,
                        realized,
                        pnl_pct * 100.0,
                        1 if realized > 0 else (0 if realized < 0 else None),
                        hold_seconds,
                        strategy_id,
                        confidence,
                        fill.mode,
                        fill.closed_at_iso,
                    ),
                )
                conn.commit()
                result["created"]["trade_performance"] = oid
        except Exception as exc:
            result["errors"].append(f"trade_performance:{exc}")
            logger.warning("RECOVERED_CLOSE_PERF_FAILED buy=%s err=%s", fill.buy_trade_id, exc)
    elif perf_exists:
        result["existing"]["trade_performance"] = fill.exchange_sell_order_id

    try:
        from backend.services.ai_post_trade_feature_review import record_post_trade_feature_review

        record_post_trade_feature_review(
            trade_id=fill.buy_trade_id,
            symbol=fill.symbol,
            closed_at_utc=fill.closed_at_iso,
            explainability=explain_for_review,
            hold_seconds=float(hold_seconds),
            net_profit_usd=realized,
            net_profit_pct=net_pct,
            db_path=db_path,
        )
    except Exception as exc:
        result["errors"].append(f"post_review:{exc}")

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
