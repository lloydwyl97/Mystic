"""Live close integrity — venue fill is the only completed-trade authority.

A live strategy close is finalized only after:

    EXIT DECISION → EXCHANGE ORDER ACCEPTED → VENUE FILL RECONCILED
    → POSITION QTY REDUCED → REALIZED P&L RECORDED

``paper_trades`` rows are evidence, not the economic ledger. Historical
venue-less and duplicate rows are classified in an append-only table and
are never deleted.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.utils.sqlite_runtime import connect_rw

logger = logging.getLogger(__name__)

CLASS_VENUE_BACKED = "VENUE_BACKED_STRATEGY_TRADE"
CLASS_NON_VENUE_GHOST = "NON_VENUE_GHOST"
CLASS_DUPLICATE_MIRROR = "DUPLICATE_LOCAL_MIRROR"
CLASS_RECONCILE_MIRROR = "RECONCILE_MIRROR"
CLASS_MANUAL_UNMATCHED = "MANUAL_UNMATCHED"

STRATEGY_CLASSES = frozenset({CLASS_VENUE_BACKED})

CLASSIFICATION_TABLE = "live_close_classifications"
EVENT_TABLE = "live_close_events"
ECONOMIC_TABLE = "live_economic_closes"

TRAIL_ACTIVATION = 0.0025
TRAIL_DISTANCE = 0.0025

_HONEST_RT = {
    "BTC/USDT": 0.00060359,
    "BTCUSDT": 0.00060359,
    "ETH/USDT": 0.00061697,
    "ETHUSDT": 0.00061697,
    "SOL/USDT": 0.00076317,
    "SOLUSDT": 0.00076317,
    "XRP/USDT": 0.00074463,
    "XRPUSDT": 0.00074463,
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_live_close_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {CLASSIFICATION_TABLE} (
            trade_id TEXT PRIMARY KEY,
            accounting_class TEXT NOT NULL,
            symbol TEXT,
            timestamp TEXT,
            exchange_order_id TEXT,
            reason TEXT NOT NULL,
            classified_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {EVENT_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            symbol TEXT NOT NULL,
            event_type TEXT NOT NULL,
            exit_trigger TEXT,
            quantity REAL,
            price_snapshot REAL,
            exchange_order_id TEXT,
            detail TEXT
        )
        """
    )
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {ECONOMIC_TABLE} (
            exchange_order_id TEXT NOT NULL,
            side TEXT NOT NULL,
            mystic_trade_id TEXT,
            symbol TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (exchange_order_id, side)
        )
        """
    )


def live_order_has_venue_fill(order: dict[str, Any] | None) -> bool:
    """True only when the exchange accepted an order and filled quantity > 0."""
    if not isinstance(order, dict):
        return False
    oid = str(order.get("id") or (order.get("info") or {}).get("orderId") or "").strip()
    if not oid:
        return False
    try:
        filled = float(order.get("filled") or (order.get("info") or {}).get("executedQty") or 0.0)
    except (TypeError, ValueError):
        filled = 0.0
    return filled > 1e-15


def classify_paper_sell_row(
    row: dict[str, Any],
    *,
    seen_order_ids: set[str] | None = None,
) -> str:
    """Assign exactly one accounting class. Does not mutate the row."""
    exit_type = str(row.get("exit_type") or "").upper()
    exit_reason = str(row.get("exit_reason") or "").upper()
    status = str(row.get("status") or "").lower()
    oid = str(row.get("order_id") or "").strip()
    mode = str(row.get("mode") or "").lower()

    if status == "dust_writeoff" or exit_type == "DUST_WRITEOFF":
        return CLASS_NON_VENUE_GHOST
    if "RECONCILE" in exit_type or "RECONCILE" in exit_reason or "RECOVERED" in str(row.get("trade_id") or "").upper():
        return CLASS_RECONCILE_MIRROR
    if "HUMAN_MANUAL" in exit_type or "HUMAN_MANUAL" in exit_reason:
        return CLASS_MANUAL_UNMATCHED
    if oid and seen_order_ids is not None and oid in seen_order_ids:
        return CLASS_DUPLICATE_MIRROR
    if oid and mode == "live":
        if seen_order_ids is not None:
            seen_order_ids.add(oid)
        return CLASS_VENUE_BACKED
    if mode == "live" and not oid:
        return CLASS_NON_VENUE_GHOST
    if oid:
        if seen_order_ids is not None:
            seen_order_ids.add(oid)
        return CLASS_VENUE_BACKED
    return CLASS_NON_VENUE_GHOST


def classify_existing_rows(db_path: str) -> dict[str, int]:
    """Append-only classify every paper_trades SELL. Never updates paper_trades."""
    counts: dict[str, int] = {}
    with connect_rw(db_path) as conn:
        ensure_live_close_tables(conn)
        rows = conn.execute(
            """
            SELECT trade_id, symbol, timestamp, order_id, mode, status,
                   exit_type, exit_reason
            FROM paper_trades WHERE UPPER(side) = 'SELL'
            ORDER BY timestamp, id
            """
        ).fetchall()
        seen: set[str] = set()
        already = {str(r[0]) for r in conn.execute(f"SELECT trade_id FROM {CLASSIFICATION_TABLE}")}
        for r in rows:
            rec = {
                "trade_id": r[0],
                "symbol": r[1],
                "timestamp": r[2],
                "order_id": r[3],
                "mode": r[4],
                "status": r[5],
                "exit_type": r[6],
                "exit_reason": r[7],
            }
            klass = classify_paper_sell_row(rec, seen_order_ids=seen)
            counts[klass] = counts.get(klass, 0) + 1
            tid = str(rec["trade_id"] or "")
            if not tid or tid in already:
                continue
            conn.execute(
                f"""
                INSERT OR IGNORE INTO {CLASSIFICATION_TABLE}
                (trade_id, accounting_class, symbol, timestamp, exchange_order_id, reason, classified_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tid,
                    klass,
                    rec["symbol"],
                    rec["timestamp"],
                    str(rec["order_id"] or "") or None,
                    f"auto:{klass}",
                    _now_iso(),
                ),
            )
        conn.commit()
    return counts


def persist_pending_close_event(
    db_path: str,
    *,
    symbol: str,
    event_type: str,
    exit_trigger: str = "",
    quantity: float = 0.0,
    price_snapshot: float = 0.0,
    exchange_order_id: str = "",
    detail: str = "",
) -> None:
    with connect_rw(db_path) as conn:
        ensure_live_close_tables(conn)
        conn.execute(
            f"""
            INSERT OR IGNORE INTO {EVENT_TABLE}
            (ts, symbol, event_type, exit_trigger, quantity, price_snapshot, exchange_order_id, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now_iso(),
                symbol,
                event_type,
                exit_trigger,
                float(quantity or 0.0),
                float(price_snapshot or 0.0),
                str(exchange_order_id or "") or None,
                detail,
            ),
        )
        conn.commit()


def claim_economic_close(
    db_path: str,
    exchange_order_id: str,
    side: str,
    *,
    mystic_trade_id: str = "",
    symbol: str = "",
) -> bool:
    """Return True if this venue fill is newly claimed. False = already booked."""
    oid = str(exchange_order_id or "").strip()
    if not oid:
        return False
    with connect_rw(db_path) as conn:
        ensure_live_close_tables(conn)
        cur = conn.execute(
            f"""
            INSERT OR IGNORE INTO {ECONOMIC_TABLE}
            (exchange_order_id, side, mystic_trade_id, symbol, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (oid, str(side or "SELL").upper(), mystic_trade_id or None, symbol or None, _now_iso()),
        )
        conn.commit()
        return int(cur.rowcount or 0) == 1


def dust_writeoff_already_recorded(db_path: str, symbol: str, quantity: float) -> bool:
    with connect_rw(db_path) as conn:
        row = conn.execute(
            """
            SELECT 1 FROM dust_writeoffs
            WHERE symbol = ? AND ABS(quantity - ?) < 1e-12
            LIMIT 1
            """,
            (symbol, float(quantity or 0.0)),
        ).fetchone()
        return row is not None


def persist_dust_inventory_event(
    db_path: str,
    *,
    symbol: str,
    quantity: float,
    entry_price: float,
    price_snapshot: float,
    reason: str,
    est_notional: float,
) -> bool:
    """Record leftover dust once. Returns False if already booked. No strategy P&L."""
    if dust_writeoff_already_recorded(db_path, symbol, quantity):
        return False
    ts = _now_iso()
    with connect_rw(db_path) as conn:
        ensure_live_close_tables(conn)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dust_writeoffs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                price_snapshot REAL NOT NULL,
                est_notional REAL NOT NULL,
                reason TEXT NOT NULL,
                sell_trade_id TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO dust_writeoffs
            (timestamp, symbol, quantity, entry_price, price_snapshot, est_notional, reason, sell_trade_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, symbol, float(quantity or 0.0), float(entry_price or 0.0), float(price_snapshot or 0.0), float(est_notional or 0.0), reason, ""),
        )
        conn.commit()
    persist_pending_close_event(
        db_path,
        symbol=symbol,
        event_type="DUST_PENDING",
        exit_trigger=reason,
        quantity=quantity,
        price_snapshot=price_snapshot,
        detail="dust leftover; not a strategy close",
    )
    return True


def strategy_scorecard_sql_predicate(alias: str = "t") -> str:
    """SQL fragment: only venue-backed live strategy closes."""
    return f"""
        COALESCE({alias}.mode, '') = 'live'
        AND COALESCE({alias}.status, '') NOT IN ('dust_writeoff', 'pending', 'rejected')
        AND COALESCE({alias}.exit_type, '') NOT IN (
            'DUST_WRITEOFF', 'EXCHANGE_RECONCILE_CLOSE', 'HUMAN_MANUAL_SELL',
            'ADMIN_POSITION_CLEAR', 'STALE_PRE_CORRECTION_POSITION_CLEAR',
            'RESEARCH_RESET_EXIT', 'STALE_LIVE_GHOST_POSITION_CLEAR'
        )
        AND COALESCE({alias}.exit_reason, '') NOT IN (
            'EXCHANGE_RECONCILE_CLOSE', 'HUMAN_MANUAL_SELL'
        )
        AND TRIM(COALESCE({alias}.order_id, '')) != ''
        AND COALESCE({alias}.is_synthetic, 0) = 0
    """


def venue_backed_unique_aggregate_sql(day_clause: str = "") -> str:
    """One economic close per venue order_id. Earliest local row wins."""
    pred = strategy_scorecard_sql_predicate("t")
    pred2 = strategy_scorecard_sql_predicate("t2")
    return f"""
        SELECT COUNT(*) AS n, COALESCE(SUM(net), 0) AS pnl FROM (
            SELECT COALESCE(t.pnl_usd_net, t.pnl) AS net
            FROM paper_trades t
            WHERE UPPER(t.side) = 'SELL'
              AND {pred}
              {day_clause}
              AND t.rowid = (
                  SELECT MIN(t2.rowid)
                  FROM paper_trades t2
                  WHERE TRIM(t2.order_id) = TRIM(t.order_id)
                    AND UPPER(t2.side) = 'SELL'
                    AND {pred2}
                    {day_clause.replace("t.", "t2.") if day_clause else ""}
              )
        )
    """


def venue_backed_window_stats(
    db_path: str,
    start_iso: str,
    end_iso: str,
) -> dict[str, Any]:
    """Count unique venue-backed SELL fills in [start, end)."""
    sql = venue_backed_unique_aggregate_sql("AND t.timestamp >= ? AND t.timestamp < ?")
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        n, pnl = conn.execute(sql, (start_iso, end_iso, start_iso, end_iso)).fetchone()
        ghosts = conn.execute(
            """
            SELECT COUNT(*), COALESCE(SUM(COALESCE(pnl_usd_net, pnl)), 0)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND timestamp >= ? AND timestamp < ?
              AND (status = 'dust_writeoff' OR exit_type = 'DUST_WRITEOFF' OR TRIM(COALESCE(order_id,'')) = '')
            """,
            (start_iso, end_iso),
        ).fetchone()
        recon = conn.execute(
            """
            SELECT COUNT(*)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND timestamp >= ? AND timestamp < ?
              AND (exit_type = 'EXCHANGE_RECONCILE_CLOSE' OR exit_reason = 'EXCHANGE_RECONCILE_CLOSE')
            """,
            (start_iso, end_iso),
        ).fetchone()
        manual = conn.execute(
            """
            SELECT COUNT(*)
            FROM paper_trades
            WHERE UPPER(side) = 'SELL' AND timestamp >= ? AND timestamp < ?
              AND (exit_type = 'HUMAN_MANUAL_SELL' OR exit_reason = 'HUMAN_MANUAL_SELL')
            """,
            (start_iso, end_iso),
        ).fetchone()
        return {
            "venue_backed_closes": int(n or 0),
            "stored_venue_backed_pnl": float(pnl or 0.0),
            "raw_non_venue_rows": int(ghosts[0] or 0),
            "raw_non_venue_stored_pnl": float(ghosts[1] or 0.0),
            "reconcile_mirror_rows": int((recon[0] if recon else 0) or 0),
            "manual_rows": int((manual[0] if manual else 0) or 0),
        }
    finally:
        conn.close()


def honest_rt(symbol: str) -> float:
    return float(_HONEST_RT.get(str(symbol or "").upper(), 0.0007))


def classify_trailing_exit(
    *,
    symbol: str,
    entry: float,
    highest_executable: float,
    fill_price: float,
    executable_bid: float | None = None,
) -> dict[str, Any]:
    """Classify one venue-backed trailing SELL. Does not change the trail."""
    from backend.services.day_controlled_exits import _trail_activation_price

    activation = _trail_activation_price(entry=entry, trail_distance=TRAIL_DISTANCE, symbol=symbol)
    activated = highest_executable + 1e-12 >= activation
    raw = highest_executable * (1.0 - TRAIL_DISTANCE) if activated else 0.0
    be = entry * (1.0 + honest_rt(symbol))
    controlling = max(raw, be) if activated else None
    bid = float(executable_bid) if executable_bid else fill_price
    slip = (controlling - fill_price) if controlling else None
    if not activated:
        label = "TRAIL_NOT_ACTIVATED"
    elif controlling is not None and fill_price + 1e-8 < controlling:
        if bid + 1e-8 >= controlling:
            label = "VALID_TRAIL_GAP_OR_SLIPPAGE"
        else:
            label = "COST_FLOOR_NOT_APPLIED"
    else:
        label = "VALID_TRAIL_GAP_OR_SLIPPAGE"
    return {
        "activation_threshold": activation,
        "activated": activated,
        "raw_trailing_stop": raw,
        "cost_aware_floor": be,
        "controlling_stop": controlling,
        "fill_price": fill_price,
        "executable_bid": bid,
        "slippage_vs_controlling": slip,
        "classification": label,
    }
