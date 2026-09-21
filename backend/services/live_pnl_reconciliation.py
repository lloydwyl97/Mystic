"""Reconcile the recorded live result against actual Binance.US fills.

``paper_trades`` stores paper and live rows in one table, separated only by the
``mode`` column, and the stored ``portfolio_engine_ledger.realized_pnl`` is a
historical sum of both. Presenting that total as trading performance credits the
live account with simulated profit earned before live execution began.

This module produces three clearly separated figures and never rewrites a row:

* ``live_reconciled`` - venue gross minus venue quote fees, from Binance.US
  fills only. This is the real trading result.
* ``paper_realized`` - simulated rows, reported on its own.
* ``legacy_mixed_total`` - the stored ledger value, labelled historical.

Venue calls are slow and rate limited, so results are cached.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DAY_SYMBOLS: tuple[str, ...] = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")

CACHE_TTL_SEC = 300.0
_QUOTE_ASSETS = {"USDT", "USD", "BUSD", "USDC"}

# Quantity tolerance when pairing a recorded row to a venue fill. Binance.US
# reports base quantity at the symbol's step size; our rows carry the same
# value, so this only absorbs float representation drift.
_QTY_REL_TOL = 1e-6
_QTY_ABS_TOL = 1e-12

# A venue fill or recorded row counts as accounted for once all but this
# fraction of its quantity is consumed. Step-size rounding leaves sub-unit
# residuals that are not real unreconciled inventory.
_RESIDUAL_REL_TOL = 1e-4

_cache: dict[str, Any] = {"at": 0.0, "payload": None}


@dataclass
class SymbolRecon:
    symbol: str
    recorded_buy_rows: int = 0
    recorded_sell_rows: int = 0
    recorded_buy_qty: float = 0.0
    recorded_sell_qty: float = 0.0
    recorded_buy_notional: float = 0.0
    recorded_sell_notional: float = 0.0
    venue_buy_fills: int = 0
    venue_sell_fills: int = 0
    venue_buy_qty: float = 0.0
    venue_sell_qty: float = 0.0
    venue_buy_notional: float = 0.0
    venue_sell_notional: float = 0.0
    venue_fee_quote_usd: float = 0.0
    venue_fee_base: dict[str, float] = field(default_factory=dict)
    venue_gross_usd: float = 0.0
    matched_fills: int = 0
    matched_recorded_rows: int = 0
    id_matched_rows: int = 0
    unmatched_recorded_rows: int = 0
    unmatched_venue_fills: int = 0
    unmatched_records: list[dict[str, Any]] = field(default_factory=list)
    qty_coverage_pct: float = 0.0
    unmatched_fill_groups: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LivePnlReconciliation:
    ok: bool = False
    error: str = ""
    generated_at: float = 0.0
    stale: bool = False

    window_start: str = ""
    window_end: str = ""

    # The real trading result: venue fills only.
    live_reconciled_usd: float = 0.0
    live_venue_gross_usd: float = 0.0
    live_venue_fee_quote_usd: float = 0.0

    # What the engine recorded for mode=live, for comparison.
    live_recorded_usd: float = 0.0
    live_dust_writeoff_usd: float = 0.0

    # Simulated. Never part of the live result.
    paper_realized_usd: float = 0.0

    # Stored ledger value. Historical, mixes both. Not live profit.
    legacy_mixed_total_usd: float = 0.0

    matched_fills: int = 0
    matched_recorded_rows: int = 0
    id_matched_rows: int = 0
    unmatched_recorded_rows: int = 0
    unmatched_venue_fills: int = 0
    unmatched_records: list[dict[str, Any]] = field(default_factory=list)
    recorded_rows_with_exchange_order_id: int = 0
    recorded_live_rows: int = 0
    qty_coverage_pct: float = 0.0
    unmatched_fill_groups: dict[str, Any] = field(default_factory=dict)

    per_symbol: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(ms: float) -> str:
    try:
        return datetime.fromtimestamp(float(ms) / 1000.0, tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError):
        return ""


def _epoch_ms(ts: str) -> int | None:
    raw = str(ts or "").strip()
    if not raw:
        return None
    try:
        return int(datetime.fromisoformat(raw).timestamp() * 1000)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            dt = datetime.strptime(raw[:26], fmt).replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except ValueError:
            continue
    return None


def _qty_close(a: float, b: float) -> bool:
    return abs(a - b) <= max(_QTY_ABS_TOL, abs(b) * _QTY_REL_TOL)


_FEE_FRAGMENT_REL = 0.0025


def classify_unmatched_venue_fills(
    unmatched: list[dict[str, Any]],
    *,
    known_order_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Group leftover venue fills without claiming the same quantity twice."""
    known = {str(x).strip() for x in (known_order_ids or set()) if str(x).strip()}
    groups: dict[str, list[dict[str, Any]]] = {
        "base_asset_fee_fragments": [],
        "partial_fills_of_known_orders": [],
        "full_buys_lacking_local": [],
        "full_sells_lacking_local": [],
        "genuine_unexplained": [],
    }
    for row in unmatched or []:
        if str(row.get("source") or "") != "venue_fill":
            continue
        qty = float(row.get("quantity") or 0.0)
        leftover = float(row.get("unmatched_quantity") or 0.0)
        if leftover <= 0:
            continue
        oid = str(row.get("exchange_order_id") or "").strip()
        side = str(row.get("side") or "").upper()
        consumed = max(0.0, qty - leftover)
        residual = max(_QTY_ABS_TOL, qty * _RESIDUAL_REL_TOL)
        is_full = consumed <= residual
        fee_like = leftover <= max(residual, qty * _FEE_FRAGMENT_REL)
        if oid in known and not is_full:
            group = "base_asset_fee_fragments" if fee_like else "partial_fills_of_known_orders"
        elif fee_like and not is_full:
            group = "base_asset_fee_fragments"
        elif is_full and side == "BUY":
            group = "full_buys_lacking_local"
        elif is_full and side == "SELL":
            group = "full_sells_lacking_local"
        else:
            group = "genuine_unexplained"
        item = dict(row)
        item["group"] = group
        groups[group].append(item)
    summary = {
        name: {
            "count": len(rows),
            "quantity": sum(float(r.get("unmatched_quantity") or 0.0) for r in rows),
            "dollar_value": sum(float(r.get("dollar_discrepancy") or 0.0) for r in rows),
        }
        for name, rows in groups.items()
    }
    return {**groups, "summary": summary}


def read_recorded_live(db_path: str) -> dict[str, Any]:
    """Recorded mode=live rows, plus the paper and legacy figures. Read-only."""
    out: dict[str, Any] = {
        "rows": {},
        "first_ts": "",
        "last_ts": "",
        "with_order_id": 0,
        "total_rows": 0,
        "paper_realized_usd": 0.0,
        "live_recorded_usd": 0.0,
        "live_dust_writeoff_usd": 0.0,
        "legacy_mixed_total_usd": 0.0,
    }
    uri = f"file:{db_path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=10)
    try:
        conn.row_factory = sqlite3.Row
        live = "LOWER(COALESCE(mode,'')) = 'live' AND COALESCE(is_synthetic,0) = 0"
        cols = {str(c[1]) for c in conn.execute("PRAGMA table_info(paper_trades)")}
        trade_expr = "COALESCE(trade_id,'')" if "trade_id" in cols else "''"
        order_expr = "COALESCE(order_id,'')" if "order_id" in cols else "''"
        exit_expr = "COALESCE(exit_type,'')" if "exit_type" in cols else "''"
        for r in conn.execute(
            f"""
            SELECT rowid AS local_id, symbol, UPPER(side) AS side, quantity, price, timestamp,
                   {order_expr} AS order_id, {exit_expr} AS exit_type, {trade_expr} AS trade_id
            FROM paper_trades
            WHERE {live}
            ORDER BY timestamp
            """
        ):
            out["rows"].setdefault((r["symbol"], r["side"]), []).append(
                {
                    "local_id": r["local_id"],
                    "trade_id": str(r["trade_id"] or ""),
                    "qty": float(r["quantity"] or 0.0),
                    "price": float(r["price"] or 0.0),
                    "ts": _epoch_ms(r["timestamp"]),
                    "timestamp": str(r["timestamp"] or ""),
                    "order_id": str(r["order_id"] or ""),
                    "client_order_id": "",
                    "exit_type": str(r["exit_type"] or ""),
                }
            )
            out["total_rows"] += 1
            if str(r["order_id"] or "").strip():
                out["with_order_id"] += 1

        win = conn.execute(f"SELECT MIN(timestamp), MAX(timestamp) FROM paper_trades WHERE {live}").fetchone()
        out["first_ts"] = str((win[0] if win else "") or "")
        out["last_ts"] = str((win[1] if win else "") or "")

        excluded = "('ADMIN_POSITION_CLEAR','STALE_PRE_CORRECTION_POSITION_CLEAR','RESEARCH_RESET_EXIT','DUST_WRITEOFF')"
        row = conn.execute(
            f"""SELECT COALESCE(SUM(pnl),0) FROM paper_trades
                WHERE {live} AND UPPER(side)='SELL' AND pnl IS NOT NULL
                  AND COALESCE(exit_type,'') NOT IN {excluded}"""
        ).fetchone()
        out["live_recorded_usd"] = float((row or (0.0,))[0] or 0.0)

        row = conn.execute(
            f"""SELECT COALESCE(SUM(pnl),0) FROM paper_trades
                WHERE {live} AND UPPER(side)='SELL' AND COALESCE(exit_type,'')='DUST_WRITEOFF'"""
        ).fetchone()
        out["live_dust_writeoff_usd"] = float((row or (0.0,))[0] or 0.0)

        row = conn.execute(
            f"""SELECT COALESCE(SUM(pnl),0) FROM paper_trades
                WHERE LOWER(COALESCE(mode,''))='paper' AND COALESCE(is_synthetic,0)=0
                  AND UPPER(side)='SELL' AND pnl IS NOT NULL
                  AND COALESCE(exit_type,'') NOT IN {excluded}"""
        ).fetchone()
        out["paper_realized_usd"] = float((row or (0.0,))[0] or 0.0)

        try:
            row = conn.execute("SELECT realized_pnl FROM portfolio_engine_ledger WHERE id=1").fetchone()
            out["legacy_mixed_total_usd"] = float((row or (0.0,))[0] or 0.0)
        except sqlite3.Error:
            out["legacy_mixed_total_usd"] = 0.0
    finally:
        conn.close()
    return out


async def _fetch_venue_fills(since_ms: int) -> dict[str, Any]:
    """Binance.US fills per DAY symbol since ``since_ms``. Read-only."""
    from backend.services.live_trading_service import LiveTradingService, _to_binance_pair

    svc = LiveTradingService()
    await svc._ensure_initialized()
    ex = svc.binance
    if ex is None:
        raise RuntimeError("live trading service has no exchange client")

    result: dict[str, Any] = {}
    for sym in DAY_SYMBOLS:
        pair = _to_binance_pair(sym)
        try:
            trades = await asyncio.to_thread(ex.fetch_my_trades, pair, since_ms, 1000)
        except Exception as exc:  # one symbol failing must not void the rest
            result[sym] = {"error": str(exc)[:200], "fills": []}
            continue
        fills = []
        for t in trades or []:
            fee = t.get("fee") or {}
            fills.append(
                {
                    "id": str(t.get("id") or ""),
                    "order": str(t.get("order") or ""),
                    "side": str(t.get("side") or "").upper(),
                    "qty": float(t.get("amount") or 0.0),
                    "cost": float(t.get("cost") or 0.0),
                    "ts": int(t.get("timestamp") or 0),
                    "fee_cost": float(fee.get("cost") or 0.0),
                    "fee_ccy": str(fee.get("currency") or "").upper(),
                }
            )
        result[sym] = {"fills": fills}
    return result


def _reconcile_symbol(symbol: str, recorded: dict[str, list[dict[str, Any]]], venue: dict[str, Any]) -> SymbolRecon:
    rec = SymbolRecon(symbol=symbol)
    if venue.get("error"):
        rec.error = str(venue["error"])
    fills = list(venue.get("fills") or [])

    for side in ("BUY", "SELL"):
        rows = list(recorded.get((symbol, side)) or [])
        vfills = [f for f in fills if f["side"] == side]

        if side == "BUY":
            rec.recorded_buy_rows = len(rows)
            rec.recorded_buy_qty = sum(r["qty"] for r in rows)
            rec.recorded_buy_notional = sum(r["qty"] * r["price"] for r in rows)
            rec.venue_buy_fills = len(vfills)
            rec.venue_buy_qty = sum(f["qty"] for f in vfills)
            rec.venue_buy_notional = sum(f["cost"] for f in vfills)
        else:
            rec.recorded_sell_rows = len(rows)
            rec.recorded_sell_qty = sum(r["qty"] for r in rows)
            rec.recorded_sell_notional = sum(r["qty"] * r["price"] for r in rows)
            rec.venue_sell_fills = len(vfills)
            rec.venue_sell_qty = sum(f["qty"] for f in vfills)
            rec.venue_sell_notional = sum(f["cost"] for f in vfills)

        for f in vfills:
            if f["fee_ccy"] in _QUOTE_ASSETS:
                rec.venue_fee_quote_usd += f["fee_cost"]
            elif f["fee_cost"] > 0:
                rec.venue_fee_base[f["fee_ccy"]] = rec.venue_fee_base.get(f["fee_ccy"], 0.0) + f["fee_cost"]

        queue = sorted(vfills, key=lambda x: x["ts"] or 0)
        remaining = [float(f["qty"]) for f in queue]
        touched = [False] * len(queue)
        row_need = [0.0 if r.get("exit_type") == "DUST_WRITEOFF" else float(r["qty"]) for r in rows]

        def _identity_hit(row: dict[str, Any], fill: dict[str, Any]) -> bool:
            oid = str(row.get("order_id") or "").strip()
            vid = str(row.get("venue_trade_id") or "").strip()
            cid = str(row.get("client_order_id") or "").strip()
            if oid and oid == str(fill.get("order") or "").strip():
                return True
            if vid and vid == str(fill.get("id") or "").strip():
                return True
            return bool(cid and cid == str(fill.get("client_order_id") or "").strip())

        for i, r in enumerate(rows):
            if r.get("exit_type") == "DUST_WRITEOFF":
                continue
            if not (str(r.get("order_id") or "").strip() or str(r.get("venue_trade_id") or "").strip() or str(r.get("client_order_id") or "").strip()):
                continue
            for j, f in enumerate(queue):
                if remaining[j] <= max(_QTY_ABS_TOL, float(f["qty"]) * _RESIDUAL_REL_TOL):
                    continue
                if not _identity_hit(r, f):
                    continue
                take = min(row_need[i], remaining[j])
                if take <= 0:
                    continue
                remaining[j] -= take
                row_need[i] -= take
                touched[j] = True
                rec.id_matched_rows += 1

        leftover_rows = [rows[i] for i, need in enumerate(row_need) if need > max(_QTY_ABS_TOL, float(rows[i]["qty"]) * _RESIDUAL_REL_TOL)]
        leftover_idx = [j for j, f in enumerate(queue) if remaining[j] > max(_QTY_ABS_TOL, float(f["qty"]) * _RESIDUAL_REL_TOL)]
        cursor = 0
        for r in leftover_rows:
            i = rows.index(r)
            need = row_need[i]
            tol = max(_QTY_ABS_TOL, float(r["qty"]) * _RESIDUAL_REL_TOL)
            while need > tol and cursor < len(leftover_idx):
                j = leftover_idx[cursor]
                spent = max(_QTY_ABS_TOL, queue[j]["qty"] * _RESIDUAL_REL_TOL)
                if remaining[j] <= spent:
                    cursor += 1
                    continue
                take = min(need, remaining[j])
                remaining[j] -= take
                need -= take
                touched[j] = True
            row_need[i] = need

        for i, r in enumerate(rows):
            if r.get("exit_type") == "DUST_WRITEOFF":
                continue
            leftover = row_need[i]
            tol = max(_QTY_ABS_TOL, float(r["qty"]) * _RESIDUAL_REL_TOL)
            if leftover > tol:
                rec.unmatched_recorded_rows += 1
                reason = "local quantity exceeds venue fills for this identity"
                if not str(r.get("order_id") or "").strip():
                    reason = "historical row lacks exchange order id; qty/time pairing incomplete"
                rec.unmatched_records.append(
                    {
                        "source": "local_recorded",
                        "symbol": symbol,
                        "timestamp": r.get("timestamp") or _iso(r.get("ts") or 0),
                        "side": side,
                        "quantity": r["qty"],
                        "unmatched_quantity": leftover,
                        "price": r["price"],
                        "exchange_order_id": r.get("order_id") or "",
                        "venue_trade_id": "",
                        "local_record_id": r.get("local_id"),
                        "dollar_discrepancy": leftover * float(r["price"] or 0.0),
                        "unmatched_reason": reason,
                    }
                )
            else:
                rec.matched_recorded_rows += 1
        rec.matched_fills += sum(1 for t in touched if t)
        for j, f in enumerate(queue):
            leftover = remaining[j]
            if leftover > max(_QTY_ABS_TOL, f["qty"] * _RESIDUAL_REL_TOL):
                rec.unmatched_venue_fills += 1
                rec.unmatched_records.append(
                    {
                        "source": "venue_fill",
                        "symbol": symbol,
                        "timestamp": _iso(f.get("ts") or 0),
                        "side": side,
                        "quantity": f["qty"],
                        "unmatched_quantity": leftover,
                        "price": (f["cost"] / f["qty"]) if f["qty"] else 0.0,
                        "exchange_order_id": f.get("order") or "",
                        "venue_trade_id": f.get("id") or "",
                        "local_record_id": "",
                        "dollar_discrepancy": leftover * ((f["cost"] / f["qty"]) if f["qty"] else 0.0),
                        "unmatched_reason": "no local paper_trades row carries this exchange order id",
                    }
                )

    rec.venue_gross_usd = rec.venue_sell_notional - rec.venue_buy_notional
    denom = rec.venue_buy_qty + rec.venue_sell_qty
    leftover_v = sum(float(u.get("unmatched_quantity") or 0.0) for u in rec.unmatched_records if u.get("source") == "venue_fill")
    from backend.services.live_exchange_equity import cap_qty_coverage_pct

    rec.qty_coverage_pct = cap_qty_coverage_pct(matched_qty=max(0.0, denom - leftover_v), venue_qty=denom)
    return rec


async def build_reconciliation(db_path: str) -> LivePnlReconciliation:
    out = LivePnlReconciliation(generated_at=time.time())
    try:
        local = await asyncio.to_thread(read_recorded_live, db_path)
    except Exception as exc:
        out.error = f"local read failed: {exc}"[:200]
        return out

    out.live_recorded_usd = float(local["live_recorded_usd"])
    out.live_dust_writeoff_usd = float(local["live_dust_writeoff_usd"])
    out.paper_realized_usd = float(local["paper_realized_usd"])
    out.legacy_mixed_total_usd = float(local["legacy_mixed_total_usd"])
    out.recorded_live_rows = int(local["total_rows"])
    out.recorded_rows_with_exchange_order_id = int(local["with_order_id"])

    since_ms = _epoch_ms(local["first_ts"])
    if since_ms is None:
        out.error = "no mode=live rows to reconcile"
        out.window_start = ""
        out.window_end = ""
        return out
    out.window_start = _iso(since_ms)

    try:
        venue = await _fetch_venue_fills(since_ms)
    except Exception as exc:
        out.error = f"venue fetch failed: {exc}"[:200]
        out.notes.append("Live reconciliation unavailable; recorded figures shown without venue confirmation.")
        return out

    last_ts = 0
    flat_recorded: list[dict[str, Any]] = []
    flat_venue: list[dict[str, Any]] = []
    for (sym, _side), rows in (local.get("rows") or {}).items():
        for r in rows:
            item = dict(r)
            item["symbol"] = sym
            item["side"] = _side
            flat_recorded.append(item)
    for sym in DAY_SYMBOLS:
        rec = _reconcile_symbol(sym, local["rows"], venue.get(sym) or {})
        out.per_symbol.append(rec.to_dict())
        out.live_venue_gross_usd += rec.venue_gross_usd
        out.live_venue_fee_quote_usd += rec.venue_fee_quote_usd
        out.matched_fills += rec.matched_fills
        out.matched_recorded_rows += rec.matched_recorded_rows
        out.id_matched_rows += rec.id_matched_rows
        out.unmatched_recorded_rows += rec.unmatched_recorded_rows
        out.unmatched_venue_fills += rec.unmatched_venue_fills
        out.unmatched_records.extend(rec.unmatched_records)
        for f in (venue.get(sym) or {}).get("fills") or []:
            last_ts = max(last_ts, int(f["ts"] or 0))
            item = dict(f)
            item["symbol"] = sym
            flat_venue.append(item)

    try:
        from backend.services.live_exchange_equity import backfill_provable_fill_identities

        backfill_provable_fill_identities(db_path, recorded=flat_recorded, venue_fills=flat_venue)
    except Exception:
        logger.debug("provable fill identity backfill skipped", exc_info=True)

    known_ids = {str(r.get("order_id") or "").strip() for r in flat_recorded if str(r.get("order_id") or "").strip()}
    venue_unmatched = [u for u in out.unmatched_records if str(u.get("source") or "") == "venue_fill"]
    out.unmatched_fill_groups = classify_unmatched_venue_fills(venue_unmatched, known_order_ids=known_ids)
    try:
        from backend.services.live_exchange_equity import backfill_exchange_reconciled_orders

        backfill_exchange_reconciled_orders(db_path, unmatched=venue_unmatched, known_order_ids=known_ids)
    except Exception:
        logger.debug("exchange-reconciled identity backfill skipped", exc_info=True)

    out.window_end = _iso(last_ts) if last_ts else str(local["last_ts"] or "")
    # Venue gross already nets base-asset commission out of received quantity;
    # quote-denominated commission is charged on top and must be subtracted.
    out.live_reconciled_usd = out.live_venue_gross_usd - out.live_venue_fee_quote_usd

    from backend.services.live_exchange_equity import cap_qty_coverage_pct

    v_qty = sum(float(s["venue_buy_qty"]) + float(s["venue_sell_qty"]) for s in out.per_symbol)
    leftover_v = sum(float(u.get("unmatched_quantity") or 0.0) for u in out.unmatched_records if u.get("source") == "venue_fill")
    out.qty_coverage_pct = cap_qty_coverage_pct(matched_qty=max(0.0, v_qty - leftover_v), venue_qty=v_qty)

    if out.recorded_rows_with_exchange_order_id == 0 and out.recorded_live_rows > 0:
        out.notes.append(
            "No exchange order id is stored on any recorded live row, so fills are paired by quantity and time rather than by id. "
            "Rows written after identifier persistence landed carry the order id and reconcile exactly."
        )
    elif out.id_matched_rows:
        out.notes.append(f"{out.id_matched_rows} row(s) reconciled exactly by exchange order id; the remainder are paired by quantity and time.")
    if abs(out.live_reconciled_usd - out.live_recorded_usd) > 1.0:
        out.notes.append(f"Recorded live P&L (${out.live_recorded_usd:,.2f}) differs from the exchange-reconciled result (${out.live_reconciled_usd:,.2f}).")
    out.notes.append("LEGACY MIXED TOTAL is historical and mixes paper with live. It is not live trading profit.")
    out.ok = True
    return out


async def get_reconciliation(db_path: str, *, force: bool = False, cached_only: bool = False) -> dict[str, Any]:
    """Cached reconciliation. Venue calls are rate limited, so reuse results.

    ``cached_only`` lets a latency-sensitive caller reuse whatever the
    dedicated reconciliation poll already fetched without itself waiting on
    four exchange round trips.
    """
    now = time.time()
    cached = _cache.get("payload")
    if cached and not force and (now - float(_cache.get("at") or 0.0)) < CACHE_TTL_SEC:
        payload = dict(cached)
        payload["cached"] = True
        payload["cache_age_sec"] = round(now - float(_cache["at"]), 1)
        return payload
    if cached_only:
        payload = dict(cached) if cached else LivePnlReconciliation(error="reconciliation not yet fetched").to_dict()
        payload["cached"] = bool(cached)
        payload["stale"] = bool(cached)
        payload["cache_age_sec"] = round(now - float(_cache.get("at") or 0.0), 1) if cached else 0.0
        return payload

    result = await build_reconciliation(db_path)
    payload = result.to_dict()
    payload["cached"] = False
    payload["cache_age_sec"] = 0.0
    if result.ok:
        _cache["at"] = now
        _cache["payload"] = dict(payload)
    elif cached:
        # Venue unreachable: serve the last good result, clearly marked stale,
        # rather than presenting zeros as a real trading figure.
        stale = dict(cached)
        stale["cached"] = True
        stale["stale"] = True
        stale["cache_age_sec"] = round(now - float(_cache.get("at") or 0.0), 1)
        stale["error"] = result.error
        return stale
    return payload


def account_basis_for_presentation(
    db_path: str,
    *,
    current_equity: float,
    contributed_principal: float | None = None,
    forward_baseline_equity: float | None = None,
    net_liquidatable_equity: float | None = None,
    baseline_dust_known: bool = False,
) -> dict[str, Any]:
    """Forward-baseline inputs. Adoption deltas are corrections, not profit."""
    from backend.services.live_account_basis import RECON_KEY, load_operational_json

    raw = load_operational_json(db_path, RECON_KEY)
    try:
        adj = float(raw.get("adjustment_usd") or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    baseline = forward_baseline_equity if forward_baseline_equity is not None else contributed_principal
    return {
        "current_equity": float(current_equity or 0.0),
        "forward_baseline_equity": float(baseline or 0.0),
        "net_liquidatable_equity": float(net_liquidatable_equity) if net_liquidatable_equity is not None else None,
        "baseline_dust_known": bool(baseline_dust_known),
        "reconciliation_adjustment_usd": adj,
    }


def presentation_fields(
    recon: dict[str, Any],
    *,
    is_live: bool,
    current_equity: float | None = None,
    contributed_principal: float | None = None,
    forward_baseline_equity: float | None = None,
    net_liquidatable_equity: float | None = None,
    baseline_dust_known: bool = False,
    reconciliation_adjustment_usd: float | None = None,
) -> dict[str, Any]:
    """Separated live economic, completeness, correction and paper figures."""
    reconciled_ok = bool(recon.get("ok")) and not recon.get("error")
    fill_tape = float(recon.get("live_reconciled_usd") or 0.0) if reconciled_ok else None
    recorded = float(recon.get("live_recorded_usd") or 0.0)
    dust = float(recon.get("live_dust_writeoff_usd") or 0.0)
    recon_adj = float(reconciliation_adjustment_usd or 0.0)
    accounting_correction = dust + recon_adj
    baseline = forward_baseline_equity if forward_baseline_equity is not None else contributed_principal
    cash_change = None
    if current_equity is not None and baseline is not None:
        cash_change = float(current_equity) - float(baseline)
    comparable_change = None
    if is_live and baseline_dust_known and net_liquidatable_equity is not None and baseline is not None:
        comparable_change = float(net_liquidatable_equity) - float(baseline)
    if comparable_change is not None:
        primary = comparable_change
        label = "LIVE NET LIQUIDATABLE CHANGE vs comparable forward baseline"
        primary_is_recon = True
    elif fill_tape is not None:
        primary = fill_tape
        label = "LIVE (exchange-reconciled fill tape; not total-account P&L)"
        primary_is_recon = True
    else:
        primary = recorded
        label = "LIVE (recorded, not exchange-reconciled; not total-account P&L)"
        primary_is_recon = False
    return {
        "primary_result_label": label,
        "primary_result_usd": primary,
        "primary_result_is_exchange_reconciled": primary_is_recon,
        "live_economic_pnl_usd": comparable_change,
        "live_economic_label": (
            "LIVE NET LIQUIDATABLE CHANGE vs comparable forward baseline" if comparable_change is not None else "INCOMPLETE: cash-only change is not total-account P&L because baseline dust is unknown"
        ),
        "forward_cash_change_usd": cash_change,
        "forward_baseline_equity_usd": baseline,
        "forward_baseline_label": "forward baseline equity (adopted cash at SHA 9039923; not contributed principal)",
        "lifetime_contributed_capital": "UNKNOWN",
        "contributed_principal_usd": None,
        "net_liquidatable_equity_usd": net_liquidatable_equity,
        "current_equity_usd": current_equity,
        "live_reconciled_usd": fill_tape,
        "live_reconciled_label": "LOCAL-RECORD COMPLETENESS DIAGNOSTIC (venue fill tape gross - quote fees; not cash-identity)",
        "live_recorded_usd": recorded,
        "live_recorded_label": "LOCAL-RECORD COMPLETENESS DIAGNOSTIC (mode=live SELL pnl excluding dust)",
        "live_dust_writeoff_usd": dust,
        "live_dust_writeoff_label": "ACCOUNTING CORRECTION (historical dust write-off; not market P&L)",
        "reconciliation_adjustment_usd": recon_adj,
        "accounting_correction_total_usd": accounting_correction,
        "accounting_correction_label": "ACCOUNTING CORRECTION TOTAL (dust write-off + cash-adoption adjustment; not trading profit)",
        "paper_realized_usd": float(recon.get("paper_realized_usd") or 0.0),
        "legacy_mixed_total_usd": float(recon.get("legacy_mixed_total_usd") or 0.0),
        "legacy_mixed_total_label": "LEGACY MIXED TOTAL (historical, not live profit)",
        "paper_is_not_live_performance": True,
        "account_execution_mode": "live" if is_live else "paper",
        "matched_fills": int(recon.get("matched_fills") or 0),
        "matched_recorded_rows": int(recon.get("matched_recorded_rows") or 0),
        "id_matched_rows": int(recon.get("id_matched_rows") or 0),
        "unmatched_recorded_rows": int(recon.get("unmatched_recorded_rows") or 0),
        "unmatched_venue_fills": int(recon.get("unmatched_venue_fills") or 0),
        "unmatched_records": list(recon.get("unmatched_records") or []),
        "exchange_fees_quote_usd": float(recon.get("live_venue_fee_quote_usd") or 0.0),
        "reconciliation_window_start": str(recon.get("window_start") or ""),
        "reconciliation_window_end": str(recon.get("window_end") or ""),
        "qty_coverage_pct": round(float(recon.get("qty_coverage_pct") or 0.0), 2),
        "unmatched_fill_groups": dict(recon.get("unmatched_fill_groups") or {}),
        "reconciliation_stale": bool(recon.get("stale")),
        "reconciliation_error": str(recon.get("error") or ""),
        "notes": list(recon.get("notes") or []),
    }
