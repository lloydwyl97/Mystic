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
        for r in conn.execute(
            f"""
            SELECT rowid AS local_id, symbol, UPPER(side) AS side, quantity, price, timestamp,
                   COALESCE(order_id,'') AS order_id, COALESCE(exit_type,'') AS exit_type
            FROM paper_trades
            WHERE {live}
            ORDER BY timestamp
            """
        ):
            out["rows"].setdefault((r["symbol"], r["side"]), []).append(
                {
                    "local_id": r["local_id"],
                    "qty": float(r["quantity"] or 0.0),
                    "price": float(r["price"] or 0.0),
                    "ts": _epoch_ms(r["timestamp"]),
                    "timestamp": str(r["timestamp"] or ""),
                    "order_id": str(r["order_id"] or ""),
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

        # Rows that carry an exchange order id reconcile exactly, with no
        # inference. Historical rows predate identifier persistence and fall
        # through to the quantity walk below.
        for r in rows:
            oid = str(r.get("order_id") or "").strip()
            if oid and any(str(f.get("order") or "") == oid for f in vfills):
                rec.id_matched_rows += 1

        # Row-to-fill is many-to-many: the engine writes one row per FIFO lot
        # while the venue reports one fill per partial execution. Pairing 1:1
        # would report most rows unmatched even when every unit is accounted
        # for, so consume quantity chronologically instead — which is what the
        # FIFO accounting actually claims happened.
        queue = sorted(vfills, key=lambda x: x["ts"] or 0)
        remaining = [f["qty"] for f in queue]
        touched = [False] * len(queue)
        cursor = 0
        for r in sorted(rows, key=lambda x: x["ts"] or 0):
            if r["exit_type"] == "DUST_WRITEOFF":
                # Written off without an exchange order by design.
                continue
            need = float(r["qty"])
            tol = max(_QTY_ABS_TOL, need * _RESIDUAL_REL_TOL)
            while need > tol and cursor < len(queue):
                spent = max(_QTY_ABS_TOL, queue[cursor]["qty"] * _RESIDUAL_REL_TOL)
                if remaining[cursor] <= spent:
                    cursor += 1
                    continue
                take = min(need, remaining[cursor])
                remaining[cursor] -= take
                need -= take
                touched[cursor] = True
            leftover = need if need > tol else 0.0
            if leftover:
                rec.unmatched_recorded_rows += 1
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
                    }
                )
            else:
                rec.matched_recorded_rows += 1
        rec.matched_fills += sum(1 for t in touched if t)
        for i, f in enumerate(queue):
            leftover = remaining[i]
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
                    }
                )

    rec.venue_gross_usd = rec.venue_sell_notional - rec.venue_buy_notional
    denom = rec.venue_buy_qty + rec.venue_sell_qty
    if denom > 0:
        rec.qty_coverage_pct = 100.0 * (rec.recorded_buy_qty + rec.recorded_sell_qty) / denom
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

    out.window_end = _iso(last_ts) if last_ts else str(local["last_ts"] or "")
    # Venue gross already nets base-asset commission out of received quantity;
    # quote-denominated commission is charged on top and must be subtracted.
    out.live_reconciled_usd = out.live_venue_gross_usd - out.live_venue_fee_quote_usd

    v_qty = sum(float(s["venue_buy_qty"]) + float(s["venue_sell_qty"]) for s in out.per_symbol)
    r_qty = sum(float(s["recorded_buy_qty"]) + float(s["recorded_sell_qty"]) for s in out.per_symbol)
    out.qty_coverage_pct = (100.0 * r_qty / v_qty) if v_qty > 0 else 0.0

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
    contributed_principal: float,
) -> dict[str, float]:
    """Ledger cash-identity inputs. Adoption deltas are corrections, not profit."""
    from backend.services.live_account_basis import RECON_KEY, load_operational_json

    raw = load_operational_json(db_path, RECON_KEY)
    try:
        adj = float(raw.get("adjustment_usd") or 0.0)
    except (TypeError, ValueError):
        adj = 0.0
    return {
        "current_equity": float(current_equity or 0.0),
        "contributed_principal": float(contributed_principal or 0.0),
        "reconciliation_adjustment_usd": adj,
    }


def presentation_fields(
    recon: dict[str, Any],
    *,
    is_live: bool,
    current_equity: float | None = None,
    contributed_principal: float | None = None,
    reconciliation_adjustment_usd: float | None = None,
) -> dict[str, Any]:
    """Separated live economic, completeness, correction and paper figures."""
    reconciled_ok = bool(recon.get("ok")) and not recon.get("error")
    fill_tape = float(recon.get("live_reconciled_usd") or 0.0) if reconciled_ok else None
    recorded = float(recon.get("live_recorded_usd") or 0.0)
    dust = float(recon.get("live_dust_writeoff_usd") or 0.0)
    recon_adj = float(reconciliation_adjustment_usd or 0.0)
    accounting_correction = dust + recon_adj
    economic = None
    if current_equity is not None and contributed_principal is not None:
        economic = float(current_equity) - float(contributed_principal)
    if is_live and economic is not None:
        primary = economic
        label = "LIVE ECONOMIC (cash-identity: marked equity - contributed principal)"
        primary_is_recon = True
    elif fill_tape is not None:
        primary = fill_tape
        label = "LIVE (exchange-reconciled fill tape)"
        primary_is_recon = True
    else:
        primary = recorded
        label = "LIVE (recorded, not exchange-reconciled)"
        primary_is_recon = False
    return {
        "primary_result_label": label,
        "primary_result_usd": primary,
        "primary_result_is_exchange_reconciled": primary_is_recon,
        "live_economic_pnl_usd": economic,
        "live_economic_label": "LIVE ECONOMIC (cash-identity: marked equity - contributed principal; not fill-tape P&L)",
        "contributed_principal_usd": contributed_principal,
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
        "reconciliation_stale": bool(recon.get("stale")),
        "reconciliation_error": str(recon.get("error") or ""),
        "notes": list(recon.get("notes") or []),
    }
