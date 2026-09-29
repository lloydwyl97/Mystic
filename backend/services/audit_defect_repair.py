"""Deterministic historical repairs for the 2026-09-29 audit defects.

Pure decision helpers; the script (scripts/repair_audit_defects_20260929.py)
does the I/O, backups and dry run. A repair is applied only where venue or
stored provenance proves it exactly; anything else is reported, not guessed.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

MIN_EXECUTABLE_SELL_NOTIONAL = 1.0
SELL_MATCH_WINDOW_SEC = 180.0
CHUNK_QTY_TOL = 1e-9
CHUNK_VWAP_REL_TOL = 1e-6


def _epoch(ts: Any) -> float:
    if ts is None or ts == "":
        return 0.0
    if isinstance(ts, (int, float)):
        return float(ts)
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def real_close_evidence(exit_epoch: float, sell_fills: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The venue SELL fill that makes a learning row a real strategy close.

    ``sell_fills`` are live_exchange_fills SELL rows for the lot's trade id. A
    close is real when an executable venue sell (notional >= $1) settled within
    the match window of the learning row's exit. A residual left afterwards is
    dust inventory, not the outcome.
    """
    best: dict[str, Any] | None = None
    for f in sell_fills or []:
        qty = float(f.get("executed_qty") or 0.0)
        px = float(f.get("avg_fill_price") or 0.0)
        ts = _epoch(f.get("event_ts_exchange") or f.get("event_ts_recorded"))
        if qty * px < MIN_EXECUTABLE_SELL_NOTIONAL or not ts:
            continue
        gap = abs(ts - float(exit_epoch or 0.0))
        if gap > SELL_MATCH_WINDOW_SEC:
            continue
        if best is None or gap < best["gap_sec"]:
            best = {
                "fill_row_id": f.get("id"),
                "exchange_order_id": str(f.get("exchange_order_id") or ""),
                "sell_qty": qty,
                "avg_fill_price": px,
                "notional": qty * px,
                "gap_sec": gap,
            }
    return best


def repaired_learning_extra(extra: dict[str, Any], evidence: dict[str, Any], *, residual_qty: float, setup: str | None = None) -> dict[str, Any]:
    """Metadata-only correction of a real close that was labeled dust."""
    out = dict(extra or {})
    strategy = str(out.get("strategy") or ("scalp" if str(out.get("engine_id") or "").upper() == "SCALP_V2" else "day"))
    out["is_dust"] = False
    out["label_strategy"] = strategy
    out["is_strategy_close"] = True
    out["original_trade_id"] = out.get("trade_id")
    out["pre_close_status"] = "ACTIVE"
    out["sell_qty"] = float(evidence["sell_qty"])
    out["residual_qty"] = float(residual_qty or 0.0)
    out["exit_provenance"] = {
        "exchange_order_id": evidence["exchange_order_id"],
        "fill_row_id": evidence["fill_row_id"],
        "sell_notional": round(float(evidence["notional"]), 8),
    }
    if setup and str(out.get("setup") or "UNKNOWN").upper() == "UNKNOWN":
        out["setup"] = setup
    out["provenance_repair"] = "audit_20260929_dust_mislabel"
    return out


def verify_chunk_trades(fill_row: dict[str, Any], venue_trades: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Venue trades that exactly reconstruct one aggregated sell, else None.

    The recorded row already holds the total executed qty and VWAP; only the
    chunk orders' cost and fees were dropped. The venue set must include the
    recorded order and match qty and VWAP exactly, or nothing is repaired.
    """
    want_qty = float(fill_row.get("executed_qty") or 0.0)
    want_px = float(fill_row.get("avg_fill_price") or 0.0)
    oid = str(fill_row.get("exchange_order_id") or "")
    sells = [t for t in venue_trades or [] if str(t.get("side") or "").lower() == "sell"]
    if not sells or want_qty <= 0 or want_px <= 0:
        return None
    orders = sorted({str(t.get("order") or "") for t in sells})
    if oid not in orders:
        return None
    qty = sum(float(t.get("amount") or 0.0) for t in sells)
    cost = sum(float(t.get("cost") or float(t.get("amount") or 0.0) * float(t.get("price") or 0.0)) for t in sells)
    if abs(qty - want_qty) > CHUNK_QTY_TOL:
        return None
    vwap = cost / qty
    if abs(vwap - want_px) / want_px > CHUNK_VWAP_REL_TOL:
        return None
    fees: dict[str, float] = {}
    for t in sells:
        fee = t.get("fee") or {}
        ccy = str(fee.get("currency") or "").upper()
        fees[ccy] = fees.get(ccy, 0.0) + float(fee.get("cost") or 0.0)
    return {
        "order_ids": orders,
        "trade_ids": [str(t.get("id") or "") for t in sells],
        "qty": qty,
        "cost": cost,
        "vwap": vwap,
        "fees": fees,
        "chunks": len(orders),
    }


def fee_correction(recorded_fee: float, venue: dict[str, Any]) -> float | None:
    """Missing quote fee (USD) for a sell, or None when the fee is not all quote-denominated."""
    fees = venue.get("fees") or {}
    if set(fees) - {"USDT", "USD"}:
        return None
    total = sum(fees.values())
    return max(0.0, total - float(recorded_fee or 0.0))


def load_json(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        val = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return val if isinstance(val, dict) else {}
