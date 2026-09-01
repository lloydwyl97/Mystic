"""Observability-only BUY reference-price telemetry.

Does not change order price, quantity, or whether an order is sent.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

TELEMETRY_KEY = "entry_reference_telemetry"


def build_entry_reference_telemetry(
    *,
    best_bid: float | None,
    best_ask: float | None,
    submitted_order_price: float | None,
    fill_price: float,
    decision_ts: float | None = None,
    fill_ts: float | None = None,
    is_maker: bool | None = None,
    live_order: dict[str, Any] | None = None,
) -> dict[str, Any]:
    bid = float(best_bid or 0.0)
    ask = float(best_ask or 0.0)
    mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else 0.0
    fill = float(fill_price or 0.0)
    submitted = float(submitted_order_price or 0.0)
    if is_maker is None and isinstance(live_order, dict):
        raw = live_order.get("isMaker")
        if raw is None:
            raw = (live_order.get("info") or {}).get("isMaker") if isinstance(live_order.get("info"), dict) else None
        if raw is not None:
            is_maker = bool(raw)
    decision_ts = float(decision_ts or 0.0)
    fill_ts = float(fill_ts or time.time())
    latency = (fill_ts - decision_ts) if decision_ts > 0 and fill_ts > 0 else None
    slip_from_mid = ((fill - mid) / mid) if mid > 0 and fill > 0 else None
    spread_cross = ((ask - mid) / mid) if mid > 0 and ask > 0 else None
    return {
        TELEMETRY_KEY: True,
        "decision_best_bid": bid or None,
        "decision_best_ask": ask or None,
        "decision_midpoint": mid or None,
        "submitted_order_price": submitted or None,
        "fill_price": fill or None,
        "is_maker": is_maker,
        "fill_latency_sec": round(latency, 6) if latency is not None else None,
        "entry_slippage_from_midpoint_pct": slip_from_mid,
        "entry_spread_crossing_cost_pct": spread_cross,
        "mark_price": mid or None,
    }


def merge_into_json_blob(raw: str | dict[str, Any] | None, telemetry: dict[str, Any]) -> str:
    payload: dict[str, Any]
    if isinstance(raw, dict):
        payload = dict(raw)
    elif isinstance(raw, str) and raw.strip():
        try:
            loaded = json.loads(raw)
            payload = loaded if isinstance(loaded, dict) else {"_raw": raw}
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
    else:
        payload = {}
    payload[TELEMETRY_KEY] = telemetry
    return json.dumps(payload, separators=(",", ":"))


def persist_entry_reference_row(
    conn: Any,
    *,
    trade_id: str,
    telemetry: dict[str, Any],
    context_snapshot_json: str | None = None,
    diagnostics_json: str | None = None,
) -> None:
    """Write mark_price + JSON blobs. Ignore missing columns (older DBs)."""
    mid = telemetry.get("decision_midpoint") or telemetry.get("mark_price")
    slip = telemetry.get("entry_slippage_from_midpoint_pct")
    cols = {row[1] for row in conn.execute("PRAGMA table_info(paper_trades)")}
    sets: list[str] = []
    args: list[Any] = []
    if "mark_price" in cols:
        sets.append("mark_price = ?")
        args.append(mid)
    if "slippage_pct_implied" in cols:
        sets.append("slippage_pct_implied = ?")
        args.append(slip)
    if "spread_pct_used" in cols:
        sets.append("spread_pct_used = ?")
        args.append(telemetry.get("entry_spread_crossing_cost_pct"))
    if "context_snapshot_json" in cols and context_snapshot_json is not None:
        sets.append("context_snapshot_json = ?")
        args.append(merge_into_json_blob(context_snapshot_json, telemetry))
    if "diagnostics_json" in cols and diagnostics_json is not None:
        sets.append("diagnostics_json = ?")
        args.append(merge_into_json_blob(diagnostics_json, telemetry))
    if not sets:
        return
    args.append(trade_id)
    conn.execute(f"UPDATE paper_trades SET {', '.join(sets)} WHERE trade_id = ? AND side = 'BUY'", args)
    logger.info(
        "ENTRY_REFERENCE_TELEMETRY trade_id=%s mid=%s fill=%s maker=%s slip_from_mid=%s",
        trade_id,
        mid,
        telemetry.get("fill_price"),
        telemetry.get("is_maker"),
        slip,
    )
