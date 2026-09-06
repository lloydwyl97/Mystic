"""V5 authoritative calibration-fill identity. Readiness only.

One counted event is one logical live production ENTRY that actually filled
from a v5 DEVELOPMENT decision group. Selections, valid labels, HOLD, exits,
and dust SELLs are never substitutes. Does not train and does not inspect
sealed 4H outcomes.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import HOLD_SYMBOL
from backend.services.day_clock_v2_labels import load_v5_label_presence
from backend.services.day_clock_v2_partition import DEVELOPMENT
from backend.services.day_decision_observability import TABLE_GROUPS
from backend.services.day_path_clock_v2_capture import TABLE_ARTIFACT

TABLE_AUDIT = "portfolio_engine_audit"

CALIBRATION_KEY = "decision_group_id+fill_trade_id"
CALIBRATION_EVENT_DEFINITION = "one logical live production ENTRY that actually filled from a v5 DEVELOPMENT decision group, keyed by decision_group_id + fill_trade_id"


def api_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper().replace("/", "").replace("-", "")


def _loads(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _truthy(raw: Any) -> bool:
    return raw in (True, 1, "1")


def _has_quote_spread(quote: dict[str, Any]) -> bool:
    if quote.get("spread_bps") is not None:
        return True
    bid, ask = quote.get("best_bid"), quote.get("best_ask")
    try:
        return bid is not None and ask is not None and float(bid) > 0 and float(ask) > 0
    except (TypeError, ValueError):
        return False


def _empty_funnel() -> dict[str, Any]:
    return {
        "authoritative_calibration_fills": 0,
        "AUTHORITATIVE_CALIBRATION_FILLS": 0,
        "non_hold_selected_groups": 0,
        "execute_authorized_groups": 0,
        "actual_logical_buy_fills": 0,
        "groups_with_authoritative_fill_linkage": 0,
        "groups_with_commission_provenance": 0,
        "groups_with_spread_provenance": 0,
        "groups_with_slippage_provenance": 0,
        "groups_with_valid_selected_v2_label": 0,
        "groups_with_complete_execution_provenance": 0,
        "v5_development_groups": 0,
        "calibration_keys": [],
        "per_symbol_calibration_support": {},
        "calibration_event_definition": CALIBRATION_EVENT_DEFINITION,
        "calibration_key": CALIBRATION_KEY,
    }


def _load_development_partitions(conn: sqlite3.Connection) -> dict[str, str]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if TABLE_ARTIFACT not in tables:
        return {}
    out: dict[str, str] = {}
    for gid, part in conn.execute(f"SELECT decision_group_id, clock_v2_partition FROM {TABLE_ARTIFACT}"):
        if gid and str(gid) not in out and part:
            out[str(gid)] = str(part)
    return out


def _load_selected_quotes(conn: sqlite3.Connection) -> dict[tuple[str, str], dict[str, Any]]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if TABLE_ARTIFACT not in tables:
        return {}
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for gid, sym, raw in conn.execute(f"SELECT decision_group_id, symbol, quote_json FROM {TABLE_ARTIFACT}"):
        out[(str(gid), api_symbol(str(sym)))] = _loads(raw)
    return out


def _load_buy_audits(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if TABLE_AUDIT not in tables:
        return {}
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_AUDIT})")}
    needed = {"action", "trade_id", "qty", "price"}
    if not needed.issubset(cols):
        return {}
    by_trade: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conn.execute(f"SELECT * FROM {TABLE_AUDIT}"):
        payload = dict(row)
        if str(payload.get("action") or "").upper() != "BUY":
            continue
        trade_id = str(payload.get("trade_id") or "").strip()
        if not trade_id:
            continue
        by_trade[trade_id].append(payload)
    return by_trade


def _aggregate_buy(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    qty = 0.0
    price = None
    fees = None
    slippage = None
    symbol = None
    for row in rows:
        try:
            qty += float(row.get("qty") or 0.0)
        except (TypeError, ValueError):
            continue
        if price is None:
            try:
                candidate = float(row.get("price"))
            except (TypeError, ValueError):
                candidate = None
            if candidate is not None and candidate > 0:
                price = candidate
        if fees is None and row.get("fees") is not None:
            fees = row.get("fees")
        if slippage is None and row.get("slippage") is not None:
            slippage = row.get("slippage")
        if symbol is None:
            symbol = row.get("symbol")
    if qty <= 0 or price is None:
        return None
    return {"qty": qty, "price": price, "fees": fees, "slippage": slippage, "symbol": symbol}


def count_v5_authoritative_calibration_fills(db_path: str | Path) -> dict[str, Any]:
    """Count logical live BUY fills that qualify for v5 execution calibration."""
    out = _empty_funnel()
    if not db_path or not Path(db_path).exists():
        return out
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if TABLE_GROUPS not in tables:
            return out
        partitions = _load_development_partitions(conn)
        quotes = _load_selected_quotes(conn)
        audits = _load_buy_audits(conn)
        labels = load_v5_label_presence(db_path)
        groups = [dict(r) for r in conn.execute(f"SELECT * FROM {TABLE_GROUPS}")]
    finally:
        conn.close()

    out["v5_development_groups"] = sum(1 for part in partitions.values() if part == DEVELOPMENT)
    keys: set[tuple[str, str]] = set()
    per_symbol: dict[str, int] = defaultdict(int)
    for group in groups:
        gid = str(group.get("decision_group_id") or "")
        if not gid or partitions.get(gid) != DEVELOPMENT:
            continue
        selected = api_symbol(str(group.get("selected_symbol") or ""))
        action = str(group.get("selected_action") or "")
        if not selected or selected == HOLD_SYMBOL or action.upper() == HOLD_SYMBOL:
            continue
        out["non_hold_selected_groups"] += 1
        if labels.get((gid, selected)):
            out["groups_with_valid_selected_v2_label"] += 1
        if _has_quote_spread(quotes.get((gid, selected), {})):
            out["groups_with_spread_provenance"] += 1
        if not _truthy(group.get("execute_authorized")):
            continue
        out["execute_authorized_groups"] += 1
        fill_id = str(group.get("fill_trade_id") or "").strip()
        if not fill_id:
            continue
        buy = _aggregate_buy(audits.get(fill_id, []))
        if buy is None:
            continue
        if api_symbol(str(buy.get("symbol") or "")) != selected:
            continue
        out["actual_logical_buy_fills"] += 1
        out["groups_with_authoritative_fill_linkage"] += 1
        commission_ok = group.get("commission") is not None or buy.get("fees") is not None
        spread_ok = _has_quote_spread(quotes.get((gid, selected), {}))
        slippage_ok = buy.get("slippage") is not None
        if commission_ok:
            out["groups_with_commission_provenance"] += 1
        if slippage_ok:
            out["groups_with_slippage_provenance"] += 1
        if commission_ok and spread_ok and slippage_ok:
            out["groups_with_complete_execution_provenance"] += 1
        if not (commission_ok and spread_ok and slippage_ok):
            continue
        if not labels.get((gid, selected)):
            continue
        key = (gid, fill_id)
        if key in keys:
            continue
        keys.add(key)
        per_symbol[selected] += 1

    n = len(keys)
    out["authoritative_calibration_fills"] = n
    out["AUTHORITATIVE_CALIBRATION_FILLS"] = n
    out["calibration_keys"] = sorted(f"{gid}:{trade}" for gid, trade in keys)
    out["per_symbol_calibration_support"] = dict(per_symbol)
    return out


__all__ = [
    "CALIBRATION_EVENT_DEFINITION",
    "CALIBRATION_KEY",
    "api_symbol",
    "count_v5_authoritative_calibration_fills",
]
