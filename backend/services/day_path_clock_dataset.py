"""Decision-group research dataset for clock-consistent path modeling.

Unit of observation is one ranking group: BTC, ETH, SOL, XRP, HOLD.
Sealed-lock groups may be listed for provenance but never receive outcome labels.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.services.day_4h_entry_features import COINS, HOLD_SYMBOL
from backend.services.day_forward_lock import FORWARD_LOCK_START, HISTORICAL_66_END, HISTORICAL_66_START
from backend.services.day_forward_lock import TABLE as TABLE_LOCK
from backend.services.day_path_clock_features import build_clock_features, parse_as_of
from backend.services.day_path_clock_labels import build_clock_labels
from backend.services.day_path_clock_v2 import PRIMARY_TARGET, SCHEMA_VERSION, clock_challenger_export_schema

RESEARCH_GROUPS_TABLE = "day_path_clock_research_groups"


def load_asof_1m_bars(db_path: str | Path, symbol: str, as_of: Any, *, limit: int = 240) -> list[dict[str, Any]]:
    """Point-in-time 1m rows with ts <= as_of. Read-only."""
    when = parse_as_of(as_of)
    if when is None or not Path(db_path).exists():
        return []
    names = [symbol, symbol.replace("/", "-"), symbol.replace("/", ""), symbol.replace("-", "/")]
    seen: set[str] = set()
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        rows: list[tuple[Any, ...]] = []
        for name in names:
            if not name or name in seen:
                continue
            seen.add(name)
            rows = conn.execute(
                """
                SELECT open, high, low, close, volume, ts
                FROM feature_ohlcv
                WHERE interval='1m' AND symbol=? AND ts<=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (name, when.isoformat(), int(limit)),
            ).fetchall()
            if rows:
                break
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = []
    for o, h, low, c, v, ts in reversed(rows):
        out.append({"open": o, "high": h, "low": low, "close": c, "volume": v, "ts": ts})
    return out


def _utc_iso(ts: Any) -> str | None:
    parsed = parse_as_of(ts)
    return parsed.isoformat() if parsed else None


def lock_window_status(db_path: str | Path | None = None) -> dict[str, Any]:
    out = {
        "experiment_id": None,
        "dataset_cutoff": FORWARD_LOCK_START,
        "inspected": False,
        "historical_66_excluded": True,
        "historical_66_window": [HISTORICAL_66_START, HISTORICAL_66_END],
        "locks": 0,
    }
    if not db_path or not Path(db_path).exists():
        return out
    conn = sqlite3.connect(f"file:{Path(db_path)}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if TABLE_LOCK not in tables:
            return out
        rows = list(conn.execute(f"SELECT experiment_id, dataset_cutoff, inspected, meta_json FROM {TABLE_LOCK}"))
    finally:
        conn.close()
    if not rows:
        return out
    latest = rows[-1]
    meta = {}
    try:
        meta = json.loads(latest[3] or "{}")
    except (TypeError, ValueError):
        meta = {}
    out.update(
        {
            "experiment_id": latest[0],
            "dataset_cutoff": latest[1] or FORWARD_LOCK_START,
            "inspected": bool(latest[2]),
            "historical_66_excluded": bool(meta.get("historical_66_excluded", True)),
            "historical_66_window": meta.get("historical_66_window") or [HISTORICAL_66_START, HISTORICAL_66_END],
            "locks": len(rows),
        }
    )
    return out


def in_sealed_lock(decision_ts: Any, *, cutoff: str = FORWARD_LOCK_START) -> bool:
    when = parse_as_of(decision_ts)
    edge = parse_as_of(cutoff)
    if when is None or edge is None:
        return False
    return when >= edge


def build_group_record(
    *,
    decision_group_id: str,
    decision_ts: Any,
    bars_by_symbol: dict[str, Any],
    context_by_symbol: dict[str, dict[str, Any]] | None = None,
    eligible_by_symbol: dict[str, bool] | None = None,
    quotes_by_symbol: dict[str, dict[str, Any]] | None = None,
    production_exits: dict[str, dict[str, Any]] | None = None,
    lock_cutoff: str = FORWARD_LOCK_START,
    now: datetime | None = None,
    attach_labels: bool = True,
) -> dict[str, Any]:
    locked = in_sealed_lock(decision_ts, cutoff=lock_cutoff)
    allow_labels = bool(attach_labels) and not locked
    allow_production = allow_labels and bool(production_exits)
    ctx = context_by_symbol or {}
    elig = eligible_by_symbol or {}
    quotes = quotes_by_symbol or {}
    exits = production_exits or {}
    btc_bars = bars_by_symbol.get("BTCUSDT") or bars_by_symbol.get("BTC/USDT")
    candidates: list[dict[str, Any]] = []
    for symbol in (*COINS, HOLD_SYMBOL):
        raw_bars = [] if symbol == HOLD_SYMBOL else (bars_by_symbol.get(symbol) or [])
        info = ctx.get(symbol) or {}
        feats = build_clock_features(
            raw_bars,
            as_of=decision_ts,
            symbol=symbol,
            btc_bars=btc_bars,
            p_buy=info.get("p_buy"),
            legacy_path_ev=info.get("legacy_path_ev") if symbol != HOLD_SYMBOL else 0.0,
            final_rank_score=info.get("final_rank_score") if symbol != HOLD_SYMBOL else 0.0,
            structure=info.get("structure"),
            quote_spread_bps=(quotes.get(symbol) or {}).get("spread_bps"),
        )
        label = None
        if allow_labels:
            exit_row = exits.get(symbol) if allow_production else None
            label = build_clock_labels(
                raw_bars,
                decision_ts=decision_ts,
                symbol=symbol,
                cost_bps=float(feats.get("estimated_all_in_cost_bps") or 0.0),
                commission_bps=float(feats.get("commission_rt_bps") or 0.0),
                spread_bps=float(feats.get("spread_bps") or 0.0),
                slippage_bps=float(feats.get("expected_slippage_bps") or 0.0),
                entry_px=(quotes.get(symbol) or {}).get("entry_px"),
                production_exit_net_bps=(exit_row or {}).get("production_exit_net_bps") if symbol != HOLD_SYMBOL else 0.0,
                production_exit_reason=(exit_row or {}).get("exit_reason") if symbol != HOLD_SYMBOL else None,
                now=now,
            )
        elif symbol == HOLD_SYMBOL:
            label = build_clock_labels([], decision_ts=decision_ts, symbol=HOLD_SYMBOL)
        candidates.append(
            {
                "symbol": symbol,
                "eligible": True if symbol == HOLD_SYMBOL else bool(elig.get(symbol, True)),
                "features": feats,
                "quotes": quotes.get(symbol) or {},
                "cost_assumptions": {
                    "commission_rt_bps": feats.get("commission_rt_bps"),
                    "spread_bps": feats.get("spread_bps"),
                    "slippage_bps": feats.get("expected_slippage_bps"),
                    "all_in_cost_bps": feats.get("estimated_all_in_cost_bps"),
                },
                "label": label,
                "provenance": {
                    "schema_version": SCHEMA_VERSION,
                    "lock_excluded": locked,
                    "labels_attached": allow_labels or symbol == HOLD_SYMBOL,
                    "production_exit_attached": bool(allow_production and symbol != HOLD_SYMBOL and exits.get(symbol)),
                },
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_group_id": str(decision_group_id),
        "decision_timestamp": _utc_iso(decision_ts),
        "primary_unit": "decision_group",
        "lock_excluded": locked,
        "lock_cutoff": lock_cutoff,
        "target": PRIMARY_TARGET,
        "challenger_inputs": clock_challenger_export_schema()["inputs"],
        "candidates": candidates,
    }


def dataset_counts(groups: list[dict[str, Any]]) -> dict[str, Any]:
    labeled = [g for g in groups if not g.get("lock_excluded")]
    locked = [g for g in groups if g.get("lock_excluded")]
    available = 0
    for group in labeled:
        for cand in group.get("candidates") or []:
            if cand.get("symbol") == HOLD_SYMBOL:
                continue
            if (cand.get("features") or {}).get("feature_available"):
                available += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_groups": len(groups),
        "research_labeled_groups": len(labeled),
        "lock_excluded_groups": len(locked),
        "candidate_rows_including_hold": sum(len(g.get("candidates") or []) for g in groups),
        "independent_decisions": len(groups),
        "clock_feature_available_coin_rows": available,
        "primary_unit": "decision_group",
    }


def assert_group_integrity(group: dict[str, Any]) -> None:
    symbols = [c["symbol"] for c in group.get("candidates") or []]
    if symbols != [*COINS, HOLD_SYMBOL]:
        raise ValueError(f"group must contain BTC/ETH/SOL/XRP/HOLD in order, got {symbols}")
    hold = next(c for c in group["candidates"] if c["symbol"] == HOLD_SYMBOL)
    if hold.get("label") and float(hold["label"].get("clock_net_bps", {}).get("4h") or 0.0) != 0.0:
        raise ValueError("HOLD label must be zero")
    if group.get("lock_excluded"):
        for cand in group["candidates"]:
            if cand["symbol"] == HOLD_SYMBOL:
                continue
            if cand.get("label") not in (None, {}):
                raise ValueError("sealed lock groups must not carry coin outcome labels")
            if (cand.get("provenance") or {}).get("production_exit_attached"):
                raise ValueError("sealed lock groups must not carry production exits")
