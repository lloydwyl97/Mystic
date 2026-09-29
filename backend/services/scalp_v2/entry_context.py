"""Entry-time ranking context persisted on SCALP V2 BUY rows.

Telemetry only: written after the fill, never read by entry, sizing or exit.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

CONTEXT_KEY = "scalp_entry_context"

_ROW_KEYS = (
    "rank_score",
    "static_rank_score",
    "entry_eligible",
    "strategy_passed",
    "best_setup",
    "soft_reason",
    "hard_block",
    "selection_confidence",
    "reachability_surplus",
    "arm_penalty_mult",
    "mtf_penalty_mult",
    "regime_mismatch",
    "symbol_stall_risk",
    "microstructure_adjustment",
    "learned_adjustment",
    "selection_version",
    "EV_1s",
    "EV_5s",
    "EV_10s",
    "EV_30s",
    "EV_60s",
)
_META_KEYS = (
    "regime",
    "expected_move_pct",
    "net_edge_after_costs_pct",
    "edge_source",
    "mtf_5m_trend_pct",
    "mtf_5m_aligned",
    "mtf_15m_trend_pct",
    "mtf_15m_aligned",
    "micro_quality_mult",
    "momentum_insufficient_windows",
    "momentum_sample_age_error_sec",
)
_SIGNAL_KEYS = ("score", "confidence", "expected_move_pct", "required_target_pct", "spread_pct", "impact_pct", "depth_sufficient")
_SNAP_KEYS = ("best_bid", "best_ask", "mid", "spread_pct")


def build_entry_context(row: dict[str, Any] | None, *, cycle_ts: float) -> dict[str, Any]:
    row = row or {}
    meta = row.get("rank_meta") or {}
    ctx: dict[str, Any] = {"cycle_ts": cycle_ts}
    ctx.update({k: row.get(k) for k in _ROW_KEYS if row.get(k) is not None})
    ctx.update({k: meta.get(k) for k in _META_KEYS if meta.get(k) is not None})
    if row.get("rank_components"):
        ctx["rank_components"] = row["rank_components"]
    signal = row.get("signal")
    for k in _SIGNAL_KEYS:
        v = getattr(signal, k, None)
        if v is not None:
            ctx[f"signal_{k}"] = v
    snap = row.get("snap")
    for k in _SNAP_KEYS:
        v = getattr(snap, k, None)
        if v is not None:
            ctx[f"book_{k}"] = v
    return json.loads(json.dumps(ctx, default=str))


def persist_entry_context(db_path: str | Path, *, order_id: str, context: dict[str, Any]) -> bool:
    """Merge the context into the SCALP V2 BUY row for ``order_id``; never overwrites an existing context."""
    if not order_id:
        return False
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        found = conn.execute(
            "SELECT id, explainability_json FROM paper_trades WHERE order_id=? AND engine_id='SCALP_V2' AND UPPER(side)='BUY' ORDER BY id DESC LIMIT 1",
            (str(order_id),),
        ).fetchone()
        if found is None:
            return False
        try:
            existing = json.loads(found[1]) if found[1] else {}
        except ValueError:
            existing = {}
        if not isinstance(existing, dict) or CONTEXT_KEY in existing:
            return False
        existing[CONTEXT_KEY] = context
        conn.execute("UPDATE paper_trades SET explainability_json=? WHERE id=?", (json.dumps(existing), found[0]))
        conn.commit()
        return True
    finally:
        conn.close()
