"""Offline DAY decision label contract. Does not alter the live ranker.

Stores matured markouts / MFE / MAE / regret with provenance:
authoritative | reconstructed | estimated | unknown.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

TABLE_LABELS = "day_decision_outcome_labels"
PROVENANCE = frozenset({"authoritative", "reconstructed", "estimated", "unknown"})
MARKOUTS = ("15m", "30m", "1h", "2h", "4h")

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_LABELS} (
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    provenance TEXT NOT NULL,
    markout_15m_net_bps REAL,
    markout_30m_net_bps REAL,
    markout_1h_net_bps REAL,
    markout_2h_net_bps REAL,
    markout_4h_net_bps REAL,
    mfe_bps REAL,
    mae_bps REAL,
    time_to_mfe_sec REAL,
    time_to_mae_sec REAL,
    cost_cover INTEGER,
    production_exit_gross_bps REAL,
    commission_bps REAL,
    spread_bps REAL,
    slippage_bps REAL,
    production_exit_net_bps REAL,
    holding_time_sec REAL,
    capture_ratio REAL,
    exit_reason TEXT,
    regret_vs_best_eligible_bps REAL,
    regret_vs_hold_bps REAL,
    label_json TEXT,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_day_obs_labels_created ON {TABLE_LABELS}(created_at);
"""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_label_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def normalize_label(payload: dict[str, Any]) -> dict[str, Any]:
    prov = str(payload.get("provenance") or "unknown")
    if prov not in PROVENANCE:
        prov = "unknown"
    markouts = dict(payload.get("markouts") or {})
    return {
        "decision_group_id": str(payload.get("decision_group_id") or ""),
        "symbol": str(payload.get("symbol") or ""),
        "provenance": prov,
        "markouts": {k: markouts.get(k) for k in MARKOUTS},
        "mfe_bps": payload.get("mfe_bps"),
        "mae_bps": payload.get("mae_bps"),
        "time_to_mfe_sec": payload.get("time_to_mfe_sec"),
        "time_to_mae_sec": payload.get("time_to_mae_sec"),
        "cost_cover": bool(payload.get("cost_cover") or payload.get("covered_genuine_cost")),
        # Carried so a label written through the contract cannot lose the flag that separates a
        # simulated ranking loser from a real fill. The offline runner already persists it.
        "counterfactual": bool(payload.get("counterfactual")),
        "production_exit_gross_bps": payload.get("production_exit_gross_bps"),
        "commission_bps": payload.get("commission_bps"),
        "spread_bps": payload.get("spread_bps"),
        "slippage_bps": payload.get("slippage_bps"),
        "production_exit_net_bps": payload.get("production_exit_net_bps"),
        "holding_time_sec": payload.get("holding_time_sec"),
        "capture_ratio": payload.get("capture_ratio"),
        "exit_reason": payload.get("exit_reason"),
        "regret_vs_best_eligible_bps": payload.get("regret_vs_best_eligible_bps"),
        "regret_vs_hold_bps": payload.get("regret_vs_hold_bps"),
    }


def write_outcome_label(db_path: str | Path, payload: dict[str, Any]) -> None:
    """Offline writer. Fail-open. Never called from ranking."""
    row = normalize_label(payload)
    if not row["decision_group_id"] or not row["symbol"]:
        return
    try:
        ensure_label_schema(db_path)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            m = row["markouts"]
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_LABELS}(
                    decision_group_id, symbol, created_at, provenance,
                    markout_15m_net_bps, markout_30m_net_bps, markout_1h_net_bps,
                    markout_2h_net_bps, markout_4h_net_bps,
                    mfe_bps, mae_bps, time_to_mfe_sec, time_to_mae_sec, cost_cover,
                    production_exit_gross_bps, commission_bps, spread_bps, slippage_bps,
                    production_exit_net_bps, holding_time_sec, capture_ratio, exit_reason,
                    regret_vs_best_eligible_bps, regret_vs_hold_bps, label_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    row["decision_group_id"],
                    row["symbol"],
                    _now_iso(),
                    row["provenance"],
                    m.get("15m"),
                    m.get("30m"),
                    m.get("1h"),
                    m.get("2h"),
                    m.get("4h"),
                    row["mfe_bps"],
                    row["mae_bps"],
                    row["time_to_mfe_sec"],
                    row["time_to_mae_sec"],
                    1 if row["cost_cover"] else 0,
                    row["production_exit_gross_bps"],
                    row["commission_bps"],
                    row["spread_bps"],
                    row["slippage_bps"],
                    row["production_exit_net_bps"],
                    row["holding_time_sec"],
                    row["capture_ratio"],
                    row["exit_reason"],
                    row["regret_vs_best_eligible_bps"],
                    row["regret_vs_hold_bps"],
                    json.dumps(row, default=str)[:16000],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        return
