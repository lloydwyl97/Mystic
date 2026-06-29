"""SCALP post-trade outcome attribution — separate from DAY day_outcome_attribution."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any

from backend.services.binance_scalp.config import get_scalp_config
from backend.services.scalp_block_scores import compute_block_scores_from_intelligence
from backend.services.scalp_feature_health import entry_feature_health_pass

logger = logging.getLogger(__name__)

TABLE = "scalp_outcome_attribution"

_CREATE = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    trade_id TEXT NOT NULL UNIQUE,
    symbol TEXT NOT NULL,
    scalp_setup TEXT NOT NULL DEFAULT '',
    micro_regime TEXT NOT NULL DEFAULT '',
    entry_time TEXT NOT NULL DEFAULT '',
    exit_time TEXT NOT NULL DEFAULT '',
    gross_pnl REAL,
    fees REAL,
    net_pnl_after_fees REAL,
    hold_seconds REAL,
    exit_reason TEXT NOT NULL DEFAULT '',
    outcome_reason TEXT NOT NULL DEFAULT '',
    entry_scalp_vector_json TEXT,
    entry_feature_health_json TEXT,
    entry_block_scores_json TEXT,
    setup_score REAL,
    execution_quality_score REAL,
    model_probabilities_json TEXT,
    final_scalp_selection_score REAL,
    feature_health_pass INTEGER NOT NULL DEFAULT 0,
    slippage_estimate REAL,
    realized_slippage REAL,
    spread_at_entry REAL,
    spread_at_exit REAL,
    attribution_json TEXT NOT NULL DEFAULT '{{}}',
    created_at TEXT NOT NULL DEFAULT ''
)
"""

_INDEXES = (
    f"CREATE INDEX IF NOT EXISTS ix_scalp_outcome_trade ON {TABLE}(trade_id)",
    f"CREATE INDEX IF NOT EXISTS ix_scalp_outcome_symbol ON {TABLE}(symbol, created_at)",
    f"CREATE INDEX IF NOT EXISTS ix_scalp_outcome_setup ON {TABLE}(micro_regime, scalp_setup)",
)

OUTCOME_REASONS: tuple[str, ...] = (
    "GOOD_SCALP_GOOD_ENTRY",
    "GOOD_SCALP_BAD_EXIT",
    "BAD_SCALP_SETUP",
    "ENTRY_TOO_LATE",
    "ENTRY_TOO_EARLY",
    "SPREAD_COST_TOO_HIGH",
    "SLIPPAGE_TOO_HIGH",
    "PRICE_IMPACT_TOO_HIGH",
    "MOMENTUM_FADED",
    "VOLUME_BURST_FAILED",
    "MICRO_REGIME_SHIFT_AGAINST_TRADE",
    "LIQUIDITY_VANISHED",
    "EXIT_TOO_EARLY",
    "EXIT_TOO_LATE",
    "FEATURE_HEALTH_WEAK",
)


def ensure_scalp_outcome_attribution_table(db_path: str | None = None) -> None:
    path = db_path or get_scalp_config().database_path
    with sqlite3.connect(path) as conn:
        conn.execute(_CREATE)
        for stmt in _INDEXES:
            conn.execute(stmt)
        conn.commit()


def classify_scalp_outcome(
    *,
    intelligence: dict[str, Any],
    net_pnl: float,
    hold_seconds: float,
    exit_reason: str,
) -> str:
    ex = intelligence or {}
    if not entry_feature_health_pass(ex):
        return "FEATURE_HEALTH_WEAK"
    spread_in = float(ex.get("spread_at_entry") or ex.get("spread_pct") or 0.0)
    spread_out = float(ex.get("spread_at_exit") or 0.0)
    slip = float(ex.get("realized_slippage") or ex.get("slippage_estimate") or 0.0)
    impact = float(ex.get("impact_pct") or 0.0)
    if spread_in > 0.003 or spread_out > 0.004:
        return "SPREAD_COST_TOO_HIGH"
    if slip > 0.002:
        return "SLIPPAGE_TOO_HIGH"
    if impact > 0.004 and net_pnl <= 0:
        return "PRICE_IMPACT_TOO_HIGH"
    setup_score = float(ex.get("setup_score") or 0.5)
    if setup_score < 0.35 and net_pnl <= 0:
        return "BAD_SCALP_SETUP"
    mom = float(ex.get("mid_change_30s") or 0.0)
    if mom < 0 and net_pnl <= 0:
        return "MOMENTUM_FADED"
    vol = float(ex.get("kline_volume_ratio") or 1.0)
    if vol < 0.8 and net_pnl <= 0:
        return "VOLUME_BURST_FAILED"
    trans = float(ex.get("scalp_regime_transition_score") or 0.0)
    if trans < 0.25 and net_pnl < 0:
        return "MICRO_REGIME_SHIFT_AGAINST_TRADE"
    if net_pnl > 0 and hold_seconds < 30:
        return "EXIT_TOO_EARLY"
    if net_pnl < 0 and hold_seconds > 900:
        return "EXIT_TOO_LATE"
    if net_pnl > 0:
        return "GOOD_SCALP_GOOD_ENTRY"
    return "GOOD_SCALP_BAD_EXIT"


def build_attribution_payload(
    *,
    trade_id: str,
    symbol: str,
    intelligence: dict[str, Any],
    gross_pnl: float,
    fees: float,
    net_pnl: float,
    hold_seconds: float,
    exit_reason: str,
) -> dict[str, Any]:
    ex = dict(intelligence or {})
    blocks = compute_block_scores_from_intelligence(ex)
    outcome = classify_scalp_outcome(
        intelligence=ex,
        net_pnl=net_pnl,
        hold_seconds=hold_seconds,
        exit_reason=exit_reason,
    )
    fh_pass = entry_feature_health_pass(ex)
    return {
        "trade_id": trade_id,
        "symbol": symbol,
        "scalp_setup": str(ex.get("scalp_setup") or ex.get("setup_name") or ""),
        "micro_regime": str(ex.get("micro_regime") or ""),
        "entry_time": str(ex.get("entry_time") or ""),
        "exit_time": datetime.now(timezone.utc).isoformat(),
        "gross_pnl": gross_pnl,
        "fees": fees,
        "net_pnl_after_fees": net_pnl,
        "hold_seconds": hold_seconds,
        "exit_reason": exit_reason,
        "outcome_reason": outcome,
        "entry_scalp_vector": list(ex.get("entry_scalp_vector") or []),
        "entry_feature_health": json.loads(ex.get("feature_health_json") or "{}") if isinstance(ex.get("feature_health_json"), str) else ex.get("entry_feature_health") or {},
        "entry_block_scores": blocks,
        "setup_score": float(ex.get("setup_score") or 0.5),
        "execution_quality_score": float(ex.get("scalp_execution_quality_score") or ex.get("execution_quality_score") or 0.5),
        "model_probabilities": {
            "signal_score": ex.get("signal_score"),
            "signal_confidence": ex.get("signal_confidence"),
        },
        "final_scalp_selection_score": float(ex.get("final_scalp_selection_score") or ex.get("signal_score") or 0.0),
        "feature_health_pass": fh_pass,
        "slippage_estimate": float(ex.get("slippage_estimate") or 0.0),
        "realized_slippage": float(ex.get("realized_slippage") or 0.0),
        "spread_at_entry": float(ex.get("spread_at_entry") or ex.get("spread_pct") or 0.0),
        "spread_at_exit": float(ex.get("spread_at_exit") or 0.0),
    }


def record_scalp_outcome_attribution(
    *,
    trade_id: str,
    symbol: str,
    intelligence: dict[str, Any] | None,
    gross_pnl: float,
    fees: float,
    net_pnl: float,
    hold_seconds: float,
    exit_reason: str,
    db_path: str | None = None,
) -> int | None:
    if not trade_id:
        return None
    path = db_path or get_scalp_config().database_path
    try:
        ensure_scalp_outcome_attribution_table(path)
        payload = build_attribution_payload(
            trade_id=trade_id,
            symbol=symbol,
            intelligence=intelligence or {},
            gross_pnl=gross_pnl,
            fees=fees,
            net_pnl=net_pnl,
            hold_seconds=hold_seconds,
            exit_reason=exit_reason,
        )
        created = payload.get("exit_time") or datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(path) as conn:
            cur = conn.execute(
                f"""
                INSERT INTO {TABLE} (
                    trade_id, symbol, scalp_setup, micro_regime, entry_time, exit_time,
                    gross_pnl, fees, net_pnl_after_fees, hold_seconds, exit_reason, outcome_reason,
                    entry_scalp_vector_json, entry_feature_health_json, entry_block_scores_json,
                    setup_score, execution_quality_score, model_probabilities_json,
                    final_scalp_selection_score, feature_health_pass,
                    slippage_estimate, realized_slippage, spread_at_entry, spread_at_exit,
                    attribution_json, created_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(trade_id) DO UPDATE SET
                    net_pnl_after_fees=excluded.net_pnl_after_fees,
                    outcome_reason=excluded.outcome_reason,
                    attribution_json=excluded.attribution_json
                """,
                (
                    trade_id,
                    symbol,
                    payload["scalp_setup"],
                    payload["micro_regime"],
                    payload["entry_time"],
                    payload["exit_time"],
                    payload["gross_pnl"],
                    payload["fees"],
                    payload["net_pnl_after_fees"],
                    payload["hold_seconds"],
                    payload["exit_reason"],
                    payload["outcome_reason"],
                    json.dumps(payload.get("entry_scalp_vector") or []),
                    json.dumps(payload.get("entry_feature_health") or {}),
                    json.dumps(payload.get("entry_block_scores") or {}),
                    payload["setup_score"],
                    payload["execution_quality_score"],
                    json.dumps(payload.get("model_probabilities") or {}),
                    payload["final_scalp_selection_score"],
                    1 if payload["feature_health_pass"] else 0,
                    payload["slippage_estimate"],
                    payload["realized_slippage"],
                    payload["spread_at_entry"],
                    payload["spread_at_exit"],
                    json.dumps(payload, separators=(",", ":")),
                    created,
                ),
            )
            conn.commit()
            return int(cur.lastrowid or 0) or None
    except Exception as exc:
        logger.warning("record_scalp_outcome_attribution failed %s: %s", trade_id, exc)
        return None


def dry_run_insert_and_delete(db_path: str | None = None) -> dict[str, Any]:
    path = db_path or get_scalp_config().database_path
    test_id = "__scalp_schema_verify__"
    record_scalp_outcome_attribution(
        trade_id=test_id,
        symbol="BTCUSDT",
        intelligence={
            "scalp_setup": "MICRO_BREAKOUT",
            "micro_regime": "bull_trend",
            "setup_score": 0.7,
            "scalp_execution_quality_score": 0.8,
            "feature_health_pass": True,
            "feature_health_json": json.dumps({"pass": True, "health_pct": 90}),
            "entry_scalp_vector": [0.1] * 40,
            "spread_pct": 0.0005,
        },
        gross_pnl=1.0,
        fees=0.1,
        net_pnl=0.9,
        hold_seconds=120.0,
        exit_reason="NET_PROFIT_TARGET",
        db_path=path,
    )
    with sqlite3.connect(path) as conn:
        row = conn.execute(f"SELECT trade_id, scalp_setup FROM {TABLE} WHERE trade_id=?", (test_id,)).fetchone()
        conn.execute(f"DELETE FROM {TABLE} WHERE trade_id=?", (test_id,))
        conn.commit()
    return {"read_back_ok": bool(row), "scalp_setup": row[1] if row else None, "deleted_test_row": True}


__all__ = [
    "OUTCOME_REASONS",
    "dry_run_insert_and_delete",
    "ensure_scalp_outcome_attribution_table",
    "record_scalp_outcome_attribution",
]
