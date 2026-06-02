"""
Profit-after-cost (PAC) validation metrics for model promotion candidates.

Reads filtered ai_outcome_training_rows for top-4 DAY symbols only.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables

FEATURE_VERSION_DAY_HTF = 5
FEATURE_DIM_V2 = 145
MIN_PAC_SAMPLES = 5


def _symbol_forms(symbol_bus: str) -> tuple[str, str]:
    bus = (symbol_bus or "").strip().upper()
    if "/" in bus:
        ccxt = bus
        if not bus.endswith("USDT"):
            return bus, bus
        bus = bus.replace("/", "")
    else:
        ccxt = f"{bus[:-4]}/USDT" if bus.endswith("USDT") else f"{bus}/USDT"
    return bus, ccxt


def _row_net_pnl(row: sqlite3.Row | dict[str, Any]) -> float | None:
    for key in ("net_pnl_pct", "actual_net_outcome", "realized_pct"):
        raw = row[key] if isinstance(row, sqlite3.Row) else row.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def _row_feature_version(row: sqlite3.Row | dict[str, Any]) -> int:
    ctx_raw = row["context_json"] if isinstance(row, sqlite3.Row) else row.get("context_json")
    ctx_fv = 0
    if ctx_raw:
        try:
            ctx = json.loads(ctx_raw) if isinstance(ctx_raw, str) else {}
            if isinstance(ctx, dict):
                ctx_fv = int(ctx.get("_feature_version") or ctx.get("feature_version") or 0)
        except (json.JSONDecodeError, TypeError, ValueError):
            ctx_fv = 0
    dim_fv = 0
    fj = row["features_json"] if isinstance(row, sqlite3.Row) else row.get("features_json")
    if fj:
        try:
            feats = json.loads(fj) if isinstance(fj, str) else fj
            if isinstance(feats, list):
                if len(feats) >= FEATURE_DIM_V2:
                    dim_fv = FEATURE_VERSION_DAY_HTF
                elif len(feats) >= 124:
                    dim_fv = 1
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return max(ctx_fv, dim_fv)


def _row_passes_filters(row: sqlite3.Row, *, symbol_bus: str, min_fv: int, min_dim: int) -> bool:
    bus, ccxt = _symbol_forms(symbol_bus)
    sym = str(row["symbol"] or "").strip().upper()
    if sym not in (bus, ccxt, bus.replace("USDT", "/USDT")):
        return False
    if str(row["strategy_id"] or "").strip().lower() not in ("", "day"):
        return False
    oc = str(row["outcome_class"] or "").strip().upper()
    if oc in ("ADMIN", "VOID", "VOIDED"):
        return False
    gb = str(row["good_bad_memory_class"] or "").strip().upper()
    if gb == "ADMIN":
        return False
    if int(row["churn_flag"] or 0) == 1:
        return False
    if not row["features_json"]:
        return False
    try:
        feats = json.loads(row["features_json"])
        if not isinstance(feats, list) or len(feats) != min_dim:
            return False
    except (json.JSONDecodeError, TypeError, ValueError):
        return False
    if _row_feature_version(row) < min_fv:
        return False
    if _row_net_pnl(row) is None:
        return False
    return True


def build_pac_validation_metrics(
    *,
    strategy_id: str,
    symbol_bus: str,
    candidate_accuracy: float,
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
    db_path: str = DATABASE_PATH,
    min_samples: int = MIN_PAC_SAMPLES,
    active_accuracy: float | None = None,
    rf_val_samples: int | None = None,
) -> dict[str, Any]:
    """
    Build promotion validation metrics from filtered DAY outcome rows for one top-4 symbol.
    """
    sid = (strategy_id or "day").strip().lower()
    bus, ccxt = _symbol_forms(symbol_bus)
    if bus not in TRADING_SYMBOLS:
        return {
            "symbol": bus,
            "strategy": sid,
            "feature_version": int(feature_version),
            "feature_dim": int(feature_dim),
            "candidate_accuracy": round(float(candidate_accuracy), 6),
            "active_accuracy": round(float(active_accuracy), 6) if active_accuracy is not None else None,
            "profit_after_cost": None,
            "avg_net_pnl_pct": None,
            "avg_actual_net": None,
            "win_rate_after_cost": None,
            "bad_trade_rate": None,
            "sample_count": 0,
            "pac_status": "PAC_UNAVAILABLE",
            "pac_source": "ai_outcome_training_rows",
        }

    ensure_ai_canonical_tables(db_path)
    nets: list[float] = []
    actuals: list[float] = []
    wins = 0
    bad = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT symbol, strategy_id, outcome_class, good_bad_memory_class, churn_flag,
                   features_json, context_json, net_pnl_pct, actual_net_outcome, realized_pct
            FROM ai_outcome_training_rows
            WHERE strategy_id = ?
              AND symbol IN (?, ?)
              AND features_json IS NOT NULL
            ORDER BY id DESC
            """,
            (sid, ccxt, bus),
        ).fetchall()

    for row in rows:
        if not _row_passes_filters(row, symbol_bus=bus, min_fv=feature_version, min_dim=feature_dim):
            continue
        net = _row_net_pnl(row)
        if net is None:
            continue
        nets.append(net)
        act = row["actual_net_outcome"]
        if act is not None:
            try:
                actuals.append(float(act))
            except (TypeError, ValueError):
                actuals.append(net)
        else:
            actuals.append(net)
        gb = str(row["good_bad_memory_class"] or "").strip().upper()
        if net > 0:
            wins += 1
        if gb == "BAD" or net <= 0:
            bad += 1

    sample_count = len(nets)
    if sample_count < min_samples:
        pac_status = "PAC_UNAVAILABLE"
        profit_after_cost = None
        avg_net = None
        avg_actual = None
        win_rate = None
        bad_rate = None
    else:
        pac_status = "OK"
        profit_after_cost = sum(nets) / sample_count
        avg_net = profit_after_cost
        avg_actual = sum(actuals) / len(actuals)
        win_rate = wins / sample_count
        bad_rate = bad / sample_count

    out: dict[str, Any] = {
        "symbol": bus,
        "strategy": sid,
        "feature_version": int(feature_version),
        "feature_dim": int(feature_dim),
        "candidate_accuracy": round(float(candidate_accuracy), 6),
        "active_accuracy": round(float(active_accuracy), 6) if active_accuracy is not None else None,
        "candidate_profit_after_cost": round(profit_after_cost, 6) if profit_after_cost is not None else None,
        "profit_after_cost": round(profit_after_cost, 6) if profit_after_cost is not None else None,
        "avg_net_pnl_pct": round(avg_net, 6) if avg_net is not None else None,
        "avg_actual_net": round(avg_actual, 6) if avg_actual is not None else None,
        "win_rate_after_cost": round(win_rate, 6) if win_rate is not None else None,
        "bad_trade_rate": round(bad_rate, 6) if bad_rate is not None else None,
        "sample_count": int(sample_count),
        "pac_status": pac_status,
        "pac_source": "ai_outcome_training_rows",
        "pac_columns": "net_pnl_pct|actual_net_outcome|realized_pct",
    }
    if rf_val_samples is not None:
        out["rf_val_samples"] = int(rf_val_samples)
    return out


__all__ = [
    "FEATURE_DIM_V2",
    "FEATURE_VERSION_DAY_HTF",
    "MIN_PAC_SAMPLES",
    "build_pac_validation_metrics",
]
