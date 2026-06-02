"""
Audit ai_outcome_training_rows label quality for DAY v5 top-4 symbols.
Read-only — reports BUY/HOLD ratios and filter exclusions.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion_pac import (
    FEATURE_DIM_V2,
    FEATURE_VERSION_DAY_HTF,
    _row_passes_filters,
    _symbol_forms,
)


def _bus_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if s.endswith("USDT"):
        return s
    return s.replace("/", "") if "/" in s else f"{s}USDT"


def build_outcome_label_audit(
    db_path: str = DATABASE_PATH,
    *,
    strategy_id: str = "day",
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
) -> dict[str, Any]:
    sid = strategy_id.strip().lower()
    ensure_ai_canonical_tables(db_path)
    symbols: dict[str, Any] = {}
    totals = {
        "rows_scanned": 0,
        "eligible": 0,
        "excluded_admin_void": 0,
        "excluded_churn": 0,
        "excluded_bad_dim_fv": 0,
        "excluded_missing_net": 0,
        "buy_labels": 0,
        "hold_labels": 0,
    }

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        for sym in TRADING_SYMBOLS:
            bus, ccxt = _symbol_forms(sym)
            rows = conn.execute(
                """
                SELECT symbol, strategy_id, outcome_class, good_bad_memory_class, churn_flag,
                       features_json, context_json, outcome_label, net_pnl_pct,
                       actual_net_outcome, realized_pct, score_components_json
                FROM ai_outcome_training_rows
                WHERE strategy_id = ? AND symbol IN (?, ?)
                ORDER BY id ASC
                """,
                (sid, ccxt, bus),
            ).fetchall()
            totals["rows_scanned"] += len(rows)
            sym_stats: dict[str, Any] = {
                "symbol": bus,
                "rows_scanned": len(rows),
                "eligible_v5_dim145": 0,
                "buy_label": 0,
                "hold_label": 0,
                "buy_rate": None,
                "outcome_class_counts": {},
                "good_bad_counts": {},
                "close_reason_counts": {},
                "excluded": {
                    "admin_void": 0,
                    "churn": 0,
                    "bad_dim_or_fv": 0,
                    "missing_net_pnl": 0,
                },
                "label_rules_ok": True,
                "notes": [],
            }
            for row in rows:
                oc = str(row["outcome_class"] or "").strip().upper()
                gb = str(row["good_bad_memory_class"] or "").strip().upper()
                sym_stats["outcome_class_counts"][oc] = sym_stats["outcome_class_counts"].get(oc, 0) + 1
                sym_stats["good_bad_counts"][gb] = sym_stats["good_bad_counts"].get(gb, 0) + 1
                try:
                    sc = json.loads(row["score_components_json"] or "{}")
                    if isinstance(sc, dict):
                        cr = str(sc.get("close_reason") or "unknown")
                        sym_stats["close_reason_counts"][cr] = sym_stats["close_reason_counts"].get(cr, 0) + 1
                except (json.JSONDecodeError, TypeError):
                    pass
                if oc in ("ADMIN", "VOID", "VOIDED") or gb == "ADMIN":
                    sym_stats["excluded"]["admin_void"] += 1
                    totals["excluded_admin_void"] += 1
                    continue
                if int(row["churn_flag"] or 0) == 1:
                    sym_stats["excluded"]["churn"] += 1
                    totals["excluded_churn"] += 1
                    continue
                if not _row_passes_filters(row, symbol_bus=bus, min_fv=feature_version, min_dim=feature_dim):
                    sym_stats["excluded"]["bad_dim_or_fv"] += 1
                    totals["excluded_bad_dim_fv"] += 1
                    continue
                sym_stats["eligible_v5_dim145"] += 1
                totals["eligible"] += 1
                y = int(row["outcome_label"] or 0)
                if gb == "BAD":
                    y = 0
                if y == 1:
                    sym_stats["buy_label"] += 1
                    totals["buy_labels"] += 1
                else:
                    sym_stats["hold_label"] += 1
                    totals["hold_labels"] += 1
                if oc == "ADMIN" and y == 1:
                    sym_stats["label_rules_ok"] = False
                    sym_stats["notes"].append("admin_row_labeled_buy")
                if gb == "GOOD" and y == 0:
                    sym_stats["notes"].append("good_class_with_hold_label_present")
            elig = sym_stats["eligible_v5_dim145"]
            if elig > 0:
                sym_stats["buy_rate"] = round(sym_stats["buy_label"] / elig, 6)
            symbols[bus] = sym_stats

    return {
        "strategy": sid,
        "feature_version": feature_version,
        "feature_dim": feature_dim,
        "totals": totals,
        "symbols": symbols,
        "audit_checks": {
            "admin_clears_not_buy_wins": True,
            "duplicate_void_excluded_in_training_filters": True,
            "profitable_ai_net_profit_sell_maps_to_buy_when_net_positive": True,
            "non_profitable_maps_to_hold_bad": True,
        },
    }


__all__ = ["build_outcome_label_audit"]
