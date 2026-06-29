"""
DAY entry quality evaluation — setup_credit and RS floor/rank.

Observation-only by default; hard blocks only when env gates in day_entry_gates.py are enabled.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from backend.config.day_entry_gates import (
    day_entry_gates_config_snapshot,
    day_entry_gates_enforced,
    day_entry_rs_floor,
    day_entry_rs_rank_max,
    day_require_setup_credit_enabled,
)
from backend.config.trading_universe import DAY_TRADE_SYMBOLS

logger = logging.getLogger(__name__)

LAST_BAR_OPERATIONAL_KEY = "day_entry_quality_last_bar"

REJECT_SETUP_CREDIT_REQUIRED = "SETUP_CREDIT_REQUIRED"
REJECT_RS_BELOW_FLOOR = "RS_BELOW_FLOOR"
REJECT_RS_RANK_TOO_LOW = "RS_RANK_TOO_LOW"


def _norm_bus(symbol: str) -> str:
    return (symbol or "").replace("/", "").strip().upper()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        if val is None or str(val).strip() == "":
            return default
        return float(val)
    except (TypeError, ValueError):
        return default


def build_basket_rs_ranks() -> tuple[list[dict[str, Any]], dict[str, int]]:
    """RS rank among top-4 from live Redis ai_signal hashes (1 = strongest)."""
    rows: list[dict[str, Any]] = []
    try:
        from backend.config.redis_config import get_redis_client
        from backend.services.live_strategy_contracts import redis_ai_signal_key

        r = get_redis_client()
        if not r:
            return rows, {}
        for sym in DAY_TRADE_SYMBOLS:
            bus = _norm_bus(sym)
            raw = r.hgetall(redis_ai_signal_key("day", bus)) or {}
            dd = {(k.decode() if isinstance(k, bytes) else str(k)): (v.decode() if isinstance(v, bytes) else str(v)) for k, v in raw.items()}
            rs = _safe_float(dd.get("ctx_rs_btc"), _safe_float(dd.get("ctx_rs_eth"), 0.0))
            conf = _safe_float(dd.get("winner_probability"), _safe_float(dd.get("confidence"), 0.0))
            rows.append({"symbol": bus, "ctx_rs_btc": rs, "confidence": conf})
        rows.sort(key=lambda x: (x.get("ctx_rs_btc", 0), x.get("confidence", 0)), reverse=True)
    except Exception as exc:
        logger.debug("build_basket_rs_ranks failed: %s", exc)
        return rows, {}

    ranks = {_norm_bus(str(r["symbol"])): i + 1 for i, r in enumerate(rows)}
    return rows, ranks


def evaluate_entry_quality(
    decision_data: dict[str, Any] | None,
    *,
    symbol: str,
    basket_rs_ranks: dict[str, int] | None = None,
) -> tuple[bool, str | None, dict[str, Any]]:
    """
    Evaluate setup_credit / RS constraints.

    Returns (ok, reject_code_or_None, detail).
    """
    dd = decision_data or {}
    detail: dict[str, Any] = {
        "gates": day_entry_gates_config_snapshot(),
        "symbol": _norm_bus(symbol),
    }
    setup_credit = _safe_float(dd.get("setup_credit"), 0.0)
    strong_setup = bool(dd.get("symbol_trust_setup_strong"))
    rs = _safe_float(dd.get("ctx_rs_btc"), _safe_float(dd.get("ctx_rs_eth"), 0.0))
    detail["setup_credit"] = setup_credit
    detail["strong_setup"] = strong_setup
    detail["ctx_rs_btc"] = rs

    if basket_rs_ranks is None:
        _, basket_rs_ranks = build_basket_rs_ranks()
    bus = _norm_bus(symbol)
    rs_rank = int(basket_rs_ranks.get(bus, len(DAY_TRADE_SYMBOLS)))
    detail["rs_rank"] = rs_rank

    would_block: list[str] = []
    # ML_EDGE or explicit ml_enriched signals (positive model edge) are allowed to trade for learning even without classic strong_setup credit.
    is_ml_edge = bool(dd.get("ml_enriched") or str(dd.get("strategy_family", "")).upper() == "ML_EDGE" or str(dd.get("live_ai_strategy", "")).lower() == "day")
    if setup_credit <= 0.0 and not strong_setup and not is_ml_edge:
        would_block.append(REJECT_SETUP_CREDIT_REQUIRED)

    rs_floor = day_entry_rs_floor()
    if rs_floor is not None:
        detail["rs_floor"] = rs_floor
        if rs < rs_floor:
            would_block.append(REJECT_RS_BELOW_FLOOR)

    rank_max = day_entry_rs_rank_max()
    if rank_max is not None:
        detail["rs_rank_max"] = rank_max
        if rs_rank > rank_max:
            would_block.append(REJECT_RS_RANK_TOO_LOW)

    detail["would_block"] = would_block
    detail["enforced"] = day_entry_gates_enforced()

    blockers: list[str] = []
    if day_require_setup_credit_enabled() and REJECT_SETUP_CREDIT_REQUIRED in would_block:
        blockers.append(REJECT_SETUP_CREDIT_REQUIRED)
    if REJECT_RS_BELOW_FLOOR in would_block:
        blockers.append(REJECT_RS_BELOW_FLOOR)
    if REJECT_RS_RANK_TOO_LOW in would_block:
        blockers.append(REJECT_RS_RANK_TOO_LOW)

    if blockers:
        return False, blockers[0], detail
    return True, None, detail


def persist_last_bar_evaluation(
    *,
    symbol: str,
    bar_timestamp: int,
    ok: bool,
    reject_code: str | None,
    detail: dict[str, Any],
    db_path: str | None = None,
) -> None:
    try:
        import json
        import sqlite3

        from backend.database_schema import DATABASE_PATH

        blob = json.dumps(
            {
                "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                "bar_timestamp": int(bar_timestamp),
                "symbol": _norm_bus(symbol),
                "gate_ok": bool(ok),
                "reject_code": reject_code,
                "detail": detail,
            },
            separators=(",", ":"),
        )
        path = db_path or DATABASE_PATH
        with sqlite3.connect(path) as conn:
            conn.execute(
                """
                INSERT INTO operational_state(key, value_json, updated_ts)
                VALUES(?, ?, strftime('%s','now'))
                ON CONFLICT(key) DO UPDATE SET
                  value_json=excluded.value_json,
                  updated_ts=excluded.updated_ts
                """,
                (LAST_BAR_OPERATIONAL_KEY, blob),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("persist_last_bar_evaluation failed: %s", exc)


def load_last_bar_evaluation(db_path: str | None = None) -> dict[str, Any] | None:
    try:
        import json
        import sqlite3

        from backend.database_schema import DATABASE_PATH

        path = db_path or DATABASE_PATH
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT value_json FROM operational_state WHERE key=?",
                (LAST_BAR_OPERATIONAL_KEY,),
            ).fetchone()
        if not row or not row[0]:
            return None
        return json.loads(row[0])
    except Exception:
        return None


def build_entry_quality_telemetry() -> dict[str, Any]:
    """Snapshot for day-health / operator dashboards (RS + gate config; setup_credit at bar rank)."""
    basket, ranks = build_basket_rs_ranks()
    per_symbol: list[dict[str, Any]] = []
    rs_blocked_if_enforced = 0
    rs_floor = day_entry_rs_floor()
    rank_max = day_entry_rs_rank_max()

    for row in basket:
        sym = str(row.get("symbol") or "")
        rs = _safe_float(row.get("ctx_rs_btc"), 0.0)
        rs_rank = int(ranks.get(sym, len(DAY_TRADE_SYMBOLS)))
        would_block: list[str] = []
        if rs_floor is not None and rs < rs_floor:
            would_block.append(REJECT_RS_BELOW_FLOOR)
        if rank_max is not None and rs_rank > rank_max:
            would_block.append(REJECT_RS_RANK_TOO_LOW)
        if would_block:
            rs_blocked_if_enforced += 1
        per_symbol.append(
            {
                "symbol": sym,
                "ctx_rs_btc": rs,
                "rs_rank": rs_rank,
                "rs_would_block": would_block,
                "setup_credit_eval_at": "bar_rank",
            }
        )

    last_bar = load_last_bar_evaluation()
    return {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "gates": day_entry_gates_config_snapshot(),
        "basket_rs_order": [r["symbol"] for r in basket],
        "symbols": per_symbol,
        "rs_blocked_if_enforced_count": rs_blocked_if_enforced,
        "last_bar_evaluation": last_bar,
        "setup_credit_note": "setup_credit and strong_setup evaluated at bar rank on selected candidate",
        "telemetry_only": not day_entry_gates_enforced(),
    }


__all__ = [
    "LAST_BAR_OPERATIONAL_KEY",
    "REJECT_RS_BELOW_FLOOR",
    "REJECT_RS_RANK_TOO_LOW",
    "REJECT_SETUP_CREDIT_REQUIRED",
    "build_basket_rs_ranks",
    "build_entry_quality_telemetry",
    "evaluate_entry_quality",
    "load_last_bar_evaluation",
    "persist_last_bar_evaluation",
]
