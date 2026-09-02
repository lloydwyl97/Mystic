"""DAY ranking observability. Persistence only — never changes the decision.

Writes a point-in-time group contract after production has already chosen
the action. Fail-open. Does not size, rank, authorize, or exit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "day_decision_obs_v1"
TABLE_GROUPS = "day_decision_group_records"
TABLE_CANDIDATES = "day_decision_candidate_records"
FEATURE_SCHEMA = "day_path_ev_candidate_v1"

_COINS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
_RANK_DELTAS = (
    "block_score_rank_delta",
    "setup_score_rank_delta",
    "execution_rank_delta",
    "regime_transition_rank_delta",
    "memory_rank_delta",
    "chart_pattern_rank_delta",
    "cross_sectional_rank_delta",
    "intelligence_rank_delta",
    "thesis_rank_delta",
    "ml_rank_adjustment",
    "outcome_low_mfe_stall_ev_factor",
)

SCHEMA_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_GROUPS} (
    decision_group_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    account_execution_mode TEXT NOT NULL,
    selected_action TEXT,
    selected_symbol TEXT,
    selected_ranking_action TEXT,
    execute_authorized INTEGER,
    lifecycle_state TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    feature_schema TEXT,
    model_version TEXT,
    feature_artifact_ref TEXT,
    slot_count INTEGER,
    cash_balance REAL,
    contract_json TEXT,
    order_id TEXT,
    client_order_id TEXT,
    fill_trade_id TEXT,
    maker_taker TEXT,
    commission REAL,
    commission_asset TEXT
);
CREATE INDEX IF NOT EXISTS idx_day_obs_groups_created ON {TABLE_GROUPS}(created_at);
CREATE TABLE IF NOT EXISTS {TABLE_CANDIDATES} (
    decision_group_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    eligible INTEGER,
    exclusion_reason TEXT,
    base_score REAL,
    p_buy REAL,
    path_ev REAL,
    rank_deltas_json TEXT,
    final_rank_score REAL,
    feature_json TEXT,
    feature_hash TEXT,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_day_obs_cands_created ON {TABLE_CANDIDATES}(created_at);
"""


def observability_enabled() -> bool:
    raw = os.getenv("DAY_DECISION_OBSERVABILITY", "true")
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def runtime_account_execution_mode() -> str:
    """Account mode from runtime authority. Telemetry only — never changes execution."""
    try:
        from backend.config.trading_mode import resolve_trading_mode

        return str(resolve_trading_mode().value)
    except Exception:
        for key in ("MYSTIC_TRADING_MODE", "TRADING_MODE", "EXECUTION_MODE"):
            raw = str(os.getenv(key) or "").strip().lower()
            if raw in {"paper", "live"}:
                return raw
        return "unknown"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").replace("_", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def _num(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ensure_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


def _candidate_map(candidates: list[Any] | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cand in list(candidates or []):
        sym = _api(getattr(cand, "symbol", "") or "")
        if sym:
            out[sym] = cand
    return out


def _feature_payload(dd: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "prob_buy",
        "p_buy",
        "p_sell",
        "p_hold",
        "final_selection_score",
        "ml_score",
        "buy_margin",
        "selected_net_expected_value",
        "predicted_net_return",
        "setup_type",
        "feature_version",
        "features_json_hash",
        "artifact_sha256",
        "path_net_status",
        "outcome_low_mfe_stall_penalty_applied",
        "outcome_low_mfe_stall_ev_factor",
        "first_hard_block",
        "true_safety_reject_reason",
    )
    payload = {k: dd.get(k) for k in keys if dd.get(k) not in (None, "")}
    deltas = {k: _num(dd.get(k)) for k in _RANK_DELTAS if dd.get(k) not in (None, "")}
    if deltas:
        payload["rank_deltas"] = deltas
    return payload


def _feature_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _book_fields(symbol: str) -> dict[str, Any]:
    empty = {
        "best_bid": None,
        "best_ask": None,
        "mid": None,
        "spread_pct": None,
        "predicted_impact": None,
        "book_ts": None,
    }
    if symbol == "HOLD":
        return empty
    try:
        from backend.services.decision_book_tape import snapshot_book

        book = snapshot_book(symbol)
    except Exception:
        return empty
    return {
        "best_bid": book.get("best_bid"),
        "best_ask": book.get("best_ask"),
        "mid": book.get("mid"),
        "spread_pct": book.get("spread_pct"),
        "predicted_impact": book.get("microprice_pressure"),
        "book_ts": _now_iso(),
    }


def _account_state(engine: Any) -> dict[str, Any]:
    if engine is None:
        return {"slot_count": None, "cash_balance": None, "open_symbols": []}
    positions = list(getattr(engine, "positions", None) or [])
    open_syms = []
    for pos in positions:
        qty = float(getattr(pos, "quantity", 0) or 0)
        if qty > 0:
            open_syms.append(_api(getattr(pos, "symbol", "") or ""))
    cash = getattr(engine, "cash_balance", None)
    return {
        "slot_count": len(open_syms),
        "cash_balance": _num(cash),
        "open_symbols": open_syms,
    }


def _lifecycle_from_action(selected_action: str) -> str:
    if str(selected_action or "").upper().startswith("BUY"):
        return "selected_execute"
    return "HOLD"


def build_group_contract(
    *,
    decision: dict[str, Any],
    candidates: list[Any] | None = None,
    bar_timestamp: int | None = None,
    engine: Any = None,
    account_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Pure builder. Never mutates ``decision``."""
    dec = dict(decision or {})
    selected = str(dec.get("selected_action") or "HOLD")
    selected_symbol = _api(str(dec.get("selected_symbol") or "")) if dec.get("selected_symbol") else ""
    if bar_timestamp is not None:
        group_id = f"daygrp_{int(bar_timestamp)}"
    else:
        group_id = str(
            dec.get("decision_group_id")
            or dec.get("decision_id")
            or f"daygrp_{dec.get('prediction_timestamp') or _now_iso()}"
        )
    cand_map = _candidate_map(candidates)
    model_version = str(dec.get("path_net_model_id") or dec.get("forward_net_model_version") or "")
    rows = []
    for sym in (*_COINS, "HOLD"):
        cand = cand_map.get(sym)
        dd = dict(getattr(cand, "decision_data", None) or {}) if cand is not None else {}
        ev_key = {"BTCUSDT": "btc_path_ev", "ETHUSDT": "eth_path_ev", "SOLUSDT": "sol_path_ev", "XRPUSDT": "xrp_path_ev"}.get(sym)
        path_ev = _num(dec.get(ev_key)) if ev_key else 0.0
        exclusion = str(dd.get("first_hard_block") or dd.get("true_safety_reject_reason") or "") or None
        if cand is None and sym != "HOLD":
            exclusion = exclusion or "NO_SCORED_CANDIDATE"
        feats = _feature_payload(dd) if sym != "HOLD" else {}
        deltas = {k: _num(dd.get(k)) for k in _RANK_DELTAS if dd.get(k) not in (None, "")}
        base = _num(dd.get("ml_score") if dd.get("ml_score") not in (None, "") else dd.get("buy_margin"))
        p_buy = _num(dd.get("prob_buy") if dd.get("prob_buy") not in (None, "") else dd.get("p_buy"))
        final_score = _num(dd.get("final_selection_score") if dd.get("final_selection_score") not in (None, "") else path_ev)
        book = _book_fields(sym)
        rows.append(
            {
                "symbol": sym,
                "eligible": cand is not None or sym == "HOLD",
                "exclusion_reason": None if (cand is not None or sym == "HOLD") else exclusion,
                "base_score": base,
                "p_buy": p_buy,
                "path_ev": path_ev if sym != "HOLD" else 0.0,
                "rank_deltas": deltas,
                "final_rank_score": 0.0 if sym == "HOLD" else final_score,
                "feature_schema": FEATURE_SCHEMA,
                "feature_values": feats,
                "feature_hash": _feature_hash(feats) if feats else None,
                **book,
            }
        )
    acct = dict(account_state or _account_state(engine))
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_group_id": group_id,
        "account_execution_mode": runtime_account_execution_mode(),
        "bar_timestamp": bar_timestamp,
        "prediction_timestamp": dec.get("prediction_timestamp"),
        "model_version": model_version,
        "feature_schema": FEATURE_SCHEMA,
        "feature_artifact_ref": f"{FEATURE_SCHEMA}:{model_version or 'unknown'}",
        "selected_ranking_action": selected,
        "selected_action": selected,
        "selected_symbol": selected_symbol or "HOLD",
        "execute_authorized": None,
        "lifecycle_state": _lifecycle_from_action(selected),
        "order_submitted": False,
        "order_id": None,
        "client_order_id": None,
        "fill_trade_id": None,
        "maker_taker": None,
        "commission": None,
        "commission_asset": None,
        "slot_count": acct.get("slot_count"),
        "cash_balance": acct.get("cash_balance"),
        "open_symbols": acct.get("open_symbols") or [],
        "candidates": rows,
    }


def record_day_ranking_group(
    db_path: str | Path,
    *,
    decision: dict[str, Any],
    candidates: list[Any] | None = None,
    bar_timestamp: int | None = None,
    engine: Any = None,
    account_state: dict[str, Any] | None = None,
) -> str | None:
    """Persist the group contract. Returns group id. Never mutates ``decision``."""
    if not observability_enabled():
        return None
    if not db_path:
        return None
    try:
        contract = build_group_contract(
            decision=decision,
            candidates=candidates,
            bar_timestamp=bar_timestamp,
            engine=engine,
            account_state=account_state,
        )
        _ensure_schema(db_path)
        created = _now_iso()
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            conn.execute(
                f"""
                INSERT OR REPLACE INTO {TABLE_GROUPS}(
                    decision_group_id, created_at, account_execution_mode,
                    selected_action, selected_symbol, selected_ranking_action,
                    execute_authorized, lifecycle_state, schema_version,
                    feature_schema, model_version, feature_artifact_ref,
                    slot_count, cash_balance, contract_json,
                    order_id, client_order_id, fill_trade_id, maker_taker,
                    commission, commission_asset
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    contract["decision_group_id"],
                    created,
                    contract["account_execution_mode"],
                    contract["selected_action"],
                    contract["selected_symbol"],
                    contract["selected_ranking_action"],
                    None,
                    contract["lifecycle_state"],
                    SCHEMA_VERSION,
                    FEATURE_SCHEMA,
                    contract["model_version"],
                    contract["feature_artifact_ref"],
                    contract["slot_count"],
                    contract["cash_balance"],
                    json.dumps(contract, default=str)[:32000],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            for row in contract["candidates"]:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {TABLE_CANDIDATES}(
                        decision_group_id, symbol, created_at, eligible, exclusion_reason,
                        base_score, p_buy, path_ev, rank_deltas_json, final_rank_score,
                        feature_json, feature_hash
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        contract["decision_group_id"],
                        row["symbol"],
                        created,
                        1 if row["eligible"] else 0,
                        row["exclusion_reason"],
                        row["base_score"],
                        row["p_buy"],
                        row["path_ev"],
                        json.dumps(row["rank_deltas"], default=str),
                        row["final_rank_score"],
                        json.dumps(row["feature_values"], default=str)[:16000],
                        row["feature_hash"],
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        return str(contract["decision_group_id"])
    except Exception as exc:
        logger.debug("day decision observability record failed: %s", exc)
        return None


def update_day_decision_lifecycle(
    db_path: str | Path,
    *,
    decision_group_id: str,
    execute_authorized: bool | None = None,
    lifecycle_state: str | None = None,
    order_id: str | None = None,
    client_order_id: str | None = None,
    fill_trade_id: str | None = None,
    maker_taker: str | None = None,
    commission: float | None = None,
    commission_asset: str | None = None,
) -> None:
    """Patch order/fill fields after ranking. Does not change trading behavior."""
    if not observability_enabled() or not db_path or not decision_group_id:
        return
    try:
        _ensure_schema(db_path)
        conn = sqlite3.connect(str(db_path), timeout=30)
        try:
            row = conn.execute(
                f"SELECT contract_json FROM {TABLE_GROUPS} WHERE decision_group_id=?",
                (decision_group_id,),
            ).fetchone()
            contract = {}
            if row and row[0]:
                try:
                    contract = json.loads(row[0])
                except Exception:
                    contract = {}
            if execute_authorized is not None:
                contract["execute_authorized"] = bool(execute_authorized)
            if lifecycle_state:
                contract["lifecycle_state"] = lifecycle_state
            if order_id:
                contract["order_id"] = order_id
                contract["order_submitted"] = True
            if client_order_id:
                contract["client_order_id"] = client_order_id
                contract["order_submitted"] = True
            if fill_trade_id:
                contract["fill_trade_id"] = fill_trade_id
            if maker_taker:
                contract["maker_taker"] = maker_taker
            if commission is not None:
                contract["commission"] = commission
            if commission_asset:
                contract["commission_asset"] = commission_asset
            sets = [
                "contract_json=?",
                "execute_authorized=COALESCE(?, execute_authorized)",
                "lifecycle_state=COALESCE(?, lifecycle_state)",
                "order_id=COALESCE(?, order_id)",
                "client_order_id=COALESCE(?, client_order_id)",
                "fill_trade_id=COALESCE(?, fill_trade_id)",
                "maker_taker=COALESCE(?, maker_taker)",
                "commission=COALESCE(?, commission)",
                "commission_asset=COALESCE(?, commission_asset)",
            ]
            conn.execute(
                f"UPDATE {TABLE_GROUPS} SET {', '.join(sets)} WHERE decision_group_id=?",
                (
                    json.dumps(contract, default=str)[:32000],
                    None if execute_authorized is None else (1 if execute_authorized else 0),
                    lifecycle_state,
                    order_id,
                    client_order_id,
                    fill_trade_id,
                    maker_taker,
                    commission,
                    commission_asset,
                    decision_group_id,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception as exc:
        logger.debug("day decision observability update failed: %s", exc)


def classify_terminal_fill(*, status: str, filled_qty: float, requested_qty: float) -> str:
    """Lifecycle label. Only canceled/rejected/expired with zero fill is terminal_fill_failure."""
    st = str(status or "").strip().lower()
    filled = float(filled_qty or 0)
    requested = float(requested_qty or 0)
    if st in {"canceled", "cancelled", "rejected", "expired"} and filled <= 0:
        return "terminal_fill_failure"
    if filled > 0 and requested > 0 and filled + 1e-12 < requested:
        return "partial_fill"
    if filled > 0:
        return "filled"
    if st in {"submitted", "new", "open", "ack"}:
        return "order_submitted"
    return "no_order_match"
