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

from backend.services.day_clock_v2_action_contract import (
    CONTRACT_VERSION as ACTION_CONTRACT_VERSION,
)
from backend.services.day_clock_v2_action_contract import (
    evaluate_action_row,
    selected_action_invariant,
)
from backend.services.day_clock_v2_partition import partition_for

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "day_decision_obs_v2_action_corrected"
LEGACY_SCHEMA_VERSION = "day_decision_obs_v1"
TABLE_GROUPS = "day_decision_group_records"
TABLE_CANDIDATES = "day_decision_candidate_records"
TABLE_FEATURE_ARTIFACTS = "day_decision_feature_artifacts"
FEATURE_SCHEMA = "day_145_feature_vector_v1"
FEATURE_SCHEMA_FALLBACK = "day_path_ev_candidate_v1"
STRATEGY_ID = "day"

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
_HAIRCUTS = (
    "outcome_low_mfe_stall_ev_factor",
    "quality_opinion_penalty",
    "signal_side_penalty",
    "pnl_adapt_penalty",
    "veto_opinion_penalty",
    "confidence_floor_penalty",
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
    action_available INTEGER,
    action_unavailable_reason TEXT,
    legacy_rank_candidate_present INTEGER,
    legacy_rank_candidate_reason TEXT,
    legacy_final_rank_score REAL,
    legacy_final_rank_score_valid INTEGER,
    legacy_final_rank_reason TEXT,
    production_selected INTEGER,
    execution_resolvable_candidate_present INTEGER,
    action_contract_version TEXT,
    PRIMARY KEY (decision_group_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_day_obs_cands_created ON {TABLE_CANDIDATES}(created_at);
CREATE TABLE IF NOT EXISTS {TABLE_FEATURE_ARTIFACTS} (
    feature_artifact_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    feature_dim INTEGER,
    feature_values_json TEXT NOT NULL
);
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


def _tri(value: Any) -> int | None:
    """Three-valued store: None stays NULL so 'unknown' is not written as false."""
    return None if value is None else (1 if value else 0)


# Corrected action-semantics columns. Additive only: historical rows keep NULL,
# so `eligible` / `exclusion_reason` / `final_rank_score` retain their v1 meaning
# and are never silently redefined.
_CANDIDATE_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("action_available", "INTEGER"),
    ("action_unavailable_reason", "TEXT"),
    ("legacy_rank_candidate_present", "INTEGER"),
    ("legacy_rank_candidate_reason", "TEXT"),
    ("legacy_final_rank_score", "REAL"),
    ("legacy_final_rank_score_valid", "INTEGER"),
    ("legacy_final_rank_reason", "TEXT"),
    ("production_selected", "INTEGER"),
    ("execution_resolvable_candidate_present", "INTEGER"),
    ("action_contract_version", "TEXT"),
)


def _migrate_candidate_columns(conn: sqlite3.Connection) -> None:
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({TABLE_CANDIDATES})")}
    for name, decl in _CANDIDATE_MIGRATIONS:
        if name not in existing:
            conn.execute(f"ALTER TABLE {TABLE_CANDIDATES} ADD COLUMN {name} {decl}")


def _ensure_schema(db_path: str | Path) -> None:
    conn = sqlite3.connect(str(db_path), timeout=30)
    try:
        conn.executescript(SCHEMA_SQL)
        _migrate_candidate_columns(conn)
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


def _coerce_feature_vector(raw: Any) -> list[float] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return None
    if isinstance(raw, dict) and "features" in raw:
        raw = raw.get("features")
    if not isinstance(raw, (list, tuple)):
        return None
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            return None
    return out or None


def _lookup_inference_vector(
    db_path: str | Path | None,
    *,
    symbol: str,
    bar_timestamp: int | None,
) -> tuple[list[float] | None, str | None]:
    """Read-only lookup of the stored 145-vector. Never calls the live model."""
    if not db_path or not symbol:
        return None, None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        try:
            cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(ai_inference_log)")}
            if "features_json" not in cols:
                return None, None
            params: list[Any] = [symbol, _api(symbol)]
            if symbol.endswith("USDT"):
                params.append(f"{symbol[:-4]}/USDT")
            placeholders = ",".join("?" * len(params))
            sql = f"""
                SELECT id, features_json FROM ai_inference_log
                WHERE symbol IN ({placeholders}) AND features_json IS NOT NULL
            """
            if bar_timestamp is not None and "ts_utc" in cols:
                epoch = float(bar_timestamp)
                if epoch > 1e12:
                    epoch = epoch / 1000.0
                sql += " AND ts_utc <= datetime(?, 'unixepoch') ORDER BY ts_utc DESC LIMIT 1"
                params.append(epoch)
            else:
                sql += " ORDER BY id DESC LIMIT 1"
            row = conn.execute(sql, params).fetchone()
            if not row:
                return None, None
            vec = _coerce_feature_vector(row[1])
            return vec, str(row[0])
        finally:
            conn.close()
    except Exception:
        return None, None


def _feature_payload(
    dd: dict[str, Any],
    *,
    db_path: str | Path | None = None,
    symbol: str = "",
    bar_timestamp: int | None = None,
) -> dict[str, Any]:
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
        "inference_log_id",
    )
    payload = {k: dd.get(k) for k in keys if dd.get(k) not in (None, "")}
    deltas = {k: _num(dd.get(k)) for k in _RANK_DELTAS if dd.get(k) not in (None, "")}
    if deltas:
        payload["rank_deltas"] = deltas
    vec = None
    for key in ("feature_vector", "features", "features_json"):
        vec = _coerce_feature_vector(dd.get(key))
        if vec:
            break
    inf_id = None
    if vec is None:
        vec, inf_id = _lookup_inference_vector(db_path, symbol=symbol, bar_timestamp=bar_timestamp)
        if inf_id:
            payload["inference_log_id"] = inf_id
    if vec:
        payload["feature_vector"] = vec
        payload["feature_dim"] = len(vec)
        payload["feature_schema_version"] = FEATURE_SCHEMA
    return payload


def _feature_hash(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _artifact_id_for_vector(vector: list[float], schema: str) -> str:
    blob = json.dumps({"schema": schema, "vector": vector}, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _book_fields(symbol: str) -> dict[str, Any]:
    empty = {
        "bid": None,
        "ask": None,
        "best_bid": None,
        "best_ask": None,
        "midpoint": None,
        "mid": None,
        "spread_bps": None,
        "spread_pct": None,
        "predicted_impact": None,
        "expected_slippage": None,
        "quote_timestamp": None,
        "quote_age": None,
        "book_ts": None,
    }
    if symbol == "HOLD":
        return empty
    try:
        from backend.services.decision_book_tape import snapshot_book

        book = snapshot_book(symbol)
    except Exception:
        return empty
    bid = book.get("best_bid")
    ask = book.get("best_ask")
    mid = book.get("mid")
    spread_pct = book.get("spread_pct")
    age = book.get("book_age_sec")
    return {
        "bid": bid,
        "ask": ask,
        "best_bid": bid,
        "best_ask": ask,
        "midpoint": mid,
        "mid": mid,
        "spread_bps": (float(spread_pct) * 1e4) if spread_pct is not None else None,
        "spread_pct": spread_pct,
        "predicted_impact": book.get("microprice_pressure"),
        "expected_slippage": book.get("expected_slippage"),
        "quote_timestamp": book.get("ts_utc") or _now_iso(),
        "quote_age": age,
        "book_ts": _now_iso(),
    }


def _account_state(engine: Any) -> dict[str, Any]:
    try:
        from backend.config.trading_economics import DAY_MAX_OPEN_SLOTS
    except Exception:
        DAY_MAX_OPEN_SLOTS = 4
    if engine is None:
        return {
            "slot_count": DAY_MAX_OPEN_SLOTS,
            "slots_used": 0,
            "cash_balance": None,
            "cash_available": None,
            "open_symbols": [],
            "capital_state": None,
        }
    positions = list(getattr(engine, "positions", None) or [])
    open_syms = []
    for pos in positions:
        qty = float(getattr(pos, "quantity", 0) or 0)
        if qty > 0:
            open_syms.append(_api(getattr(pos, "symbol", "") or ""))
    cash = _num(getattr(engine, "cash_balance", None))
    return {
        "slot_count": DAY_MAX_OPEN_SLOTS,
        "slots_used": len(open_syms),
        "cash_balance": cash,
        "cash_available": cash,
        "open_symbols": open_syms,
        "capital_state": getattr(engine, "account_status", None),
    }


def _lifecycle_from_action(selected_action: str) -> str:
    if str(selected_action or "").upper().startswith("BUY"):
        return "ranking_selected"
    return "HOLD"


def _haircuts(dd: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _HAIRCUTS:
        val = _num(dd.get(key))
        if val is not None:
            out[key] = val
    for key, val in dd.items():
        if "haircut" in str(key).lower():
            num = _num(val)
            if num is not None:
                out[str(key)] = num
    return out


def _collect_4h_group(decision: dict[str, Any]) -> dict[str, Any]:
    """Fail-open 4H telemetry. Never mutates decision or ranking scores."""
    try:
        from backend.services.day_4h_entry_telemetry import collect_4h_entry_telemetry

        return collect_4h_entry_telemetry(decision)
    except Exception as exc:
        logger.debug("DAY_DECISION_OBSERVABILITY 4h telemetry failed: %s", exc)
        return {}


def _proposed_notional(symbol: str) -> float:
    if symbol == "HOLD":
        return 0.0
    try:
        from backend.config.trading_economics import DAY_TARGET_NOTIONAL_PER_SLOT_USD

        return float(DAY_TARGET_NOTIONAL_PER_SLOT_USD)
    except Exception:
        return 0.0


def build_group_contract(
    *,
    decision: dict[str, Any],
    candidates: list[Any] | None = None,
    bar_timestamp: int | None = None,
    engine: Any = None,
    account_state: dict[str, Any] | None = None,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pure builder. Never mutates ``decision``."""
    dec = dict(decision or {})
    selected = str(dec.get("selected_action") or "HOLD")
    selected_symbol = _api(str(dec.get("selected_symbol") or "")) if dec.get("selected_symbol") else ""
    if bar_timestamp is not None:
        group_id = f"daygrp_{int(bar_timestamp)}"
    else:
        group_id = str(dec.get("decision_group_id") or dec.get("decision_id") or f"daygrp_{dec.get('prediction_timestamp') or _now_iso()}")
    cand_map = _candidate_map(candidates)
    # The executor resolves the selected symbol against
    # `valid_candidates + self.current_bar_candidates` (portfolio_engine
    # _select_direct_path_ev_candidate). The recorder is handed only the ranked
    # snapshot, which is why production could fill a symbol this table marked
    # NO_SCORED_CANDIDATE. Capture the wider resolvable set separately.
    resolvable_map = dict(cand_map)
    try:
        resolvable_map.update(_candidate_map(list(getattr(engine, "current_bar_candidates", None) or [])))
    except Exception:
        pass
    model_version = str(dec.get("path_net_model_id") or dec.get("forward_net_model_version") or "")
    acct = dict(account_state or _account_state(engine))
    open_syms = {_api(s) for s in (acct.get("open_symbols") or [])}
    slots_used = int(acct.get("slots_used") if acct.get("slots_used") is not None else len(open_syms))
    slot_count = int(acct["slot_count"]) if acct.get("slot_count") is not None else 4
    cash = acct.get("cash_available") if acct.get("cash_available") is not None else acct.get("cash_balance")
    rows = []
    artifacts: list[dict[str, Any]] = []
    scored: list[tuple[str, float]] = []
    fourh_group = _collect_4h_group(dec)
    fourh_by_symbol = dict(fourh_group.get("4h_entry_telemetry") or {})
    for sym in (*_COINS, "HOLD"):
        cand = cand_map.get(sym)
        dd = dict(getattr(cand, "decision_data", None) or {}) if cand is not None else {}
        ev_key = {
            "BTCUSDT": "btc_path_ev",
            "ETHUSDT": "eth_path_ev",
            "SOLUSDT": "sol_path_ev",
            "XRPUSDT": "xrp_path_ev",
        }.get(sym)
        path_ev = _num(dec.get(ev_key)) if ev_key else 0.0
        exclusion = str(dd.get("first_hard_block") or dd.get("true_safety_reject_reason") or "") or None
        if cand is None and sym != "HOLD":
            exclusion = exclusion or "NO_SCORED_CANDIDATE"
        feats = _feature_payload(dd, db_path=db_path, symbol=sym, bar_timestamp=bar_timestamp) if sym != "HOLD" else {}
        deltas = {k: _num(dd.get(k)) for k in _RANK_DELTAS if dd.get(k) not in (None, "")}
        base = _num(dd.get("ml_score") if dd.get("ml_score") not in (None, "") else dd.get("buy_margin"))
        p_buy = _num(dd.get("prob_buy") if dd.get("prob_buy") not in (None, "") else dd.get("p_buy"))
        final_score = _num(dd.get("final_selection_score") if dd.get("final_selection_score") not in (None, "") else path_ev)
        book = _book_fields(sym)
        vector = feats.get("feature_vector") if isinstance(feats.get("feature_vector"), list) else None
        artifact_id = None
        schema = str(feats.get("feature_schema_version") or FEATURE_SCHEMA_FALLBACK)
        if vector:
            artifact_id = _artifact_id_for_vector(vector, FEATURE_SCHEMA)
            artifacts.append(
                {
                    "feature_artifact_id": artifact_id,
                    "feature_schema_version": FEATURE_SCHEMA,
                    "feature_dim": len(vector),
                    "feature_values": vector,
                }
            )
            schema = FEATURE_SCHEMA
        elif feats:
            artifact_id = _feature_hash(feats)
        if sym != "HOLD" and final_score is not None:
            scored.append((sym, float(final_score)))
        already_open = sym in open_syms
        slot_available = slots_used < slot_count and not already_open
        capital_available = cash is None or float(cash) > 0
        fourh_tel = fourh_by_symbol.get(sym)
        path_tel = dict((dec.get("path_input_by_symbol") or {}).get(sym) or {})
        action_state = evaluate_action_row(
            symbol=sym,
            candidate_present=cand is not None,
            exclusion_reason=exclusion,
            path_input_valid=path_tel.get("path_input_valid"),
            path_invalid_reason=path_tel.get("path_invalid_reason"),
            open_symbols=open_syms,
            slots_used=slots_used,
            slot_count=slot_count,
            final_selection_score=(dd.get("final_selection_score") if cand is not None else None),
            recorded_final_rank_score=final_score,
            path_ev=path_ev,
            production_selected=(sym == (selected_symbol or "HOLD")),
        )
        rows.append(
            {
                "symbol": sym,
                "eligible": cand is not None or sym == "HOLD",
                "exclusion_reason": None if (cand is not None or sym == "HOLD") else exclusion,
                **action_state,
                "execution_resolvable_candidate_present": (True if sym == "HOLD" else sym in resolvable_map),
                "base_score": base,
                "p_buy": p_buy,
                "path_ev": 0.0 if sym == "HOLD" else path_ev,
                "path_input_valid": path_tel.get("path_input_valid"),
                "path_invalid_reason": path_tel.get("path_invalid_reason"),
                "path_row_count": path_tel.get("path_row_count"),
                "path_first_bar_ts": path_tel.get("path_first_bar_ts"),
                "path_last_bar_ts": path_tel.get("path_last_bar_ts"),
                "path_actual_lookback_seconds": path_tel.get("path_actual_lookback_seconds"),
                "path_max_gap_seconds": path_tel.get("path_max_gap_seconds"),
                "path_latest_bar_age_seconds": path_tel.get("path_latest_bar_age_seconds"),
                "path_model_version": path_tel.get("path_model_version") or model_version,
                "path_feature_schema_version": path_tel.get("path_feature_schema_version"),
                "legacy_btc_ret_5": path_tel.get("legacy_btc_ret_5"),
                "correct_btc_ret_5": path_tel.get("correct_btc_ret_5"),
                "legacy_path_ev": path_tel.get("legacy_path_ev"),
                "shadow_correct_btc_path_ev": path_tel.get("shadow_correct_btc_path_ev"),
                "path_max_abs_z": path_tel.get("path_max_abs_z"),
                "path_ood_feature_count_at_8": path_tel.get("path_ood_feature_count_at_8"),
                "rank_deltas": deltas,
                "all_rank_deltas": deltas,
                "all_haircuts": {} if sym == "HOLD" else _haircuts(dd),
                "final_rank_score": 0.0 if sym == "HOLD" else final_score,
                "rank_position": None,
                "feature_schema": schema,
                "feature_schema_version": schema,
                "feature_values": feats,
                "feature_hash": artifact_id,
                "feature_artifact_id": artifact_id,
                "symbol_already_open": already_open,
                "slot_available": True if sym == "HOLD" else slot_available,
                "capital_available": True if sym == "HOLD" else capital_available,
                "proposed_notional": 0.0 if sym == "HOLD" else _proposed_notional(sym),
                "gross_value": 0.0 if sym == "HOLD" else None,
                "net_value": 0.0 if sym == "HOLD" else None,
                "capital_usage": 0.0 if sym == "HOLD" else None,
                "4h_entry_telemetry": fourh_tel,
                **book,
            }
        )
    scored.sort(key=lambda item: item[1], reverse=True)
    rank_map = {sym: idx + 1 for idx, (sym, _) in enumerate(scored)}
    for row in rows:
        if row["symbol"] == "HOLD":
            row["rank_position"] = len(scored) + 1
        else:
            row["rank_position"] = rank_map.get(row["symbol"])
    mode = runtime_account_execution_mode()
    data_ts = dec.get("prediction_timestamp") or bar_timestamp
    decision_iso = _now_iso()
    invariant = selected_action_invariant(
        rows=rows,
        selected_symbol=selected_symbol or "HOLD",
        filled=False,
    )
    if not invariant.get("pass"):
        logger.warning(
            "DAY_CLOCK_V2_ACTION_INVARIANT group=%s selected=%s violations=%s reason=%s",
            group_id,
            invariant.get("selected_symbol"),
            invariant.get("violations"),
            invariant.get("action_unavailable_reason"),
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "clock_v2_partition": partition_for(decision_iso),
        "selected_action_invariant": invariant,
        "decision_group_id": group_id,
        "decision_timestamp": decision_iso,
        "runtime_trading_mode": mode,
        "account_execution_mode": mode,
        "strategy_id": STRATEGY_ID,
        "bar_timestamp": bar_timestamp,
        "prediction_timestamp": dec.get("prediction_timestamp"),
        "data_timestamp": data_ts,
        "data_freshness": dec.get("data_freshness") or dec.get("path_net_status"),
        "feature_schema_version": FEATURE_SCHEMA,
        "model_version": model_version,
        "calibration_version": str(dec.get("calibration_version") or dec.get("path_net_status") or ""),
        "ranking_version": str(dec.get("ranking_version") or dec.get("why_selected") or "direct_four_coin_path_ev"),
        "feature_schema": FEATURE_SCHEMA,
        "feature_artifact_ref": artifacts[0]["feature_artifact_id"] if artifacts else f"{FEATURE_SCHEMA}:{model_version or 'unknown'}",
        "feature_artifacts": artifacts,
        "selected_ranking_action": selected,
        "selected_action": selected,
        "selected_symbol": selected_symbol or "HOLD",
        "execute_authorization": None,
        "execute_authorized": None,
        "final_lifecycle_state": _lifecycle_from_action(selected),
        "lifecycle_state": _lifecycle_from_action(selected),
        "order_submitted": False,
        "order_id": None,
        "client_order_id": None,
        "fill_trade_id": None,
        "maker_taker": None,
        "commission": None,
        "commission_asset": None,
        "cash_available": cash,
        "slot_count": slot_count,
        "slots_used": slots_used,
        "cash_balance": acct.get("cash_balance"),
        "open_symbols": list(acct.get("open_symbols") or []),
        "capital_state": acct.get("capital_state"),
        "candidates": rows,
        "4h_entry_telemetry": fourh_by_symbol,
        "4h_peer_structure": fourh_group.get("4h_peer_structure"),
        "selected_already_broken_at_ranking": fourh_group.get("selected_already_broken_at_ranking"),
        "selected_4h_state": fourh_group.get("selected_4h_state"),
        "selected_distance_to_break_bps": fourh_group.get("selected_distance_to_break_bps"),
        "healthiest_peer_symbol": fourh_group.get("healthiest_peer_symbol"),
        "healthiest_peer_distance_bps": fourh_group.get("healthiest_peer_distance_bps"),
        "all_four_already_broken": fourh_group.get("all_four_already_broken"),
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
            db_path=db_path,
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
                    json.dumps(contract, default=str)[:64000],
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                ),
            )
            for art in contract.get("feature_artifacts") or []:
                conn.execute(
                    f"""
                    INSERT OR IGNORE INTO {TABLE_FEATURE_ARTIFACTS}(
                        feature_artifact_id, created_at, feature_schema_version,
                        feature_dim, feature_values_json
                    ) VALUES (?,?,?,?,?)
                    """,
                    (
                        art["feature_artifact_id"],
                        created,
                        art["feature_schema_version"],
                        art["feature_dim"],
                        json.dumps(art["feature_values"], default=str),
                    ),
                )
            for row in contract["candidates"]:
                stored_features = dict(row["feature_values"] or {})
                if row.get("4h_entry_telemetry"):
                    stored_features["4h_entry_telemetry"] = row["4h_entry_telemetry"]
                if stored_features.get("feature_vector") and row.get("feature_artifact_id"):
                    stored_features = {
                        **{k: v for k, v in stored_features.items() if k != "feature_vector"},
                        "feature_artifact_id": row["feature_artifact_id"],
                        "feature_dim": stored_features.get("feature_dim"),
                    }
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {TABLE_CANDIDATES}(
                        decision_group_id, symbol, created_at, eligible, exclusion_reason,
                        base_score, p_buy, path_ev, rank_deltas_json, final_rank_score,
                        feature_json, feature_hash,
                        action_available, action_unavailable_reason,
                        legacy_rank_candidate_present, legacy_rank_candidate_reason,
                        legacy_final_rank_score, legacy_final_rank_score_valid,
                        legacy_final_rank_reason, production_selected,
                        execution_resolvable_candidate_present, action_contract_version
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                        json.dumps(stored_features, default=str)[:32000],
                        row["feature_hash"],
                        _tri(row.get("action_available")),
                        row.get("action_unavailable_reason"),
                        _tri(row.get("legacy_rank_candidate_present")),
                        row.get("legacy_rank_candidate_reason"),
                        row.get("legacy_final_rank_score"),
                        _tri(row.get("legacy_final_rank_score_valid")),
                        row.get("legacy_final_rank_reason"),
                        _tri(row.get("production_selected")),
                        _tri(row.get("execution_resolvable_candidate_present")),
                        ACTION_CONTRACT_VERSION,
                    ),
                )
            conn.commit()
        finally:
            conn.close()
        try:
            from backend.services.day_path_clock_v2_capture import capture_clock_v2_fail_open

            capture_clock_v2_fail_open(db_path, contract)
        except Exception as cap_exc:
            logger.warning("DAY_CLOCK_V2_CAPTURE hook failed: %s", cap_exc)
        return str(contract["decision_group_id"])
    except Exception as exc:
        logger.warning("DAY_DECISION_OBSERVABILITY record failed: %s", exc)
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
    block_reason: str | None = None,
    requested_qty: float | None = None,
    filled_qty: float | None = None,
    fill_timestamp: str | None = None,
    fill_price: float | None = None,
    trade_id: str | None = None,
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
                contract["execute_authorization"] = bool(execute_authorized)
            if lifecycle_state:
                contract["lifecycle_state"] = lifecycle_state
                contract["final_lifecycle_state"] = lifecycle_state
            if order_id:
                contract["order_id"] = order_id
                contract["order_submitted"] = True
            if client_order_id:
                contract["client_order_id"] = client_order_id
                contract["order_submitted"] = True
            if fill_trade_id:
                contract["fill_trade_id"] = fill_trade_id
            if trade_id:
                contract["trade_id"] = trade_id
            if maker_taker:
                contract["maker_taker"] = maker_taker
            if commission is not None:
                contract["commission"] = commission
            if commission_asset:
                contract["commission_asset"] = commission_asset
            if block_reason:
                contract["block_reason"] = block_reason
            if requested_qty is not None:
                contract["requested_qty"] = requested_qty
            if filled_qty is not None:
                contract["filled_qty"] = filled_qty
            if fill_timestamp:
                contract["fill_timestamp"] = fill_timestamp
            if fill_price is not None:
                contract["fill_price"] = fill_price
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
                    json.dumps(contract, default=str)[:64000],
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
        logger.warning("DAY_DECISION_OBSERVABILITY update failed: %s", exc)


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


def classify_execute_lifecycle(
    *,
    result: dict[str, Any] | None,
    block_reason: str | None = None,
) -> str:
    """Map an execution result to a spec lifecycle state. Telemetry only."""
    if result is None:
        return "blocked_after_ranking" if block_reason else "execute_decision"
    status = str(result.get("status") or "")
    filled = float(result.get("filled_qty") or result.get("quantity") or 0)
    requested = float(result.get("requested_qty") or result.get("quantity") or 0)
    labeled = classify_terminal_fill(status=status, filled_qty=filled, requested_qty=requested)
    if labeled != "no_order_match":
        return labeled
    if result.get("trade_id") or result.get("fill_id"):
        return "filled"
    if result.get("exchange_order_id") or result.get("order_id") or result.get("client_order_id"):
        return "order_submitted"
    return "execute_decision"


def estimate_observability_storage(
    *,
    groups_per_day: float,
    group_bytes: float,
    candidate_bytes: float,
    artifact_bytes: float,
    index_overhead: float = 0.20,
    current_db_bytes: int,
    disk_free_bytes: int,
    reserve_bytes: int,
    candidates_per_group: int = 5,
) -> dict[str, Any]:
    """Project observability growth from measured write sizes. Prospective only."""
    bytes_per_group = group_bytes + (candidate_bytes * candidates_per_group) + artifact_bytes
    bytes_per_day = groups_per_day * bytes_per_group * (1.0 + index_overhead)
    out: dict[str, Any] = {
        "rows_per_day": groups_per_day,
        "bytes_per_day": bytes_per_day,
        "current_db_bytes": current_db_bytes,
        "disk_free_bytes": disk_free_bytes,
        "reserve_bytes": reserve_bytes,
        "horizons": {},
    }
    selected = 30
    for days in (30, 60, 90):
        growth = bytes_per_day * days
        remaining = disk_free_bytes - growth - reserve_bytes
        fits = remaining > 0
        out["horizons"][str(days)] = {
            "days": days,
            "growth_bytes": growth,
            "remaining_after_reserve_bytes": remaining,
            "fits": fits,
        }
        if fits:
            selected = days
    out["selected_retention_days"] = selected
    return out
