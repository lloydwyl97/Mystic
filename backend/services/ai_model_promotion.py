from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.ai_artifact_contract_gate import evaluate_signal_hash_artifact_contract
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion_pac import _symbol_forms

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_meta(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"accuracy": 0.0, "feature_version": 0, "feature_dim": 0, "strategy_id": "", "ok": False}
    if not path.exists():
        return out
    payload = pickle.loads(path.read_bytes())
    if isinstance(payload, dict):
        out["accuracy"] = float(payload.get("accuracy") or 0.0)
        out["feature_version"] = int(payload.get("feature_version") or 0)
        out["feature_dim"] = int(payload.get("feature_dim") or 0)
        out["strategy_id"] = str(payload.get("live_strategy_id") or "").strip().lower()
        for key in ("profit_after_cost", "profit_after_cost_pct", "avg_net_pnl_pct"):
            if payload.get(key) is not None:
                out[key] = float(payload.get(key))
    out["ok"] = True
    return out


def _holdout_accuracy(metrics: dict[str, Any] | None, *, role: str) -> float | None:
    if not metrics:
        return None
    key = "candidate_accuracy" if role == "candidate" else "active_accuracy"
    if metrics.get(key) is not None:
        try:
            return float(metrics[key])
        except (TypeError, ValueError):
            pass
    holdout_key = "candidate_holdout" if role == "candidate" else "active_holdout"
    holdout = metrics.get(holdout_key)
    if isinstance(holdout, dict) and holdout.get("accuracy") is not None:
        try:
            return float(holdout["accuracy"])
        except (TypeError, ValueError):
            pass
    return None


def holdout_is_causal_for(candidate_path: Path, window: dict[str, Any] | None) -> tuple[bool, str]:
    """True only when everything the candidate trained on was known before the holdout opened."""
    try:
        art = pickle.loads(Path(candidate_path).read_bytes())
    except Exception:
        return False, "candidate_unreadable"
    end = str((art or {}).get("training_data_end") or "") if isinstance(art, dict) else ""
    start = str((window or {}).get("first_opened_at") or "")
    if not end:
        return False, "candidate_training_data_end_missing"
    if not start:
        return False, "holdout_start_missing"
    try:
        end_dt = datetime.fromisoformat(end.replace(" ", "T").replace("Z", "+00:00"))
        start_dt = datetime.fromisoformat(start.replace(" ", "T").replace("Z", "+00:00"))
    except ValueError:
        return False, "unparseable_window"
    end_dt = end_dt if end_dt.tzinfo else end_dt.replace(tzinfo=timezone.utc)
    start_dt = start_dt if start_dt.tzinfo else start_dt.replace(tzinfo=timezone.utc)
    if end_dt > start_dt:
        return False, f"training_data_end_after_holdout_start:{end_dt.isoformat()}>{start_dt.isoformat()}"
    return True, "causal"


def register_candidate_and_maybe_promote(
    *,
    strategy_id: str,
    symbol: str,
    candidate_path: Path,
    active_path: Path,
    validation_metrics: dict[str, Any] | None = None,
    db_path: str = DATABASE_PATH,
) -> tuple[bool, str]:
    ensure_ai_canonical_tables(db_path)
    if not candidate_path.exists():
        return False, "candidate_missing"
    sid = strategy_id.strip().lower()
    sym = symbol.strip().upper()
    c_hash = _hash_file(candidate_path)
    c_meta = _artifact_meta(candidate_path)
    gate_ok, gate_reason, _detail = evaluate_signal_hash_artifact_contract(
        {
            "live_ai_strategy": sid,
            "feature_version": str(c_meta.get("feature_version") or 0),
            "feature_dim": str(c_meta.get("feature_dim") or 0),
            "artifact_sha256": c_hash,
            "model_artifact_path": str(candidate_path),
        },
        redis_strategy_id=sid,
        symbol_bus=sym,
    )
    # Promotion candidates are evaluated from candidate paths first; the strict
    # contract gate expects active canonical paths and can emit false path
    # mismatches before the file is promoted. Keep hash/version/dim checks but
    # allow path mismatch for candidate staging.
    if (not gate_ok) and str(gate_reason or "") == "ARTIFACT_CONTRACT_PATH_MISMATCH":
        gate_ok = True
        gate_reason = None
    if not gate_ok:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO ai_model_promotion_events (strategy_id, symbol, from_model_id, to_model_id, event_type, reason, metrics_json, created_at)
                VALUES (?, ?, ?, ?, 'reject', ?, ?, datetime('now'))
                """,
                (sid, sym, None, None, f"artifact_invalid:{gate_reason}", json.dumps(validation_metrics or {}, separators=(",", ":"))),
            )
            conn.commit()
        return False, f"artifact_invalid:{gate_reason}"

    from backend.services import ai_model_registry as registry

    reg_root = registry.registry_root(active_path.parent.parent if active_path.parent.name == sid else active_path.parent)
    metrics = dict(validation_metrics or {})
    has_active = active_path.exists()
    for role, key in (("candidate", "candidate_accuracy"), ("active", "active_accuracy")):
        acc = _holdout_accuracy(metrics, role=role)
        if acc is not None:
            metrics.setdefault(key, round(acc, 6))
    cand_holdout = metrics.get("candidate_holdout") if isinstance(metrics.get("candidate_holdout"), dict) else {}
    holdout_count = int(metrics.get("holdout_sample_count") or 0)
    buy_sig = int(cand_holdout.get("buy_signal_count") or 0)
    candidate_always_buy = holdout_count > 0 and buy_sig >= holdout_count
    metrics["candidate_always_buy"] = candidate_always_buy
    metrics["candidate_always_hold"] = holdout_count > 0 and buy_sig == 0

    if not has_active:
        decision: dict[str, Any] = {"verdict": "COLD_START", "promote": True}
        metrics["promotion_path"] = "cold_start_bootstrap"
    else:
        raw = metrics.get("paired_decision")
        decision = dict(raw) if isinstance(raw, dict) else {"verdict": "NO_SHARED_HOLDOUT", "promote": False}
        causal_ok, causal_why = holdout_is_causal_for(candidate_path, metrics.get("holdout_window"))
        metrics["holdout_causality"] = causal_why
        if decision.get("verdict") != "PROMOTE":
            decision["promote"] = False
        elif not causal_ok:
            decision.update(verdict="CAUSALITY_UNVERIFIED", promote=False)
        elif candidate_always_buy:
            decision.update(verdict="REJECT_ALWAYS_BUY", promote=False)
        metrics["promotion_path"] = "paired_forward_lcb"
    metrics["promotion_decision"] = {k: v for k, v in decision.items() if k not in ("candidate", "incumbent")}

    try:
        cand_reg = registry.register_artifact(sid, sym, candidate_path, {"role": "candidate", **registry.artifact_meta(candidate_path), "validation": metrics}, reg_root)
        if has_active:
            registry.ensure_incumbent_registered(sid, sym, active_path, reg_root)
    except Exception as exc:
        logger.warning("MODEL_REGISTRY_WRITE_FAILED strategy=%s symbol=%s err=%s", sid, sym, exc)
        return False, f"registry_write_failed:{type(exc).__name__}"

    bus_sym, ccxt_sym = _symbol_forms(sym)
    model_id = f"{sid}:{bus_sym}:{c_hash[:16]}"
    active_model_id = f"{sid}:{bus_sym}:{_hash_file(active_path)[:16]}" if has_active else None

    enabled, _ctl = registry.promotion_enabled(sid, reg_root)
    metrics["promotion_enabled"] = enabled
    promote = False
    if not decision.get("promote"):
        reason = f"keep_incumbent:{decision.get('verdict')}"
    elif has_active and not enabled:
        reason = "would_promote_promotion_disabled"
    else:
        ok, why, _info = registry.promote_atomic(sid, sym, candidate_path, active_path, decision=metrics["promotion_decision"], root=reg_root)
        promote = ok
        reason = f"promoted:{metrics['promotion_path']}" if ok else f"keep_incumbent:{why}"
    registry.append_event(sid, sym, {"event": "evaluated", "candidate": cand_reg["version"], "decision": metrics["promotion_decision"], "outcome": reason, "promotion_enabled": enabled}, reg_root)
    registry.prune_candidates(sid, sym, root=reg_root)

    status = "active" if promote else "candidate"
    registry_path = cand_reg["path"]

    with sqlite3.connect(db_path) as conn:
        if promote:
            # Single-active invariant: archive every prior active for this pair.
            conn.execute(
                """
                UPDATE ai_model_versions
                SET status = 'archived',
                    retired_at = datetime('now')
                WHERE strategy_id = ?
                  AND symbol IN (?, ?)
                  AND status = 'active'
                """,
                (sid, bus_sym, ccxt_sym),
            )
        if promote or model_id != active_model_id:
            conn.execute(
                """
                INSERT OR REPLACE INTO ai_model_versions (
                    model_id, strategy_id, symbol, feature_version, artifact_hash, path, status,
                    created_at, promoted_at, validation_metrics_json, promotion_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), ?, ?, ?)
                """,
                (
                    model_id,
                    sid,
                    bus_sym,
                    int(c_meta.get("feature_version") or 0),
                    c_hash,
                    registry_path,
                    status,
                    (datetime.now(timezone.utc).isoformat() if promote else None),
                    json.dumps(metrics, separators=(",", ":")),
                    reason,
                ),
            )
        conn.execute(
            """
            INSERT INTO ai_model_promotion_events (strategy_id, symbol, from_model_id, to_model_id, event_type, reason, metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """,
            (
                sid,
                bus_sym,
                active_model_id,
                model_id,
                ("promote" if promote else "reject"),
                reason,
                json.dumps(metrics, separators=(",", ":")),
            ),
        )
        conn.commit()
    return promote, reason


ROLLBACK_Z = 1.645


def attributable_live_evidence(rows: list[tuple[Any, Any]], model_version: str) -> dict[str, Any]:
    """Split live outcomes into model-attributable evidence and telemetry.

    ``rows`` are ``(net_pnl_pct, score_components_json)``. Only an outcome whose
    decision recorded that ``model_version`` changed its score, rank, size or
    action counts. The verdict is ``HARMFUL`` only when the upper confidence
    bound of the attributable mean net is below zero.
    """
    from backend.services.day_model_attribution import attributable_to

    nets: list[float] = []
    for net, comps in rows:
        if attributable_to(comps, model_version):
            nets.append(float(net or 0.0))
    out: dict[str, Any] = {"live_outcomes": len(rows), "attributable": len(nets), "telemetry_only": len(rows) - len(nets)}
    if not nets:
        out["verdict"] = "NO_ATTRIBUTABLE_EVIDENCE"
        return out
    n = len(nets)
    mean = sum(nets) / n
    sample_var = sum((x - mean) ** 2 for x in nets) / (n - 1) if n > 1 else 0.0
    var = max(sample_var, sum(x * x for x in nets) / n)
    ucb = mean + ROLLBACK_Z * (var / n) ** 0.5
    out.update(mean_net=mean, ucb=ucb, verdict="HARMFUL" if ucb < 0 else "NOT_PROVEN_HARMFUL")
    return out


def maybe_rollback_underperforming_model(
    *,
    strategy_id: str,
    symbol: str,
    db_path: str = DATABASE_PATH,
    active_dir: Path | None = None,
) -> tuple[bool, str]:
    """Roll back to the registry PREVIOUS artifact only on attributable harm.

    A live outcome is evidence against the serving model only when its decision
    recorded that this model version changed score, rank, size or action. Every
    other outcome is telemetry and has no rollback authority. Suppressed while
    auto-promotion is disabled.
    """
    from backend.services import ai_model_registry as registry
    from backend.services.live_strategy_contracts import per_coin_artifact_file
    from backend.utils.path_helpers import ensure_model_directories

    ensure_ai_canonical_tables(db_path)
    sid = strategy_id.strip().lower()
    bus_sym, ccxt_sym = _symbol_forms(symbol)
    active_dir = Path(active_dir or ensure_model_directories()["active"])
    reg_root = registry.registry_root(active_dir)
    active_pkl_path = per_coin_artifact_file(active_dir, sid, bus_sym)
    ptr = registry.read_pointer(sid, bus_sym, "ACTIVE", reg_root)
    since = str(ptr.get("set_at") or "").replace("T", " ")[:19]
    with sqlite3.connect(db_path) as conn:
        if not since:
            row = conn.execute(
                "SELECT COALESCE(promoted_at, created_at) FROM ai_model_versions WHERE strategy_id=? AND symbol IN (?, ?) AND status='active' ORDER BY id DESC LIMIT 1",
                (sid, bus_sym, ccxt_sym),
            ).fetchone()
            since = str(row[0]) if row and row[0] else ""
        if not since:
            return False, "no_active_model"
        rows = conn.execute(
            """
            SELECT net_pnl_pct, score_components_json
            FROM ai_outcome_training_rows
            WHERE strategy_id = ?
              AND UPPER(symbol) IN (?, ?)
              AND julianday(closed_at_utc) >= julianday(?)
            """,
            (sid, bus_sym.upper(), ccxt_sym.upper(), since),
        ).fetchall()
    serving = str(ptr.get("version") or "")
    if not serving and active_pkl_path.exists():
        serving = _hash_file(active_pkl_path)[:16]
    evidence = attributable_live_evidence([(r[0], r[1]) for r in rows], serving)
    if evidence["verdict"] == "NO_ATTRIBUTABLE_EVIDENCE":
        return False, "no_attributable_live_outcomes"
    if evidence["verdict"] != "HARMFUL":
        return False, "no_rollback_needed"
    prev = registry.read_pointer(sid, bus_sym, "PREVIOUS", reg_root)
    if not prev or registry.was_rolled_back(sid, bus_sym, str(prev.get("version") or ""), reg_root):
        return False, "no_previous_model"
    enabled, _ = registry.promotion_enabled(sid, reg_root)
    if not enabled:
        registry.append_event(sid, bus_sym, {"event": "rollback_suppressed", "reason": "promotion_disabled", "evidence": evidence}, reg_root)
        return False, "rollback_suppressed_promotion_disabled"
    ok, why = registry.rollback_to_previous(sid, bus_sym, active_pkl_path, reason=f"attributable_live_harm mean_net={evidence['mean_net']:.6f} ucb={evidence['ucb']:.6f}", root=reg_root)
    if not ok:
        return False, why
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO ai_model_promotion_events (strategy_id, symbol, from_model_id, to_model_id, event_type, reason, metrics_json, created_at)
            VALUES (?, ?, ?, ?, 'rollback', ?, ?, datetime('now'))
            """,
            (
                sid,
                bus_sym,
                ptr.get("version"),
                registry.read_pointer(sid, bus_sym, "ACTIVE", reg_root).get("version"),
                "attributable_live_harm",
                json.dumps(evidence, separators=(",", ":")),
            ),
        )
        conn.commit()
    logger.info("[ROLLBACK] %s %s restored registry PREVIOUS", sid, bus_sym)
    return True, "rollback_executed"
