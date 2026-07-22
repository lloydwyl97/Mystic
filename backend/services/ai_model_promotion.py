from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.database_schema import DATABASE_PATH
from backend.services.ai_artifact_contract_gate import evaluate_signal_hash_artifact_contract
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables


def _active_age_hours(path: Path) -> float | None:
    """Age of active artifact in hours (max of mtime and embedded trained_at)."""
    if not path.exists():
        return None
    ages: list[float] = [(time.time() - path.stat().st_mtime) / 3600.0]
    try:
        payload = pickle.loads(path.read_bytes())
        if isinstance(payload, dict):
            trained_at = str(payload.get("trained_at") or "").strip()
            if trained_at:
                dt = datetime.fromisoformat(trained_at.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                ages.append((datetime.now(timezone.utc) - dt).total_seconds() / 3600.0)
    except Exception:
        pass
    return max(ages) if ages else None


def _stale_hours_threshold() -> float:
    return float(os.getenv("MODEL_STALE_HOURS", "72") or "72")


def _accuracy_margin(holdout_count: int) -> float:
    """
    Statistical tolerance for the candidate-vs-active accuracy comparison.

    A raw `c_acc >= a_acc` test with zero tolerance blocks genuinely-improved
    candidates whenever the (often tiny) real holdout set draws unlucky —
    e.g. SOL/XRP validation sets run ~20-22 rows, where a couple of flipped
    predictions swings measured accuracy by >=10 points. Use a one-standard-
    error band on a proportion estimate (p=0.5, max variance, conservative)
    so the gate isn't rejecting on pure sampling noise, floored/ceilinged so
    it's never a no-op and never absurdly loose for very small n.
    """
    n = max(int(holdout_count or 0), 1)
    z = float(os.getenv("MODEL_PROMOTION_ACCURACY_MARGIN_Z", "1.0") or "1.0")
    se_band = z * math.sqrt(0.25 / n)
    min_margin = float(os.getenv("MODEL_PROMOTION_ACCURACY_MIN_MARGIN", "0.01") or "0.01")
    max_margin = float(os.getenv("MODEL_PROMOTION_ACCURACY_MAX_MARGIN", "0.15") or "0.15")
    return max(min_margin, min(se_band, max_margin))


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


def _metrics_profit_after_cost(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    for key in (
        "candidate_profit_after_cost",
        "profit_after_cost_if_followed",
        "profit_after_cost",
        "profit_after_cost_pct",
        "avg_net_pnl_pct_if_followed",
        "avg_net_pnl_pct",
    ):
        if metrics.get(key) is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    holdout = metrics.get("candidate_holdout")
    if isinstance(holdout, dict) and holdout.get("profit_after_cost_if_followed") is not None:
        try:
            return float(holdout["profit_after_cost_if_followed"])
        except (TypeError, ValueError):
            pass
    return None


def _metrics_bad_trade_rate(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    for key in ("candidate_bad_trade_rate", "bad_trade_rate_if_followed", "bad_trade_rate"):
        if metrics.get(key) is not None:
            try:
                return float(metrics[key])
            except (TypeError, ValueError):
                continue
    holdout = metrics.get("candidate_holdout")
    if isinstance(holdout, dict) and holdout.get("bad_trade_rate_if_followed") is not None:
        try:
            return float(holdout["bad_trade_rate_if_followed"])
        except (TypeError, ValueError):
            pass
    return None


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


def _compose_promotion_reason(
    *,
    accuracy_ok: bool,
    c_acc: float,
    a_acc: float,
    holdout_status: str,
    c_profit: float | None,
    a_profit: float | None,
    bad_ok: bool,
    c_bad: float | None,
    a_bad: float | None,
    accuracy_margin: float = 0.0,
) -> str:
    parts: list[str] = []
    if holdout_status == "HOLDOUT_PAC_UNAVAILABLE":
        parts.append("HOLDOUT_PAC_UNAVAILABLE")
    if not accuracy_ok:
        parts.append(f"candidate_accuracy_below_active:{c_acc:.4f}<{a_acc:.4f}-{accuracy_margin:.4f}margin")
    if c_profit is not None and a_profit is not None and c_profit < (a_profit - 0.0005):
        parts.append(f"candidate_profit_after_cost_below_active:{c_profit:.6f}<{a_profit:.6f}")
    if not bad_ok and c_bad is not None and a_bad is not None:
        parts.append(f"candidate_bad_trade_rate_above_active:{c_bad:.6f}>{a_bad:.6f}")
    if not parts:
        return "validation_fail"
    return ";".join(parts)


def _load_active_validation_metrics(db_path: str, strategy_id: str, symbol: str) -> dict[str, Any]:
    sid = strategy_id.strip().lower()
    sym = symbol.strip().upper()
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute(
                """
                SELECT validation_metrics_json FROM ai_model_versions
                WHERE strategy_id=? AND symbol=? AND status='active'
                ORDER BY id DESC LIMIT 1
                """,
                (sid, sym),
            ).fetchone()
            if row and row[0]:
                parsed = json.loads(row[0])
                if isinstance(parsed, dict):
                    return parsed
    except Exception:
        pass
    return {}


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

    cur_meta = _artifact_meta(active_path) if active_path.exists() else {"accuracy": -1.0, "ok": False}
    metrics = dict(validation_metrics or {})
    holdout_status = str(metrics.get("holdout_status") or "HOLDOUT_PAC_UNAVAILABLE")
    c_acc = _holdout_accuracy(metrics, role="candidate")
    a_acc = _holdout_accuracy(metrics, role="active")
    if c_acc is None:
        c_acc = float(c_meta.get("accuracy") or 0.0)
        metrics.setdefault("candidate_accuracy", round(c_acc, 6))
    if active_path.exists() and a_acc is None:
        a_acc = float(cur_meta.get("accuracy") or -1.0)
        metrics.setdefault("active_accuracy", round(a_acc, 6))
    c_profit = _metrics_profit_after_cost(metrics)
    a_profit = metrics.get("active_profit_after_cost")
    if a_profit is not None:
        try:
            a_profit = float(a_profit)
        except (TypeError, ValueError):
            a_profit = None
    if a_profit is None:
        active_holdout = metrics.get("active_holdout")
        if isinstance(active_holdout, dict) and active_holdout.get("profit_after_cost_if_followed") is not None:
            a_profit = float(active_holdout["profit_after_cost_if_followed"])
            metrics["active_profit_after_cost"] = round(a_profit, 6)
    c_bad = _metrics_bad_trade_rate(metrics)
    a_bad = metrics.get("active_bad_trade_rate")
    if a_bad is not None:
        try:
            a_bad = float(a_bad)
        except (TypeError, ValueError):
            a_bad = None
    if a_bad is None:
        active_holdout = metrics.get("active_holdout")
        if isinstance(active_holdout, dict) and active_holdout.get("bad_trade_rate_if_followed") is not None:
            a_bad = float(active_holdout["bad_trade_rate_if_followed"])
            metrics["active_bad_trade_rate"] = round(a_bad, 6)

    has_active = active_path.exists()
    holdout_ok = holdout_status in ("OK", "STALE_EVAL_BYPASS")
    holdout_count = int(metrics.get("holdout_sample_count") or metrics.get("sample_count") or 0)
    accuracy_margin = _accuracy_margin(holdout_count)
    metrics["accuracy_margin_applied"] = round(accuracy_margin, 6)
    if not has_active:
        accuracy_ok = True
        pac_ok = holdout_ok and c_profit is not None
        bad_ok = holdout_ok and c_bad is not None
    else:
        accuracy_ok = holdout_ok and c_acc is not None and a_acc is not None and c_acc >= (a_acc - accuracy_margin)
        pac_ok = holdout_ok and c_profit is not None and a_profit is not None and c_profit >= (a_profit - 0.0005)
        bad_ok = holdout_ok and c_bad is not None and a_bad is not None and c_bad <= (a_bad + 0.0005)

    promote = holdout_ok and accuracy_ok and pac_ok and bad_ok
    holdout_low_confidence = bool(metrics.get("holdout_low_confidence")) or holdout_count < 20
    metrics["holdout_low_confidence"] = holdout_low_confidence
    cand_holdout = metrics.get("candidate_holdout")
    buy_sig = 0
    if isinstance(cand_holdout, dict):
        buy_sig = int(cand_holdout.get("buy_signal_count") or 0)
        candidate_always_buy = holdout_count > 0 and buy_sig >= holdout_count
    else:
        candidate_always_buy = False
    metrics["candidate_always_buy"] = candidate_always_buy
    metrics["candidate_not_always_buy"] = not candidate_always_buy
    holdout_buy_labels = int(metrics.get("holdout_buy_label_count") or 0)
    candidate_always_hold = holdout_count > 0 and buy_sig == 0
    metrics["candidate_always_hold"] = candidate_always_hold
    metrics["candidate_not_always_hold"] = not candidate_always_hold
    reject_reason = _compose_promotion_reason(
        accuracy_ok=accuracy_ok,
        c_acc=float(c_acc or 0.0),
        a_acc=float(a_acc or -1.0),
        holdout_status=holdout_status,
        c_profit=c_profit,
        a_profit=a_profit,
        bad_ok=bad_ok,
        c_bad=c_bad,
        a_bad=a_bad,
        accuracy_margin=accuracy_margin,
    )
    if promote and candidate_always_buy:
        promote = False
        reject_reason = "candidate_always_buy_on_holdout"
    elif promote and candidate_always_hold and holdout_buy_labels > 0:
        promote = False
        reject_reason = "candidate_always_hold_on_holdout"
    elif promote and holdout_low_confidence and has_active:
        # Tiered fallback (learning starvation fix): when real closed-trade
        # holdout is scarce, a Tier C synthetic holdout (labeled rejected /
        # no-trade forward returns) may approve promotion. Real PnL metrics
        # are never derived from the synthetic rows.
        if bool(metrics.get("tiered_holdout_pass")):
            metrics["promotion_path"] = "tiered_holdout_fallback"
        else:
            promote = False
            reject_reason = "holdout_low_confidence"
    elif promote and has_active:
        tied = (
            abs(float(c_acc or 0.0) - float(a_acc or -1.0)) <= 0.0005
            and c_profit is not None
            and a_profit is not None
            and abs(c_profit - a_profit) <= 0.0005
            and c_bad is not None
            and a_bad is not None
            and abs(c_bad - a_bad) <= 0.0005
        )
        if tied:
            age_h = _active_age_hours(active_path)
            stale_h = _stale_hours_threshold()
            metrics["active_age_hours"] = round(float(age_h), 3) if age_h is not None else None
            # Tied candidates are normally rejected to avoid churn — but when the
            # active model is stale, promote so live inference keeps absorbing
            # recent coin/market structure from outcome-weighted retrains.
            if age_h is not None and age_h >= stale_h:
                metrics["promotion_path"] = "stale_refresh_tie"
                metrics["stale_hours_threshold"] = stale_h
            else:
                # Buy-coverage edge: only when active holdout proves always-HOLD
                # and candidate emits a healthy non-zero BUY share.
                active_holdout = metrics.get("active_holdout")
                active_always_hold = False
                if isinstance(active_holdout, dict):
                    a_buy = int(active_holdout.get("buy_signal_count") or 0)
                    a_hold = int(active_holdout.get("hold_signal_count") or 0)
                    active_always_hold = a_buy == 0 and a_hold > 0
                if active_always_hold and 0 < buy_sig < max(1, int(holdout_count * 0.55)):
                    metrics["promotion_path"] = "buy_coverage_edge"
                else:
                    promote = False
                    reject_reason = "candidate_not_improved_over_active"
    # Soft promote: accuracy within margin already (accuracy_ok), but PAC/precision
    # clearly better — prefer tradable buy edge over pure accuracy ties.
    if (not promote) and holdout_ok and has_active and accuracy_ok and pac_ok and bad_ok:
        cand_h = metrics.get("candidate_holdout") if isinstance(metrics.get("candidate_holdout"), dict) else {}
        act_h = metrics.get("active_holdout") if isinstance(metrics.get("active_holdout"), dict) else {}
        c_bp = cand_h.get("buy_precision_if_followed")
        a_bp = act_h.get("buy_precision_if_followed")
        try:
            c_bp_f = float(c_bp) if c_bp is not None else None
            a_bp_f = float(a_bp) if a_bp is not None else None
        except (TypeError, ValueError):
            c_bp_f = a_bp_f = None
        precision_edge = c_bp_f is not None and a_bp_f is not None and buy_sig >= 3 and c_bp_f >= (a_bp_f + 0.02)
        pac_edge = c_profit is not None and a_profit is not None and float(c_profit) >= (float(a_profit) + 0.0002)
        if (precision_edge or pac_edge) and not candidate_always_buy and not candidate_always_hold:
            promote = True
            reject_reason = "validation_pass"
            metrics["promotion_path"] = "buy_precision_or_pac_edge"
            metrics["candidate_buy_precision"] = c_bp_f
            metrics["active_buy_precision"] = a_bp_f
    model_id = f"{sid}:{sym}:{c_hash[:16]}"
    active_model_id = f"{sid}:{sym}:{_hash_file(active_path)[:16]}" if active_path.exists() else None
    status = "candidate"
    reason = "validation_pass"
    if promote:
        active_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate_path, active_path)
        status = "active"
        reason = str(metrics.get("promotion_path") or "promoted")
    else:
        reason = reject_reason

    with sqlite3.connect(db_path) as conn:
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
                sym,
                int(c_meta.get("feature_version") or 0),
                c_hash,
                str(active_path if promote else candidate_path),
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
                sym,
                active_model_id,
                model_id,
                ("promote" if promote else "reject"),
                reason,
                json.dumps(metrics, separators=(",", ":")),
            ),
        )
        conn.commit()
    return promote, reason


def maybe_rollback_underperforming_model(
    *,
    strategy_id: str,
    symbol: str,
    min_samples: int = 20,
    db_path: str = DATABASE_PATH,
) -> tuple[bool, str]:
    """Rollback active model when recent live net outcomes materially degrade."""
    ensure_ai_canonical_tables(db_path)
    sid = strategy_id.strip().lower()
    sym = symbol.strip().upper()
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT net_pnl_pct
            FROM ai_outcome_training_rows
            WHERE strategy_id = ? AND UPPER(symbol) = UPPER(?)
            ORDER BY id DESC
            LIMIT ?
            """,
            (sid, sym, int(min_samples)),
        ).fetchall()
        if len(rows) < min_samples:
            return False, "insufficient_live_samples"
        avg_net = sum(float(r[0] or 0.0) for r in rows) / max(1, len(rows))
        if avg_net >= -0.0015:
            return False, "no_rollback_needed"
        active = conn.execute(
            """
            SELECT model_id, path FROM ai_model_versions
            WHERE strategy_id = ? AND symbol = ? AND status = 'active'
            ORDER BY id DESC
            LIMIT 1
            """,
            (sid, sym),
        ).fetchone()
        prev = conn.execute(
            """
            SELECT model_id, path FROM ai_model_versions
            WHERE strategy_id = ? AND symbol = ? AND status IN ('archived', 'rollback')
            ORDER BY id DESC
            LIMIT 1
            """,
            (sid, sym),
        ).fetchone()
        if not active or not prev:
            return False, "no_previous_model"
        conn.execute("UPDATE ai_model_versions SET status='rollback', rollback_reason=?, retired_at=datetime('now') WHERE model_id=?", (f"avg_net={avg_net:.6f}", active[0]))
        conn.execute("UPDATE ai_model_versions SET status='active', promoted_at=datetime('now') WHERE model_id=?", (prev[0],))
        conn.execute(
            """
            INSERT INTO ai_model_promotion_events (strategy_id, symbol, from_model_id, to_model_id, event_type, reason, metrics_json, created_at)
            VALUES (?, ?, ?, ?, 'rollback', ?, ?, datetime('now'))
            """,
            (sid, sym, active[0], prev[0], "live_underperformance", json.dumps({"avg_recent_net_pnl_pct": avg_net}, separators=(",", ":"))),
        )
        conn.commit()
    # Restore previous model artifact to the live active path on disk.
    from backend.services.live_strategy_contracts import per_coin_artifact_file  # local import to avoid circular
    prev_artifact_path = prev[1] if prev and len(prev) > 1 else None
    active_pkl_path = per_coin_artifact_file(Path("models/active"), sid, sym)
    if prev_artifact_path and os.path.exists(prev_artifact_path):
        active_pkl_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(prev_artifact_path, active_pkl_path)
        logger.info("[ROLLBACK] Restored artifact %s -> %s", prev_artifact_path, active_pkl_path)
    else:
        logger.warning("[ROLLBACK] Previous artifact not found: %s", prev_artifact_path)
    return True, "rollback_executed"
