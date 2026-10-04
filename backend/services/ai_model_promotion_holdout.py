"""
Model-specific holdout PAC validation for promotion candidates.

Builds a per-symbol holdout set from filtered DAY v5 outcome rows, scores both
active and candidate artifacts on the same rows, and returns comparable metrics.
"""

from __future__ import annotations

import json
import os
import pickle
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_canonical_storage import ensure_ai_canonical_tables
from backend.services.ai_model_promotion_pac import (
    FEATURE_DIM_V2,
    FEATURE_VERSION_DAY_HTF,
    MIN_PAC_SAMPLES,
    _row_feature_version,
    _row_net_pnl,
    _row_passes_filters,
    _symbol_forms,
)

HOLDOUT_FRACTION = 0.2
MIN_HOLDOUT_SAMPLES = MIN_PAC_SAMPLES
# Larger holdout = more statistical power to detect real model improvements.
# 20 was too small: two models with different underlying accuracy routinely
# tied at ceiling (both 1.0) because they ranked the same 12 BUY predictions
# in the same order on 20 rows. Env-overrideable. Falls back down to
# MIN_HOLDOUT_SAMPLES when data is genuinely thin.
try:
    _confidence_env = int(os.getenv("MODEL_PROMOTION_MIN_CONFIDENCE_SAMPLES", "60") or "60")
except (TypeError, ValueError):
    _confidence_env = 60
try:
    _target_env = int(os.getenv("MODEL_PROMOTION_TARGET_HOLDOUT_SAMPLES", "60") or "60")
except (TypeError, ValueError):
    _target_env = 60
MIN_HOLDOUT_CONFIDENCE_SAMPLES = max(20, _confidence_env)
TARGET_HOLDOUT_SAMPLES = max(20, _target_env)


def _canonical_bus_symbol(raw: str) -> str:
    s = (raw or "").strip().upper()
    if s.endswith("USDT"):
        return s
    if "/" in s:
        return s.replace("/", "")
    return f"{s}USDT"


def _outcome_label(row: sqlite3.Row) -> int:
    y_label = int(row["outcome_label"] or 0)
    mem_class = str(row["good_bad_memory_class"] or "").strip().upper()
    if mem_class == "BAD":
        y_label = 0
    net_ev_entry = row["selected_net_expected_value"]
    rank_snapshot_id = row["rank_snapshot_id"]
    try:
        if y_label > 0 and rank_snapshot_id not in (None, "", 0) and float(net_ev_entry or 0.0) < 0.0:
            y_label = 0
    except (TypeError, ValueError):
        pass
    return y_label


def _parse_utc(raw: Any) -> datetime | None:
    s = str(raw or "").strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace(" ", "T", 1))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def incumbent_seen_cutoff(active_path: Path | None) -> dict[str, Any]:
    """Outcome rows the incumbent could see when it was trained or selected.

    An incumbent scored on rows it was trained on, or promoted for scoring well
    on, carries a home-field advantage over a candidate that never saw them.
    """
    art = _load_artifact(active_path) if active_path is not None else None
    if art is None:
        return {}
    window = art.get("holdout_window") if isinstance(art.get("holdout_window"), dict) else {}
    ids = [v for v in (art.get("train_outcome_max_id"), window.get("max_id")) if v not in (None, "")]
    trained_at = _parse_utc(art.get("trained_at"))
    out: dict[str, Any] = {}
    if ids:
        out["seen_max_id"] = max(int(v) for v in ids)
    if trained_at is not None:
        out["trained_at"] = trained_at.isoformat()
    return out


def _row_unseen_by_incumbent(row: sqlite3.Row, cutoff: dict[str, Any]) -> bool:
    seen_max_id = cutoff.get("seen_max_id")
    if seen_max_id is not None and int(row["id"]) <= int(seen_max_id):
        return False
    trained_at = _parse_utc(cutoff.get("trained_at"))
    if trained_at is not None:
        closed = _parse_utc(row["closed_at_utc"])
        if closed is None or closed <= trained_at:
            return False
    return True


def _holdout_split(
    sid: str,
    bus: str,
    ccxt: str,
    feature_version: int,
    feature_dim: int,
    db_path: str,
    holdout_fraction: float,
    seen_cutoff: dict[str, Any] | None = None,
) -> tuple[list[sqlite3.Row], int]:
    """(holdout rows, total eligible). Holdout = chronologically last eligible rows
    that closed after everything the incumbent saw (``seen_cutoff``)."""
    ensure_ai_canonical_tables(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, symbol, strategy_id, outcome_class, good_bad_memory_class, churn_flag,
                   features_json, context_json, outcome_label, rank_snapshot_id,
                   selected_net_expected_value, net_pnl_pct, actual_net_outcome, realized_pct, opened_at_utc, closed_at_utc
            FROM ai_outcome_training_rows
            WHERE strategy_id = ?
              AND symbol IN (?, ?)
              AND features_json IS NOT NULL
            ORDER BY id ASC
            """,
            (sid, ccxt, bus),
        ).fetchall()
    eligible = [row for row in rows if _row_passes_filters(row, symbol_bus=bus, min_fv=feature_version, min_dim=feature_dim)]
    if seen_cutoff:
        eligible = [row for row in eligible if _row_unseen_by_incumbent(row, seen_cutoff)]
    total_eligible = len(eligible)
    if total_eligible == 0:
        return [], 0
    if total_eligible >= TARGET_HOLDOUT_SAMPLES:
        split_idx = total_eligible - TARGET_HOLDOUT_SAMPLES
    else:
        split_idx = max(1, int(total_eligible * (1.0 - holdout_fraction)))
        if total_eligible - split_idx < MIN_HOLDOUT_SAMPLES:
            split_idx = max(0, total_eligible - MIN_HOLDOUT_SAMPLES)
    return eligible[split_idx:], total_eligible


def holdout_window(
    *,
    strategy_id: str,
    symbol_bus: str,
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
    db_path: str = DATABASE_PATH,
    holdout_fraction: float = HOLDOUT_FRACTION,
    active_path: Path | None = None,
) -> dict[str, Any]:
    """Row ids and time range of the promotion holdout. Training must exclude ``ids``."""
    sid = (strategy_id or "day").strip().lower()
    bus, ccxt = _symbol_forms(symbol_bus)
    if bus not in TRADING_SYMBOLS:
        return {"ids": [], "n": 0}
    cutoff = incumbent_seen_cutoff(active_path)
    rows, _total = _holdout_split(sid, bus, ccxt, feature_version, feature_dim, db_path, holdout_fraction, cutoff)
    if not rows:
        return {"ids": [], "n": 0, "incumbent_seen_cutoff": cutoff, "forward_rows": 0}
    ids = [int(r["id"]) for r in rows]
    opened = sorted(str(r["opened_at_utc"] or "") for r in rows if r["opened_at_utc"])
    return {
        "ids": ids,
        "n": len(ids),
        "min_id": min(ids),
        "max_id": max(ids),
        "first_closed_at": str(rows[0]["closed_at_utc"] or ""),
        "last_closed_at": str(rows[-1]["closed_at_utc"] or ""),
        "first_opened_at": opened[0] if opened else str(rows[0]["closed_at_utc"] or ""),
        "incumbent_seen_cutoff": cutoff,
    }


def load_symbol_holdout_rows(
    *,
    strategy_id: str,
    symbol_bus: str,
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
    db_path: str = DATABASE_PATH,
    holdout_fraction: float = HOLDOUT_FRACTION,
    active_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Return holdout X, y, net_pnl, good_bad flags, and total eligible count.
    Holdout = chronologically last ``holdout_fraction`` of eligible outcome rows
    the incumbent at ``active_path`` never saw.
    """
    sid = (strategy_id or "day").strip().lower()
    bus, ccxt = _symbol_forms(symbol_bus)
    empty = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
        0,
    )
    if bus not in TRADING_SYMBOLS:
        return empty

    holdout_rows, total_eligible = _holdout_split(sid, bus, ccxt, feature_version, feature_dim, db_path, holdout_fraction, incumbent_seen_cutoff(active_path))
    if total_eligible == 0:
        return empty
    if not holdout_rows:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            total_eligible,
        )

    xs: list[list[float]] = []
    ys: list[int] = []
    nets: list[float] = []
    gbs: list[str] = []
    for row in holdout_rows:
        feats = json.loads(row["features_json"])
        net = _row_net_pnl(row)
        if net is None:
            continue
        try:
            from backend.services.day_feature_health import zero_learning_blocked_feature_dims

            xs.append(zero_learning_blocked_feature_dims([float(x) for x in feats]))
        except Exception:
            xs.append([float(x) for x in feats])
        ys.append(_outcome_label(row))
        nets.append(float(net))
        gbs.append(str(row["good_bad_memory_class"] or "").strip().upper())

    if not xs:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
            total_eligible,
        )
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        np.asarray(nets, dtype=np.float64),
        np.asarray(gbs, dtype=object),
        total_eligible,
    )


def load_forward_rows(
    *,
    strategy_id: str,
    symbol_bus: str,
    opened_after_utc: str,
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
    db_path: str = DATABASE_PATH,
) -> dict[str, Any]:
    """Closed outcomes whose trade opened strictly after ``opened_after_utc`` (a model's training-data end)."""
    sid = (strategy_id or "day").strip().lower()
    bus, ccxt = _symbol_forms(symbol_bus)
    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, symbol, strategy_id, outcome_class, good_bad_memory_class, churn_flag,
                   features_json, context_json, outcome_label, rank_snapshot_id,
                   selected_net_expected_value, net_pnl_pct, actual_net_outcome, realized_pct, opened_at_utc, closed_at_utc
            FROM ai_outcome_training_rows
            WHERE strategy_id = ? AND symbol IN (?, ?) AND features_json IS NOT NULL
              AND julianday(opened_at_utc) > julianday(?)
            ORDER BY id ASC
            """,
            (sid, ccxt, bus, opened_after_utc),
        ).fetchall()
    rows = [r for r in rows if _row_passes_filters(r, symbol_bus=bus, min_fv=feature_version, min_dim=feature_dim)]
    xs, ys, nets, gbs, ids = [], [], [], [], []
    from backend.services.day_feature_health import zero_learning_blocked_feature_dims

    for r in rows:
        net = _row_net_pnl(r)
        if net is None:
            continue
        xs.append(zero_learning_blocked_feature_dims([float(x) for x in json.loads(r["features_json"])]))
        ys.append(_outcome_label(r))
        nets.append(float(net))
        gbs.append(str(r["good_bad_memory_class"] or "").strip().upper())
        ids.append(int(r["id"]))
    return {
        "X": np.asarray(xs, dtype=np.float64),
        "y": np.asarray(ys, dtype=np.int64),
        "nets": np.asarray(nets, dtype=np.float64),
        "good_bad": np.asarray(gbs, dtype=object),
        "ids": ids,
        "first_closed_at": str(rows[0]["closed_at_utc"]) if rows else None,
        "last_closed_at": str(rows[-1]["closed_at_utc"]) if rows else None,
    }


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    scaler = payload.get("scaler")
    if model is None or scaler is None:
        return None
    return payload


def artifact_predictions(artifact_path: Path | None, X: np.ndarray) -> np.ndarray | None:
    """Per-row predictions of one artifact on the holdout matrix (None if unservable)."""
    if artifact_path is None or len(X) == 0:
        return None
    art = _load_artifact(Path(artifact_path))
    if art is None:
        return None
    try:
        return np.asarray(art["model"].predict(art["scaler"].transform(X)), dtype=np.int64).reshape(-1)
    except Exception:
        return None


def evaluate_artifact_on_holdout(
    artifact_path: Path,
    X: np.ndarray,
    y: np.ndarray,
    nets: np.ndarray,
    good_bad: np.ndarray,
) -> dict[str, Any]:
    """Score one artifact on a shared holdout matrix."""
    art = _load_artifact(artifact_path)
    if art is None or len(X) == 0:
        return {
            "accuracy": None,
            "profit_after_cost_if_followed": None,
            "avg_net_pnl_pct_if_followed": None,
            "win_rate_after_cost_if_followed": None,
            "bad_trade_rate_if_followed": None,
            "sample_count": len(X),
            "buy_signal_count": 0,
            "hold_signal_count": 0,
            "artifact_ok": False,
        }

    scaler = art["scaler"]
    model = art["model"]
    Xs = scaler.transform(X)
    preds = model.predict(Xs)
    preds = np.asarray(preds, dtype=np.int64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    nets_arr = np.asarray(nets, dtype=np.float64).reshape(-1)

    sample_count = len(y_arr)
    accuracy = float(np.mean(preds == y_arr)) if sample_count else None

    followed_pnls: list[float] = []
    wins = 0
    bad = 0
    buy_count = 0
    hold_count = 0
    for pred, net, gb in zip(preds, nets_arr, good_bad, strict=False):
        gb_u = str(gb or "").strip().upper()
        if int(pred) == 1:
            buy_count += 1
            followed_pnls.append(float(net))
            if float(net) > 0:
                wins += 1
            if gb_u == "BAD" or float(net) <= 0:
                bad += 1
        else:
            hold_count += 1
            followed_pnls.append(0.0)

    profit = sum(followed_pnls) / sample_count if sample_count else None
    win_rate = wins / sample_count if sample_count else None
    bad_rate = bad / sample_count if sample_count else None
    # Precision among predicted BUYs (what matters for buy_margin edge).
    buy_precision = (wins / buy_count) if buy_count > 0 else None

    return {
        "accuracy": round(accuracy, 6) if accuracy is not None else None,
        "profit_after_cost_if_followed": round(profit, 6) if profit is not None else None,
        "avg_net_pnl_pct_if_followed": round(profit, 6) if profit is not None else None,
        "win_rate_after_cost_if_followed": round(win_rate, 6) if win_rate is not None else None,
        "bad_trade_rate_if_followed": round(bad_rate, 6) if bad_rate is not None else None,
        "buy_precision_if_followed": round(buy_precision, 6) if buy_precision is not None else None,
        "sample_count": sample_count,
        "buy_signal_count": int(buy_count),
        "hold_signal_count": int(hold_count),
        "artifact_ok": True,
        "artifact_accuracy_stored": round(float(art.get("accuracy") or 0.0), 6),
    }


MIN_TIERED_HOLDOUT_SAMPLES = 40


def _tiered_holdout_comparison(
    *,
    strategy_id: str,
    symbol_bus: str,
    candidate_path: Path,
    active_path: Path | None,
    feature_dim: int,
    feature_version: int,
    db_path: str = DATABASE_PATH,
) -> dict[str, Any]:
    """Tier C synthetic holdout (labeled candidate snapshots) for promotion fallback."""
    out: dict[str, Any] = {"tiered_holdout": {}, "tiered_holdout_pass": False}
    try:
        from backend.services.ai_learning_ingestion import tiered_holdout_eval_rows

        xs, ys = tiered_holdout_eval_rows(
            strategy_id=strategy_id,
            symbol=symbol_bus,
            feature_dim=feature_dim,
            min_feature_version=feature_version,
            db_path=db_path,
        )
    except Exception:
        return out
    if len(xs) < MIN_TIERED_HOLDOUT_SAMPLES:
        out["tiered_holdout"] = {"sample_count": len(xs), "status": "INSUFFICIENT"}
        return out

    X = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.int64)

    def _preds(path: Path | None) -> np.ndarray | None:
        if path is None or not path.exists():
            return None
        art = _load_artifact(path)
        if art is None:
            return None
        try:
            return np.asarray(art["model"].predict(art["scaler"].transform(X)), dtype=np.int64).reshape(-1)
        except Exception:
            return None

    c_preds = _preds(candidate_path)
    a_preds = _preds(active_path)
    c_acc = float(np.mean(c_preds == y)) if c_preds is not None else None
    a_acc = float(np.mean(a_preds == y)) if a_preds is not None else None

    # Degeneracy guard: a constant-output classifier can score deceptively high
    # accuracy on the tiered set purely because HOLD (or BUY) happens to be the
    # majority label — that is not a model that learned anything, and this
    # tiered path was built to be trusted as a promotion signal (see
    # register_candidate_and_maybe_promote's tiered_holdout_fallback), so it
    # must not pass a candidate that never varies its output. Mirrors the
    # candidate_always_buy/candidate_always_hold guard already applied to the
    # real-holdout path.
    c_buy_rate = float(np.mean(c_preds == 1)) if c_preds is not None else None
    candidate_degenerate = c_buy_rate is not None and (c_buy_rate <= 0.0 or c_buy_rate >= 1.0)

    out["tiered_holdout"] = {
        "status": "OK",
        "source": "ai_candidate_snapshots_tier_c",
        "sample_count": len(y),
        "buy_label_count": int(np.sum(y == 1)),
        "candidate_accuracy": round(c_acc, 6) if c_acc is not None else None,
        "active_accuracy": round(a_acc, 6) if a_acc is not None else None,
        "candidate_buy_rate": round(c_buy_rate, 6) if c_buy_rate is not None else None,
        "candidate_degenerate": candidate_degenerate,
    }
    if c_acc is not None and not candidate_degenerate and (a_acc is None or c_acc >= a_acc):
        out["tiered_holdout_pass"] = True
    return out


def build_holdout_validation_metrics(
    *,
    strategy_id: str,
    symbol_bus: str,
    candidate_path: Path,
    active_path: Path | None,
    feature_version: int = FEATURE_VERSION_DAY_HTF,
    feature_dim: int = FEATURE_DIM_V2,
    db_path: str = DATABASE_PATH,
    rf_val_samples: int | None = None,
) -> dict[str, Any]:
    """Evaluate active and candidate models on the same holdout rows."""
    sid = (strategy_id or "day").strip().lower()
    bus = _canonical_bus_symbol(symbol_bus)
    X, y, nets, gbs, total_eligible = load_symbol_holdout_rows(
        strategy_id=sid,
        symbol_bus=bus,
        feature_version=feature_version,
        feature_dim=feature_dim,
        db_path=db_path,
        active_path=active_path,
    )
    holdout_count = len(y)
    holdout_low_confidence = holdout_count < MIN_HOLDOUT_CONFIDENCE_SAMPLES
    holdout_buy_labels = int(np.sum(np.asarray(y, dtype=np.int64) == 1)) if holdout_count else 0
    base: dict[str, Any] = {
        "symbol": bus,
        "strategy": sid,
        "feature_version": int(feature_version),
        "feature_dim": int(feature_dim),
        "holdout_eligible_rows": int(total_eligible),
        "holdout_sample_count": holdout_count,
        "holdout_target_samples": TARGET_HOLDOUT_SAMPLES,
        "holdout_buy_label_count": holdout_buy_labels,
        "holdout_low_confidence": holdout_low_confidence,
        "holdout_status": "HOLDOUT_PAC_UNAVAILABLE",
        "pac_source": "ai_outcome_training_rows_holdout",
        "active_holdout": {},
        "candidate_holdout": {},
    }
    window = holdout_window(strategy_id=sid, symbol_bus=bus, feature_version=feature_version, feature_dim=feature_dim, db_path=db_path, active_path=active_path)
    base["holdout_window"] = {k: v for k, v in window.items() if k != "ids"}
    if rf_val_samples is not None:
        base["rf_val_samples"] = int(rf_val_samples)

    # Tier C synthetic comparison: diagnostic only, never promotion authority.
    if holdout_low_confidence:
        base.update(
            _tiered_holdout_comparison(
                strategy_id=sid,
                symbol_bus=bus,
                candidate_path=candidate_path,
                active_path=active_path,
                feature_dim=int(feature_dim),
                feature_version=int(feature_version),
                db_path=db_path,
            )
        )

    if holdout_count == 0:
        return base

    from backend.services.ai_model_promotion_decision import compare_paired

    c_preds = artifact_predictions(candidate_path, X)
    a_preds = artifact_predictions(active_path, X) if active_path is not None and active_path.exists() else None
    if c_preds is not None and a_preds is not None:
        base["paired_decision"] = compare_paired(c_preds, a_preds, nets, gbs, y)
    elif c_preds is None:
        base["paired_decision"] = {"verdict": "CANDIDATE_UNSERVABLE", "promote": False, "n": holdout_count}
    else:
        base["paired_decision"] = {"verdict": "NO_INCUMBENT", "promote": False, "n": holdout_count}

    candidate_holdout = evaluate_artifact_on_holdout(candidate_path, X, y, nets, gbs)
    active_holdout: dict[str, Any]
    if active_path is not None and active_path.exists():
        active_holdout = evaluate_artifact_on_holdout(active_path, X, y, nets, gbs)
    else:
        active_holdout = {
            "accuracy": None,
            "profit_after_cost_if_followed": None,
            "avg_net_pnl_pct_if_followed": None,
            "win_rate_after_cost_if_followed": None,
            "bad_trade_rate_if_followed": None,
            "sample_count": holdout_count,
            "artifact_ok": False,
        }

    if not candidate_holdout.get("artifact_ok"):
        return {**base, "active_holdout": active_holdout, "candidate_holdout": candidate_holdout}

    c_acc = candidate_holdout.get("accuracy")
    a_acc = active_holdout.get("accuracy")
    c_profit = candidate_holdout.get("profit_after_cost_if_followed")
    a_profit = active_holdout.get("profit_after_cost_if_followed")
    c_bad = candidate_holdout.get("bad_trade_rate_if_followed")
    a_bad = active_holdout.get("bad_trade_rate_if_followed")

    if c_acc is None or c_profit is None or c_bad is None:
        return {
            **base,
            "active_holdout": active_holdout,
            "candidate_holdout": candidate_holdout,
        }
    if active_path is not None and active_path.exists() and (a_acc is None or a_profit is None or a_bad is None):
        return {
            **base,
            "active_holdout": active_holdout,
            "candidate_holdout": candidate_holdout,
        }

    base.update(
        {
            "holdout_status": "OK",
            "pac_status": "OK",
            "active_holdout": active_holdout,
            "candidate_holdout": candidate_holdout,
            "active_accuracy": a_acc,
            "candidate_accuracy": c_acc,
            "active_profit_after_cost": a_profit,
            "candidate_profit_after_cost": c_profit,
            "active_bad_trade_rate": a_bad,
            "candidate_bad_trade_rate": c_bad,
            "profit_after_cost": c_profit,
            "avg_net_pnl_pct": candidate_holdout.get("avg_net_pnl_pct_if_followed"),
            "win_rate_after_cost": candidate_holdout.get("win_rate_after_cost_if_followed"),
            "bad_trade_rate": c_bad,
            "sample_count": holdout_count,
            "win_rate_after_cost_if_followed": candidate_holdout.get("win_rate_after_cost_if_followed"),
            "bad_trade_rate_if_followed": c_bad,
        }
    )
    return base


__all__ = [
    "HOLDOUT_FRACTION",
    "MIN_HOLDOUT_CONFIDENCE_SAMPLES",
    "MIN_HOLDOUT_SAMPLES",
    "TARGET_HOLDOUT_SAMPLES",
    "artifact_predictions",
    "build_holdout_validation_metrics",
    "evaluate_artifact_on_holdout",
    "incumbent_seen_cutoff",
    "load_symbol_holdout_rows",
]
