"""
Read-only model behavior diagnostics on promotion holdout rows.

Explains prediction distributions, active vs candidate disagreement, BUY bias,
and whether holdout size is sufficient for promotion confidence.
"""

from __future__ import annotations

import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from backend.config.trading_universe import TRADING_SYMBOLS
from backend.database_schema import DATABASE_PATH
from backend.services.ai_model_promotion_holdout import (
    FEATURE_DIM_V2,
    FEATURE_VERSION_DAY_HTF,
    load_symbol_holdout_rows,
)
from backend.services.live_strategy_contracts import per_coin_artifact_file

MIN_HOLDOUT_CONFIDENCE_SAMPLES = 20
BUY_BIAS_THRESHOLD = 0.75
THRESHOLD_SIM_LEVELS = (0.50, 0.55, 0.60, 0.65, 0.70)


def _load_artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = pickle.loads(path.read_bytes())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("model") is None or payload.get("scaler") is None:
        return None
    return payload


def _action_from_pred(pred: int, n_classes: int) -> str:
    p = int(pred)
    if n_classes >= 3:
        return ("SELL", "HOLD", "BUY")[min(max(p, 0), 2)]
    return "BUY" if p == 1 else "HOLD"


def _predict_artifact(
    art: dict[str, Any],
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    scaler = art["scaler"]
    model = art["model"]
    Xs = scaler.transform(X)
    preds = np.asarray(model.predict(Xs), dtype=np.int64).reshape(-1)
    proba = None
    n_classes = 2
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(Xs), dtype=np.float64)
        n_classes = int(proba.shape[1])
    return preds, proba, n_classes


def _distribution(preds: np.ndarray, n_classes: int) -> dict[str, int]:
    dist = {"BUY": 0, "HOLD": 0, "SELL": 0}
    for pred in preds:
        action = _action_from_pred(int(pred), n_classes)
        dist[action] += 1
    return dist


def _label_distribution(y: np.ndarray) -> dict[str, int]:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    buy = int(np.sum(y_arr == 1))
    hold = int(np.sum(y_arr == 0))
    return {"BUY": buy, "HOLD": hold, "SELL": 0}


def _avg_confidence_by_class(
    preds: np.ndarray,
    proba: np.ndarray | None,
    n_classes: int,
) -> dict[str, float | None]:
    buckets: dict[str, list[float]] = {"BUY": [], "HOLD": [], "SELL": []}
    if proba is None:
        return dict.fromkeys(buckets)
    for i, pred in enumerate(preds):
        action = _action_from_pred(int(pred), n_classes)
        idx = int(pred)
        if n_classes == 2:
            idx = 1 if int(pred) == 1 else 0
        if 0 <= idx < proba.shape[1]:
            buckets[action].append(float(proba[i, idx]))
    return {k: round(sum(v) / len(v), 6) if v else None for k, v in buckets.items()}


def _row_follow_pnl(pred: int, net: float, n_classes: int) -> float:
    action = _action_from_pred(int(pred), n_classes)
    if action == "BUY":
        return float(net)
    return 0.0


def _compare_rows(
    active_preds: np.ndarray,
    candidate_preds: np.ndarray,
    y: np.ndarray,
    nets: np.ndarray,
    n_classes: int,
) -> dict[str, int]:
    improved = 0
    worsened = 0
    pac_improved = 0
    pac_worsened = 0
    disagreement = 0
    for a_p, c_p, label, net in zip(active_preds, candidate_preds, y, nets, strict=False):
        if int(a_p) != int(c_p):
            disagreement += 1
        a_ok = int(a_p) == int(label)
        c_ok = int(c_p) == int(label)
        if c_ok and not a_ok:
            improved += 1
        elif a_ok and not c_ok:
            worsened += 1
        a_pnl = _row_follow_pnl(int(a_p), float(net), n_classes)
        c_pnl = _row_follow_pnl(int(c_p), float(net), n_classes)
        if c_pnl > a_pnl + 1e-9:
            pac_improved += 1
        elif c_pnl < a_pnl - 1e-9:
            pac_worsened += 1
    return {
        "disagreement_count": disagreement,
        "rows_candidate_improved": improved,
        "rows_candidate_worsened": worsened,
        "rows_pac_improved": pac_improved,
        "rows_pac_worsened": pac_worsened,
    }


def _prob_stats(proba: np.ndarray | None, n_classes: int) -> dict[str, Any]:
    if proba is None or n_classes < 2:
        return {
            "prob_buy": None,
            "prob_hold": None,
            "buy_hold_margin_mean": None,
            "prob_buy_distribution": None,
            "prob_hold_distribution": None,
        }
    prob_buy = proba[:, 1]
    prob_hold = proba[:, 0]
    margin = prob_buy - prob_hold

    def _dist(arr: np.ndarray) -> dict[str, float]:
        return {
            "min": round(float(np.min(arr)), 6),
            "max": round(float(np.max(arr)), 6),
            "mean": round(float(np.mean(arr)), 6),
            "p50": round(float(np.percentile(arr, 50)), 6),
            "p90": round(float(np.percentile(arr, 90)), 6),
        }

    return {
        "prob_buy": _dist(prob_buy),
        "prob_hold": _dist(prob_hold),
        "buy_hold_margin_mean": round(float(np.mean(margin)), 6),
        "prob_buy_distribution": _dist(prob_buy),
        "prob_hold_distribution": _dist(prob_hold),
    }


def _false_buy_stats(preds: np.ndarray, y: np.ndarray) -> dict[str, int]:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    p_arr = np.asarray(preds, dtype=np.int64).reshape(-1)
    false_buy = int(np.sum((p_arr == 1) & (y_arr == 0)))
    missed_good = int(np.sum((p_arr == 0) & (y_arr == 1)))
    return {"false_buy_count": false_buy, "missed_good_buy_count": missed_good}


def _simulate_thresholds(
    proba: np.ndarray | None,
    y: np.ndarray,
    nets: np.ndarray,
    n_classes: int,
) -> list[dict[str, Any]]:
    if proba is None or n_classes < 2:
        return []
    prob_buy = proba[:, 1]
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    nets_arr = np.asarray(nets, dtype=np.float64).reshape(-1)
    rows: list[dict[str, Any]] = []
    for thr in THRESHOLD_SIM_LEVELS:
        preds = (prob_buy >= thr).astype(np.int64)
        buy_count = int(np.sum(preds == 1))
        hold_count = int(len(preds) - buy_count)
        accuracy = float(np.mean(preds == y_arr)) if len(y_arr) else None
        false_buy = int(np.sum((preds == 1) & (y_arr == 0)))
        missed_good = int(np.sum((preds == 0) & (y_arr == 1)))
        followed: list[float] = []
        bad = 0
        for pred, net in zip(preds, nets_arr, strict=False):
            if int(pred) == 1:
                followed.append(float(net))
                if float(net) <= 0:
                    bad += 1
            else:
                followed.append(0.0)
        n = max(1, len(y_arr))
        pac = sum(followed) / n
        bad_rate = bad / n
        rows.append(
            {
                "threshold": thr,
                "buy_count": buy_count,
                "hold_count": hold_count,
                "accuracy": round(accuracy, 6) if accuracy is not None else None,
                "profit_after_cost_if_followed": round(pac, 6),
                "bad_trade_rate_if_followed": round(bad_rate, 6),
                "false_buy_count": false_buy,
                "missed_good_buy_count": missed_good,
            }
        )
    return rows


def _class_balance_summary(y: np.ndarray) -> dict[str, Any]:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    buy = int(np.sum(y_arr == 1))
    hold = int(np.sum(y_arr == 0))
    total = len(y_arr)
    return {
        "BUY": buy,
        "HOLD": hold,
        "buy_rate": round(buy / total, 6) if total else None,
        "hold_rate": round(hold / total, 6) if total else None,
    }


def _buy_bias_score(dist: dict[str, int], sample_count: int) -> float | None:
    if sample_count <= 0:
        return None
    return round(dist["BUY"] / sample_count, 6)


def _buy_bias_status(dist: dict[str, int], sample_count: int) -> dict[str, Any]:
    if sample_count <= 0:
        return {"buy_biased": False, "buy_rate": None, "reason": "no_holdout_samples"}
    buy_rate = dist["BUY"] / sample_count
    buy_biased = buy_rate >= BUY_BIAS_THRESHOLD
    reason = "over_predicting_buy" if buy_biased else "balanced_or_hold_heavy"
    if dist["BUY"] == sample_count:
        reason = "always_buy_on_holdout"
    return {
        "buy_biased": buy_biased,
        "buy_rate": round(buy_rate, 6),
        "reason": reason,
    }


def _tie_explanation(
    *,
    disagreement_count: int,
    active_dist: dict[str, int],
    candidate_dist: dict[str, int],
    sample_count: int,
) -> str:
    if sample_count == 0:
        return "no_holdout_samples"
    if disagreement_count > 0:
        return "models_disagree_on_holdout"
    if active_dist == candidate_dist:
        if active_dist["BUY"] == sample_count:
            return "identical_always_buy_predictions"
        if active_dist["HOLD"] == sample_count:
            return "identical_always_hold_predictions"
        return "identical_prediction_distribution"
    return "identical_argmax_on_holdout"


def _latest_candidate_path(version_dir: Path, symbol: str) -> Path | None:
    pattern = sorted(version_dir.glob(f"day_{symbol}_*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pattern[0] if pattern else None


def build_symbol_model_behavior(
    symbol_bus: str,
    *,
    strategy_id: str = "day",
    db_path: str = DATABASE_PATH,
    models_dir: Path | str = "models/active",
    version_dir: Path | str = "models/versions/per_coin",
) -> dict[str, Any]:
    sym = symbol_bus.strip().upper()
    sid = strategy_id.strip().lower()
    X, y, nets, _gbs, eligible = load_symbol_holdout_rows(
        strategy_id=sid,
        symbol_bus=sym,
        feature_version=FEATURE_VERSION_DAY_HTF,
        feature_dim=FEATURE_DIM_V2,
        db_path=db_path,
    )
    sample_count = len(y)
    holdout_low_confidence = sample_count < MIN_HOLDOUT_CONFIDENCE_SAMPLES
    active_path = per_coin_artifact_file(models_dir, sid, sym)
    cand_path = _latest_candidate_path(Path(version_dir), sym)

    base: dict[str, Any] = {
        "symbol": sym,
        "strategy": sid,
        "feature_version": FEATURE_VERSION_DAY_HTF,
        "feature_dim": FEATURE_DIM_V2,
        "holdout_eligible_rows": int(eligible),
        "holdout_sample_count": sample_count,
        "holdout_low_confidence": holdout_low_confidence,
        "holdout_confidence_status": ("low_confidence" if holdout_low_confidence else "adequate"),
        "holdout_target_samples": 20,
        "holdout_buy_label_count": int(np.sum(y == 1)) if sample_count else 0,
        "min_holdout_confidence_samples": MIN_HOLDOUT_CONFIDENCE_SAMPLES,
        "active_path": str(active_path),
        "candidate_path": str(cand_path) if cand_path else None,
        "actual_label_distribution": _label_distribution(y) if sample_count else {"BUY": 0, "HOLD": 0, "SELL": 0},
        "active_prediction_distribution": {"BUY": 0, "HOLD": 0, "SELL": 0},
        "candidate_prediction_distribution": {"BUY": 0, "HOLD": 0, "SELL": 0},
        "active_avg_confidence_by_class": {"BUY": None, "HOLD": None, "SELL": None},
        "candidate_avg_confidence_by_class": {"BUY": None, "HOLD": None, "SELL": None},
        "disagreement_count": 0,
        "rows_candidate_improved": 0,
        "rows_candidate_worsened": 0,
        "rows_pac_improved": 0,
        "rows_pac_worsened": 0,
        "active_buy_bias": {"buy_biased": False, "buy_rate": None, "reason": "no_holdout_samples"},
        "candidate_buy_bias": {"buy_biased": False, "buy_rate": None, "reason": "no_holdout_samples"},
        "holdout_tie_explanation": "no_holdout_samples",
        "holdout_class_balance": _class_balance_summary(y) if sample_count else {"BUY": 0, "HOLD": 0, "buy_rate": None, "hold_rate": None},
        "active_buy_bias_score": None,
        "candidate_buy_bias_score": None,
        "active_false_buy": {"false_buy_count": 0, "missed_good_buy_count": 0},
        "candidate_false_buy": {"false_buy_count": 0, "missed_good_buy_count": 0},
        "active_probability_diagnostics": {},
        "candidate_probability_diagnostics": {},
        "active_threshold_simulation": [],
        "candidate_threshold_simulation": [],
        "candidate_not_always_buy": False,
        "candidate_always_buy": False,
        "candidate_always_hold": False,
        "candidate_not_always_hold": False,
        "candidate_class_weight_mode": None,
        "candidate_train_class_distribution": None,
        "training_data_balance": None,
        "raw_self_supervised_class_distribution": None,
        "raw_outcome_class_distribution": None,
        "final_training_class_distribution": None,
        "effective_sample_weights": None,
        "candidate_holdout_pac": None,
        "active_holdout_pac": None,
        "false_buy_before": None,
        "false_buy_after": None,
        "diagnostics_ok": False,
    }

    if sample_count == 0:
        return base

    active_art = _load_artifact(active_path)
    candidate_art = _load_artifact(cand_path) if cand_path else None
    if active_art is None:
        base["error"] = "active_artifact_missing_or_invalid"
        return base

    active_preds, active_proba, n_classes = _predict_artifact(active_art, X)
    active_dist = _distribution(active_preds, n_classes)
    active_conf = _avg_confidence_by_class(active_preds, active_proba, n_classes)
    active_bias = _buy_bias_status(active_dist, sample_count)
    active_false = _false_buy_stats(active_preds, y)

    base["active_prediction_distribution"] = active_dist
    base["active_avg_confidence_by_class"] = active_conf
    base["active_buy_bias"] = active_bias
    base["active_buy_bias_score"] = _buy_bias_score(active_dist, sample_count)
    base["active_false_buy"] = active_false
    base["false_buy_before"] = dict(active_false)
    base["active_probability_diagnostics"] = _prob_stats(active_proba, n_classes)
    base["active_threshold_simulation"] = _simulate_thresholds(active_proba, y, nets, n_classes)
    base["active_artifact_accuracy_stored"] = round(float(active_art.get("accuracy") or 0.0), 6)

    if candidate_art is None:
        base["holdout_tie_explanation"] = "no_candidate_artifact"
        base["diagnostics_ok"] = True
        return base

    cand_preds, cand_proba, cand_classes = _predict_artifact(candidate_art, X)
    n_classes = max(n_classes, cand_classes)
    cand_dist = _distribution(cand_preds, cand_classes)
    cand_conf = _avg_confidence_by_class(cand_preds, cand_proba, cand_classes)
    cand_bias = _buy_bias_status(cand_dist, sample_count)
    cand_false = _false_buy_stats(cand_preds, y)
    always_buy = cand_dist["BUY"] >= sample_count > 0
    compare = _compare_rows(active_preds, cand_preds, y, nets, n_classes)

    from backend.services.ai_model_promotion_holdout import build_holdout_validation_metrics

    holdout_metrics = build_holdout_validation_metrics(
        strategy_id=sid,
        symbol_bus=sym,
        candidate_path=cand_path,
        active_path=active_path,
    )
    active_h = holdout_metrics.get("active_holdout") or {}
    cand_h = holdout_metrics.get("candidate_holdout") or {}

    base.update(
        {
            "candidate_prediction_distribution": cand_dist,
            "candidate_avg_confidence_by_class": cand_conf,
            "candidate_buy_bias": cand_bias,
            "candidate_buy_bias_score": _buy_bias_score(cand_dist, sample_count),
            "candidate_false_buy": cand_false,
            "false_buy_after": dict(cand_false),
            "candidate_probability_diagnostics": _prob_stats(cand_proba, cand_classes),
            "candidate_threshold_simulation": _simulate_thresholds(cand_proba, y, nets, cand_classes),
            "candidate_not_always_buy": not always_buy,
            "candidate_always_buy": always_buy,
            "candidate_always_hold": cand_dist["HOLD"] >= sample_count > 0,
            "candidate_not_always_hold": not (cand_dist["HOLD"] >= sample_count > 0),
            "candidate_class_weight_mode": candidate_art.get("class_weight_mode"),
            "candidate_train_class_distribution": candidate_art.get("train_class_distribution"),
            "training_data_balance": candidate_art.get("training_balance"),
            "raw_self_supervised_class_distribution": (candidate_art.get("training_balance") or {}).get("raw_self_supervised"),
            "raw_outcome_class_distribution": (candidate_art.get("training_balance") or {}).get("raw_outcome"),
            "final_training_class_distribution": (candidate_art.get("training_balance") or {}).get("final_training"),
            "effective_sample_weights": {
                "outcome_row_weight": candidate_art.get("outcome_row_weight"),
                "self_supervised_row_weight": candidate_art.get("self_supervised_row_weight"),
                "class_effective_weights": candidate_art.get("class_effective_weights"),
            },
            "active_holdout_pac": {
                "accuracy": active_h.get("accuracy"),
                "profit_after_cost_if_followed": active_h.get("profit_after_cost_if_followed"),
                "bad_trade_rate_if_followed": active_h.get("bad_trade_rate_if_followed"),
                "sample_count": active_h.get("sample_count"),
            },
            "candidate_holdout_pac": {
                "accuracy": cand_h.get("accuracy"),
                "profit_after_cost_if_followed": cand_h.get("profit_after_cost_if_followed"),
                "bad_trade_rate_if_followed": cand_h.get("bad_trade_rate_if_followed"),
                "sample_count": cand_h.get("sample_count"),
            },
            "candidate_artifact_accuracy_stored": round(float(candidate_art.get("accuracy") or 0.0), 6),
            "disagreement_count": compare["disagreement_count"],
            "rows_candidate_improved": compare["rows_candidate_improved"],
            "rows_candidate_worsened": compare["rows_candidate_worsened"],
            "rows_pac_improved": compare["rows_pac_improved"],
            "rows_pac_worsened": compare["rows_pac_worsened"],
            "holdout_tie_explanation": _tie_explanation(
                disagreement_count=compare["disagreement_count"],
                active_dist=active_dist,
                candidate_dist=cand_dist,
                sample_count=sample_count,
            ),
            "diagnostics_ok": True,
        }
    )
    return base


def build_model_behavior_report(
    db_path: str = DATABASE_PATH,
    *,
    models_dir: Path | str = "models/active",
    version_dir: Path | str = "models/versions/per_coin",
) -> dict[str, Any]:
    from backend.services.ai_outcome_label_audit import build_outcome_label_audit

    symbols = {
        sym: build_symbol_model_behavior(
            sym,
            db_path=db_path,
            models_dir=models_dir,
            version_dir=version_dir,
        )
        for sym in TRADING_SYMBOLS
    }
    low_conf = [s for s, d in symbols.items() if d.get("holdout_low_confidence")]
    buy_biased = [s for s, d in symbols.items() if d.get("active_buy_bias", {}).get("buy_biased")]
    cand_always_buy = [s for s, d in symbols.items() if d.get("candidate_always_buy")]
    cand_always_hold = [s for s, d in symbols.items() if d.get("candidate_always_hold")]
    cand_not_always_buy = [s for s, d in symbols.items() if d.get("candidate_not_always_buy")]
    tied = [s for s, d in symbols.items() if d.get("disagreement_count") == 0 and d.get("diagnostics_ok") and d.get("candidate_path")]
    return {
        "strategy": "day",
        "feature_version": FEATURE_VERSION_DAY_HTF,
        "feature_dim": FEATURE_DIM_V2,
        "min_holdout_confidence_samples": MIN_HOLDOUT_CONFIDENCE_SAMPLES,
        "buy_bias_threshold": BUY_BIAS_THRESHOLD,
        "threshold_simulation_levels": list(THRESHOLD_SIM_LEVELS),
        "label_audit": build_outcome_label_audit(db_path),
        "symbols_low_holdout_confidence": low_conf,
        "symbols_active_buy_biased": buy_biased,
        "symbols_candidate_always_buy": cand_always_buy,
        "symbols_candidate_always_hold": cand_always_hold,
        "symbols_candidate_not_always_buy": cand_not_always_buy,
        "symbols_holdout_tied": tied,
        "symbols": symbols,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "BUY_BIAS_THRESHOLD",
    "MIN_HOLDOUT_CONFIDENCE_SAMPLES",
    "build_model_behavior_report",
    "build_symbol_model_behavior",
]
