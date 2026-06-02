"""
Outcome-weighted, self-supervised BUY-capped training mix for per-coin RF models.

Keeps all realized outcome rows; caps/down-samples self-supervised BUY rows so
combined training BUY rate stays near real outcome priors.
"""

from __future__ import annotations

from typing import Any

import numpy as np

OUTCOME_ROW_WEIGHT = 5.0
SELF_SUPERVISED_ROW_WEIGHT = 1.0
MAX_SELF_SUPERVISED_BUY_RATIO = 0.50
MAX_COMBINED_BUY_RATE = 0.35
MIN_COMBINED_BUY_RATE = 0.15
OUTCOME_BUY_RATE_HEADROOM = 3.0
OUTCOME_BUY_RATE_BUFFER = 0.08


def _count_labels(y: np.ndarray) -> dict[str, int]:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    buy = int(np.sum(y_arr == 1))
    hold = int(np.sum(y_arr == 0))
    return {"BUY": buy, "HOLD": hold}


def _buy_rate(y: np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    if len(y_arr) == 0:
        return 0.0
    return float(np.mean(y_arr == 1))


def resolve_combined_buy_rate_cap(
    raw_outcome: dict[str, int] | None,
    *,
    max_cap: float = MAX_COMBINED_BUY_RATE,
    min_cap: float = MIN_COMBINED_BUY_RATE,
) -> float:
    """Derive combined BUY cap from realized outcome priors (~5–9%) plus SS headroom."""
    if not raw_outcome:
        return max_cap
    buy = int(raw_outcome.get("BUY") or 0)
    hold = int(raw_outcome.get("HOLD") or 0)
    total = buy + hold
    if total <= 0:
        return max_cap
    oc_rate = buy / total
    target = (oc_rate * OUTCOME_BUY_RATE_HEADROOM) + OUTCOME_BUY_RATE_BUFFER
    return round(max(min_cap, min(max_cap, target)), 6)


def cap_self_supervised_buy_rows(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_buy_ratio: float = MAX_SELF_SUPERVISED_BUY_RATIO,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Limit self-supervised BUY share to ``max_buy_ratio`` of SS rows."""
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64).reshape(-1)
    meta: dict[str, Any] = {
        "before": _count_labels(y_arr),
        "after": _count_labels(y_arr),
        "rows_removed": 0,
        "max_buy_ratio": max_buy_ratio,
    }
    if len(y_arr) == 0:
        return X_arr, y_arr, meta
    buy_idx = np.flatnonzero(y_arr == 1)
    hold_idx = np.flatnonzero(y_arr == 0)
    max_buy = int(np.floor(len(y_arr) * max_buy_ratio))
    if len(buy_idx) <= max_buy:
        return X_arr, y_arr, meta
    if max_buy <= 0:
        keep = hold_idx
    else:
        rng = np.random.default_rng(random_state)
        keep_buy = rng.choice(buy_idx, size=max_buy, replace=False)
        keep = np.sort(np.concatenate([hold_idx, keep_buy]))
    meta["after"] = _count_labels(y_arr[keep])
    meta["rows_removed"] = int(len(y_arr) - len(keep))
    return X_arr[keep], y_arr[keep], meta


def enforce_combined_buy_rate_cap(
    X_ss: np.ndarray,
    y_ss: np.ndarray,
    X_oc: np.ndarray,
    y_oc: np.ndarray,
    *,
    max_combined_buy_rate: float = MAX_COMBINED_BUY_RATE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Downsample only self-supervised BUY rows until combined BUY rate <= cap."""
    Xs = np.asarray(X_ss, dtype=np.float64)
    ys = np.asarray(y_ss, dtype=np.int64).reshape(-1)
    Xo = np.asarray(X_oc, dtype=np.float64)
    yo = np.asarray(y_oc, dtype=np.int64).reshape(-1)
    meta: dict[str, Any] = {
        "combined_buy_rate_before": round(_buy_rate(np.concatenate([ys, yo])), 6) if len(ys) + len(yo) else None,
        "combined_buy_rate_after": None,
        "ss_buy_rows_removed": 0,
        "max_combined_buy_rate": max_combined_buy_rate,
    }
    if len(ys) == 0:
        meta["combined_buy_rate_after"] = meta["combined_buy_rate_before"]
        return Xs, ys, meta

    oc_buy = int(np.sum(yo == 1)) if len(yo) else 0
    oc_total = len(yo)
    ss_hold = int(np.sum(ys == 0))
    ss_buy_idx = np.flatnonzero(ys == 1)
    numer = (max_combined_buy_rate * (ss_hold + oc_total)) - oc_buy
    if numer <= 0:
        max_ss_buy = 0
    else:
        max_ss_buy = int(np.floor(numer / (1.0 - max_combined_buy_rate)))
    max_ss_buy = max(0, min(max_ss_buy, len(ss_buy_idx)))
    ss_hold_idx = np.flatnonzero(ys == 0)
    removed = max(0, len(ss_buy_idx) - max_ss_buy)
    if removed > 0:
        rng = np.random.default_rng(random_state)
        if max_ss_buy <= 0:
            keep = ss_hold_idx
        else:
            keep_buy = rng.choice(ss_buy_idx, size=max_ss_buy, replace=False)
            keep = np.sort(np.concatenate([ss_hold_idx, keep_buy]))
        Xs = Xs[keep]
        ys = ys[keep]
    final = np.concatenate([ys, yo]) if len(yo) else ys
    meta["ss_buy_rows_removed"] = removed
    meta["combined_buy_rate_after"] = round(_buy_rate(final), 6) if len(final) else None
    return Xs, ys, meta


def prepare_outcome_weighted_training_arrays(
    X_self: np.ndarray,
    y_self: np.ndarray,
    X_outcome: np.ndarray,
    y_outcome: np.ndarray,
    *,
    outcome_weight: float = OUTCOME_ROW_WEIGHT,
    self_supervised_weight: float = SELF_SUPERVISED_ROW_WEIGHT,
    max_self_supervised_buy_ratio: float = MAX_SELF_SUPERVISED_BUY_RATIO,
    max_combined_buy_rate: float = MAX_COMBINED_BUY_RATE,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Build per-symbol training matrices with SS BUY cap, combined BUY cap, and source weights.

    Returns X, y, sample_weights, diagnostics dict.
    """
    Xs = np.asarray(X_self, dtype=np.float64) if len(X_self) else np.empty((0, 0))
    ys = np.asarray(y_self, dtype=np.int64).reshape(-1) if len(y_self) else np.array([], dtype=np.int64)
    Xo = np.asarray(X_outcome, dtype=np.float64) if len(X_outcome) else np.empty((0, 0))
    yo = np.asarray(y_outcome, dtype=np.int64).reshape(-1) if len(y_outcome) else np.array([], dtype=np.int64)

    raw_ss = _count_labels(ys)
    raw_oc = _count_labels(yo)
    combined_cap = resolve_combined_buy_rate_cap(raw_oc)

    ss_cap_meta: dict[str, Any] = {"before": raw_ss, "after": raw_ss, "rows_removed": 0}
    if len(ys) > 0:
        Xs, ys, ss_cap_meta = cap_self_supervised_buy_rows(
            Xs,
            ys,
            max_buy_ratio=max_self_supervised_buy_ratio,
            random_state=random_state,
        )

    combined_meta: dict[str, Any] = {}
    if len(ys) > 0 and len(yo) > 0:
        Xs, ys, combined_meta = enforce_combined_buy_rate_cap(
            Xs,
            ys,
            Xo,
            yo,
            max_combined_buy_rate=combined_cap,
            random_state=random_state,
        )
    elif len(yo) > 0:
        combined_meta = {
            "combined_buy_rate_before": round(_buy_rate(yo), 6),
            "combined_buy_rate_after": round(_buy_rate(yo), 6),
            "ss_buy_rows_removed": 0,
            "max_combined_buy_rate": combined_cap,
            "outcome_only_training": True,
        }

    parts_x: list[np.ndarray] = []
    parts_y: list[np.ndarray] = []
    parts_w: list[np.ndarray] = []
    if len(ys) > 0:
        parts_x.append(Xs)
        parts_y.append(ys)
        parts_w.append(np.full(len(ys), float(self_supervised_weight), dtype=np.float64))
    if len(yo) > 0:
        parts_x.append(Xo)
        parts_y.append(yo)
        parts_w.append(np.full(len(yo), float(outcome_weight), dtype=np.float64))

    if not parts_x:
        return np.array([]), np.array([]), np.array([]), {}

    X_all = np.vstack(parts_x)
    y_all = np.concatenate(parts_y)
    w_all = np.concatenate(parts_w)
    final = _count_labels(y_all)
    diag: dict[str, Any] = {
        "raw_self_supervised": raw_ss,
        "raw_outcome": raw_oc,
        "after_self_supervised_buy_cap": ss_cap_meta.get("after", raw_ss),
        "self_supervised_buy_cap": ss_cap_meta,
        "combined_buy_cap": combined_meta,
        "combined_buy_rate_cap_used": combined_cap,
        "final_training": final,
        "final_buy_rate": round(_buy_rate(y_all), 6),
        "outcome_row_weight": float(outcome_weight),
        "self_supervised_row_weight": float(self_supervised_weight),
        "outcome_rows_kept": int(len(yo)),
        "self_supervised_rows_kept": int(len(ys)),
    }
    return X_all, y_all, w_all, diag


__all__ = [
    "MAX_COMBINED_BUY_RATE",
    "MAX_SELF_SUPERVISED_BUY_RATIO",
    "MIN_COMBINED_BUY_RATE",
    "OUTCOME_BUY_RATE_BUFFER",
    "OUTCOME_BUY_RATE_HEADROOM",
    "OUTCOME_ROW_WEIGHT",
    "SELF_SUPERVISED_ROW_WEIGHT",
    "cap_self_supervised_buy_rows",
    "enforce_combined_buy_rate_cap",
    "prepare_outcome_weighted_training_arrays",
    "resolve_combined_buy_rate_cap",
]
