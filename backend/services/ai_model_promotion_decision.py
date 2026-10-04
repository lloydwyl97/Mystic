"""Uncertainty-aware candidate-vs-incumbent decision on one shared causal holdout.

Per holdout row i (a realized trade outcome that closed after the incumbent's
training cutoff and is excluded from candidate training):

    u_i(model)   = net_pnl_pct_i if model predicts BUY on row i else 0   (net after cost)
    bad_i(model) = 1 if model predicts BUY and the trade was BAD or net <= 0 else 0

    d_i  = u_i(candidate) - u_i(incumbent)
    b_i  = bad_i(candidate) - bad_i(incumbent)
    var  = max(sample_var(d), mean(d^2))          (mean(d^2) alone when n == 1)
    LCB  = mean(d) - z * sqrt(var / n)
    UCB  = mean(d) + z * sqrt(var / n)

    PROMOTE   iff LCB > 0 and mean(b) <= 0
    INFERIOR  iff UCB < 0
    TIE       iff every d_i == 0 and every b_i == 0
    otherwise AMBIGUOUS

Only PROMOTE replaces the incumbent. The uncertainty penalty shrinks as evidence
grows, so there is no fixed sample-count gate: with few rows the bound is wide and
the incumbent is kept. ``mean(d^2)`` as a variance floor stops a handful of
identical differences from producing a zero-width bound. Accuracy is reported
but carries no authority.
"""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np

DEFAULT_Z = float(os.getenv("MODEL_PROMOTION_Z", "1.645") or "1.645")


def row_utilities(preds: np.ndarray, nets: np.ndarray, good_bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(preds, dtype=np.int64).reshape(-1)
    n = np.asarray(nets, dtype=np.float64).reshape(-1)
    gb = np.asarray([str(g or "").strip().upper() for g in good_bad], dtype=object)
    buy = p == 1
    util = np.where(buy, n, 0.0)
    bad = (buy & ((gb == "BAD") | (n <= 0.0))).astype(np.float64)
    return util, bad


def _bound(diff: np.ndarray, z: float) -> tuple[float, float, float]:
    n = len(diff)
    mean = float(np.mean(diff))
    second = float(np.mean(diff**2))
    var = max(float(np.var(diff, ddof=1)) if n >= 2 else 0.0, second)
    se = math.sqrt(var / n)
    return mean, mean - z * se, mean + z * se


def summarize_model(util: np.ndarray, bad: np.ndarray, preds: np.ndarray, labels: np.ndarray | None, z: float = DEFAULT_Z) -> dict[str, Any]:
    n = len(util)
    if n == 0:
        return {"n": 0}
    mean, lcb, ucb = _bound(np.asarray(util, dtype=np.float64), z)
    out = {
        "n": n,
        "net_after_cost_mean": round(mean, 6),
        "net_after_cost_lcb": round(lcb, 6),
        "net_after_cost_ucb": round(ucb, 6),
        "bad_trade_rate": round(float(np.mean(bad)), 6),
        "buy_signals": int(np.sum(np.asarray(preds) == 1)),
    }
    if labels is not None and len(labels) == n:
        out["accuracy"] = round(float(np.mean(np.asarray(preds, dtype=np.int64) == np.asarray(labels, dtype=np.int64))), 6)
    return out


def compare_paired(
    cand_preds: np.ndarray,
    inc_preds: np.ndarray,
    nets: np.ndarray,
    good_bad: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    z: float = DEFAULT_Z,
) -> dict[str, Any]:
    n = len(np.asarray(nets).reshape(-1))
    if n == 0 or len(cand_preds) != n or len(inc_preds) != n:
        return {"verdict": "NO_SHARED_HOLDOUT", "promote": False, "n": n, "z": z}
    uc, bc = row_utilities(cand_preds, nets, good_bad)
    ui, bi = row_utilities(inc_preds, nets, good_bad)
    d = uc - ui
    b = bc - bi
    mean_d, lcb, ucb = _bound(d, z)
    mean_b = float(np.mean(b))
    if not np.any(d != 0.0) and not np.any(b != 0.0):
        verdict = "TIE"
    elif lcb > 0.0 and mean_b <= 0.0:
        verdict = "PROMOTE"
    elif ucb < 0.0:
        verdict = "INFERIOR"
    else:
        verdict = "AMBIGUOUS"
    return {
        "verdict": verdict,
        "promote": verdict == "PROMOTE",
        "n": n,
        "z": z,
        "utility_diff_mean": round(mean_d, 6),
        "utility_diff_lcb": round(lcb, 6),
        "utility_diff_ucb": round(ucb, 6),
        "bad_rate_diff_mean": round(mean_b, 6),
        "rows_disagreeing": int(np.sum(np.asarray(cand_preds) != np.asarray(inc_preds))),
        "candidate": summarize_model(uc, bc, cand_preds, labels, z),
        "incumbent": summarize_model(ui, bi, inc_preds, labels, z),
        "rule": "promote iff LCB(mean(u_cand-u_inc)) > 0 and mean(bad_cand-bad_inc) <= 0",
    }


def classify_standalone(util: np.ndarray, bad: np.ndarray, baseline_util: np.ndarray, z: float = DEFAULT_Z) -> str:
    """SUPPORTED / AMBIGUOUS / UNDERPERFORMING for a serving model on forward rows.

    SUPPORTED: following it beats both abstaining and taking every trade, with the
    lower bound above zero. UNDERPERFORMING: the upper bound sits below either
    baseline. Otherwise AMBIGUOUS.
    """
    u = np.asarray(util, dtype=np.float64)
    if len(u) == 0:
        return "AMBIGUOUS"
    _m, lcb0, ucb0 = _bound(u, z)
    _m2, lcb1, ucb1 = _bound(u - np.asarray(baseline_util, dtype=np.float64), z)
    if lcb0 > 0.0 and lcb1 > 0.0:
        return "SUPPORTED"
    if ucb0 < 0.0 or ucb1 < 0.0:
        return "UNDERPERFORMING"
    return "AMBIGUOUS"


__all__ = ["DEFAULT_Z", "classify_standalone", "compare_paired", "row_utilities", "summarize_model"]
