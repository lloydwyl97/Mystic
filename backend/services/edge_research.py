"""Offline, chronological evaluation of predictive arms. Research only.

Nothing here is read by a live decision. Every arm is fitted on rows whose
label had become final before the fold starts (``label_ts < fold start``) and
scored on the fold; the online arm updates only on labels that were final at
the moment of each prediction. Metrics are reported per fold so an aggregate
cannot hide a single good fold.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

EDGE_RESEARCH_VERSION = "EDGE_RESEARCH_V1"


@dataclass
class Row:
    t: float  # decision time
    group: Any  # decision point (rows ranked against each other)
    symbol: str
    x: np.ndarray
    label: float  # net market outcome after cost
    label_ts: float  # when the label became final
    key: tuple = ()  # hierarchical key, coarse to fine
    extra: dict[str, Any] = field(default_factory=dict)


def fold_bounds(times: Sequence[float], folds: int) -> list[tuple[float, float]]:
    """``folds`` equal-count chronological blocks; block 0 is train-only."""
    ts = np.sort(np.asarray(times, dtype=float))
    if len(ts) == 0 or folds < 2:
        return []
    edges = [ts[round(i * (len(ts) - 1) / folds)] for i in range(folds + 1)]
    edges[-1] = ts[-1] + 1e-6
    return [(float(edges[i]), float(edges[i + 1])) for i in range(1, folds)]


def rank(a: Sequence[float]) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(a, dtype=float))


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.all(a == a[0]) or np.all(b == b[0]):
        return None
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def trajectory_weights(keys: Sequence[Any]) -> np.ndarray:
    """One unit of weight per trajectory, shared equally by its rows.

    A trajectory that emits 1,000 heartbeat rows then counts the same as a
    trajectory that emits one. The weights sum to the number of trajectories,
    which is the effective independent sample.
    """
    counts: dict[Any, int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    return np.array([1.0 / counts[key] for key in keys], dtype=float)


def effective_independent_weight(weights: Sequence[float]) -> float:
    return float(np.sum(np.asarray(weights, dtype=float)))


def weighted_spearman(a: Sequence[float], b: Sequence[float], weights: Sequence[float]) -> float | None:
    """Spearman correlation with one weight per row (cluster-safe rank)."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return _weighted_pearson(rank(x), rank(y), w)


def _weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float | None:
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if total <= 0:
        return None
    w = w / total
    mx, my = float(np.sum(w * x)), float(np.sum(w * y))
    cov = float(np.sum(w * (x - mx) * (y - my)))
    vx = float(np.sum(w * (x - mx) ** 2))
    vy = float(np.sum(w * (y - my) ** 2))
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def weighted_top(score: Sequence[float], gross: Sequence[float], weights: Sequence[float], cost: float, fraction: float) -> dict[str, float] | None:
    """Gross and net of the highest-scored fraction of total weight."""
    s = np.asarray(score, dtype=float)
    g = np.asarray(gross, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(s) & np.isfinite(g) & np.isfinite(w) & (w > 0)
    s, g, w = s[mask], g[mask], w[mask]
    total = float(w.sum())
    if total <= 0 or len(s) < 5:
        return None
    order = np.argsort(s)
    w, g = w[order], g[order]
    top = np.cumsum(w) > (1.0 - fraction) * total
    gw = float(w[top].sum())
    if gw <= 0:
        return None
    mean = float(np.sum(w[top] * g[top]) / gw)
    return {"gross": mean, "net": mean - float(cost)}


def row_weights(rows: Sequence[Row]) -> np.ndarray | None:
    """Per-row research weights. ``None`` when every row weighs 1 (unweighted fit)."""
    if not rows or not any("weight" in r.extra for r in rows):
        return None
    w = np.array([float(r.extra.get("weight", 1.0)) for r in rows], dtype=float)
    if np.allclose(w, 1.0):
        return None
    return w


def ranking_metrics(rows: Sequence[Row], score: Sequence[float]) -> dict[str, Any]:
    """Per decision point: chosen (top score), best, worst and regret, in label units."""
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r.group].append(i)
    chosen, best, worst, mean, within = [], [], [], [], []
    for idx in groups.values():
        if len(idx) < 2:
            continue
        lab = np.array([rows[i].label for i in idx])
        sc = np.array([score[i] for i in idx])
        if np.all(sc == sc[0]):
            continue
        chosen.append(float(lab[int(np.argmax(sc))]))
        best.append(float(lab.max()))
        worst.append(float(lab.min()))
        mean.append(float(lab.mean()))
        if len(idx) >= 3:
            s = spearman(sc, lab)
            if s is not None:
                within.append(s)
    if not chosen:
        return {"points": 0}
    lift = np.array(chosen) - np.array(mean)
    return {
        "points": len(chosen),
        "chosen": float(np.mean(chosen)),
        "best": float(np.mean(best)),
        "worst": float(np.mean(worst)),
        "point_mean": float(np.mean(mean)),
        "regret": float(np.mean(best) - np.mean(chosen)),
        "chosen_minus_mean": float(lift.mean()),
        "chosen_minus_mean_t": t_stat(lift),
        "within_point_rank": float(np.mean(within)) if within else None,
    }


def t_stat(x: Sequence[float]) -> float | None:
    a = np.asarray(x, dtype=float)
    if len(a) < 3 or float(a.std(ddof=1)) == 0.0:
        return None
    return float(a.mean() / (a.std(ddof=1) / math.sqrt(len(a))))


def closed_loop(rows: Sequence[Row], pred_net: Sequence[float], *, max_slots: int = 4) -> dict[str, Any]:
    """Chronological book: at each decision point, enter the highest predicted
    candidates with predicted net > 0, one per symbol, while slots are free; a
    position occupies its slot until its label's exit time."""
    order = sorted(range(len(rows)), key=lambda i: (rows[i].t, -pred_net[i]))
    sums: dict[Any, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in rows:
        sums[r.group][0] += r.label
        sums[r.group][1] += 1.0
    open_until: dict[str, float] = {}
    trades: list[float] = []
    alpha: list[float] = []
    for i in order:
        r = rows[i]
        if not pred_net[i] > 0:
            continue
        live = {s: u for s, u in open_until.items() if u > r.t}
        if r.symbol in live or len(live) >= max_slots:
            open_until = live
            continue
        live[r.symbol] = float(r.extra.get("exit_ts", r.label_ts))
        open_until = live
        trades.append(float(r.label))
        total, n = sums[r.group]
        alpha.append(float(r.label) - total / n)
    out = book_stats(trades)
    out["alpha_avg"] = float(np.mean(alpha)) if alpha else None
    out["alpha_t"] = t_stat(alpha)
    out["alphas"] = alpha
    out["nets"] = trades
    return out


def book_stats(trades: Sequence[float]) -> dict[str, Any]:
    if not trades:
        return {"trades": 0, "net": 0.0, "pf": None, "dd": 0.0, "avg": None, "win": None}
    t = np.asarray(trades, dtype=float)
    cum = np.cumsum(t)
    dd = float(np.max(np.maximum.accumulate(np.concatenate([[0.0], cum]))[1:] - cum))
    gains, losses = float(t[t > 0].sum()), float(-t[t < 0].sum())
    return {"trades": len(t), "net": float(t.sum()), "pf": (gains / losses) if losses > 0 else None, "dd": dd, "avg": float(t.mean()), "win": float(np.mean(t > 0))}


def quintiles(score: Sequence[float], gross: Sequence[float], cost: Sequence[float] | float) -> dict[str, Any]:
    s, g = np.asarray(score, dtype=float), np.asarray(gross, dtype=float)
    c = np.broadcast_to(np.asarray(cost, dtype=float), g.shape)
    if len(s) < 10:
        return {"n": len(s)}
    lo, hi = np.quantile(s, [0.2, 0.8])
    top, bot = s >= hi, s <= lo
    return {
        "n": len(s),
        "top_gross": float(g[top].mean()),
        "top_net": float((g[top] - c[top]).mean()),
        "bottom_gross": float(g[bot].mean()),
        "all_gross": float(g.mean()),
        "spread": float(g[top].mean() - g[bot].mean()),
    }


# --------------------------------------------------------------------------- arms


class Standardizer:
    def __init__(self, X: np.ndarray) -> None:
        self.mu = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd = np.where(sd > 1e-12, sd, 1.0)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.clip((X - self.mu) / self.sd, -5.0, 5.0)


def _inner_split(train: Sequence[Row]) -> tuple[list[Row], list[Row]]:
    cut = sorted(r.t for r in train)[int(0.75 * len(train))]
    a = [r for r in train if r.label_ts < cut]
    b = [r for r in train if r.t >= cut]
    return a, b


def fit_ridge(train: Sequence[Row], alphas: Sequence[float] = (1.0, 10.0, 100.0, 1000.0)) -> Callable[[Sequence[Row]], np.ndarray]:
    """Closed-form ridge; alpha picked on the last quarter of the train window."""

    def solve(rows: Sequence[Row], alpha: float) -> Callable[[Sequence[Row]], np.ndarray]:
        X = np.stack([r.x for r in rows])
        y = np.array([r.label for r in rows])
        st = Standardizer(X)
        Z = st(X)
        wts = row_weights(rows)
        if wts is None:
            mu = float(y.mean())
            coef = np.linalg.solve(Z.T @ Z + alpha * np.eye(Z.shape[1]), Z.T @ (y - mu))
        else:
            mu = float(np.average(y, weights=wts))
            sw = np.sqrt(wts)
            zw, yw = Z * sw[:, None], (y - mu) * sw
            coef = np.linalg.solve(zw.T @ zw + alpha * np.eye(Z.shape[1]), zw.T @ yw)
        return lambda rs: st(np.stack([r.x for r in rs])) @ coef + mu

    a, b = _inner_split(train)
    best = alphas[-1]
    if len(a) > 50 and len(b) > 20:
        errs = {al: float(np.mean((solve(a, al)(b) - np.array([r.label for r in b])) ** 2)) for al in alphas}
        best = min(errs, key=errs.get)
    return solve(train, best)


def fit_huber(train: Sequence[Row]) -> Callable[[Sequence[Row]], np.ndarray]:
    from sklearn.linear_model import HuberRegressor

    X = np.stack([r.x for r in train])
    y = np.array([r.label for r in train])
    st = Standardizer(X)
    scale = float(np.std(y)) or 1.0
    wts = row_weights(train)
    m = HuberRegressor(epsilon=1.35, alpha=1.0, max_iter=500).fit(st(X), y / scale, sample_weight=wts)
    return lambda rs: m.predict(st(np.stack([r.x for r in rs]))) * scale


def fit_tree(train: Sequence[Row]) -> Callable[[Sequence[Row]], np.ndarray]:
    from sklearn.ensemble import HistGradientBoostingRegressor

    X = np.stack([r.x for r in train])
    y = np.array([r.label for r in train])
    m = HistGradientBoostingRegressor(max_depth=3, max_iter=150, learning_rate=0.05, min_samples_leaf=max(20, len(y) // 50), l2_regularization=1.0, loss="absolute_error", random_state=0)
    m.fit(X, y, sample_weight=row_weights(train))
    return lambda rs: m.predict(np.stack([r.x for r in rs]))


def fit_pairwise(train: Sequence[Row], *, max_pairs: int = 40000) -> Callable[[Sequence[Row]], np.ndarray]:
    """Logistic model on within-decision-point differences; then a train-fold
    linear map from score to net so the book can abstain on its own scale."""
    from sklearn.linear_model import LogisticRegression

    X = np.stack([r.x for r in train])
    st = Standardizer(X)
    Z = st(X)
    groups: dict[Any, list[int]] = defaultdict(list)
    for i, r in enumerate(train):
        groups[r.group].append(i)
    dx, dy = [], []
    for idx in groups.values():
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                i, j = idx[a], idx[b]
                if train[i].label == train[j].label:
                    continue
                dx.append(Z[i] - Z[j])
                dy.append(1 if train[i].label > train[j].label else 0)
    if len(set(dy)) < 2:
        return lambda rs: np.zeros(len(rs))
    dx_a, dy_a = np.array(dx), np.array(dy)
    if len(dy_a) > max_pairs:
        keep = np.random.default_rng(0).choice(len(dy_a), max_pairs, replace=False)
        dx_a, dy_a = dx_a[keep], dy_a[keep]
    m = LogisticRegression(C=0.1, fit_intercept=False, max_iter=1000).fit(dx_a, dy_a)
    w = m.coef_[0]
    s_train = Z @ w
    y = np.array([r.label for r in train])
    slope, icpt = np.polyfit(s_train, y, 1) if np.std(s_train) > 0 else (0.0, float(y.mean()))
    return lambda rs: st(np.stack([r.x for r in rs])) @ w * slope + icpt


def fit_hierarchical(train: Sequence[Row], *, prior_n: float = 10.0) -> Callable[[Sequence[Row]], np.ndarray]:
    """Empirical-Bayes shrinkage of label means down ``Row.key`` levels (the
    current learner's structure: engine, setup, related, key)."""
    stats: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for r in train:
        for depth in range(len(r.key) + 1):
            s = stats[r.key[:depth]]
            s[0] += r.label
            s[1] += 1.0

    def one(key: tuple) -> float:
        est = stats[()][0] / stats[()][1] if stats[()][1] else 0.0
        for depth in range(1, len(key) + 1):
            total, n = stats.get(key[:depth], (0.0, 0.0))
            if n:
                est = (total + prior_n * est) / (n + prior_n)
        return est

    return lambda rs: np.array([one(r.key) for r in rs])


def online_sgd(rows: Sequence[Row], *, lr: float = 0.01, l2: float = 1e-3, warm: Sequence[Row] = ()) -> np.ndarray:
    """Predict every row in time order, learning only labels final before it."""
    ordered = sorted(range(len(rows)), key=lambda i: rows[i].t)
    pool = sorted(list(warm) + list(rows), key=lambda r: r.label_ts)
    X0 = np.stack([r.x for r in (warm or rows)])
    st = Standardizer(X0)
    w = np.zeros(X0.shape[1])
    b = 0.0
    pred = np.zeros(len(rows))
    j = 0
    for i in ordered:
        t = rows[i].t
        while j < len(pool) and pool[j].label_ts < t:
            z = st(pool[j].x[None, :])[0]
            err = float(z @ w + b - pool[j].label)
            step = lr / (1.0 + float(z @ z))
            w -= step * err * z + lr * l2 * w
            b -= lr * err
            j += 1
        pred[i] = float(st(rows[i].x[None, :])[0] @ w + b)
    return pred


def evaluate_arm(name: str, train: Sequence[Row], test: Sequence[Row], pred: np.ndarray, *, regression: bool, max_slots: int = 4) -> dict[str, Any]:
    labels = np.array([r.label for r in test])
    out: dict[str, Any] = {"arm": name, "n_train": len(train), "n_test": len(test), "rank": spearman(pred, labels)}
    if regression:
        out["bias"] = float(np.mean(pred - labels))
    out["ranking"] = ranking_metrics(test, pred)
    out["book"] = closed_loop(test, pred, max_slots=max_slots)
    return out


def calibrate(train: Sequence[Row], train_score: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Train-fold linear map from a raw score to net, so any arm abstains on its own scale."""
    y = np.array([r.label for r in train])
    s = np.asarray(train_score, dtype=float)
    if len(s) < 3 or np.std(s) == 0:
        mu = float(y.mean()) if len(y) else 0.0
        return lambda z: np.full(len(z), mu)
    slope, icpt = np.polyfit(s, y, 1)
    return lambda z: np.asarray(z, dtype=float) * slope + icpt


def fold_rows(rows: Sequence[Row], lo: float, hi: float) -> tuple[list[Row], list[Row]]:
    train = [r for r in rows if r.label_ts < lo]
    test = [r for r in rows if lo <= r.t < hi]
    return train, test


def summarize(folds: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    per = [f["arms"][arm] for f in folds if arm in f["arms"]]
    ranks = [p["rank"] for p in per if p.get("rank") is not None]
    nets = [p["book"].get("net", 0.0) for p in per]
    alphas = [a for p in per for a in p["book"].get("alphas", [])]
    trades = [x for p in per for x in p["book"].get("nets", [])]
    lifts = [p["ranking"].get("chosen_minus_mean") for p in per if p.get("ranking", {}).get("points")]
    return {
        "arm": arm,
        "fold_rank": ranks,
        "rank_positive_folds": sum(1 for r in ranks if r > 0),
        "fold_net": nets,
        "net_positive_folds": sum(1 for n in nets if n > 0),
        "fold_alpha": [p["book"].get("alpha_avg") for p in per],
        "alpha_positive_folds": sum(1 for p in per if (p["book"].get("alpha_avg") or 0) > 0),
        "fold_point_lift": lifts,
        "book": book_stats(trades),
        "alpha_avg": float(np.mean(alphas)) if alphas else None,
        "alpha_t": t_stat(alphas),
        "trades": len(trades),
    }


def is_finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


__all__ = [
    "EDGE_RESEARCH_VERSION",
    "Row",
    "Standardizer",
    "book_stats",
    "calibrate",
    "closed_loop",
    "effective_independent_weight",
    "evaluate_arm",
    "fit_hierarchical",
    "fit_huber",
    "fit_pairwise",
    "fit_ridge",
    "fit_tree",
    "fold_bounds",
    "fold_rows",
    "online_sgd",
    "quintiles",
    "ranking_metrics",
    "row_weights",
    "spearman",
    "summarize",
    "trajectory_weights",
    "weighted_spearman",
    "weighted_top",
]
