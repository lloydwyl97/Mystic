#!/usr/bin/env python3
"""SCALP direct executable move research on a research extract. Research only.

Label: buy at the ask of a 5 s microstructure snapshot, sell at the bid of
the first snapshot at or after t+h. That gross already pays the spread; net
subtracts the remaining taker fees and slippage buffer. A label is dropped
when the book was stale, crossed, or the snapshot stream has a gap inside
[t, t+h]. Features are the same snapshot's derived book/flow state plus past
mid returns, so every input is known at t.

The geometric strategy claim is tested only as an input: its rank against the
executable move, and whether adding it to the snapshot model changes OOS rank.

Usage: scalp_move_research.py --db /tmp/rx.db [--folds 5] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import edge_research as er

HORIZONS = (30, 60, 120, 300, 600, 1200)
MAX_GAP_SEC = 30.0
MATCH_TOL_SEC = 10.0
MAX_DATA_AGE_SEC = 2.0
DROP = {"ts", "symbol", "best_bid", "best_ask", "mid", "microprice", "sample_count"}
PAST = (30, 60, 300, 900)


def cost_beyond_spread() -> float:
    from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE

    return 2.0 * float(TAKER_FEE) + 2.0 * float(SLIPPAGE_BUFFER)


def load(db: str) -> dict[str, dict]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    raw: dict[str, list] = {}
    for sym, ts, fj in conn.execute("SELECT symbol, ts_utc, features_json FROM microstructure_feature_snapshots ORDER BY symbol, ts_utc"):
        try:
            raw.setdefault(str(sym), []).append((float(ts), json.loads(fj)))
        except (TypeError, ValueError):
            continue
    conn.close()
    keys = sorted({k for rows in raw.values() for _, f in rows[:200] for k, v in f.items() if k not in DROP and isinstance(v, int | float | bool)})
    out = {}
    for sym, rows in raw.items():
        ts = np.array([t for t, _ in rows])
        bid = np.array([float(f.get("best_bid") or 0) for _, f in rows])
        ask = np.array([float(f.get("best_ask") or 0) for _, f in rows])
        mid = (bid + ask) / 2.0
        mp = np.array([float(f.get("microprice") or 0) for _, f in rows])
        X = np.array([[float(f.get(k) or 0.0) for k in keys] for _, f in rows])
        disp = np.where(mid > 0, mp / np.where(mid > 0, mid, 1) - 1.0, 0.0)
        past = []
        for p in PAST:
            j = np.searchsorted(ts, ts - p, side="right") - 1
            ok = j >= 0
            prev = np.where(ok, mid[np.clip(j, 0, None)], mid)
            past.append(np.where(ok & (prev > 0), mid / np.where(prev > 0, prev, 1) - 1.0, 0.0))
        gaps = np.diff(ts, prepend=ts[0])
        bad = np.cumsum(gaps > MAX_GAP_SEC)
        age = np.array([float(f.get("data_age_sec") or 0) for _, f in rows])
        usable = (bid > 0) & (ask > bid) & (age <= MAX_DATA_AGE_SEC)
        out[sym] = {"ts": ts, "bid": bid, "ask": ask, "X": np.column_stack([X, disp, *past]), "bad": bad, "usable": usable}
    return {"symbols": out, "keys": keys + ["microprice_displacement"] + [f"mid_ret_{p}s" for p in PAST]}


def labels(s: dict, h: float) -> tuple[np.ndarray, np.ndarray]:
    ts = s["ts"]
    j = np.searchsorted(ts, ts + h, side="left")
    jj = np.clip(j, 0, len(ts) - 1)
    ok = (j < len(ts)) & (ts[jj] - (ts + h) <= MATCH_TOL_SEC) & (s["bad"][jj] == s["bad"]) & s["usable"] & s["usable"][jj]
    gross = np.where(ok, s["bid"][jj] / s["ask"] - 1.0, np.nan)
    return gross, ok


def rows_for(data: dict, h: float, stride: int = 1) -> list[er.Row]:
    out = []
    for sym, s in data["symbols"].items():
        gross, ok = labels(s, h)
        for i in np.nonzero(ok)[0][::stride]:
            out.append(er.Row(t=float(s["ts"][i]), group=int(s["ts"][i] // 5), symbol=sym, x=s["X"][i], label=float(gross[i]), label_ts=float(s["ts"][i] + h)))
    out.sort(key=lambda r: r.t)
    return out


def horizon_study(data: dict, h: float, folds: int, cost: float) -> dict:
    rows = rows_for(data, h, stride=1)
    bounds = er.fold_bounds([r.t for r in rows], folds)
    res = []
    for k, (lo, hi) in enumerate(bounds):
        train, test = er.fold_rows(rows, lo, hi)
        if len(train) < 2000 or len(test) < 500:
            continue
        y = np.array([r.label for r in test])
        sub = train[:: max(1, len(train) // 60000)]
        arms = {}
        for name, fit in (("ridge", er.fit_ridge), ("tree_hgb", er.fit_tree)):
            pred = fit(sub)(test)
            arms[name] = {"rank": er.spearman(pred, y), "bias": float(np.mean(pred - y)), "q": er.quintiles(pred, y, cost), "top_decile_net": float(np.mean(y[pred >= np.quantile(pred, 0.9)]) - cost)}
        Xtr = np.stack([r.x for r in sub])
        ytr = np.array([r.label for r in sub])
        Xte = np.stack([r.x for r in test])
        singles = []
        for d in range(Xtr.shape[1]):
            s_tr = er.spearman(Xtr[:, d], ytr)
            if s_tr is None:
                continue
            sign = 1.0 if s_tr >= 0 else -1.0
            s_te = er.spearman(sign * Xte[:, d], y)
            singles.append((data["keys"][d], s_tr, s_te))
        singles.sort(key=lambda z: -abs(z[1]))
        res.append({"fold": k + 1, "n_train": len(train), "n_test": len(test), "eff_n_test": int(len(test) * 5.0 / h), "test_mean_gross": float(y.mean()), "arms": arms, "top_single": singles[:8]})
    summary = {}
    for name in ("ridge", "tree_hgb"):
        per = [f["arms"][name] for f in res]
        summary[name] = {
            "fold_rank": [p["rank"] for p in per],
            "rank_positive_folds": sum(1 for p in per if (p["rank"] or 0) > 0),
            "fold_top_q_gross": [p["q"].get("top_gross") for p in per],
            "fold_top_q_net": [p["q"].get("top_net") for p in per],
            "top_q_net_positive_folds": sum(1 for p in per if (p["q"].get("top_net") or -1) > 0),
            "fold_top_decile_net": [p["top_decile_net"] for p in per],
            "fold_bias": [p["bias"] for p in per],
        }
    return {"h": h, "rows": len(rows), "folds": res, "summary": summary}


def claim_study(db: str, data: dict, cost: float, folds: int) -> dict:
    """Geometric strategy claim as a feature: rank vs executable move, capture, and OOS value."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    claims = conn.execute(
        "SELECT symbol, evaluated_at, raw_expected_move, markouts_json FROM adaptive_candidate_markouts "
        "WHERE engine_id='SCALP_V2' AND raw_move_source='STRATEGY_CLAIM' AND raw_expected_move>0 ORDER BY evaluated_at"
    ).fetchall()
    conn.close()
    out: dict = {"claims": len(claims), "horizons": {}}
    for h in HORIZONS:
        recs, bar_marks, exec_marks = [], [], []
        for sym, t, claim, mj in claims:
            s = data["symbols"].get(str(sym).replace("USDT", ""))
            if s is None:
                continue
            i = int(np.searchsorted(s["ts"], float(t), side="right") - 1)
            if i < 0 or float(t) - s["ts"][i] > 5.0:
                continue
            j = int(np.searchsorted(s["ts"], s["ts"][i] + h, side="left"))
            if j >= len(s["ts"]) or s["ts"][j] - (s["ts"][i] + h) > MATCH_TOL_SEC or s["bad"][j] != s["bad"][i] or not (s["usable"][i] and s["usable"][j]):
                continue
            g = float(s["bid"][j] / s["ask"][i] - 1.0)
            recs.append(er.Row(t=float(t), group=int(float(t) // 5), symbol=str(sym), x=np.concatenate([s["X"][i], [float(claim)]]), label=g, label_ts=float(t) + h, extra={"claim": float(claim)}))
            with_mark = json.loads(mj or "{}").get(str(h))
            if with_mark is not None:
                bar_marks.append(float(with_mark))
                exec_marks.append(g - cost)
        if len(recs) < 200:
            out["horizons"][h] = {"n": len(recs)}
            continue
        c = np.array([r.extra["claim"] for r in recs])
        y = np.array([r.label for r in recs])
        entry = {"n": len(recs), "claim_mean": float(c.mean()), "exec_gross_mean": float(y.mean()), "capture": float(y.mean() / c.mean()), "claim_rank": er.spearman(c, y)}
        if bar_marks:
            bm, em = np.array(bar_marks), np.array(exec_marks)
            entry["bar_mark_vs_exec"] = {"n": len(bm), "bar_mark_mean_net": float(bm.mean()), "exec_mean_net": float(em.mean()), "corr": float(np.corrcoef(bm, em)[0, 1]) if len(bm) > 2 else None}
        with_claim, without = [], []
        for lo, hi in er.fold_bounds([r.t for r in recs], folds):
            train, test = er.fold_rows(recs, lo, hi)
            if len(train) < 100 or len(test) < 30:
                continue
            yt = np.array([r.label for r in test])
            with_claim.append(er.spearman(er.fit_ridge(train)(test), yt))
            strip = lambda rs: [er.Row(t=r.t, group=r.group, symbol=r.symbol, x=r.x[:-1], label=r.label, label_ts=r.label_ts) for r in rs]  # noqa: E731
            without.append(er.spearman(er.fit_ridge(strip(train))(strip(test)), yt))
        entry["oos_rank_with_claim"] = with_claim
        entry["oos_rank_without_claim"] = without
        out["horizons"][h] = entry
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cost = cost_beyond_spread()
    data = load(args.db)
    out = {
        "version": er.EDGE_RESEARCH_VERSION,
        "cost_beyond_spread": cost,
        "features": len(data["keys"]),
        "snapshots": {s: len(v["ts"]) for s, v in data["symbols"].items()},
        "usable": {s: int(v["usable"].sum()) for s, v in data["symbols"].items()},
        "gap_segments": {s: int(v["bad"][-1]) for s, v in data["symbols"].items()},
    }
    out["horizons"] = [horizon_study(data, h, args.folds, cost) for h in HORIZONS]
    out["claim"] = claim_study(args.db, data, cost, args.folds)
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
