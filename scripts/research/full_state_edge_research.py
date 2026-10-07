#!/usr/bin/env python3
"""Full-state DAY/SCALP edge research and short-horizon label rebuild. Research only.

Reads a research extract. Writes nothing to live accounting. 30s and 60s
labels use the last executable bid at or before the horizon; a 1-minute bar
close is not a substitute. Continuation pairs are weighted so each trajectory
counts once.

Usage: full_state_edge_research.py --db /tmp/rx.db --cf /tmp/cf_states.db --out /tmp/fs_edge.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import edge_research as er
from backend.services.day_feature_catalog import causal_features
from backend.services.day_research_rows import _epoch, _VectorIndex
from backend.services.horizon_alignment import (
    SCALP_HORIZONS,
    SHORT_HORIZONS,
    align_observation,
    bar_close_known_at,
    executable_gross,
    max_early_sec,
    rebuild_disposition,
)

HALF = 0.00006
DAY_HORIZONS = (900, 1800, 3600, 7200, 14400, 21600, 43200)


def _cost() -> float:
    from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE

    return 2.0 * float(TAKER_FEE) + 2.0 * float(SLIPPAGE_BUFFER)


def _scalp_mod():
    path = Path(__file__).with_name("scalp_move_research.py")
    spec = importlib.util.spec_from_file_location("scalp_move_research", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _book(db: str) -> dict[str, dict[str, np.ndarray]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    raw: dict[str, list] = {}
    for sym, ts, fj in conn.execute("SELECT symbol, ts_utc, features_json FROM microstructure_feature_snapshots ORDER BY symbol, ts_utc"):
        try:
            feat = json.loads(fj)
        except (TypeError, ValueError):
            continue
        bid, ask = float(feat.get("best_bid") or 0), float(feat.get("best_ask") or 0)
        if bid > 0 and ask > bid:
            raw.setdefault(str(sym), []).append((float(ts), bid, ask))
    conn.close()
    return {s: {"ts": np.array([r[0] for r in v]), "bid": np.array([r[1] for r in v]), "ask": np.array([r[2] for r in v])} for s, v in raw.items()}


def _at_or_before(ts: np.ndarray, px: np.ndarray, target: float, early: float) -> tuple[float, float] | None:
    j = int(np.searchsorted(ts, target, side="right")) - 1
    if j < 0 or ts[j] > target or target - ts[j] > early:
        return None
    return float(ts[j]), float(px[j])


def rebuild_short_labels(db: str, book: dict[str, dict[str, np.ndarray]]) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute("SELECT symbol, evaluated_at, markouts_json FROM adaptive_candidate_markouts WHERE engine_id='SCALP_V2'").fetchall()
    conn.close()
    counts = {h: Counter() for h in SCALP_HORIZONS}
    err = {h: [] for h in SCALP_HORIZONS}
    sources = {h: Counter() for h in SCALP_HORIZONS}
    late = dict.fromkeys(SCALP_HORIZONS, 0)
    for sym, evaluated, raw in rows:
        base = str(sym).replace("USDT", "")
        s = book.get(base)
        marks = {}
        try:
            marks = json.loads(raw or "{}")
        except (TypeError, ValueError):
            marks = {}
        if s is None:
            for h in SHORT_HORIZONS:
                counts[h][rebuild_disposition(marks.get(str(h)) is not None, "MISSING")] += 1
            continue
        entry = _at_or_before(s["ts"], s["ask"], float(evaluated), 5.0)
        for h in SCALP_HORIZONS:
            target = float(evaluated) + h
            hit = None if entry is None else _at_or_before(s["ts"], s["bid"], target, max_early_sec(h))
            status = "OK" if hit else "MISSING"
            if h in SHORT_HORIZONS:
                counts[h][rebuild_disposition(marks.get(str(h)) is not None, status)] += 1
            if hit:
                sources[h]["book_bid"] += 1
                timing = hit[0] - target
                err[h].append(timing)
                late[h] += int(timing > 0)
            else:
                sources[h]["MISSING"] += 1
    return {
        "version": "HORIZON_LABELS_V2",
        "candidates": len(rows),
        "by_horizon": {
            str(h): {
                "requested_horizon_sec": h,
                "source": dict(sources[h]),
                "max_timing_error_sec": max(err[h]) if err[h] else None,
                "min_timing_error_sec": min(err[h]) if err[h] else None,
                "observations_after_horizon": late[h],
                "disposition": dict(counts[h]) if h in SHORT_HORIZONS else None,
            }
            for h in SCALP_HORIZONS
        },
    }


def _bars(db: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    raw: dict[str, list] = {}
    for sym, ts, close in conn.execute("SELECT symbol, ts, close FROM feature_ohlcv WHERE interval='1m'"):
        opened = _epoch(ts)
        if opened is None or close is None:
            continue
        base = str(sym).upper().replace("-", "").replace("/", "").replace("USDT", "")
        raw.setdefault(base, []).append((bar_close_known_at(opened, 60.0), float(close)))
    conn.close()
    out = {}
    for sym, rows in raw.items():
        rows.sort()
        out[sym] = (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]))
    return out


def day_full_state(db: str, book: dict, bars: dict, cost: float) -> dict:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    hi = conn.execute("SELECT MAX(ts_utc) FROM ai_inference_log").fetchone()[0]
    lo = conn.execute("SELECT MIN(ts_utc) FROM ai_inference_log WHERE strategy_id='day' AND feature_dim=145").fetchone()[0]
    until, since = _epoch(hi), _epoch(lo)
    vectors = _VectorIndex(conn, float(since), float(until))
    setups: dict[tuple[str, int], str] = {}
    for sym, evaluated, setup in conn.execute("SELECT symbol, evaluated_at, setup FROM adaptive_candidate_markouts WHERE engine_id='DAY_V2' AND setup IS NOT NULL"):
        base = str(sym).upper().replace("-", "").replace("/", "").replace("USDT", "")
        setups[(base, int(float(evaluated) // 900) * 900)] = str(setup)
    conn.close()
    symbols = sorted({s.replace("USDT", "") for s in vectors.ts})
    if until is None or since is None:
        return {"rows": 0}
    start = math.ceil(since / 900) * 900
    end = until - 43200
    grid = np.arange(start, end, 900.0)
    names = None
    packed = []
    no_setup = 0
    for t in grid:
        cross = {}
        got = {}
        for sym in symbols:
            key = sym if sym in vectors.ts else sym + "USDT"
            row = vectors.asof(key, float(t)) or vectors.asof(sym + "USDT", float(t)) or vectors.asof(sym, float(t))
            if row is None or float(t) - row[0] > 300:
                continue
            try:
                vec = json.loads(row[1])
            except (TypeError, ValueError):
                continue
            if not isinstance(vec, list) or len(vec) != 145 or float(vec[0] or 0) <= 0:
                continue
            feats = causal_features(vec)
            got[sym] = feats
            cross[sym] = math.log(float(vec[0]))
        if len(got) < 2:
            continue
        for sym, feats in got.items():
            b = bars.get(sym)
            known = _at_or_before(b[0], b[1], float(t), 60.0) if b is not None else None
            bk = book.get(sym)
            entry = _at_or_before(bk["ts"], bk["ask"], float(t), 5.0) if bk is not None else None
            if entry is not None:
                ask, entry_ts, entry_src = entry[1], entry[0], "book_ask"
            elif known is not None:
                ask, entry_ts, entry_src = known[1] * (1.0 + HALF), known[0], "bar_close"
            else:
                continue
            x = dict(feats)
            for other, log_px in cross.items():
                if other != sym:
                    x[f"x_log_price_{other}"] = log_px
            if names is None:
                names = sorted(x)
            setup = setups.get((sym, int(t)), "NO_SETUP")
            no_setup += int(setup == "NO_SETUP")
            labels = {}
            for h in DAY_HORIZONS:
                target = float(t) + h
                points = []
                if bk is not None:
                    hit = _at_or_before(bk["ts"], bk["bid"], target, max_early_sec(h))
                    if hit:
                        points.append((hit[0], hit[1], "book_bid"))
                if b is not None:
                    hit = _at_or_before(b[0], b[1], target, max_early_sec(h))
                    if hit:
                        points.append((hit[0], hit[1] * (1.0 - HALF), "bar_close"))
                aligned = align_observation(points, target, h)
                gross = executable_gross(ask, aligned["price"]) if aligned["status"] == "OK" else None
                labels[h] = None if gross is None else gross - cost
            packed.append(
                {
                    "t": float(t),
                    "sym": sym,
                    "x": np.array([x.get(n, 0.0) for n in names]),
                    "labels": labels,
                    "setup": setup,
                    "entry_src": entry_src,
                    "entry_ts": entry_ts,
                }
            )
    out: dict = {"rows": len(packed), "without_setup": no_setup, "symbols": symbols, "features": 0 if names is None else len(names), "horizons": {}}
    if len(packed) < 50:
        return out
    for h in DAY_HORIZONS:
        recs = [
            er.Row(t=r["t"], group=r["t"], symbol=r["sym"], x=r["x"], label=r["labels"][h], label_ts=r["t"] + h, key=(r["setup"],), extra={"weight": 1.0}) for r in packed if r["labels"][h] is not None
        ]
        if len(recs) < 80:
            out["horizons"][str(h)] = {"n": len(recs)}
            continue
        keys = [(r.symbol, int(r.t // h)) for r in recs]
        w = er.trajectory_weights(keys)
        for r, weight in zip(recs, w, strict=True):
            r.extra["weight"] = float(weight)
        folds = []
        for lo, hi in er.fold_bounds([r.t for r in recs], 4):
            train, test = er.fold_rows(recs, lo, hi)
            if len(train) < 40 or len(test) < 20:
                continue
            y = np.array([r.label for r in test])
            arms = {}
            for name, fit in (("ridge", er.fit_ridge), ("huber", er.fit_huber), ("tree_hgb", er.fit_tree)):
                pred = fit(train)(test)
                rank_m = er.ranking_metrics(test, pred)
                arms[name] = {
                    "global_rank": er.spearman(pred, y),
                    "within_rank": rank_m.get("within_point_rank"),
                    "chosen": rank_m.get("chosen"),
                    "point_mean": rank_m.get("point_mean"),
                    "best": rank_m.get("best"),
                    "regret": rank_m.get("regret"),
                    "bias": float(np.mean(pred - y)),
                }
            folds.append(arms)
        summary = {}
        for name in ("ridge", "huber", "tree_hgb"):
            per = [f[name] for f in folds if name in f]
            summary[name] = {
                "global_rank": [p["global_rank"] for p in per],
                "within_rank": [p["within_rank"] for p in per],
                "chosen_bps": [None if p["chosen"] is None else p["chosen"] * 1e4 for p in per],
                "mean_bps": [None if p["point_mean"] is None else p["point_mean"] * 1e4 for p in per],
                "best_bps": [None if p["best"] is None else p["best"] * 1e4 for p in per],
                "regret_bps": [None if p["regret"] is None else p["regret"] * 1e4 for p in per],
            }
        out["horizons"][str(h)] = {"n": len(recs), "trajectories": len(set(keys)), "effective_weight": er.effective_independent_weight(w), "folds": summary}
    return out


def scalp_full_state(db: str, cost: float) -> dict:
    mod = _scalp_mod()
    data = mod.load(db)
    out: dict = {"features": len(data["keys"]), "horizons": {}, "cost": cost}
    for h in SCALP_HORIZONS:
        rows = []
        for sym, s in data["symbols"].items():
            gross, ok = mod.labels(s, h)
            idx = np.nonzero(ok)[0][::6]
            for i in idx:
                rows.append(er.Row(t=float(s["ts"][i]), group=int(s["ts"][i] // 30), symbol=sym, x=s["X"][i], label=float(gross[i]), label_ts=float(s["ts"][i]) + h))
        rows.sort(key=lambda r: r.t)
        if len(rows) < 500:
            out["horizons"][str(h)] = {"n": len(rows)}
            continue
        keys = [(r.symbol, int(r.t // h)) for r in rows]
        w = er.trajectory_weights(keys)
        for r, weight in zip(rows, w, strict=True):
            r.extra["weight"] = float(weight)
        folds = []
        bounds = er.fold_bounds([r.t for r in rows], 4)
        for lo, hi in bounds:
            train, test = er.fold_rows(rows, lo, hi)
            if len(train) < 400 or len(test) < 100:
                continue
            pred = er.fit_ridge(train)(test)
            y = np.array([r.label for r in test])
            ww = np.array([r.extra["weight"] for r in test])
            top_q = er.weighted_top(pred, y, ww, cost, 0.2)
            top_d = er.weighted_top(pred, y, ww, cost, 0.1)
            folds.append(
                {
                    "rank": er.weighted_spearman(pred, y, ww),
                    "bias": float(np.average(pred - y, weights=ww)),
                    "top_quintile_gross": None if top_q is None else top_q["gross"],
                    "top_quintile_net": None if top_q is None else top_q["net"],
                    "top_decile_gross": None if top_d is None else top_d["gross"],
                    "top_decile_net": None if top_d is None else top_d["net"],
                    "mean_gross": float(np.average(y, weights=ww)),
                }
            )
        out["horizons"][str(h)] = {
            "n": len(rows),
            "trajectories": len(set(keys)),
            "effective_weight": er.effective_independent_weight(w),
            "folds": folds,
            "round_trip_cost": cost,
        }
    return out


def continuation_weights(path: str) -> dict:
    if not path or not Path(path).exists():
        return {"present": False}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    raw = conn.execute("SELECT engine_id, symbol, entry_t, features_json, advantage_json FROM counterfactual_continuation_states").fetchall()
    conn.close()
    by_engine = defaultdict(list)
    for engine, sym, entry_t, fj, aj in raw:
        try:
            feats = json.loads(fj)
            adv = json.loads(aj)
        except (TypeError, ValueError):
            continue
        by_engine[str(engine)].append((str(sym), float(entry_t), feats, adv))
    out = {"present": True, "engines": {}}
    for engine, rows in by_engine.items():
        keys = [(r[0], r[1]) for r in rows]
        w = er.trajectory_weights(keys)
        horizons = sorted({int(h) for r in rows for h in r[3]})
        per = {}
        feat_names = sorted(rows[0][2])
        for h in horizons:
            idx = [i for i, r in enumerate(rows) if str(h) in r[3]]
            if len(idx) < 200:
                per[str(h)] = {"n": len(idx)}
                continue
            sub = [rows[i] for i in idx]
            ww = w[idx]
            x = np.array([[float(r[2].get(n) or 0.0) for n in feat_names] for r in sub])
            y = np.array([float(r[3][str(h)]) for r in sub])
            recs = [er.Row(t=r[1], group=r[1], symbol=r[0], x=x[k], label=float(y[k]), label_ts=r[1] + h, extra={"weight": float(ww[k])}) for k, r in enumerate(sub)]
            recs.sort(key=lambda r: r.t)
            ranks = []
            for lo, hi in er.fold_bounds([r.t for r in recs], 4):
                train, test = er.fold_rows(recs, lo, hi)
                if len(train) < 50 or len(test) < 20:
                    continue
                pred = er.fit_ridge(train)(test)
                yt = np.array([r.label for r in test])
                wt = np.array([r.extra["weight"] for r in test])
                ranks.append({"weighted": er.weighted_spearman(pred, yt, wt), "unweighted": er.spearman(pred, yt)})
            importance = []
            for d, name in enumerate(feat_names):
                importance.append((name, er.weighted_spearman(x[:, d], y, ww)))
            importance.sort(key=lambda z: -(abs(z[1]) if z[1] is not None else -1))
            per[str(h)] = {
                "n": len(idx),
                "weighted_rank": [r["weighted"] for r in ranks],
                "unweighted_rank": [r["unweighted"] for r in ranks],
                "top_features": importance[:5],
            }
        out["engines"][engine] = {"rows": len(rows), "trajectories": len(set(keys)), "effective_weight": er.effective_independent_weight(w), "horizons": per}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--cf", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    cost = _cost()
    book = _book(args.db)
    out = {
        "cost_beyond_spread": cost,
        "short_labels": rebuild_short_labels(args.db, book),
        "continuation": continuation_weights(args.cf),
        "day": day_full_state(args.db, book, _bars(args.db), cost),
        "scalp": scalp_full_state(args.db, cost),
    }
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
