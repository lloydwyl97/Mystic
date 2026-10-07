#!/usr/bin/env python3
"""DAY edge research on a research extract (read-only). Research only.

Reconstructs every DAY context the live loop would rank (fired detector plus
context states with a valid structural zone) on each closed 15m bar, labels
each with the live exit contract from the bar-close ask over closed 1m bars,
joins the causal 145-dim vector stamped at or before the decision and the last
closed order-flow bar, then compares arms in expanding chronological folds.

Usage: day_edge_research.py --db /tmp/rx.db [--since EPOCH] [--folds 5] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import edge_research as er
from backend.services.day_feature_catalog import CAUSAL_DIMS, DAY_FEATURE_NAMES, causal_features
from backend.services.day_research_rows import VECTOR_MAX_AGE_SEC, _closed_flow, _flow_index, _VectorIndex

HALF_SPREAD = 0.00006


def reconstruct(db: str, since: float, until: float, label_now: float) -> list[dict]:
    from backend.config.trading_economics import canonical_roundtrip_cost_pct
    from backend.services.adaptive_learning import day_learned_setup, market_regime_tag
    from backend.services.day_v2.lifecycle_sim import DAY_LIFECYCLE_MAX_MIN, LifecycleParams, simulate_lifecycle
    from backend.services.day_v2.live_signal import context_entry_signals, day_state_features, evaluate_entry_signal
    from backend.services.day_v2.structural_entry import evaluate_structural_zone
    from backend.services.economic_replay import BAR_LIMITS, DAY_SYMBOLS, BarStore

    cost = canonical_roundtrip_cost_pct()
    store = BarStore(db, DAY_SYMBOLS)
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    vectors = _VectorIndex(conn, since, until)
    flow = _flow_index(conn, since, until + 900)
    conn.close()
    out: list[dict] = []
    t = float(np.ceil(since / 900.0) * 900.0)
    while t < until:
        for sym in DAY_SYMBOLS:
            b15 = store.closed(sym, "15m", BAR_LIMITS["15m"], t)
            if len(b15) < 32 or b15[-1]["ts_epoch"] + 900 < t - 1:
                continue
            b1h = store.closed(sym, "1h", BAR_LIMITS["1h"], t)
            b4h = store.closed(sym, "4h", BAR_LIMITS["4h"], t)
            fired = evaluate_entry_signal(sym, b15, b1h, b4h)
            seen: set[str] = set()
            sigs = []
            for s in ([fired] if fired is not None else []) + context_entry_signals(sym, b15, b1h, b4h):
                if s is not None and s.setup not in seen:
                    seen.add(s.setup)
                    sigs.append(s)
            if not sigs:
                continue
            vec = vectors.asof(sym, t)
            if vec is None or t - vec[0] > VECTOR_MAX_AGE_SEC:
                continue
            feats = causal_features([float(x) for x in json.loads(vec[1])])
            _, fl = _closed_flow(flow.get(sym, {}), t)
            regime = market_regime_tag(db, sym, as_of=t) or ""
            ask = float(b15[-1]["close"]) * (1.0 + HALF_SPREAD)
            for s in sigs:
                if not evaluate_structural_zone(s).valid:
                    continue
                params = LifecycleParams.from_signal(s, entry_price=ask, entry_time=t)
                lab = simulate_lifecycle(params, store.minute_path(sym, t, t + DAY_LIFECYCLE_MAX_MIN * 60 + 60), roundtrip_cost=cost, now=label_now)
                if not lab.get("final") or lab.get("net") is None:
                    continue
                out.append(
                    {
                        "t": t,
                        "symbol": sym,
                        "setup": day_learned_setup(s.setup, detector_fired=s is fired),
                        "regime": regime,
                        "feature_ts": vec[0],
                        "features": feats,
                        "state": day_state_features(b15, b1h, s.setup),
                        "flow": fl,
                        "prob_buy": vec[2],
                        "label": float(lab["net"]),
                        "exit_ts": float(lab["exit_time"]),
                        "reason": lab["reason"],
                        "mfe": lab["mfe"],
                    }
                )
        t += 900.0
    return out


def to_rows(recs: list[dict], *, vector_only: bool = False) -> list[er.Row]:
    setups = sorted({r["setup"] for r in recs})
    syms = sorted({r["symbol"] for r in recs})
    state_keys = sorted({k for r in recs for k in r["state"]})
    flow_keys = sorted({k for r in recs for k in r["flow"]})
    vec_keys = [DAY_FEATURE_NAMES[d] for d in CAUSAL_DIMS]
    rows = []
    for r in recs:
        x = [r["features"][k] for k in vec_keys]
        if not vector_only:
            x += [float(r["state"].get(k, 0.0)) for k in state_keys]
            x += [float(r["flow"].get(k, 0.0)) for k in flow_keys]
            x += [1.0 if r["setup"] == s else 0.0 for s in setups]
            x += [1.0 if r["symbol"] == s else 0.0 for s in syms]
        rows.append(
            er.Row(
                t=r["t"],
                group=r["t"],
                symbol=r["symbol"],
                x=np.nan_to_num(np.array(x, dtype=float)),
                label=r["label"],
                label_ts=r["exit_ts"],
                key=(r["setup"], r["symbol"], r["regime"]),
                extra={"exit_ts": r["exit_ts"], "prob_buy": r["prob_buy"]},
            )
        )
    return rows


def run_arms(rows: list[er.Row], rows145: list[er.Row], folds: int) -> dict:
    bounds = er.fold_bounds([r.t for r in rows], folds)
    results = []
    for k, (lo, hi) in enumerate(bounds):
        train, test = er.fold_rows(rows, lo, hi)
        train145, test145 = er.fold_rows(rows145, lo, hi)
        if len(train) < 100 or len(test) < 20:
            continue
        arms: dict[str, dict] = {}

        def add(name: str, pred: np.ndarray, *, tr=train, te=test, arms=arms) -> None:
            arms[name] = er.evaluate_arm(name, tr, te, np.asarray(pred, dtype=float), regression=True)

        add("hierarchical_current", er.fit_hierarchical(train)(test))
        add("ridge", er.fit_ridge(train)(test))
        add("ridge_145_only", er.fit_ridge(train145)(test145), tr=train145, te=test145)
        add("huber", er.fit_huber(train)(test))
        add("tree_hgb", er.fit_tree(train)(test))
        add("pairwise", er.fit_pairwise(train)(test))
        sgd = er.online_sgd(test, warm=train)
        add("online_sgd", sgd)
        pb_tr = np.array([float(r.extra["prob_buy"] or 0.0) for r in train])
        pb_te = np.array([float(r.extra["prob_buy"] or 0.0) for r in test])
        add("rf145_prob_buy", er.calibrate(train, pb_tr)(pb_te))
        arms["take_all"] = {"book": er.closed_loop(test, np.ones(len(test))), "ranking": er.ranking_metrics(test, np.random.default_rng(k).random(len(test)))}
        results.append({"fold": k + 1, "lo": lo, "hi": hi, "n_train": len(train), "n_test": len(test), "market_mean": float(np.mean([r.label for r in test])), "arms": arms})
    names = sorted({a for f in results for a in f["arms"]})
    return {"folds": results, "summary": [er.summarize(results, a) for a in names]}


def live_learner(rows_pkl: str, folds: int) -> dict:
    """The live learner's own decision-time scores against the market label."""
    recs = [r for r in pickle.load(open(rows_pkl, "rb")) if r["market_label"] is not None and r["rank_score"] is not None and r["candidate_state"] == "QUALIFIED"]
    rows = [
        er.Row(t=r["decision_ts"], group=r["decision_point"], symbol=r["symbol"], x=np.zeros(1), label=r["market_label"], label_ts=r["decision_ts"], extra={"exit_ts": r["decision_ts"] + 60 * 60})
        for r in recs
    ]
    pv = np.array([r["rank_score"] for r in recs])
    ma = np.array([r["market_alpha"] or 0.0 for r in recs])
    labels = np.array([r.label for r in rows])
    out = {
        "n": len(rows),
        "rank_score_rank": er.spearman(pv, labels),
        "market_alpha_rank": er.spearman(ma, labels),
        "rank_score_bias": float(np.mean(pv - labels)),
        "market_alpha_bias": float(np.mean(ma - labels)),
        "ranking": er.ranking_metrics(rows, pv),
        "folds": [],
    }
    for lo, hi in er.fold_bounds([r.t for r in rows], folds):
        idx = [i for i, r in enumerate(rows) if lo <= r.t < hi]
        if len(idx) < 10:
            continue
        sub = [rows[i] for i in idx]
        out["folds"].append({"lo": lo, "n": len(idx), "rank": er.spearman(pv[idx], labels[idx]), "bias": float(np.mean(pv[idx] - labels[idx])), "ranking": er.ranking_metrics(sub, pv[idx])})
    filled = [r for r in pickle.load(open(rows_pkl, "rb")) if r["policy_effect"] is not None]
    if filled:
        gaps = np.array([r["policy_gap"] or 0.0 for r in filled])
        eff = np.array([r["policy_effect"] for r in filled])
        out["policy_gap"] = {
            "n": len(filled),
            "rank": er.spearman(gaps, eff),
            "bias": float(np.mean(gaps - eff)),
            "mean_effect": float(eff.mean()),
            "realized_mean": float(np.mean([r["realized_net"] for r in filled])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--since", type=float, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--rows-pkl", default="")
    ap.add_argument("--cache", default="/tmp/day_edge_recs.pkl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    now = json.loads(conn.execute("SELECT value FROM research_extract_meta WHERE key='now'").fetchone()[0])
    vec0 = conn.execute("SELECT MIN(ts_utc) FROM ai_inference_log").fetchone()[0]
    conn.close()
    from backend.services.day_research_rows import _epoch

    since = args.since or float(np.ceil((_epoch(vec0) + 60) / 900.0) * 900.0)
    until = float(now) - 720 * 60 - 900
    t0 = time.time()
    if args.cache and Path(args.cache).exists():
        recs = pickle.load(open(args.cache, "rb"))
    else:
        recs = reconstruct(args.db, since, until, float(now))
        if args.cache:
            pickle.dump(recs, open(args.cache, "wb"))
    leak = sum(1 for r in recs if r["feature_ts"] > r["t"])
    out = {
        "version": er.EDGE_RESEARCH_VERSION,
        "since": since,
        "until": until,
        "candidates": len(recs),
        "decision_points": len({r["t"] for r in recs}),
        "feature_after_decision": leak,
        "market_mean": float(np.mean([r["label"] for r in recs])) if recs else None,
        "reconstruct_sec": round(time.time() - t0, 1),
    }
    out.update(run_arms(to_rows(recs), to_rows(recs, vector_only=True), args.folds))
    if args.rows_pkl:
        out["live_learner"] = live_learner(args.rows_pkl, 3)
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
