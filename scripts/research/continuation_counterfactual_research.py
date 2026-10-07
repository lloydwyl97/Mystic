#!/usr/bin/env python3
"""Build COUNTERFACTUAL_EXECUTABLE continuation states and test exit-now vs continue. Research only.

SCALP entries: every strategy claim and a 5-minute grid on the 5 s snapshot
window. DAY entries: every (symbol, 15m bar) with a reconstructed DAY context
(from day_edge_research's cache), entered at the bar close plus half spread.
States are written to a separate research DB; nothing reaches accounting.

Usage: continuation_counterfactual_research.py --db /tmp/rx.db --states /tmp/cf_states.db [--day-cache /tmp/day_edge_recs.pkl]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import counterfactual_continuation as cc
from backend.services import edge_research as er
from backend.services.continuation_surface import FEATURE_NAMES

HALF_SPREAD = 0.00006


def load_book(db: str) -> dict[str, dict[str, np.ndarray]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, list] = {}
    for sym, ts, fj in conn.execute("SELECT symbol, ts_utc, features_json FROM microstructure_feature_snapshots ORDER BY symbol, ts_utc"):
        f = json.loads(fj)
        b, a = float(f.get("best_bid") or 0), float(f.get("best_ask") or 0)
        if b > 0 and a > b and float(f.get("data_age_sec") or 0) <= 2.0:
            out.setdefault(str(sym), []).append((float(ts), b, a))
    conn.close()
    return {s: {"ts": np.array([r[0] for r in v]), "bid": np.array([r[1] for r in v]), "ask": np.array([r[2] for r in v])} for s, v in out.items()}


def evaluate(rows: list[dict], horizons, folds: int = 4) -> dict:
    res = {}
    for h in horizons:
        recs = [
            er.Row(t=r["state_t"], group=r["entry_t"], symbol=r["symbol"], x=np.array([r["features"][k] for k in FEATURE_NAMES]), label=r["advantage"][str(h)], label_ts=r["state_t"] + h)
            for r in rows
            if str(h) in r["advantage"]
        ]
        if len(recs) < 500:
            res[str(h)] = {"pairs": len(recs)}
            continue
        recs.sort(key=lambda r: r.t)
        per = []
        for lo, hi in er.fold_bounds([r.t for r in recs], folds):
            train, test = er.fold_rows(recs, lo, hi)
            if len(train) < 200 or len(test) < 100:
                continue
            pred = er.fit_ridge(train)(test)
            y = np.array([r.label for r in test])
            cont = pred > 0
            per.append(
                {
                    "rank": er.spearman(pred, y),
                    "continue_share": float(cont.mean()),
                    "continue_mean_adv": float(y[cont].mean()) if cont.any() else None,
                    "policy_gain_per_state": float(np.where(cont, y, 0.0).mean()),
                    "always_continue": float(y.mean()),
                }
            )
        res[str(h)] = {"pairs": len(recs), "mean_adv": float(np.mean([r.label for r in recs])), "folds": per}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--states", required=True)
    ap.add_argument("--day-cache", default="/tmp/day_edge_recs.pkl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from backend.config.trading_economics import SLIPPAGE_BUFFER, TAKER_FEE, canonical_roundtrip_cost_pct
    from backend.services.continuation_surface import DAY_ADVANTAGE_HORIZONS, SCALP_ADVANTAGE_HORIZONS
    from backend.services.day_v2.lifecycle_sim import ohlcv_bars_1m
    from backend.services.economic_replay import BarStore

    Path(args.states).unlink(missing_ok=True)
    out: dict = {"kind": cc.CF_KIND}
    book = load_book(args.db)
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    scalp_rows: list[dict] = []
    for sym, b in book.items():
        lo, hi = float(b["ts"][0]), float(b["ts"][-1])
        claims = [
            float(t)
            for (t,) in conn.execute(
                "SELECT evaluated_at FROM adaptive_candidate_markouts WHERE engine_id='SCALP_V2' AND raw_move_source='STRATEGY_CLAIM' AND symbol=? AND evaluated_at BETWEEN ? AND ?",
                (f"{sym}USDT", lo, hi),
            )
        ]
        grid = list(np.arange(np.ceil(lo / 300) * 300, hi, 300.0))
        scalp_rows.extend(cc.scalp_states(b, sym, sorted(set(claims) | set(grid)), cost=2 * TAKER_FEE + 2 * SLIPPAGE_BUFFER))
    conn.close()
    out["scalp_states"] = cc.write_states(args.states, scalp_rows)
    out["scalp_entries"] = len({(r["symbol"], r["entry_t"]) for r in scalp_rows})
    out["scalp_pairs"] = sum(len(r["advantage"]) for r in scalp_rows)
    out["scalp"] = evaluate(scalp_rows, SCALP_ADVANTAGE_HORIZONS)
    day_rows: list[dict] = []
    if Path(args.day_cache).exists():
        recs = pickle.load(open(args.day_cache, "rb"))
        store = BarStore(args.db)
        entries: dict[str, set] = {}
        for r in recs:
            entries.setdefault(r["symbol"], set()).add(float(r["t"]))
        cost = canonical_roundtrip_cost_pct() - HALF_SPREAD
        for sym, ts in entries.items():
            pts = []
            for t in sorted(ts):
                b15 = store.closed(sym, "15m", 1, t)
                if b15:
                    pts.append((t, float(b15[-1]["close"]) * (1.0 + HALF_SPREAD)))
            bars = ohlcv_bars_1m(args.db, sym, min(ts), max(ts) + 43200 + 7200)
            day_rows.extend(cc.day_states(bars, sym, pts, cost=cost, half_spread=HALF_SPREAD))
        out["day_states"] = cc.write_states(args.states, day_rows)
        out["day_entries"] = len({(r["symbol"], r["entry_t"]) for r in day_rows})
        out["day_pairs"] = sum(len(r["advantage"]) for r in day_rows)
        out["day"] = evaluate(day_rows, DAY_ADVANTAGE_HORIZONS)
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
