#!/usr/bin/env python3
"""Maker vs taker counterfactual on the snapshot window. Research only; no orders.

A post-only bid is placed at the best bid of the 5 s microstructure snapshot at
decision time. With no historical queue depth, a fill is counted only when a
seller-initiated print trades strictly below that bid inside the wait window:
the whole queue at the price was consumed. Prints at the bid itself may or may
not have reached the order and are not counted (conservative). The position
exits as a taker at the bid of the first snapshot at or after fill + h.

The taker counterfactual buys the ask at t and sells the bid at t + h. Every
fill here is COUNTERFACTUAL_EXECUTABLE and never touches accounting.

Usage: maker_counterfactual_research.py --db /tmp/rx.db --trades /tmp/rx_trades.db [--fetch]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services import edge_research as er

FILL_KIND = "COUNTERFACTUAL_EXECUTABLE"
WAITS = (30, 60, 120)
HOLDS = (60, 300)
MARKOUT_SEC = 30
SAMPLE_EVERY_SEC = 60
API = "https://api.binance.us/api/v3/aggTrades"


def fetch(trades_db: str, symbols: list[str], start: float, end: float) -> dict[str, int]:
    conn = sqlite3.connect(trades_db)
    conn.execute("CREATE TABLE IF NOT EXISTS agg_trades (symbol TEXT, agg_id INTEGER, price REAL, qty REAL, t_ms INTEGER, buyer_maker INTEGER, PRIMARY KEY(symbol, agg_id))")
    counts = {}
    for sym in symbols:
        n = 0
        lo = int(start * 1000)
        while lo < int(end * 1000):
            hi = min(lo + 3600_000 - 1, int(end * 1000))
            url = f"{API}?symbol={sym}USDT&startTime={lo}&endTime={hi}&limit=1000"
            while True:
                with urllib.request.urlopen(url, timeout=30) as r:
                    rows = json.loads(r.read())
                conn.executemany("INSERT OR IGNORE INTO agg_trades VALUES (?,?,?,?,?,?)", [(sym, int(x["a"]), float(x["p"]), float(x["q"]), int(x["T"]), 1 if x["m"] else 0) for x in rows])
                n += len(rows)
                if len(rows) < 1000:
                    break
                url = f"{API}?symbol={sym}USDT&fromId={int(rows[-1]['a']) + 1}&limit=1000"
                if int(rows[-1]["T"]) > hi:
                    break
                time.sleep(0.1)
            conn.commit()
            lo = hi + 1
            time.sleep(0.1)
        counts[sym] = n
    conn.close()
    return counts


def load_book(db: str) -> dict[str, dict[str, np.ndarray]]:
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out: dict[str, list] = {}
    for sym, ts, fj in conn.execute("SELECT symbol, ts_utc, features_json FROM microstructure_feature_snapshots ORDER BY symbol, ts_utc"):
        f = json.loads(fj)
        b, a = float(f.get("best_bid") or 0), float(f.get("best_ask") or 0)
        if b > 0 and a > b:
            out.setdefault(str(sym), []).append((float(ts), b, a))
    conn.close()
    return {s: {"ts": np.array([r[0] for r in v]), "bid": np.array([r[1] for r in v]), "ask": np.array([r[2] for r in v])} for s, v in out.items()}


def load_sells(trades_db: str) -> dict[str, dict[str, np.ndarray]]:
    conn = sqlite3.connect(f"file:{trades_db}?mode=ro", uri=True)
    out = {}
    for sym in [r[0] for r in conn.execute("SELECT DISTINCT symbol FROM agg_trades")]:
        rows = conn.execute("SELECT t_ms, price FROM agg_trades WHERE symbol=? AND buyer_maker=1 ORDER BY t_ms, agg_id", (sym,)).fetchall()
        out[sym] = {"t": np.array([r[0] / 1000.0 for r in rows]), "p": np.array([r[1] for r in rows])}
    conn.close()
    return out


def snap_at(book: dict[str, np.ndarray], t: float, tol: float = 10.0) -> int | None:
    j = int(np.searchsorted(book["ts"], t, side="left"))
    return j if j < len(book["ts"]) and book["ts"][j] - t <= tol else None


def snap_before(book: dict[str, np.ndarray], t: float, tol: float = 5.0) -> int | None:
    i = int(np.searchsorted(book["ts"], t, side="right") - 1)
    return i if i >= 0 and t - book["ts"][i] <= tol else None


def first_trade_through(sells: dict[str, np.ndarray], bid: float, t0: float, t1: float) -> float | None:
    lo = int(np.searchsorted(sells["t"], t0, side="right"))
    hi = int(np.searchsorted(sells["t"], t1, side="right"))
    if hi <= lo:
        return None
    hit = np.nonzero(sells["p"][lo:hi] < bid)[0]
    return float(sells["t"][lo + hit[0]]) if len(hit) else None


def simulate(book, sells, times: list[tuple[str, float]], *, wait: float, hold: float, maker_fee: float, taker_fee: float, slip: float) -> list[dict]:
    out = []
    for sym, t in times:
        b, s = book.get(sym), sells.get(sym)
        if b is None or s is None or not len(s["t"]):
            continue
        i = snap_before(b, t)
        if i is None:
            continue
        bid, ask = float(b["bid"][i]), float(b["ask"][i])
        j = snap_at(b, t + hold)
        if j is None or t + wait > s["t"][-1]:
            continue
        taker = float(b["bid"][j]) / ask - 1.0 - 2 * taker_fee - 2 * slip
        tf = first_trade_through(s, bid, t, t + wait)
        rec = {"symbol": sym, "t": t, "fill_kind": FILL_KIND, "taker_net": taker, "filled": tf is not None}
        if tf is not None:
            k = snap_at(b, tf + hold)
            m = snap_at(b, tf + MARKOUT_SEC)
            if k is None or m is None:
                continue
            rec["fill_delay"] = tf - t
            rec["maker_net"] = float(b["bid"][k]) / bid - 1.0 - maker_fee - taker_fee - slip
            rec["markout_mid"] = (float(b["bid"][m]) + float(b["ask"][m])) / 2.0 / bid - 1.0
        u = snap_at(b, t + MARKOUT_SEC)
        if u is None:
            continue
        rec["uncond_markout_mid"] = (float(b["bid"][u]) + float(b["ask"][u])) / 2.0 / bid - 1.0
        out.append(rec)
    return out


def summarize(recs: list[dict]) -> dict:
    if not recs:
        return {"n": 0}
    filled = [r for r in recs if r["filled"]]
    missed = [r for r in recs if not r["filled"]]
    taker = np.array([r["taker_net"] for r in recs])
    maker_policy = np.array([r.get("maker_net", 0.0) for r in recs])
    out = {
        "candidates": len(recs),
        "fills": len(filled),
        "fill_rate": len(filled) / len(recs),
        "taker_net_per_candidate": float(taker.mean()),
        "maker_net_per_candidate": float(maker_policy.mean()),
        "maker_minus_taker_per_candidate": float(maker_policy.mean() - taker.mean()),
        "maker_minus_taker_t": er.t_stat(maker_policy - taker),
        "missed_winners": sum(1 for r in missed if r["taker_net"] > 0),
        "avoided_losers": sum(1 for r in missed if r["taker_net"] < 0),
        "missed_taker_net_mean": float(np.mean([r["taker_net"] for r in missed])) if missed else None,
    }
    if filled:
        mk = np.array([r["maker_net"] for r in filled])
        tk = np.array([r["taker_net"] for r in filled])
        out.update(
            {
                "maker_net_per_fill": float(mk.mean()),
                "taker_net_same_candidates": float(tk.mean()),
                "maker_pf": er.book_stats(list(mk)).get("pf"),
                "fill_delay_median": float(np.median([r["fill_delay"] for r in filled])),
                "post_fill_markout_mid": float(np.mean([r["markout_mid"] for r in filled])),
                "uncond_markout_mid": float(np.mean([r["uncond_markout_mid"] for r in recs])),
                "adverse_selection": float(np.mean([r["markout_mid"] for r in filled]) - np.mean([r["uncond_markout_mid"] for r in recs])),
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--trades", required=True)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    from backend.config.trading_economics import MAKER_FEE, SLIPPAGE_BUFFER, TAKER_FEE

    book = load_book(args.db)
    start = max(float(v["ts"][0]) for v in book.values())
    end = min(float(v["ts"][-1]) for v in book.values())
    out: dict = {"fill_kind": FILL_KIND, "maker_fee": MAKER_FEE, "taker_fee": TAKER_FEE, "slippage": SLIPPAGE_BUFFER, "window": [start, end]}
    if args.fetch:
        out["fetched"] = fetch(args.trades, sorted(book), start, end)
    sells = load_sells(args.trades)
    out["sell_prints"] = {s: len(v["t"]) for s, v in sells.items()}
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    claims = [
        (str(s).replace("USDT", ""), float(t))
        for s, t in conn.execute(
            "SELECT symbol, evaluated_at FROM adaptive_candidate_markouts "
            "WHERE engine_id='SCALP_V2' AND raw_move_source='STRATEGY_CLAIM' AND evaluated_at>=? AND evaluated_at<=? ORDER BY evaluated_at",
            (start, end),
        )
    ]
    conn.close()
    grid = [(s, float(t)) for s in sorted(book) for t in np.arange(np.ceil(start / SAMPLE_EVERY_SEC) * SAMPLE_EVERY_SEC, end, SAMPLE_EVERY_SEC)]
    for name, times in (("claims", claims), ("grid_60s", grid)):
        out[name] = {}
        for wait in WAITS:
            for hold in HOLDS:
                for fee_label, fee in (("cfg", MAKER_FEE), ("1bp", 0.0001)):
                    recs = simulate(book, sells, times, wait=wait, hold=hold, maker_fee=fee, taker_fee=TAKER_FEE, slip=SLIPPAGE_BUFFER)
                    thirds = er.fold_bounds([r["t"] for r in recs], 3) if recs else []
                    out[name][f"w{wait}_h{hold}_{fee_label}"] = {
                        "all": summarize(recs),
                        "folds": [summarize([r for r in recs if lo <= r["t"] < hi]) for lo, hi in ([(min(r["t"] for r in recs), thirds[0][0]), *thirds] if thirds else [])],
                    }
    text = json.dumps(out, indent=1, default=float)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
