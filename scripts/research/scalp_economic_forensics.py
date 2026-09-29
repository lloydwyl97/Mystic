"""SCALP V2 economic forensics and chronological sizing counterfactuals.

Read-only. Every feature is taken as-of the entry fill (at or before the entry
timestamp); post-entry path fields are labelled ``post_`` and are used only
for loss decomposition, never by a candidate.
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import json
import math
import re
import sqlite3
import statistics
import urllib.request
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

VALID_FROM = "2026-09-26T13:44:07"
SYMBOLS = ("BTC", "ETH", "SOL", "XRP")
MICRO_KEYS = (
    "obi_l1",
    "obi_l5",
    "obi_l10",
    "microprice_pressure",
    "ofi_5s",
    "ofi_30s",
    "agg_flow_imbalance_5s",
    "agg_flow_imbalance_30s",
    "adverse_selection_score",
    "p_adverse_move",
    "bid_absorption_score",
    "ask_absorption_score",
    "spread_pct",
    "depth_fragility",
)
EVAL_RE = re.compile(r"SCALP_EVAL_TIMING symbol=(\w+) .*?passed=(\w+) reject=(\S*) hard=(\S*) setup=(\S*) edge_src=(\S*) exp_move=([-\d.]+) net_edge=([-\d.]+)")
SIGNAL_RE = re.compile(r"SCALP_V2_SIGNAL symbol=(\S+) setup=(\S+) opp=(\S+) arm_price=([\d.]+) notional=([\d.]+)")
MIN_SIZE = 0.5


def epoch(ts: str) -> float:
    return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()


def pf(values: list[float]) -> float:
    gains = sum(v for v in values if v > 0)
    losses = -sum(v for v in values if v < 0)
    return gains / losses if losses > 0 else (math.inf if gains > 0 else 0.0)


def max_drawdown(values: list[float]) -> float:
    peak = cum = worst = 0.0
    for v in values:
        cum += v
        peak = max(peak, cum)
        worst = min(worst, cum - peak)
    return worst


def auc(pos: list[float], neg: list[float]) -> float | None:
    """Tie-corrected Mann-Whitney AUC: P(pos > neg)."""
    if not pos or not neg:
        return None
    pooled = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    ranks: list[float] = [0.0] * len(pooled)
    i = 0
    while i < len(pooled):
        j = i
        while j + 1 < len(pooled) and pooled[j + 1][0] == pooled[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_pos = sum(r for r, (_, lab) in zip(ranks, pooled, strict=True) if lab == 1)
    return (rank_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 5:
        return None

    def _rank(vals: list[float]) -> list[float]:
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2.0
            i = j + 1
        return out

    rx, ry = _rank(xs), _rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def expanding_folds(n: int, n_folds: int = 4) -> list[tuple[range, range]]:
    """Chronological expanding folds: fit on everything before the block, evaluate the block."""
    block = n // (n_folds + 1)
    folds = []
    for f in range(1, n_folds + 1):
        start = f * block
        end = n if f == n_folds else (f + 1) * block
        folds.append((range(0, start), range(start, end)))
    return folds


def summarize(values: list[float]) -> dict[str, Any]:
    wins = [v for v in values if v > 0]
    losses = [v for v in values if v <= 0]
    return {
        "n": len(values),
        "w": len(wins),
        "l": len(losses),
        "net": round(sum(values), 4),
        "pf": round(pf(values), 3),
        "exp": round(sum(values) / len(values), 4) if values else 0.0,
        "avg_win": round(statistics.fmean(wins), 4) if wins else 0.0,
        "avg_loss": round(statistics.fmean(losses), 4) if losses else 0.0,
        "dd": round(max_drawdown(values), 4),
    }


# ---------------------------------------------------------------- loading


def load_trades(conn: sqlite3.Connection, since: str) -> list[dict[str, Any]]:
    conn.row_factory = sqlite3.Row
    setups = {r["opportunity_id"]: r["setup_family"] for r in conn.execute("SELECT opportunity_id, setup_family FROM scalp_v2_opportunities")}
    sells = conn.execute(
        "SELECT * FROM paper_trades WHERE engine_id='SCALP_V2' AND UPPER(side)='SELL' AND counts_toward_realized=1 AND timestamp>=? ORDER BY timestamp",
        (since,),
    ).fetchall()
    grouped: dict[str, list[sqlite3.Row]] = defaultdict(list)
    for s in sells:
        ex = json.loads(s["explainability_json"] or "{}")
        grouped[str(ex.get("paper_entry_trade_id") or s["trade_id"])].append(s)
    out = []
    for entry_id, rows in grouped.items():
        buy = conn.execute("SELECT * FROM paper_trades WHERE trade_id=? AND UPPER(side)='BUY'", (entry_id,)).fetchone()
        last = rows[-1]
        ex = json.loads(last["explainability_json"] or "{}")
        opp = last["scalp_opportunity_id"] or (buy["scalp_opportunity_id"] if buy else None)
        qty = sum(float(r["quantity"] or 0) for r in rows)
        entry_price = float((buy["price"] if buy else None) or last["entry_price"] or 0)
        out.append(
            {
                "entry_trade_id": entry_id,
                "opp": opp,
                "symbol": str(last["symbol"]).split("/")[0],
                "setup": setups.get(opp) or ex.get("setup_type") or "UNKNOWN",
                "entry_ts": epoch(buy["timestamp"]) if buy else epoch(last["entry_timestamp"]),
                "exit_ts": epoch(last["timestamp"]),
                "entry_price": entry_price,
                "exit_price": float(last["price"] or 0),
                "qty": qty,
                "notional": qty * entry_price,
                "exit": str(last["exit_reason"] or ""),
                "hold": float(last["hold_time_seconds"] or 0),
                "net": sum(float(r["pnl"] or 0) for r in rows),
                "fees": sum(float(r["fees_paid"] or 0) for r in rows) + float((buy["fees_paid"] if buy else 0) or 0),
                "slippage": sum(float(r["slippage_cost"] or 0) for r in rows),
                "mfe": float(ex.get("mfe_pct") or 0),
                "mae": float(ex.get("mae_pct") or 0),
            }
        )
    out.sort(key=lambda t: t["entry_ts"])
    return out


def attach_micro(conn: sqlite3.Connection, trades: list[dict[str, Any]]) -> None:
    """One streaming pass; only rows within [entry-65s, entry+125s] are decoded."""
    windows: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, t in enumerate(trades):
        windows[t["symbol"]].append((t["entry_ts"], i))
    for v in windows.values():
        v.sort()
    series: dict[int, list[tuple[float, dict[str, Any]]]] = defaultdict(list)
    lo = min(t["entry_ts"] for t in trades) - 70
    hi = max(t["entry_ts"] for t in trades) + 130
    cur = conn.execute("SELECT symbol, ts_utc, features_json, ranking_delta FROM microstructure_feature_snapshots WHERE ts_utc BETWEEN ? AND ?", (lo, hi))
    for sym, ts, raw, rank_delta in cur:
        entries = windows.get(sym)
        if not entries:
            continue
        keys = [e[0] for e in entries]
        j = bisect.bisect_left(keys, ts - 125)
        hits = []
        while j < len(entries) and entries[j][0] <= ts + 65:
            hits.append(entries[j][1])
            j += 1
        if not hits:
            continue
        feat = json.loads(raw or "{}")
        slim = {k: feat.get(k) for k in (*MICRO_KEYS, "mid")}
        slim["ranking_delta"] = rank_delta
        for idx in hits:
            series[idx].append((float(ts), slim))
    for i, t in enumerate(trades):
        pts = sorted(series.get(i, []), key=lambda p: p[0])
        before = [p for p in pts if p[0] <= t["entry_ts"]]

        def _mid_at(offset: float, pts=pts, t=t) -> float | None:
            cand = [p for p in pts if p[0] <= t["entry_ts"] - offset]
            if not cand or t["entry_ts"] - offset - cand[-1][0] > 10:
                return None
            return float(cand[-1][1].get("mid") or 0) or None

        t["micro_cov"] = bool(before) and t["entry_ts"] - before[-1][0] <= 10
        if t["micro_cov"]:
            snap = before[-1][1]
            for k in (*MICRO_KEYS, "ranking_delta"):
                t[k] = None if snap.get(k) is None else float(snap[k])
            mid0 = float(snap.get("mid") or 0) or None
            for n in (15, 30, 60):
                past = _mid_at(n)
                t[f"mom_{n}s"] = (mid0 / past - 1.0) if mid0 and past else None
        after = [p for p in pts if t["entry_ts"] < p[0] <= t["entry_ts"] + 60]
        if after and t["entry_price"]:
            mids = [float(p[1].get("mid") or 0) for p in after if p[1].get("mid")]
            t["post_min_60s"] = min(mids) / t["entry_price"] - 1.0 if mids else None
            t["post_end_60s"] = mids[-1] / t["entry_price"] - 1.0 if mids else None


def attach_eval_logs(log_dir: Path, trades: list[dict[str, Any]]) -> None:
    by_opp = {t["opp"]: t for t in trades if t["opp"]}
    for path in sorted(log_dir.glob("mystic_portfolio.log*")):
        opener = gzip.open if path.suffix == ".gz" else open
        last_eval: dict[str, tuple] = {}
        with opener(path, "rt", errors="replace") as fh:
            for line in fh:
                if "SCALP_EVAL_TIMING" in line:
                    m = EVAL_RE.search(line)
                    if m:
                        last_eval[m.group(1)] = m.groups()
                elif "SCALP_V2_SIGNAL" in line:
                    m = SIGNAL_RE.search(line)
                    if not m or m.group(3) not in by_opp:
                        continue
                    ev = last_eval.get(m.group(1).replace("/", ""))
                    t = by_opp[m.group(3)]
                    t["planned_notional"] = float(m.group(5))
                    if ev and ev[4] == m.group(2):
                        t["eval_cov"] = True
                        t["strategy_passed"] = ev[1] == "True"
                        t["soft_reason"] = ev[2]
                        t["edge_source"] = ev[5]
                        t["exp_move"] = float(ev[6])
                        t["net_edge"] = float(ev[7])
                        t["mtf_conflict_logged"] = ev[2].startswith("MTF_")


def _klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[dict[str, float]]:
    out: list[dict[str, float]] = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}USDT&interval={interval}&startTime={cursor}&endTime={end_ms}&limit=1000"
        with urllib.request.urlopen(url, timeout=20) as resp:
            rows = json.loads(resp.read())
        if not rows:
            break
        for r in rows:
            out.append({"open_ms": r[0], "close_ms": r[6], "open": float(r[1]), "high": float(r[2]), "low": float(r[3]), "close": float(r[4]), "volume": float(r[5])})
        cursor = rows[-1][6] + 1
        if len(rows) < 1000:
            break
    return out


def venue_min_notional(symbols: set[str]) -> dict[str, float]:
    out = {}
    for sym in symbols:
        with urllib.request.urlopen(f"https://api.binance.us/api/v3/exchangeInfo?symbol={sym}USDT", timeout=20) as resp:
            info = json.loads(resp.read())
        for flt in info["symbols"][0]["filters"]:
            if flt["filterType"] in ("NOTIONAL", "MIN_NOTIONAL"):
                out[sym] = float(flt.get("minNotional") or 0)
    return out


def attach_klines(trades: list[dict[str, Any]]) -> None:
    from backend.services.binance_scalp.scalp_regime_classifier import classify_scalp_regime

    start = int((min(t["entry_ts"] for t in trades) - 10 * 86400) * 1000)
    end = int((max(t["entry_ts"] for t in trades) + 60) * 1000)
    for sym in {t["symbol"] for t in trades}:
        bars = {iv: _klines(sym, iv, start, end) for iv in ("5m", "15m", "1h")}
        closes = {iv: [b["close_ms"] for b in v] for iv, v in bars.items()}
        for t in (t for t in trades if t["symbol"] == sym):
            ms = t["entry_ts"] * 1000
            for iv in ("5m", "15m"):
                k = bisect.bisect_right(closes[iv], ms)
                closed = bars[iv][:k]
                if len(closed) >= 6:
                    rec = sum(b["close"] for b in closed[-3:]) / 3
                    pri = sum(b["close"] for b in closed[-6:-3]) / 3
                    t[f"mtf_{iv}_trend"] = rec / pri - 1.0
            k = bisect.bisect_right(closes["1h"], ms)
            st = classify_scalp_regime(bars["1h"][:k], k - 1) if k > 31 else None
            t["regime_1h"] = st.regime if st else "unknown"


# ------------------------------------------------------------ candidates


def _full_size(_trade: dict[str, Any]) -> float:
    return 1.0


def _median(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _worse_side(fit: list[dict[str, Any]], key: str, cut: float) -> str | None:
    lo = [t["net"] for t in fit if t.get(key) is not None and t[key] < cut]
    hi = [t["net"] for t in fit if t.get(key) is not None and t[key] >= cut]
    if len(lo) < 5 or len(hi) < 5:
        return None
    return "lo" if statistics.fmean(lo) < statistics.fmean(hi) else "hi"


def _threshold_rule(key: str, fixed_side: str | None = None, fixed_cut: float | None = None):
    def rule(fit: list[dict[str, Any]]):
        cut = fixed_cut if fixed_cut is not None else _median([t.get(key) for t in fit])
        if cut is None:
            return _full_size
        side = fixed_side or _worse_side(fit, key, cut)
        if side is None:
            return _full_size

        def size(t: dict[str, Any]) -> float:
            v = t.get(key)
            if v is None:
                return 1.0
            weak = v < cut if side == "lo" else v >= cut
            return MIN_SIZE if weak else 1.0

        return size

    return rule


def _mtf_rule(_fit: list[dict[str, Any]]):
    def size(t: dict[str, Any]) -> float:
        a5, a15 = t.get("mtf_5m_trend"), t.get("mtf_15m_trend")
        conflict = (a5 is not None and a5 <= 0) or (a15 is not None and a15 <= 0)
        return MIN_SIZE if conflict else 1.0

    return size


def _shrunk_rule(keyfn, k: float = 10.0):
    def rule(fit: list[dict[str, Any]]):
        if not fit:
            return _full_size
        global_mean = statistics.fmean(t["net"] for t in fit)
        groups: dict[Any, list[float]] = defaultdict(list)
        for t in fit:
            groups[keyfn(t)].append(t["net"])
        shrunk = {g: (sum(v) + k * global_mean) / (len(v) + k) for g, v in groups.items()}

        def size(t: dict[str, Any]) -> float:
            return MIN_SIZE if shrunk.get(keyfn(t), global_mean) < global_mean else 1.0

        return size

    return rule


CANDIDATES = {
    "C0_BASELINE": lambda _fit: _full_size,
    "C1_NET_EDGE_WEAK_HALF": _threshold_rule("net_edge", fixed_side="lo"),
    "C2_MOMENTUM_60S_WORSE_SIDE": _threshold_rule("mom_60s", fixed_cut=0.0),
    "C3_ADVERSE_SELECTION_HIGH_HALF": _threshold_rule("p_adverse_move", fixed_side="hi"),
    "C4_MTF_CONFLICT_HALF": _mtf_rule,
    "C5_SETUP_SYMBOL_SHRUNK": _shrunk_rule(lambda t: (t["setup"], t["symbol"])),
    "C6_SETUP_SHRUNK": _shrunk_rule(lambda t: t["setup"]),
    "C7_MICRO_RANK_WEAK_HALF": _threshold_rule("ranking_delta", fixed_side="lo"),
}

FEATURE_OF = {
    "C1_NET_EDGE_WEAK_HALF": "net_edge",
    "C2_MOMENTUM_60S_WORSE_SIDE": "mom_60s",
    "C3_ADVERSE_SELECTION_HIGH_HALF": "p_adverse_move",
    "C4_MTF_CONFLICT_HALF": "mtf_5m_trend",
    "C7_MICRO_RANK_WEAK_HALF": "ranking_delta",
}


def scaled_net(t: dict[str, Any], size: float, min_notional: float) -> float:
    """Realized venue economics scale linearly with quantity; below min notional the trade keeps full size."""
    if size >= 1.0 or t["notional"] * size < min_notional:
        return t["net"]
    return t["net"] * size


def run_candidates(trades: list[dict[str, Any]], min_notional: dict[str, float]) -> dict[str, Any]:
    folds = expanding_folds(len(trades))
    base_oos = [trades[i]["net"] for _, ev in folds for i in ev]
    out: dict[str, Any] = {}
    for name, factory in CANDIDATES.items():
        fold_rows, oos, sizes = [], [], []
        feature = FEATURE_OF.get(name)
        for n, (fit_idx, ev_idx) in enumerate(folds, 1):
            assert not set(fit_idx) & set(ev_idx)
            sizer = factory([trades[i] for i in fit_idx])
            vals = []
            for i in ev_idx:
                s = sizer(trades[i])
                sizes.append(s)
                vals.append(scaled_net(trades[i], s, min_notional.get(trades[i]["symbol"], 1.0)))
            oos.extend(vals)
            fold_rows.append({"fold": n, "fit": f"0-{fit_idx.stop - 1}", "eval": f"{ev_idx.start}-{ev_idx.stop - 1}", **summarize(vals)})
        ev_all = [trades[i] for _, ev in folds for i in ev]
        coverage = 1.0 if feature is None else sum(t.get(feature) is not None for t in ev_all) / len(ev_all)
        agg = summarize(oos)
        agg["feature_coverage"] = round(coverage, 3)
        agg["avg_size"] = round(statistics.fmean(sizes), 3)
        agg["positive_folds"] = sum(1 for f in fold_rows if f["net"] > 0)
        base_dd = max_drawdown(base_oos)
        checks = {
            "oos_net_positive": agg["net"] > 0,
            "oos_pf_gt_1": agg["pf"] > 1.0,
            "expectancy_positive": agg["exp"] > 0,
            "majority_folds_positive": agg["positive_folds"] > len(folds) / 2,
            "drawdown_ok": agg["dd"] >= base_dd * 1.25,
            "not_collapsed": agg["avg_size"] >= MIN_SIZE,
            "feature_coverage_ok": coverage >= 0.8,
        }
        out[name] = {"folds": fold_rows, "aggregate": agg, "checks": checks, "qualified": all(checks.values()) and name != "C0_BASELINE"}
    return out


# ------------------------------------------------------------- reporting


def cohort_table(trades: list[dict[str, Any]], keyfn) -> dict[str, Any]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for t in trades:
        groups[keyfn(t)].append(t)
    table = {}
    for g, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
        s = summarize([r["net"] for r in rows])
        n = len(rows)
        s.update(
            {
                "time_stop_pct": round(100 * sum(r["exit"] == "TIME_STOP_EXIT" for r in rows) / n, 1),
                "stop_loss_pct": round(100 * sum(r["exit"] == "STOP_LOSS_EXIT" for r in rows) / n, 1),
                "avg_mfe_bps": round(1e4 * statistics.fmean(r["mfe"] for r in rows), 1),
                "avg_mae_bps": round(1e4 * statistics.fmean(r["mae"] for r in rows), 1),
                "avg_net_edge_bps": _avg_bps(rows, "net_edge"),
                "avg_micro_rank": _avg(rows, "ranking_delta"),
                "cost_drag": round(sum(r["fees"] + r["slippage"] for r in rows), 4),
            }
        )
        table[str(g)] = s
    return table


def _avg(rows: list[dict[str, Any]], key: str) -> float | None:
    vals = [r[key] for r in rows if r.get(key) is not None]
    return round(statistics.fmean(vals), 6) if vals else None


def _avg_bps(rows: list[dict[str, Any]], key: str) -> float | None:
    v = _avg(rows, key)
    return None if v is None else round(1e4 * v, 1)


def feature_separation(trades: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("net_edge", "exp_move", "mom_15s", "mom_30s", "mom_60s", "mtf_5m_trend", "mtf_15m_trend", *MICRO_KEYS, "ranking_delta")
    out = {}
    for k in keys:
        rows = [t for t in trades if t.get(k) is not None]
        if len(rows) < 10:
            out[k] = {"n": len(rows)}
            continue
        wins = [t[k] for t in rows if t["net"] > 0]
        losses = [t[k] for t in rows if t["net"] <= 0]
        rows.sort(key=lambda t: t[k])
        third = len(rows) // 3
        out[k] = {
            "n": len(rows),
            "auc_win": None if auc(wins, losses) is None else round(auc(wins, losses), 3),
            "spearman_net": None if spearman([t[k] for t in rows], [t["net"] for t in rows]) is None else round(spearman([t[k] for t in rows], [t["net"] for t in rows]), 3),
            "bottom_third_net": round(sum(t["net"] for t in rows[:third]), 4),
            "top_third_net": round(sum(t["net"] for t in rows[-third:]), 4),
        }
    return out


def loss_decomposition(trades: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for exit_reason in ("TIME_STOP_EXIT", "STOP_LOSS_EXIT", "NET_PROFIT_EXIT"):
        rows = [t for t in trades if t["exit"] == exit_reason]
        if not rows:
            continue
        post = [t["post_end_60s"] for t in rows if t.get("post_end_60s") is not None]
        out[exit_reason] = {
            **summarize([t["net"] for t in rows]),
            "avg_hold_s": round(statistics.fmean(t["hold"] for t in rows)),
            "avg_mfe_bps": round(1e4 * statistics.fmean(t["mfe"] for t in rows), 1),
            "avg_mae_bps": round(1e4 * statistics.fmean(t["mae"] for t in rows), 1),
            "immediately_adverse_pct": round(100 * sum(v < 0 for v in post) / len(post), 1) if post else None,
            "mfe_below_25bps_pct": round(100 * sum(t["mfe"] < 0.0025 for t in rows) / len(rows), 1),
            "by_setup": {k: round(sum(t["net"] for t in rows if t["setup"] == k), 4) for k in sorted({t["setup"] for t in rows})},
            "by_symbol": {k: round(sum(t["net"] for t in rows if t["symbol"] == k), 4) for k in sorted({t["symbol"] for t in rows})},
            "mtf_conflict_n": sum(1 for t in rows if (t.get("mtf_5m_trend") or 0) <= 0 or (t.get("mtf_15m_trend") or 0) <= 0),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="mystic_trading.db")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--since", default=VALID_FROM)
    ap.add_argument("--out", default="/tmp/scalp_forensics.json")
    ap.add_argument("--no-klines", action="store_true")
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True, timeout=30)
    trades = load_trades(conn, args.since)
    attach_micro(conn, trades)
    attach_eval_logs(Path(args.logs), trades)
    if not args.no_klines:
        attach_klines(trades)
    for t in trades:
        hour = datetime.fromtimestamp(t["entry_ts"], tz=UTC).hour
        t["tod"] = f"{hour // 6 * 6:02d}-{hour // 6 * 6 + 6:02d}UTC"

    folds = expanding_folds(len(trades))
    fold_of = {i: n for n, (_, ev) in enumerate(folds, 1) for i in ev}
    report = {
        "population": {"since": args.since, **summarize([t["net"] for t in trades])},
        "coverage": {
            "micro": sum(bool(t.get("micro_cov")) for t in trades),
            "eval_log": sum(bool(t.get("eval_cov")) for t in trades),
            "mtf": sum(t.get("mtf_5m_trend") is not None for t in trades),
        },
        "strategy_passed": {str(k): summarize([t["net"] for t in trades if t.get("strategy_passed") is k]) for k in (True, False, None)},
        "by_exit": cohort_table(trades, lambda t: t["exit"]),
        "setup_x_symbol": cohort_table(trades, lambda t: f"{t['setup']}|{t['symbol']}"),
        "by_setup": cohort_table(trades, lambda t: t["setup"]),
        "by_symbol": cohort_table(trades, lambda t: t["symbol"]),
        "by_setup_fold": cohort_table(trades, lambda t: f"{t['setup']}|fold{fold_of.get(trades.index(t), 0)}"),
        "by_tod": cohort_table(trades, lambda t: t["tod"]),
        "by_regime": cohort_table(trades, lambda t: t.get("regime_1h", "unknown")),
        "by_mtf": cohort_table(trades, lambda t: "conflict" if (t.get("mtf_5m_trend") or 0) <= 0 or (t.get("mtf_15m_trend") or 0) <= 0 else "aligned"),
        "loss_decomposition": loss_decomposition(trades),
        "feature_separation": feature_separation(trades),
        "min_notional": (min_notional := ({} if args.no_klines else venue_min_notional({t["symbol"] for t in trades}))),
        "candidates": run_candidates(trades, min_notional),
        "trades": trades,
    }
    Path(args.out).write_text(json.dumps(report, default=str, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "trades"}, default=str, indent=1))


if __name__ == "__main__":
    main()
