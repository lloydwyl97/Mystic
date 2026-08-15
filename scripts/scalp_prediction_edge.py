#!/usr/bin/env python3
"""Measure whether higher SCALP/DAY scores produce better realized net return.

Also replay HOLD-as-action vs the previous rank-only pick on the same rows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.services.binance_scalp.scalp_candidate_ranking import (
    HOLD_ACTION_EV,
    attach_action_predictions,
    pick_best_global_candidate,
)
from backend.services.validation_cutoff import is_strategy_acceptance_eligible

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _corr(xs, ys):
    n = len(xs)
    if n < 10:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    if dx == 0 or dy == 0:
        return None
    return round(num / (dx * dy), 4)


def _bucket(xs, ys, edges):
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pair = [(x, y) for x, y in zip(xs, ys) if lo <= x < hi]
        if not pair:
            out.append({"lo": lo, "hi": hi, "n": 0})
            continue
        realized = [y for _, y in pair]
        wins = sum(1 for y in realized if y > 0)
        out.append(
            {
                "lo": lo,
                "hi": hi,
                "n": len(pair),
                "mean_pred": round(sum(x for x, _ in pair) / len(pair), 6),
                "mean_realized": round(sum(realized) / len(realized), 6),
                "wr": round(wins / len(realized), 4),
            }
        )
    return out


def _quartile(xs, ys):
    paired = sorted(zip(xs, ys), key=lambda t: t[0])
    n = len(paired)
    if n < 20:
        return None
    q = n // 4
    bot = paired[:q]
    top = paired[-q:]
    def _summ(rows):
        vals = [y for _, y in rows]
        return {
            "n": len(vals),
            "mean_net": round(sum(vals) / len(vals), 6),
            "wr": round(sum(1 for v in vals if v > 0) / len(vals), 4),
        }
    return {"bottom": _summ(bot), "top": _summ(top), "spread": round(_summ(top)["mean_net"] - _summ(bot)["mean_net"], 6)}


def _decile(xs, ys):
    paired = sorted(zip(xs, ys), key=lambda t: t[0])
    n = len(paired)
    if n < 20:
        return None
    k = max(1, n // 10)
    bot = paired[:k]
    top = paired[-k:]
    def _summ(rows):
        vals = [y for _, y in rows]
        return {"n": len(vals), "mean_net": round(sum(vals) / len(vals), 6), "wr": round(sum(1 for v in vals if v > 0) / len(vals), 4)}
    return {"bottom_10": _summ(bot), "top_10": _summ(top), "spread": round(_summ(top)["mean_net"] - _summ(bot)["mean_net"], 6)}


def _mae(pred, real):
    pair = [(p, r) for p, r in zip(pred, real) if p is not None and r is not None]
    if len(pair) < 5:
        return None
    return round(sum(abs(p - r) for p, r in pair) / len(pair), 6)


def opportunity_edge(conn: sqlite3.Connection) -> dict:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "scalp_opportunity_snapshots" not in tables:
        return {"available": False, "reason": "no_opportunity_table"}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(scalp_opportunity_snapshots)")}
    net_col = "plus_300s_net" if "plus_300s_net" in cols else None
    if net_col is None:
        return {"available": False, "reason": "no_forward_net_column"}
    rows = list(conn.execute(
        f"""
        SELECT id, symbol, rank_score, best_passed, best_setup, best_reject,
               spread_pct, measurements_json, {net_col} AS net,
               plus_300s_mfe AS mfe, plus_300s_mae AS mae
        FROM scalp_opportunity_snapshots
        WHERE {net_col} IS NOT NULL
        """
    ))
    if not rows:
        return {"available": True, "n": 0, "reason": "no_labeled_rows"}
    by_sym = defaultdict(list)
    xs, ys, passed, mfe_p, mfe_r, mae_p, mae_r = [], [], [], [], [], [], []
    feature_rows = defaultdict(list)
    for r in rows:
        rank = float(r["rank_score"] or 0)
        net = float(r["net"] or 0)
        xs.append(rank)
        ys.append(net)
        passed.append(int(r["best_passed"] or 0))
        by_sym[str(r["symbol"])].append((rank, net))
        if r["mfe"] is not None:
            mfe_r.append(float(r["mfe"]))
            mfe_p.append(rank)
        if r["mae"] is not None:
            mae_r.append(float(r["mae"]))
            mae_p.append(rank)
        try:
            meas = json.loads(r["measurements_json"] or "{}")
        except Exception:
            meas = {}
        best = meas.get(str(r["best_setup"] or "")) or {}
        for key in (
            "reclaim_strength",
            "breakout_strength",
            "reversal_strength",
            "momentum_flip_strength",
            "compression_score",
            "volume_impulse_strength",
            "orderbook_imbalance",
            "pullback_depth",
            "projected_move",
        ):
            if key in best:
                feature_rows[key].append((float(best[key] or 0), net))
    per_symbol = {}
    for sym in SYMBOLS:
        pair = by_sym.get(sym) or []
        if len(pair) < 8:
            per_symbol[sym] = {"n": len(pair), "corr": None}
            continue
        rx = [a for a, _ in pair]
        ry = [b for _, b in pair]
        per_symbol[sym] = {
            "n": len(pair),
            "corr_rank_vs_net": _corr(rx, ry),
            "mean_net": round(sum(ry) / len(ry), 6),
            "wr": round(sum(1 for v in ry if v > 0) / len(ry), 4),
            "quartiles": _quartile(rx, ry),
            "deciles": _decile(rx, ry),
        }
    feature_value = {}
    feature_no_value = []
    for key, pair in feature_rows.items():
        if len(pair) < 20:
            continue
        fx = [a for a, _ in pair]
        fy = [b for _, b in pair]
        c = _corr(fx, fy)
        q = _quartile(fx, fy)
        rec = {"n": len(pair), "corr": c, "quartiles": q}
        feature_value[key] = rec
        if c is None or abs(c) < 0.03 or (q and q["spread"] <= 0):
            feature_no_value.append(key)
    return {
        "available": True,
        "n": len(rows),
        "corr_rank_vs_5m_net": _corr(xs, ys),
        "corr_passed_vs_5m_net": _corr(passed, ys),
        "mean_net": round(sum(ys) / len(ys), 6),
        "wr": round(sum(1 for v in ys if v > 0) / len(ys), 4),
        "quartiles": _quartile(xs, ys),
        "deciles": _decile(xs, ys),
        "prob_buckets": _bucket(xs, ys, [-1, 0.25, 0.5, 0.75, 1.0, 2.0, 9.0]),
        "mfe_rank_corr": _corr(mfe_p, mfe_r),
        "mae_rank_corr": _corr(mae_p, mae_r),
        "per_symbol": per_symbol,
        "features": feature_value,
        "features_no_value": feature_no_value,
        "useful_edge": bool(_corr(xs, ys) and _corr(xs, ys) > 0.08 and (_quartile(xs, ys) or {}).get("spread", 0) > 0),
    }


def day_edge(conn: sqlite3.Connection) -> dict:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "paper_trades" not in tables:
        return {"available": False, "reason": "no_paper_trades"}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
    pnl_col = "pnl" if "pnl" in cols else "pnl_usd"
    sells = list(conn.execute(
        f"SELECT id, symbol, {pnl_col} AS pnl, exit_reason, explainability_json FROM paper_trades WHERE UPPER(side)='SELL'"
    ))
    profit, stall = [], []
    by_sym = defaultdict(list)
    for r in sells:
        if not is_strategy_acceptance_eligible(exit_reason=r["exit_reason"], trade_id=str(r["id"])):
            continue
        try:
            ex = json.loads(r["explainability_json"] or "{}")
        except Exception:
            ex = {}
        rec = {
            "pnl": float(r["pnl"] or 0),
            "symbol": str(r["symbol"] or ""),
            "setup": str(ex.get("setup_type_canonical") or ex.get("setup_type") or ""),
            "buy_prob": float(ex.get("buy_probability") or ex.get("model_probability") or 0),
            "expected_net": float(ex.get("expected_net_edge_pct") or ex.get("predicted_net_return") or 0),
            "rank": float(ex.get("rank_score") or ex.get("final_day_selection_score") or 0),
        }
        er = str(r["exit_reason"] or "").upper()
        if er == "NET_PROFIT_EXIT":
            profit.append(rec)
        elif er in {"STALL_EXIT", "GIVEBACK_EXIT", "PROGRESS_DECAY"}:
            stall.append(rec)
        by_sym[rec["symbol"]].append(rec)
    def _mean(rows, key):
        if not rows:
            return None
        return round(sum(r[key] for r in rows) / len(rows), 6)
    xs = [r["buy_prob"] for r in profit + stall]
    ys = [r["pnl"] for r in profit + stall]
    return {
        "available": True,
        "clean_n": len(profit) + len(stall),
        "profit_n": len(profit),
        "stall_n": len(stall),
        "profit_mean_pnl": _mean(profit, "pnl"),
        "stall_mean_pnl": _mean(stall, "pnl"),
        "profit_mean_buy_prob": _mean(profit, "buy_prob"),
        "stall_mean_buy_prob": _mean(stall, "buy_prob"),
        "profit_mean_expected_net": _mean(profit, "expected_net"),
        "stall_mean_expected_net": _mean(stall, "expected_net"),
        "corr_prob_vs_pnl": _corr(xs, ys) if xs else None,
        "useful_edge": False,
        "per_symbol": {
            sym: {
                "n": len(rows),
                "mean_pnl": _mean(rows, "pnl"),
                "wr": round(sum(1 for r in rows if r["pnl"] > 0) / len(rows), 4) if rows else None,
            }
            for sym, rows in by_sym.items()
        },
    }


def replay_hold_as_action(conn: sqlite3.Connection) -> dict:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "scalp_opportunity_snapshots" not in tables:
        return {"available": False}
    cols = {r[1] for r in conn.execute("PRAGMA table_info(scalp_opportunity_snapshots)")}
    net_col = "plus_300s_net" if "plus_300s_net" in cols else None
    rows = list(conn.execute(
        f"""
        SELECT created_at, symbol, rank_score, best_passed, spread_pct, impact_pct,
               {net_col if net_col else 'NULL'} AS net
        FROM scalp_opportunity_snapshots
        ORDER BY created_at, symbol
        """
    ))
    cycles = defaultdict(list)
    for r in rows:
        cycles[str(r["created_at"])].append(r)
    old_trades = []
    new_trades = []
    hold_wins = 0
    for _, group in cycles.items():
        ranked = []
        for r in group:
            ranked.append(
                {
                    "symbol": r["symbol"],
                    "rank_score": float(r["rank_score"] or 0),
                    "entry_eligible": True,
                    "signal": SimpleNamespace(
                        passed=bool(r["best_passed"]),
                        spread_pct=float(r["spread_pct"] or 0),
                        expected_move_pct=0.0,
                        impact_pct=float(r["impact_pct"] or 0),
                        confidence=0.0,
                    ),
                    "realized_net": None if r["net"] is None else float(r["net"]),
                }
            )
        if not ranked:
            continue
        old = max(ranked, key=lambda r: float(r["rank_score"] or 0))
        if old.get("realized_net") is not None:
            old_trades.append(old["realized_net"])
        new = pick_best_global_candidate(ranked)
        if new is None:
            hold_wins += 1
            new_trades.append(0.0)
        elif new.get("realized_net") is not None:
            new_trades.append(new["realized_net"])
    def _summ(vals):
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "wr": round(sum(1 for v in vals if v > 0) / len(vals), 4),
            "net": round(sum(vals), 6),
            "expectancy": round(sum(vals) / len(vals), 6),
            "pf": round(
                (sum(v for v in vals if v > 0) / abs(sum(v for v in vals if v < 0)))
                if any(v < 0 for v in vals)
                else None,
                4,
            )
            if any(v < 0 for v in vals)
            else None,
        }
    return {
        "available": True,
        "cycles": len(cycles),
        "hold_wins": hold_wins,
        "baseline_rank_only": _summ(old_trades),
        "after_hold_as_action": _summ(new_trades),
        "hold_action_ev": HOLD_ACTION_EV,
    }


def reconstruct_soft_rank_buys(conn: sqlite3.Connection) -> list[dict]:
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if "scalp_paper_positions" not in tables:
        return []
    out = []
    for r in conn.execute("SELECT trade_id, symbol, entry_price, entry_time, diagnostics_json FROM scalp_paper_positions ORDER BY id"):
        try:
            j = json.loads(r["diagnostics_json"] or "{}")
        except Exception:
            j = {}
        if not j.get("soft_rank_entry"):
            continue
        attach_action_predictions(
            {
                "symbol": r["symbol"],
                "rank_score": j.get("rank_score"),
                "signal": SimpleNamespace(
                    passed=bool(j.get("passed")),
                    spread_pct=float(j.get("spread_pct") or 0),
                    expected_move_pct=float(j.get("expected_move_pct") or 0),
                    impact_pct=float(j.get("impact_pct") or 0),
                    confidence=float(j.get("signal_confidence") or 0),
                ),
            }
        )
        row = {
            "trade_id": r["trade_id"],
            "symbol": r["symbol"],
            "entry_time": r["entry_time"],
            "entry_price": r["entry_price"],
            "passed": j.get("passed"),
            "soft_rank_entry": j.get("soft_rank_entry"),
            "rank_score": j.get("rank_score"),
            "expected_move_pct": j.get("expected_move_pct"),
            "spread_pct": j.get("spread_pct"),
            "setup": j.get("setup_name") or j.get("scalp_setup"),
        }
        ev_row = attach_action_predictions(
            {
                "symbol": r["symbol"],
                "rank_score": j.get("rank_score"),
                "signal": SimpleNamespace(
                    passed=bool(j.get("passed")),
                    spread_pct=float(j.get("spread_pct") or 0),
                    expected_move_pct=float(j.get("expected_move_pct") or 0),
                    impact_pct=float(j.get("impact_pct") or 0),
                    confidence=float(j.get("signal_confidence") or 0),
                ),
            }
        )
        row["reconstructed_expected_net_ev"] = ev_row["expected_net_ev"]
        row["hold_would_win"] = ev_row["expected_net_ev"] <= HOLD_ACTION_EV
        out.append(row)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", action="append", default=[])
    parser.add_argument("--out", default="/tmp/mystic_phase_report/prediction_edge.json")
    args = parser.parse_args()
    dbs = args.db or [
        "/home/mystic/mystic/mystic_scalp.db",
        "/home/mystic/mystic/mystic_trading.db",
    ]
    report = {"dbs": {}, "version": "scalp_hold_as_action_v1"}
    for path in dbs:
        if not Path(path).exists():
            report["dbs"][path] = {"available": False}
            continue
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        report["dbs"][path] = {
            "scalp_opportunity": opportunity_edge(conn),
            "day": day_edge(conn),
            "replay": replay_hold_as_action(conn),
            "soft_rank_buy_reconstruction": reconstruct_soft_rank_buys(conn),
        }
        conn.close()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": True, "out": str(out), "dbs": list(report["dbs"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
