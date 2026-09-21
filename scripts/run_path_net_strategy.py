#!/usr/bin/env python3
"""Decision-book conformance + path-net ranking folds. Analysis only."""

from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.day_production_env import apply_env_map

apply_env_map(
    {
        "DAY_TARGET_NOTIONAL_PER_SLOT_USD": "4000",
        "DAY_MAX_DEPLOYED_USD": "9000",
        "DAY_MAX_OPEN_SLOTS": "4",
        "ESTIMATED_ROUNDTRIP_COST": "0.0006",
        "DAY_PATH_AWARE_EXIT": "true",
        "DAY_STALL_EXIT_ENABLED": "false",
        "DAY_GIVEBACK_EXIT_ENABLED": "false",
    },
    override=True,
)

from backend.config.execution_cost_model import LEGACY_SELL_ROUNDTRIP_PCT
from backend.services.day_4h_path_net import expected_path_net_bps
from backend.services.day_asof_4h import FOURH_KEEP_MAX, FourHAsOfTracker
from backend.services.day_decision_book_replay import candidates_from_executes, load_execute_decisions
from backend.services.day_entry_conformance import (
    load_actual_entries,
    load_actual_exits,
    match_entries,
    report_to_dict,
)
from backend.services.day_production_lifecycle_replay import (
    SYMBOLS,
    ReplayPos,
    _advance_position,
    _bar_index,
    decision_bars,
    executable_buy_price,
    load_1m_bars,
    load_inferences,
    parse_epoch,
    path_ev_asof,
    run_arm,
    summarize,
)
from backend.services.day_trade_thesis import late_4h_rise_signal
from backend.services.portfolio_engine import get_coin_profile

WINDOW_START = "2026-08-25"
WINDOW_END = "2026-09-02"
START_EPOCH = int(datetime(2026, 8, 25, tzinfo=timezone.utc).timestamp())
END_EPOCH = int(datetime(2026, 9, 2, tzinfo=timezone.utc).timestamp())
STARTING_CASH = 239.02654324
BRIEFING_N = 53
COST = 0.0006


def _api(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def nearest_inference(inferences: list[dict], symbol: str, epoch: int) -> dict | None:
    best = None
    for inf in inferences:
        if inf["symbol"] != symbol:
            continue
        if inf["epoch"] > epoch:
            break
        best = inf
    return best


def ghost_close(symbol: str, entry_epoch: int, bars: dict) -> dict:
    bset = bars.get(symbol) or []
    idx = _bar_index(bset, entry_epoch)
    if idx is None:
        return {"ok": False}
    mid = float(bset[idx][4])
    if mid <= 0:
        return {"ok": False}
    fill = executable_buy_price(mid, symbol)
    profile = get_coin_profile(symbol)
    pos = ReplayPos(
        symbol=symbol,
        entry_price=fill,
        entry_time=float(entry_epoch),
        quantity=1.0,
        notional=50.0,
        stop_price=fill * (1.0 - float(profile["sl"])),
        take_profit_1_price=fill * (1.0 + float(profile["tp"])),
        trail_pct=float(profile["trail"]),
        highest_price=fill,
        lowest_price=fill,
        max_hold_min=int(profile["max_hold_min"]),
    )
    trk = FourHAsOfTracker(bars_1m=bset, keep=FOURH_KEEP_MAX)
    closed = _advance_position(
        pos,
        bset,
        [],
        entry_epoch + 1,
        END_EPOCH,
        LEGACY_SELL_ROUNDTRIP_PCT,
        tracker=trk,
    )
    if closed is None:
        return {"ok": False}
    fav_first = closed.exit_reason in ("TRAILING_STOP_EXIT", "NET_PROFIT_EXIT") or (
        closed.mfe_pct > COST and closed.time_to_mfe_sec > 0 and closed.time_to_mfe_sec <= (closed.fourh_survival_sec or closed.hold_sec)
    )
    break_first = "STRUCTURE" in closed.exit_reason.upper() and not fav_first
    return {
        "ok": True,
        "net_bps": closed.net_pct * 1e4,
        "gross_bps": closed.gross_pct * 1e4,
        "mfe_bps": closed.mfe_pct * 1e4,
        "mae_bps": closed.mae_pct * 1e4,
        "exit_reason": closed.exit_reason,
        "hold_sec": closed.hold_sec,
        "fav_first": bool(fav_first),
        "break_first": bool(break_first),
    }


def fold_metrics(trades: list, span_sec: float) -> dict:
    s = summarize(trades, accepted=len(trades), rejected=0, span_sec=span_sec)
    n = len(trades)
    breaks = [t for t in trades if "STRUCTURE" in t.exit_reason.upper()]
    trails = [t for t in trades if t.exit_reason == "TRAILING_STOP_EXIT"]
    s["fourh_break_count"] = len(breaks)
    s["fourh_break_freq"] = round(len(breaks) / n, 4) if n else 0.0
    s["trailing_winner_count"] = len(trails)
    s["trailing_winner_freq"] = round(len(trails) / n, 4) if n else 0.0
    s["avg_4h_break_loss_bps"] = round(sum(t.net_pct for t in breaks) / len(breaks) * 1e4, 3) if breaks else 0.0
    s["avg_trailing_win_bps"] = round(sum(t.net_pct for t in trails) / len(trails) * 1e4, 3) if trails else 0.0
    s["net_pnl_usd"] = s.get("total_net_usd")
    return s


def main() -> int:
    db = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/ocean_day_replay_slim.db")
    conn = sqlite3.connect(str(db))
    actual_all = load_actual_entries(conn, start=WINDOW_START, end=WINDOW_END)
    actual_53 = actual_all[:BRIEFING_N]
    executes = load_execute_decisions(conn, start=WINDOW_START, end=WINDOW_END)
    exec_53 = executes[:BRIEFING_N]
    exits = load_actual_exits(conn, start=WINDOW_START, end=WINDOW_END)
    report_53 = match_entries(actual_53, candidates_from_executes(exec_53), actual_exits=exits, window_days=7.0)
    report_66 = match_entries(actual_all, candidates_from_executes(executes), actual_exits=exits, window_days=8.0)
    payload = {
        "conformance_53": report_to_dict(report_53),
        "conformance_66_stored": report_to_dict(report_66),
        "decision_book": {"execute": len(executes), "briefing_53": len(exec_53), "stored_rejects_not_admitted": 162},
    }

    bars = load_1m_bars(conn)
    inferences = load_inferences(conn, with_features=True)
    inferences.sort(key=lambda r: r["epoch"])

    labels = []
    for ex in executes:
        inf = nearest_inference(inferences, ex["symbol"], ex["fill_epoch"])
        ghost = ghost_close(ex["symbol"], ex["fill_epoch"], bars)
        if not ghost.get("ok"):
            continue
        feat = (inf or {}).get("features")
        if not feat:
            continue
        labels.append(
            {
                "epoch": ex["fill_epoch"],
                "symbol": ex["symbol"],
                "features": feat,
                "p_buy": float((inf or {}).get("p_buy") or 0),
                "ml_score": float(ex.get("ml_score") or 0),
                "path_ev": float(ex.get("path_ev") or 0),
                "net_bps": ghost["net_bps"],
                "fav": 1.0 if ghost["fav_first"] else 0.0,
                "brk": 1.0 if ghost["break_first"] else 0.0,
                "late_4h": 1.0 if late_4h_rise_signal(None, float(ex["fill_epoch"])) else 0.0,
            }
        )

    if len(labels) < 12:
        payload["path_net"] = {"error": "too_few_labels", "n": len(labels)}
        print(json.dumps(payload, indent=2, default=str))
        return 0

    times = sorted({int(r["epoch"]) for r in labels})
    folds = []
    n_folds = 4
    lo, hi = times[0], times[-1] + 1
    width = max(1, (hi - lo) // n_folds)
    for i in range(n_folds):
        a = lo + i * width
        b = hi if i == n_folds - 1 else lo + (i + 1) * width
        folds.append((a, b))

    x = np.asarray([[*r["features"][:145], r["p_buy"], r["ml_score"], r["path_ev"]] for r in labels], dtype=float)
    y_net = np.asarray([r["net_bps"] for r in labels], dtype=float)
    y_fav = np.asarray([r["fav"] for r in labels], dtype=float)
    y_brk = np.asarray([r["brk"] for r in labels], dtype=float)
    epochs = np.asarray([r["epoch"] for r in labels], dtype=int)

    fold_rows = []
    for i, (a, b) in enumerate(folds):
        train = epochs < a if i > 0 else epochs < b
        if i == 0:
            train = epochs < (a + width)
        valid = (epochs >= a) & (epochs < b)
        if int(train.sum()) < 8 or int(valid.sum()) < 3:
            fold_rows.append({"fold": i, "error": "too_few", "train": int(train.sum()), "valid": int(valid.sum())})
            continue
        fav_clf = LogisticRegression(max_iter=200).fit(x[train], y_fav[train])
        brk_clf = LogisticRegression(max_iter=200).fit(x[train], y_brk[train])
        ridge = Ridge(alpha=1.0).fit(x[train], y_net[train])
        fav_tr = y_fav[train] > 0.5
        brk_tr = y_brk[train] > 0.5
        e_fav = float(np.mean(y_net[train][fav_tr])) if fav_tr.any() else 8.0
        e_brk = float(abs(np.mean(y_net[train][brk_tr]))) if brk_tr.any() else 18.0
        p_fav = fav_clf.predict_proba(x[valid])[:, 1]
        p_brk = brk_clf.predict_proba(x[valid])[:, 1]
        path_net = np.array(
            [
                expected_path_net_bps(
                    probability_favorable=float(pf),
                    expected_favorable_net_bps=e_fav,
                    probability_4h_break_first=float(pb),
                    expected_break_loss_bps=e_brk,
                )
                for pf, pb in zip(p_fav, p_brk, strict=True)
            ]
        )
        direct = ridge.predict(x[valid])
        realized = y_net[valid]
        top_n = max(1, len(realized) // 3)
        pn_tb = None
        rd_tb = None
        if len(realized) >= 6:
            pn_order = np.argsort(path_net)
            rd_order = np.argsort(direct)
            pn_tb = round(float(np.mean(realized[pn_order[-top_n:]]) - np.mean(realized[pn_order[:top_n]])), 3)
            rd_tb = round(float(np.mean(realized[rd_order[-top_n:]]) - np.mean(realized[rd_order[:top_n]])), 3)
        fold_rows.append(
            {
                "fold": i,
                "start_utc": datetime.fromtimestamp(a, tz=timezone.utc).isoformat(),
                "end_utc": datetime.fromtimestamp(b, tz=timezone.utc).isoformat(),
                "n": int(valid.sum()),
                "path_net_mean_bps": round(float(np.mean(path_net)), 3),
                "direct_ridge_mean_bps": round(float(np.mean(direct)), 3),
                "realized_mean_bps": round(float(np.mean(realized)), 3),
                "fav_rate": round(float(np.mean(y_fav[valid])), 3),
                "break_rate": round(float(np.mean(y_brk[valid])), 3),
                "e_fav_train": round(e_fav, 3),
                "e_brk_train": round(e_brk, 3),
                "top_vs_bottom": {"path_net": pn_tb, "direct": rd_tb},
            }
        )

    # One-winner reconstructed policy arms on later window using last-fold model if available.
    events = decision_bars(inferences)
    fourh = {s: [] for s in SYMBOLS}
    trackers2 = {}
    for sym in SYMBOLS:
        trk = FourHAsOfTracker(bars_1m=bars.get(sym, []), keep=FOURH_KEEP_MAX)
        trackers2[sym] = trk

    def admit_path_ev(inf):
        ev = path_ev_asof(inf["symbol"], inf["epoch"], bars)
        inf["path_ev"] = ev
        return ev > 0.0, ev

    closed_a, _acc_a, _ = run_arm(
        name="A",
        events=events,
        bars=bars,
        fourh=fourh,
        admit=admit_path_ev,
        start_epoch=START_EPOCH,
        end_epoch=END_EPOCH,
        starting_cash=STARTING_CASH,
        trackers=trackers2,
        authority_mode="direct_four_coin_path_ev",
        enforce_cooldown=False,
    )
    payload["arms"] = {
        "A_current_path_ev": fold_metrics(closed_a, float(END_EPOCH - START_EPOCH)),
    }
    payload["folds"] = fold_rows
    payload["labels"] = {"n": len(labels), "fav_rate": round(float(np.mean(y_fav)), 3), "break_rate": round(float(np.mean(y_brk)), 3), "mean_net_bps": round(float(np.mean(y_net)), 3)}
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
