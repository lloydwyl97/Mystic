#!/usr/bin/env python3
"""Authoritative production-conformant DAY replay (analysis only).

Loads 1m + inferences from a SQLite DB (Ocean extract or local), seeds 4H
from Binance.US completed klines, cash-clamps like production, and prints
conformance + four-arm + ranking JSON. Does not write to production.
"""

from __future__ import annotations

import json
import os
import resource
import sqlite3
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

os.environ.setdefault("DAY_PATH_AWARE_EXIT", "true")
os.environ.setdefault("DAY_STALL_EXIT_ENABLED", "false")
os.environ.setdefault("DAY_GIVEBACK_EXIT_ENABLED", "false")
os.environ.setdefault("ESTIMATED_ROUNDTRIP_COST", "0.0006")
os.environ.setdefault("DAY_TARGET_NOTIONAL_PER_SLOT_USD", "4000")
os.environ.setdefault("DAY_MAX_DEPLOYED_USD", "9000")

from backend.config.execution_cost_model import LEGACY_SELL_ROUNDTRIP_PCT
from backend.services.day_145_path_net_ranking import attach_entry_features, evaluate_ranking
from backend.services.day_asof_4h import FOURH_KEEP_MAX, FOURH_SEC, FourHAsOfTracker, merge_seed_and_1m_completed
from backend.services.day_production_lifecycle_replay import (
    SYMBOLS,
    ClosedTrade,
    LinearCalibrator,
    decision_bars,
    fold_bounds,
    load_1m_bars,
    load_inferences,
    make_calibrated_veto,
    parse_epoch,
    run_arm,
    summarize,
    veto_current,
    veto_honest_plus_quality_floor,
    veto_honest_pure_net,
)

WINDOW_START = datetime(2026, 8, 25, tzinfo=timezone.utc).timestamp()
WINDOW_END = datetime(2026, 9, 2, tzinfo=timezone.utc).timestamp()
STARTING_CASH = 239.02654324


def fetch_completed_4h(symbol: str, end_epoch: float, limit: int = 500) -> list[list[float]]:
    end_ms = int(end_epoch * 1000)
    url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=4h&endTime={end_ms}&limit={limit}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        raw = json.loads(resp.read().decode())
    out: list[list[float]] = []
    for row in raw:
        open_ms = int(row[0])
        close_ms = int(row[6])
        if close_ms > end_ms:
            continue
        out.append(
            [
                float(open_ms // 1000),
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
                float(row[5]),
            ]
        )
    return out


def hold_stats(trades: list[ClosedTrade]) -> dict:
    holds = sorted(max(0.0, t.exit_epoch - t.entry_epoch) for t in trades)
    n = len(holds)

    def pct(lim: float) -> float:
        return round(100.0 * sum(1 for h in holds if h <= lim) / n, 2) if n else 0.0

    def perc(p: float) -> float | None:
        if not n:
            return None
        return holds[min(n - 1, int(p * (n - 1)))]

    return {
        "n": n,
        "median_s": holds[n // 2] if n else None,
        "p10_s": perc(0.10),
        "p25_s": perc(0.25),
        "p75_s": perc(0.75),
        "p90_s": perc(0.90),
        "pct_1m": pct(60),
        "pct_5m": pct(300),
        "pct_15m": pct(900),
        "pct_30m": pct(1800),
        "trades_per_day": round(n / 7.0, 2),
    }


def _api_sym(symbol: str) -> str:
    s = str(symbol or "").replace("/", "").replace("-", "").upper()
    if s.endswith("USD") and not s.endswith("USDT"):
        s += "T"
    return s


def _path_excursions(
    bars: dict[str, list[tuple[int, float, ...]]],
    symbol: str,
    entry_epoch: float,
    exit_epoch: float,
    entry_price: float,
) -> tuple[float | None, float | None]:
    if entry_price <= 0:
        return None, None
    bset = bars.get(_api_sym(symbol)) or []
    hi = lo = None
    for bar in bset:
        ep = int(bar[0])
        if ep < entry_epoch - 1e-9:
            continue
        if ep > exit_epoch + 1e-9:
            break
        h, low = float(bar[2]), float(bar[3])
        hi = h if hi is None else max(hi, h)
        lo = low if lo is None else min(lo, low)
    mfe = ((hi - entry_price) / entry_price) if hi else None
    mae = ((lo - entry_price) / entry_price) if lo else None
    return mfe, mae


def live_window_stats(conn: sqlite3.Connection, bars: dict[str, list[tuple[int, float, ...]]] | None = None) -> dict:
    rows = conn.execute(
        """
        SELECT symbol, side, quantity, price, entry_price, pnl, hold_time_seconds, exit_reason,
               timestamp, trade_id, exit_type, fees_paid, entry_timestamp, order_id
        FROM paper_trades
        WHERE mode='live' AND COALESCE(strategy_id,'')='day'
          AND timestamp>='2026-08-25' AND timestamp<'2026-09-02'
        ORDER BY timestamp
        """
    ).fetchall()
    sells = [r for r in rows if r[1] == "SELL"]
    buys = [r for r in rows if r[1] == "BUY"]
    dust = [r for r in sells if str(r[10] or "") == "DUST_WRITEOFF" or str(r[9] or "").startswith("dust_")]
    clean = [r for r in sells if r not in dust]

    def _pack(label: str, subset: list) -> dict:
        holds = sorted(float(r[6] or 0) for r in subset)
        n = len(holds)
        pnls = [float(r[5] or 0) for r in subset]
        notionals = [float(r[2] or 0) * float(r[3] or 0) for r in subset]
        reasons: dict[str, int] = defaultdict(int)
        for r in subset:
            reasons[str(r[7] or r[10] or "NONE")] += 1
        syms: dict[str, int] = defaultdict(int)
        for r in subset:
            syms[_api_sym(str(r[0]))] += 1
        wins = sum(1 for p in pnls if p > 0)
        gp = sum(p for p in pnls if p > 0)
        gl = abs(sum(p for p in pnls if p <= 0))
        gross_bps = []
        net_bps = []
        comm_bps = []
        mfes = []
        maes = []
        for r in subset:
            ep = float(r[4] or 0)
            xp = float(r[3] or 0)
            qty = float(r[2] or 0)
            notional = qty * xp
            if ep > 0:
                gross_bps.append((xp - ep) / ep * 1e4)
            if notional > 0:
                net_bps.append(float(r[5] or 0) / notional * 1e4)
                comm_bps.append(float(r[11] or 0) / notional * 1e4)
            if bars is not None:
                ent = parse_epoch(r[12]) or parse_epoch(r[8])
                ex = parse_epoch(r[8])
                if ent and ex and ep > 0:
                    mfe, mae = _path_excursions(bars, str(r[0]), float(ent), float(ex), ep)
                    if mfe is not None:
                        mfes.append(mfe * 1e4)
                    if mae is not None:
                        maes.append(mae * 1e4)
        capture = None
        if mfes and gross_bps:
            mean_mfe = sum(mfes) / len(mfes)
            mean_g = sum(gross_bps) / len(gross_bps)
            capture = round(mean_g / mean_mfe, 4) if abs(mean_mfe) > 1e-9 else None
        return {
            "population": label,
            "entry_count": len(buys) if label != "dust" else 0,
            "exit_count": len(subset),
            "trades_per_day": round(len(subset) / 7.0, 2),
            "symbol_distribution": dict(syms),
            "median_hold_s": holds[n // 2] if n else None,
            "hold_p10_s": holds[int(0.10 * (n - 1))] if n > 1 else (holds[0] if n else None),
            "hold_p25_s": holds[int(0.25 * (n - 1))] if n > 1 else (holds[0] if n else None),
            "hold_p75_s": holds[int(0.75 * (n - 1))] if n > 1 else (holds[0] if n else None),
            "hold_p90_s": holds[int(0.90 * (n - 1))] if n > 1 else (holds[0] if n else None),
            "pct_1m": round(100.0 * sum(1 for h in holds if h <= 60) / n, 2) if n else 0,
            "pct_5m": round(100.0 * sum(1 for h in holds if h <= 300) / n, 2) if n else 0,
            "pct_15m": round(100.0 * sum(1 for h in holds if h <= 900) / n, 2) if n else 0,
            "pct_30m": round(100.0 * sum(1 for h in holds if h <= 1800) / n, 2) if n else 0,
            "exit_reasons": dict(reasons),
            "trailing_stop_count": reasons.get("TRAILING_STOP_EXIT", 0),
            "fourh_break_count": reasons.get("DAY_4H_STRUCTURE_BREAK_EXIT", 0),
            "risk_floor_count": reasons.get("DAY_RISK_FLOOR_EXIT", 0),
            "win_rate_pct": round(100.0 * wins / len(subset), 2) if subset else 0,
            "sum_pnl_usd": round(sum(pnls), 4),
            "avg_filled_notional": round(sum(notionals) / len(notionals), 2) if notionals else 0,
            "gross_bps": round(sum(gross_bps) / len(gross_bps), 3) if gross_bps else None,
            "commission_bps": round(sum(comm_bps) / len(comm_bps), 3) if comm_bps else None,
            "spread_bps": None,
            "slippage_bps": None,
            "net_bps": round(sum(net_bps) / len(net_bps), 3) if net_bps else None,
            "profit_factor": round(gp / gl, 4) if gl > 1e-12 else None,
            "mfe_bps": round(sum(mfes) / len(mfes), 3) if mfes else None,
            "mae_bps": round(sum(maes) / len(maes), 3) if maes else None,
            "realized_mfe_capture": capture,
            "binance_order_id_column": "paper_trades.order_id is NULL on all rows; mystic trade_id is local",
        }

    return {
        "all_live_day_including_dust": _pack("all", sells),
        "clean_engine_managed": _pack("clean", clean),
        "dust_writeoffs": _pack("dust", dust),
        "entry_count": len(buys),
        "exit_count": len(sells),
        "clean_exit_count": len(clean),
    }


def pnl_reconciliation(conn: sqlite3.Connection) -> dict:
    q = """
    SELECT mode, COALESCE(strategy_id,''), side, COUNT(*),
           ROUND(SUM(COALESCE(pnl,0)),4),
           ROUND(AVG(quantity*price),2),
           MIN(timestamp), MAX(timestamp)
    FROM paper_trades
    {where}
    GROUP BY 1,2,3
    """
    out = {}
    for label, where in (
        ("all_history", ""),
        ("post_cutoff_aug16", "WHERE timestamp>='2026-08-16T15:08:35'"),
        ("window_aug25_sep1", "WHERE timestamp>='2026-08-25' AND timestamp<'2026-09-02'"),
        ("last_7d_from_sep1", "WHERE timestamp>='2026-08-25' AND timestamp<'2026-09-02'"),
    ):
        out[label] = conn.execute(q.format(where=where)).fetchall()
    day_sells = conn.execute(
        """
        SELECT mode, COUNT(*), ROUND(SUM(COALESCE(pnl,0)),4), ROUND(AVG(quantity*price),2)
        FROM paper_trades
        WHERE side='SELL' AND COALESCE(strategy_id,'')='day'
        GROUP BY 1
        """
    ).fetchall()
    ffa = conn.execute(
        """
        SELECT mode, side, COUNT(*), ROUND(SUM(COALESCE(net_pnl_after_actual_fees,0)),4),
               ROUND(SUM(COALESCE(actual_fee_usd,0)),4)
        FROM portfolio_engine_fill_fee_audit GROUP BY 1,2
        """
    ).fetchall()
    scalp = None
    try:
        scalp = conn.execute(
            """
            SELECT side, COUNT(*), ROUND(SUM(COALESCE(pnl,0)),4)
            FROM scalp_paper_trades GROUP BY 1
            """
        ).fetchall()
    except sqlite3.OperationalError:
        scalp = "table_or_column_missing"
    return {
        "filters": {
            "day": "strategy_id='day'",
            "live": "mode='live'",
            "paper": "mode='paper'",
            "cutoff": "2026-08-16T15:08:35Z clean-paper-2f4f212",
            "window": "2026-08-25 .. 2026-09-01",
            "is_synthetic": "column exists but all NULL — not used",
        },
        "grouped": {k: [list(r) for r in v] for k, v in out.items()},
        "day_sells_by_mode": [list(r) for r in day_sells],
        "fill_fee_audit": [list(r) for r in ffa],
        "scalp": scalp,
        "plus_966_source": (
            "paper mode DAY sells 2026-08-16 to 2026-08-23: 116 sells, +$973.98, "
            "avg notional ~$2044. Live DAY sells: 84, -$8.22, avg notional ~$34. "
            "The +$966 figure mixed paper large-notional history with live."
        ),
        "pairing_note": ("paper_trades.order_id is NULL. Pairing uses mystic trade_id + symbol + timestamp. Authenticated Binance.US fills are a separate recon file if present."),
    }


def main() -> int:
    db = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "mystic_trading.db")
    t0 = time.time()
    conn = sqlite3.connect(db)
    bars = load_1m_bars(conn)
    inferences = [i for i in load_inferences(conn, with_features=True) if WINDOW_START <= i["epoch"] < WINDOW_END]
    events = decision_bars(inferences)
    fourh_seed: dict[str, list[list[float]]] = {}
    for sym in SYMBOLS:
        try:
            fetched = fetch_completed_4h(sym, WINDOW_START)
        except Exception as exc:
            fetched = []
            fourh_seed[sym] = []
            print(json.dumps({"warn_4h_fetch": str(exc), "symbol": sym}))
            continue
        fourh_seed[sym] = merge_seed_and_1m_completed(fetched, bars.get(sym, []))

    trackers: dict[str, FourHAsOfTracker] = {}
    for sym in SYMBOLS:
        tr = FourHAsOfTracker(bars_1m=bars.get(sym, []), keep=FOURH_KEEP_MAX)
        tr.seed_completed(fourh_seed.get(sym, []))
        if events:
            tr.advance(float(events[0]["bar_epoch"]))
        trackers[sym] = tr

    runtime_stats = {
        "db": db,
        "inference_rows": len(inferences),
        "decision_bars": len(events),
        "bars_1m": {s: len(bars[s]) for s in SYMBOLS},
        "seed_4h_completed": {s: len(fourh_seed.get(s, [])) for s in SYMBOLS},
        "align_ready_at_start": {s: bool((trackers[s].bundle(WINDOW_START).get("_asof") or {}).get("align_ready")) for s in SYMBOLS},
        "starting_cash": STARTING_CASH,
        "target_notional": 4000.0,
        "max_deployed": 9000.0,
        "max_cash_per_coin_pct": 0.25,
    }

    live = live_window_stats(conn, bars)
    recon = pnl_reconciliation(conn)

    def _fresh_trackers() -> dict[str, FourHAsOfTracker]:
        out: dict[str, FourHAsOfTracker] = {}
        for sym in SYMBOLS:
            tr = FourHAsOfTracker(bars_1m=bars.get(sym, []), keep=FOURH_KEEP_MAX)
            tr.seed_completed(fourh_seed.get(sym, []))
            out[sym] = tr
        return out

    closed_a, acc_a, rej_a = run_arm(
        name="A",
        events=events,
        bars=bars,
        fourh=fourh_seed,
        admit=veto_current,
        start_epoch=int(WINDOW_START),
        end_epoch=int(WINDOW_END),
        sell_cost=LEGACY_SELL_ROUNDTRIP_PCT,
        starting_cash=STARTING_CASH,
        trackers=_fresh_trackers(),
    )
    span = float(WINDOW_END - WINDOW_START)
    base = summarize(closed_a, accepted=acc_a, rejected=rej_a, span_sec=span)
    base["hold"] = hold_stats(closed_a)
    base["accepted"] = acc_a
    base["rejected"] = rej_a

    folds = fold_bounds(events, n_folds=3)
    split = folds[1][1] if len(folds) > 1 else int(WINDOW_END)
    train_c, _, _ = run_arm(
        name="cal",
        events=events,
        bars=bars,
        fourh=fourh_seed,
        admit=veto_honest_pure_net,
        start_epoch=int(WINDOW_START),
        end_epoch=split,
        starting_cash=STARTING_CASH,
        trackers=_fresh_trackers(),
    )
    cal = LinearCalibrator()
    cal.fit([(t.p_buy, t.net_pct) for t in train_c])
    arms = {
        "A_current_22bp_veto": veto_current,
        "B_honest_quality_floor": veto_honest_plus_quality_floor,
        "C_honest_pure_net": veto_honest_pure_net,
        "D_calibrated_path_net": make_calibrated_veto(cal),
    }
    arm_out = {}
    fold_out = {}
    for name, admit in arms.items():
        cl, ac, rj = run_arm(
            name=name,
            events=events,
            bars=bars,
            fourh=fourh_seed,
            admit=admit,
            start_epoch=int(WINDOW_START),
            end_epoch=int(WINDOW_END),
            starting_cash=STARTING_CASH,
            trackers=_fresh_trackers(),
        )
        row = summarize(cl, accepted=ac, rejected=rj, span_sec=span)
        row["hold"] = hold_stats(cl)
        arm_out[name] = row
        fr = []
        for i, (a, b) in enumerate(folds):
            fc, fa, frr = run_arm(
                name=f"{name}_f{i}",
                events=events,
                bars=bars,
                fourh=fourh_seed,
                admit=admit,
                start_epoch=a,
                end_epoch=b,
                starting_cash=STARTING_CASH,
                trackers=_fresh_trackers(),
            )
            fs = summarize(fc, accepted=fa, rejected=frr, span_sec=float(b - a))
            fs["fold"] = i
            fr.append(fs)
        fold_out[name] = fr

    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux ru_maxrss is KB
    report = {
        "runtime_sec": round(time.time() - t0, 2),
        "peak_rss_mb": round(rss / 1024.0, 1),
        "tracker_stats": {s: trackers[s].stats for s in SYMBOLS},
        "runtime": runtime_stats,
        "live_ocean_window": live,
        "replay_arm_a_baseline": base,
        "pnl_reconciliation": recon,
        "calibrator": {"a": cal.a, "b": cal.b, "fitted": cal.fitted, "train_n": len(train_c)},
        "arms": arm_out,
        "folds": fold_out,
        "ranking": evaluate_ranking(attach_entry_features(closed_a, inferences), inferences, folds),
        "conformance_note": (
            "Baseline uses production pre-buy, streaming as-of 4H with Binance.US completed "
            "pre-roll, cash=$239.03, MAX_CASH_PER_COIN_PCT=0.25. Replay will not match live "
            "fill-for-fill: live also uses confidence/thesis size, dust occupancy, and "
            "exchange-sync cash."
        ),
    }
    print(json.dumps(report, indent=2, default=str))
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
