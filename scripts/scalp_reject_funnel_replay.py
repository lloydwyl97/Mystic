#!/usr/bin/env python3
"""Reproducible SCALP reject-funnel + rejected-forward replay. Exit 0 on success."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SCALP_PAPER_ENABLED", "true")

from backend.services.binance_scalp.config import ScalpConfig
from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.historical_forensic import _ohlcv_symbol, load_ohlcv
from backend.services.binance_scalp.scalp_setup_measurements import measure_all_setups
from backend.services.binance_scalp.strategies import ALL_STRATEGIES
from backend.services.binance_scalp.strategies.base import StrategyMarketContext
from backend.services.binance_scalp.strategy_module_replay import LOOKBACK, NOTIONAL, _mom_from_bars, _snapshot
from backend.services.validation_cutoff import is_strategy_acceptance_eligible

COST = 0.0006
HORIZONS = (1, 3, 5, 10, 20)
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")


def _summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    n = len(rows)
    wins = sum(1 for r in rows if r["net"] > 0)
    return {
        "n": n,
        "wr": round(wins / n, 4),
        "net_exp": round(sum(r["net"] for r in rows) / n, 6),
        "mfe": round(sum(r["mfe"] for r in rows) / n, 6),
        "mae": round(sum(r["mae"] for r in rows) / n, 6),
        "hit_target": round(sum(1 for r in rows if r["hit_target"]) / n, 4),
    }


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


def forward_stats(future, entry, minutes):
    window = future[:minutes]
    if not window or entry <= 0:
        return None
    mfe = max((float(b["high"]) - entry) / entry for b in window)
    mae = min((float(b["low"]) - entry) / entry for b in window)
    end = float(window[-1]["close"])
    gross = (end - entry) / entry
    return {"gross": gross, "net": gross - COST, "mfe": mfe, "mae": mae, "hit_target": mfe >= 0.0025}


def _default_db() -> Path:
    for candidate in (
        Path("/tmp/ocean_forensic.db"),
        ROOT / "mystic_trading.db",
        Path(os.environ.get("TRADING_DB_PATH") or ""),
    ):
        if candidate and candidate.exists():
            return candidate
    return ROOT / "mystic_trading.db"


def main() -> int:
    parser = argparse.ArgumentParser(description="SCALP reject-funnel replay")
    parser.add_argument("--db", default=str(_default_db()))
    parser.add_argument("--out", default="/tmp/mystic_phase_report/reject_funnel.json")
    parser.add_argument("--step", type=int, default=8)
    args = parser.parse_args()
    db = Path(args.db)
    if not db.exists():
        print(f"FAIL db_missing path={db}", file=sys.stderr)
        return 1
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE name='feature_ohlcv'").fetchone()
        if not has:
            print(f"FAIL feature_ohlcv_missing path={db}", file=sys.stderr)
            return 1
        n_bars = int(conn.execute("SELECT COUNT(*) FROM feature_ohlcv WHERE interval='1m'").fetchone()[0] or 0)
        if n_bars < 200:
            print(f"FAIL feature_ohlcv_too_small n={n_bars} path={db}", file=sys.stderr)
            return 1
        bars_by = load_ohlcv(conn)
    except Exception as exc:
        print(f"FAIL load_ohlcv err={exc}", file=sys.stderr)
        return 1

    config = ScalpConfig.from_env()
    econ = ScalpEconomics.from_env()
    funnels = {s.name: Counter() for s in ALL_STRATEGIES}
    evals = Counter()
    evals_by_symbol = {s.name: Counter() for s in ALL_STRATEGIES}
    rejected_fwd = {s.name: defaultdict(list) for s in ALL_STRATEGIES}
    passed_fwd = {s.name: defaultdict(list) for s in ALL_STRATEGIES}
    rank_vs = []

    for sym in SYMBOLS:
        raw = bars_by.get(_ohlcv_symbol(sym), [])
        for i in range(LOOKBACK, max(LOOKBACK, len(raw) - 21), max(1, args.step)):
            window = raw[i - LOOKBACK : i + 1]
            mid = float(window[-1]["close"] or 0) if window else 0.0
            if mid <= 0:
                continue
            snap = _snapshot(sym, mid, window)
            mom = _mom_from_bars(window)
            ctx = StrategyMarketContext(
                symbol=sym, snap=snap, mom=mom, bars_1m=window, econ=econ, config=config, notional_usd=NOTIONAL
            )
            meas_all = measure_all_setups(ctx)
            future = raw[i + 1 : i + 21]
            entry = float(snap.best_ask)
            fwd = {m: forward_stats(future, entry, m) for m in HORIZONS}
            st5 = fwd.get(5)
            for strat in ALL_STRATEGIES:
                evals[strat.name] += 1
                evals_by_symbol[strat.name][sym] += 1
                sig = strat.evaluate(ctx)
                meas = meas_all.get(strat.name) or {}
                reason = sig.reject_reason if not sig.passed else "PASSED"
                funnels[strat.name][reason] += 1
                bucket = passed_fwd if sig.passed else rejected_fwd
                for m, st in fwd.items():
                    if st:
                        bucket[strat.name][m].append(st)
                if st5 and meas:
                    rank_vs.append(
                        (
                            float(meas.get("reclaim_strength") or meas.get("momentum_flip_strength") or 0),
                            st5["net"],
                            sig.passed,
                            strat.name,
                            sym,
                        )
                    )

    scalp_report = {}
    for name in [s.name for s in ALL_STRATEGIES]:
        rej5 = rejected_fwd[name].get(5) or []
        profitable_rejected = sum(1 for s in rej5 if s["net"] > 0)
        scalp_report[name] = {
            "evaluations": evals[name],
            "evaluations_by_symbol": dict(evals_by_symbol[name]),
            "funnel": dict(funnels[name]),
            "passed": funnels[name].get("PASSED", 0),
            "rejected_forward": {str(k): _summarize(v) for k, v in rejected_fwd[name].items()},
            "passed_forward": {str(k): _summarize(v) for k, v in passed_fwd[name].items()},
            "rejected_5m_n": len(rej5),
            "rejected_5m_profitable_frac": round(profitable_rejected / len(rej5), 4) if rej5 else None,
            "rejected_5m_expectancy": round(sum(s["net"] for s in rej5) / len(rej5), 6) if rej5 else None,
        }

    xs = [a for a, b, *_ in rank_vs]
    ys = [b for a, b, *_ in rank_vs]
    paired = sorted(rank_vs, key=lambda t: t[0])
    n = len(paired)
    low = paired[: n // 3] if n >= 30 else []
    high = paired[-n // 3 :] if n >= 30 else []
    calib = {
        "n": n,
        "feature_vs_5m_net_corr": _corr(xs, ys),
        "low_tercile_exp": round(sum(t[1] for t in low) / len(low), 6) if low else None,
        "high_tercile_exp": round(sum(t[1] for t in high) / len(high), 6) if high else None,
    }

    conn.row_factory = sqlite3.Row
    try:
        sells = list(conn.execute("SELECT id, symbol, pnl, exit_reason, explainability_json FROM paper_trades WHERE UPPER(side)='SELL'"))
    except sqlite3.OperationalError:
        sells = []
    profit, stall = [], []
    for r in sells:
        if not is_strategy_acceptance_eligible(exit_reason=r["exit_reason"], trade_id=str(r["id"])):
            continue
        try:
            ex = json.loads(r["explainability_json"] or "{}")
        except Exception:
            ex = {}
        rec = {"pnl": float(r["pnl"] or 0), "setup": str(ex.get("setup_type_canonical") or ex.get("setup_type") or "")}
        er = str(r["exit_reason"] or "").upper()
        if er == "NET_PROFIT_EXIT":
            profit.append(rec)
        elif er in {"STALL_EXIT", "GIVEBACK_EXIT", "PROGRESS_DECAY"}:
            stall.append(rec)

    report = {
        "db": str(db),
        "step": args.step,
        "scalp_funnels": scalp_report,
        "scalp_calibration": calib,
        "day_stall_vs_profit": {"profit_n": len(profit), "stall_n": len(stall)},
        "command": f"{sys.executable} {Path(__file__).resolve()} --db {db} --out {args.out} --step {args.step}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"ok": True, "db": str(db), "out": str(out), "funnel_passed": {k: v["passed"] for k, v in scalp_report.items()}, "evals": {k: v["evaluations"] for k, v in scalp_report.items()}}, indent=2))
    for name, body in scalp_report.items():
        print(name, "evals", body["evaluations"], "passed", body["passed"], "rej5_exp", body["rejected_5m_expectancy"], "syms", body["evaluations_by_symbol"])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
