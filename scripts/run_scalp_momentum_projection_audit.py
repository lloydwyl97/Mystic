#!/usr/bin/env python3
"""Phase 3e — 8-hour momentum projection audit vs target-reachable windows."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics
from backend.services.binance_scalp.entry_gate import evaluate_buy_entry_gate
from backend.services.binance_scalp.momentum_gross_estimate import (
    compute_from_1m_bars,
)
from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics

HOURS = 8
SYMBOLS = ("BTCUSDT", "ETHUSDT")
FIXED_SPREAD = 0.0004


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        url = f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m&startTime={cursor}&endTime={end_ms}&limit=1000"
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "30", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        rows = json.loads(proc.stdout)
        if isinstance(rows, dict) or not rows:
            break
        for r in rows:
            bars.append(
                {
                    "ts": datetime.fromtimestamp(r[0] / 1000, tz=timezone.utc),
                    "open": float(r[1]),
                    "high": float(r[2]),
                    "low": float(r[3]),
                    "close": float(r[4]),
                }
            )
        last_ms = int(rows[-1][0])
        if last_ms <= cursor:
            break
        cursor = last_ms + 60_000
        time.sleep(0.12)
    return bars


def momentum_from_bars(bars: list[dict], idx: int) -> MomentumDiagnostics:
    est = compute_from_1m_bars(bars, idx)
    if idx < 6:
        return MomentumDiagnostics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, False, True)

    def chg(i0: int, i1: int) -> float:
        a, b = float(bars[i0]["close"]), float(bars[i1]["close"])
        return (b - a) / a if a > 0 else 0.0

    mid15 = chg(idx - 1, idx)
    mid30 = chg(idx - 2, idx) / 2.0
    mid60 = chg(idx - 4, idx) / 4.0
    up = sum(1 for j in range(idx - 5, idx) if float(bars[j + 1]["close"]) > float(bars[j]["close"]))
    flat = abs(mid15) <= 0.00002 and abs(mid15) <= 0.00002
    confirmed = est.data_sufficient and est.breakout_confirmed and mid30 > 0 and mid60 > 0 and mid15 >= 0.00003 and up >= 2 and not flat
    return MomentumDiagnostics(
        mid_change_15s=mid15,
        mid_change_30s=mid30,
        mid_change_60s=mid60,
        bid_change_15s=mid15,
        bid_change_30s=mid30,
        bid_change_60s=mid60,
        last_n_ticks_up_count=up,
        sample_count=6,
        history_sec=300.0,
        recent_range_pct=est.recent_range_pct,
        realized_volatility_pct=est.realized_volatility_pct,
        momentum_confirmed=confirmed,
        flat_regime=flat,
    )


def analyze_symbol(symbol: str, bars: list[dict], econ: ScalpEconomics) -> dict:
    step = 5
    windows: list[dict] = []
    for i in range(6, len(bars) - step, step):
        entry = float(bars[i]["open"])
        if entry <= 0:
            continue
        chunk = bars[i : i + step]
        max_high = max(float(b["high"]) for b in chunk)
        max_fav = (max_high - entry) / entry
        rt = econ.roundtrip_cost_pct(FIXED_SPREAD, 0.0, 0.0)
        target_reachable = (max_fav - rt) >= econ.net_profit_target_pct

        estimate = compute_from_1m_bars(bars, i, spread_pct=FIXED_SPREAD)
        mom = momentum_from_bars(bars, i)
        ok, reason, reach = evaluate_buy_entry_gate(
            econ,
            spread_pct=FIXED_SPREAD,
            buy_impact_pct=0.0,
            sell_impact_pct=0.0,
            estimate=estimate,
            momentum=mom,
            apply_entry_gate=True,
            selected_symbol=symbol,
        )

        windows.append(
            {
                "ts": bars[i]["ts"].isoformat(),
                "entry": entry,
                "max_fav_pct": max_fav * 100,
                "target_reachable": target_reachable,
                "projection_pass": ok,
                "reject_reason": reason or None,
                "projected_gross_pct": estimate.projected_gross_move_pct * 100,
                "required_gross_pct": reach.get("required_gross_move_pct", 0) * 100,
                "projected_surplus_pct": reach.get("projected_surplus_pct", 0) * 100,
            }
        )

    n = len(windows)
    reachable = [w for w in windows if w["target_reachable"]]
    passes = [w for w in windows if w["projection_pass"]]
    fn = [w for w in reachable if not w["projection_pass"]]
    fp = [w for w in passes if not w["target_reachable"]]
    tp = [w for w in passes if w["target_reachable"]]

    best_missed = max(fn, key=lambda w: w["max_fav_pct"]) if fn else None
    best_tp = max(tp, key=lambda w: w["max_fav_pct"]) if tp else None

    return {
        "symbol": symbol,
        "five_min_windows": n,
        "target_reachable_windows": len(reachable),
        "projection_pass_windows": len(passes),
        "true_positives": len(tp),
        "false_negatives": len(fn),
        "false_positives": len(fp),
        "recall_on_reachable_pct": round(100 * len(tp) / len(reachable), 1) if reachable else 0.0,
        "false_positive_rate_pct": round(100 * len(fp) / n, 1) if n else 0.0,
        "best_true_positive": best_tp,
        "best_false_negative": best_missed,
        "reject_reason_counts": _reason_counts([w for w in windows if not w["projection_pass"]]),
    }


def _reason_counts(fails: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for w in fails:
        r = w.get("reject_reason") or "UNKNOWN"
        out[r] = out.get(r, 0) + 1
    return dict(sorted(out.items(), key=lambda x: -x[1]))


def audit_passes(summary: dict) -> bool:
    tp = summary["total_true_positives"]
    fp = summary["total_false_positives"]
    reachable = summary["total_target_reachable"]
    if reachable == 0:
        return False
    if tp < 2:
        return False
    return not fp > max(tp * 3, 10)


def main() -> int:
    econ = ScalpEconomics.from_env()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    symbols_out = {}
    for sym in SYMBOLS:
        bars = fetch_klines(sym, start_ms, end_ms)
        symbols_out[sym] = analyze_symbol(sym, bars, econ)

    summary = {
        "total_target_reachable": sum(s["target_reachable_windows"] for s in symbols_out.values()),
        "total_projection_pass": sum(s["projection_pass_windows"] for s in symbols_out.values()),
        "total_true_positives": sum(s["true_positives"] for s in symbols_out.values()),
        "total_false_negatives": sum(s["false_negatives"] for s in symbols_out.values()),
        "total_false_positives": sum(s["false_positives"] for s in symbols_out.values()),
    }
    summary["audit_passes_for_paper_soak"] = audit_passes(summary)
    summary["projected_gross_formula"] = "0.30*trend_30s + 0.25*trend_60s + 0.20*trend_15s + 0.15*breakout + 0.10*realized_vol + imbalance_boost(if trend_30s>0)"

    out = {
        "window_hours": HOURS,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "economics": econ.as_dict(),
        "summary": summary,
        "symbols": symbols_out,
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
