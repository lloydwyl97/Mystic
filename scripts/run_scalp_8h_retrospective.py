#!/usr/bin/env python3
"""8-hour scalp retrospective — public klines + Phase 3d gate simulation."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402

HOURS = 8
THRESHOLDS = [0.0030, 0.0035, 0.0040, 0.0045]
SYMBOLS = ("BTCUSDT", "ETHUSDT")


def fetch_klines(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    bars: list[dict] = []
    cursor = start_ms
    while cursor < end_ms:
        url = (
            f"https://api.binance.us/api/v3/klines?symbol={symbol}&interval=1m"
            f"&startTime={cursor}&endTime={end_ms}&limit=1000"
        )
        proc = subprocess.run(
            ["curl", "-s", "--max-time", "30", url],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0 or not proc.stdout.strip():
            break
        rows = json.loads(proc.stdout)
        if isinstance(rows, dict):
            break
        if not rows:
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
        time.sleep(0.15)
    return bars


def spread_est(bar: dict) -> float:
    c = bar["close"]
    if c <= 0:
        return 1.0
    return (bar["high"] - bar["low"]) / c


def momentum_pass(bars: list[dict], idx: int) -> tuple[bool, dict]:
    if idx < 4:
        return False, {}
    b0, b1, b2 = bars[idx], bars[idx - 1], bars[idx - 2]

    def chg(cur: float, old: float) -> float:
        return (cur - old) / old if old > 0 else 0.0

    mid15 = chg(b0["close"], b1["close"])
    mid30 = chg(b0["close"], b2["close"])
    bid15, bid30 = mid15, mid30
    up = sum(
        1
        for j in range(idx - 5, idx)
        if j >= 1 and bars[j]["close"] > bars[j - 1]["close"]
    )
    ok = (
        bid15 >= 0.00003
        and mid15 >= 0.00003
        and bid30 > 0
        and mid30 > 0
        and up >= 3
        and not (abs(mid15) <= 0.00002 and abs(bid15) <= 0.00002)
    )
    return ok, {
        "mid_change_15s": mid15,
        "mid_change_30s": mid30,
        "bid_change_15s": bid15,
        "bid_change_30s": bid30,
        "up_tick_count": up,
    }


def analyze_symbol(symbol: str, bars: list[dict], econ: ScalpEconomics) -> dict:
    if len(bars) < 10:
        return {"symbol": symbol, "error": "insufficient_bars", "bars": len(bars)}

    favs: list[float] = []
    spreads: list[float] = []
    windows: list[dict] = []
    step = 5

    for i in range(0, len(bars) - step, step):
        entry = bars[i]["open"]
        if entry <= 0:
            continue
        chunk = bars[i : i + step]
        max_high = max(b["high"] for b in chunk)
        max_fav = (max_high - entry) / entry
        favs.append(max_fav)
        sp = statistics.mean(spread_est(b) for b in chunk)
        spreads.append(sp)
        req = econ.entry_required_gross_edge_pct(sp, 0.0, 0.0)
        rt = econ.roundtrip_cost_pct(sp, 0.0, 0.0)
        gross_best = max_fav
        net_best = gross_best - rt
        profit_hit = net_best >= econ.net_profit_target_pct
        mom_ok, mom = momentum_pass(bars, i)
        spread_ok = sp <= econ.spread_cap_pct
        imb = 0.10
        projected = econ.projected_entry_edge_pct(sp, imb)
        surplus = projected - req
        gate_reason = "PASS"
        if projected < req:
            gate_reason = "GROSS_EDGE_BELOW_REQUIRED"
        elif surplus < econ.min_projected_surplus_pct:
            gate_reason = "PROJECTED_SURPLUS_TOO_SMALL"
        elif not mom_ok:
            gate_reason = "SCALP_NO_MOMENTUM_CONFIRMATION"
        gate_block = gate_reason != "PASS"

        windows.append(
            {
                "ts": bars[i]["ts"].isoformat(),
                "entry": entry,
                "max_fav_pct": max_fav * 100,
                "max_high": max_high,
                "spread_pct": sp * 100,
                "required_gross_pct": req * 100,
                "net_best_pct": net_best * 100,
                "profit_hit": profit_hit,
                "spread_ok": spread_ok,
                "mom_ok": mom_ok,
                "gate_block": gate_block,
                "gate_reason": gate_reason,
                "projected_surplus_pct": surplus * 100,
                **mom,
            }
        )

    n = len(favs)

    def pctile(p: float) -> float:
        s = sorted(favs)
        if not s:
            return 0.0
        k = int(round((p / 100) * (len(s) - 1)))
        return s[k] * 100

    exceed = {f"{int(t * 10000) / 100}%": sum(1 for f in favs if f >= t) for t in THRESHOLDS}
    best = max(windows, key=lambda w: w["max_fav_pct"]) if windows else None
    missed_profit = [w for w in windows if w["profit_hit"] and w["gate_block"]]

    return {
        "symbol": symbol,
        "bars_1m": len(bars),
        "five_min_windows": n,
        "max_favorable_move_pct": {
            "average": statistics.mean(favs) * 100 if favs else 0,
            "median": statistics.median(favs) * 100 if favs else 0,
            "p75": pctile(75),
            "p90": pctile(90),
            "max": max(favs) * 100 if favs else 0,
        },
        "windows_exceeding_gross_threshold": exceed,
        "profit_target_reachable_windows": sum(1 for w in windows if w["profit_hit"]),
        "avg_spread_pct": statistics.mean(spreads) * 100 if spreads else 0,
        "spread_cap_pass_pct": round(
            100 * sum(1 for w in windows if w["spread_ok"]) / n, 1
        )
        if n
        else 0,
        "momentum_pass_pct": round(
            100 * sum(1 for w in windows if w["mom_ok"]) / n, 1
        )
        if n
        else 0,
        "phase3d_gate_block_pct": round(
            100 * sum(1 for w in windows if w["gate_block"]) / n, 1
        )
        if n
        else 0,
        "gate_blocked_would_have_won": len(missed_profit),
        "best_window": best,
        "best_missed_scalp": missed_profit[0] if missed_profit else best,
    }


def main() -> int:
    econ = ScalpEconomics.from_env()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    out: dict = {
        "window_hours": HOURS,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "economics": econ.as_dict(),
        "data_source": "binance_us_public_klines_1m",
        "symbols": {},
    }
    for sym in SYMBOLS:
        bars = fetch_klines(sym, start_ms, end_ms)
        out["symbols"][sym] = analyze_symbol(sym, bars, econ)

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
