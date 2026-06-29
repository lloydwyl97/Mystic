#!/usr/bin/env python3
"""Replay scalp gate comparison A/B/C over recent Binance.US klines (paper-only analysis)."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.calibration_profiles import (
    CALIBRATION_PROFILES,
    apply_profile,
)
from backend.services.binance_scalp.economics import ScalpEconomics

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HOURS = 4
MOVE_THRESHOLDS = (0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0040)
PROFILES = ("strict", "moderate", "fast")


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
        if not isinstance(rows, list) or not rows:
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


def spread_est(bar: dict) -> float:
    c = bar["close"]
    if c <= 0:
        return 1.0
    return (bar["high"] - bar["low"]) / c


def momentum_pass(bars: list[dict], idx: int) -> bool:
    if idx < 4:
        return False
    b0, b1, b2 = bars[idx], bars[idx - 1], bars[idx - 2]

    def chg(cur: float, old: float) -> float:
        return (cur - old) / old if old > 0 else 0.0

    mid15 = chg(b0["close"], b1["close"])
    mid30 = chg(b0["close"], b2["close"])
    up = sum(1 for j in range(idx - 5, idx) if j >= 1 and bars[j]["close"] > bars[j - 1]["close"])
    return mid15 >= 0.00003 and mid30 > 0 and up >= 3 and not (abs(mid15) <= 0.00002)


def gate_pass(
    econ: ScalpEconomics,
    *,
    spread_pct: float,
    projected_gross: float,
    mom_ok: bool,
) -> tuple[bool, str]:
    if spread_pct > econ.spread_cap_pct:
        return False, "SPREAD_TOO_WIDE"
    if not mom_ok:
        return False, "SCALP_NO_MOMENTUM_CONFIRMATION"
    req = econ.entry_required_gross_edge_pct(spread_pct, 0.0, 0.0)
    surplus = projected_gross - req
    if projected_gross < req:
        return False, "MOMENTUM_GROSS_BELOW_REQUIRED"
    if surplus < econ.min_projected_surplus_pct:
        return False, "PROJECTED_SURPLUS_TOO_SMALL"
    return True, "PASS"


def analyze_symbol_profile(symbol: str, bars: list[dict], econ: ScalpEconomics, profile: str) -> dict:
    if len(bars) < 10:
        return {"symbol": symbol, "profile": profile, "error": "insufficient_bars"}

    step = 5
    windows: list[dict] = []
    spreads: list[float] = []

    for i in range(0, len(bars) - step, step):
        entry = bars[i]["open"]
        if entry <= 0:
            continue
        chunk = bars[i : i + step]
        max_high = max(b["high"] for b in chunk)
        max_fav = (max_high - entry) / entry
        sp = statistics.mean(spread_est(b) for b in chunk)
        spreads.append(sp)
        rt = econ.roundtrip_cost_pct(sp, 0.0, 0.0)
        net_best = max_fav - rt
        profit_hit = net_best >= econ.net_profit_target_pct
        mom_ok = momentum_pass(bars, i)
        projected = max(max_fav * 0.55, econ.projected_entry_edge_pct(sp, 0.12))
        passed, reason = gate_pass(econ, spread_pct=sp, projected_gross=projected, mom_ok=mom_ok)

        windows.append(
            {
                "max_fav_pct": max_fav,
                "spread_pct": sp,
                "net_best_pct": net_best,
                "profit_hit": profit_hit,
                "gate_pass": passed,
                "gate_reason": reason,
            }
        )

    n = len(windows)
    move_counts = {f"+{int(t * 10000) / 100}%": sum(1 for w in windows if w["max_fav_pct"] >= t) for t in MOVE_THRESHOLDS}
    profitable = sum(1 for w in windows if w["profit_hit"])
    gate_pass_n = sum(1 for w in windows if w["gate_pass"])
    gate_blocked_profitable = sum(1 for w in windows if w["profit_hit"] and not w["gate_pass"])

    sim = _simulate_trades(bars, econ)
    return {
        "symbol": symbol,
        "profile": profile,
        "economics": CALIBRATION_PROFILES[profile],
        "five_min_windows": n,
        "move_threshold_counts": move_counts,
        "profitable_after_fees_windows": profitable,
        "gate_pass_windows": gate_pass_n,
        "gate_pass_pct": round(100 * gate_pass_n / n, 2) if n else 0,
        "gate_blocked_profitable_windows": gate_blocked_profitable,
        "avg_spread_pct": round(statistics.mean(spreads) * 100, 4) if spreads else 0,
        "median_spread_pct": round(statistics.median(spreads) * 100, 4) if spreads else 0,
        "spread_cap_pass_pct": round(100 * sum(1 for w in windows if w["spread_pct"] <= econ.spread_cap_pct) / n, 2) if n else 0,
        "top_gate_blocks": _top_blocks(windows),
        "simulated_trades": sim,
    }


def _simulate_trades(bars: list[dict], econ: ScalpEconomics) -> dict:
    """One paper trade per gate-pass 5m window; exit on target or window close."""
    step = 5
    wins = losses = 0
    pnl_pct_sum = 0.0
    trades = 0
    for i in range(0, len(bars) - step, step):
        chunk = bars[i : i + step]
        entry = chunk[0]["open"]
        if entry <= 0:
            continue
        sp = statistics.mean(spread_est(b) for b in chunk)
        mom_ok = momentum_pass(bars, i)
        max_high = max(b["high"] for b in chunk)
        max_fav = (max_high - entry) / entry
        projected = max(max_fav * 0.55, econ.projected_entry_edge_pct(sp, 0.12))
        passed, _ = gate_pass(econ, spread_pct=sp, projected_gross=projected, mom_ok=mom_ok)
        if not passed:
            continue
        rt = econ.roundtrip_cost_pct(sp, 0.0, 0.0)
        target = entry * (1 + econ.net_profit_target_pct + rt)
        exit_px = chunk[-1]["close"]
        for b in chunk:
            if b["high"] >= target:
                exit_px = target
                break
        gross = (exit_px - entry) / entry
        net = gross - rt
        trades += 1
        pnl_pct_sum += net
        if net >= econ.net_profit_target_pct:
            wins += 1
        else:
            losses += 1
    return {
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / trades, 1) if trades else 0,
        "total_net_pnl_pct": round(pnl_pct_sum * 100, 4),
        "avg_net_pnl_pct": round((pnl_pct_sum / trades) * 100, 4) if trades else 0,
    }


def _top_blocks(windows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for w in windows:
        if w["gate_pass"]:
            continue
        r = w["gate_reason"]
        counts[r] = counts.get(r, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1])[:5])


def main() -> int:
    base = ScalpEconomics.from_env()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    klines: dict[str, list[dict]] = {}
    for sym in SYMBOLS:
        klines[sym] = fetch_klines(sym, start_ms, end_ms)

    out: dict = {
        "window_hours": HOURS,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "symbols": list(SYMBOLS),
        "profiles": {},
        "symbol_ranking": [],
    }

    profile_totals: dict[str, dict] = {}
    for profile in PROFILES:
        econ = apply_profile(base, profile)
        sym_results = []
        for sym in SYMBOLS:
            sym_results.append(analyze_symbol_profile(sym, klines[sym], econ, profile))
        out["profiles"][profile] = sym_results
        total_pass = sum(r.get("gate_pass_windows", 0) for r in sym_results)
        total_prof = sum(r.get("profitable_after_fees_windows", 0) for r in sym_results)
        profile_totals[profile] = {
            "gate_pass_windows": total_pass,
            "profitable_windows": total_prof,
        }

    # Rank symbols by moderate profile gate passes + spread cleanliness
    moderate = {r["symbol"]: r for r in out["profiles"]["moderate"]}
    ranking = sorted(
        SYMBOLS,
        key=lambda s: (
            -moderate[s].get("gate_pass_windows", 0),
            -moderate[s].get("spread_cap_pass_pct", 0),
            moderate[s].get("avg_spread_pct", 999),
        ),
    )
    out["symbol_ranking"] = [
        {
            "symbol": s,
            "moderate_gate_pass": moderate[s].get("gate_pass_windows", 0),
            "moderate_spread_cap_pass_pct": moderate[s].get("spread_cap_pass_pct", 0),
        }
        for s in ranking
    ]
    out["profile_totals"] = profile_totals

    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
