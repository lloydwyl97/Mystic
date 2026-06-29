#!/usr/bin/env python3
"""Phase 4b replay: XRP cap sweep + stale timeout + improved ranking."""

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

from backend.services.binance_scalp.calibration_profiles import apply_profile
from backend.services.binance_scalp.economics import ScalpEconomics

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HOURS = 6
XRP_CAPS = (0.0010, 0.0009, 0.0008, 0.0007)
STALE_TIMEOUTS = (180, 210, 240, 300)
BASE_CAPS = {"BTCUSDT": 0.0008, "ETHUSDT": 0.0006, "SOLUSDT": 0.0005}
NOTIONAL = 25.0
MIN_SURPLUS_CUSHION = 0.0005  # 0.05% paper quality floor


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
                    "ts_ms": int(r[0]),
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
        time.sleep(0.08)
    return bars


def spread_est(bar: dict) -> float:
    c = bar["close"]
    return (bar["high"] - bar["low"]) / c if c > 0 else 1.0


def bar_features(bars: list[dict], idx: int) -> dict:
    b0 = bars[idx]
    window = bars[max(0, idx - 5) : idx + 1]
    highs = [b["high"] for b in window]
    lows = [b["low"] for b in window]
    hi, lo, close = max(highs), min(lows), b0["close"]
    recent_range = (hi - lo) / close if close > 0 else 0.0

    def chg(i: int, j: int) -> float:
        return (bars[i]["close"] - bars[j]["close"]) / bars[j]["close"] if j >= 0 and bars[j]["close"] > 0 else 0.0

    mid15 = chg(idx, idx - 1) if idx >= 1 else 0.0
    mid30 = chg(idx, idx - 2) if idx >= 2 else 0.0
    up = sum(1 for k in range(max(1, idx - 5), idx + 1) if k >= 1 and bars[k]["close"] > bars[k - 1]["close"])
    pos_in_range = (close - lo) / (hi - lo) if hi > lo else 0.0
    breakout = pos_in_range >= 0.7 and recent_range >= 0.0008
    momentum = mid15 >= 0.00003 and mid30 > 0 and up >= 3 and abs(mid15) > 0.00002 and recent_range >= 0.0005
    sp = spread_est(b0)
    projected = max(recent_range * 0.45, mid15 * 2.5, sp * 0.5)
    return {
        "spread_pct": sp,
        "recent_range_pct": recent_range,
        "projected_gross": projected,
        "momentum_ok": momentum,
        "breakout_ok": breakout,
    }


def caps_with_xrp(xrp_cap: float) -> dict[str, float]:
    return {**BASE_CAPS, "XRPUSDT": xrp_cap}


def spread_cap(symbol: str, econ: ScalpEconomics) -> float:
    if econ.paper_spread_caps:
        return econ.paper_spread_caps.get(symbol, econ.spread_cap_pct)
    return econ.spread_cap_pct


def entry_pass(symbol: str, feat: dict, econ: ScalpEconomics) -> bool:
    cap = spread_cap(symbol, econ)
    if feat["spread_pct"] > cap:
        return False
    if not feat["breakout_ok"] or not feat["momentum_ok"]:
        return False
    req = econ.entry_required_gross_edge_pct(feat["spread_pct"], 0.0, 0.0)
    surplus = feat["projected_gross"] - req
    return not (feat["projected_gross"] < req or surplus < econ.min_projected_surplus_pct)


def rank_score_v2(symbol: str, feat: dict, econ: ScalpEconomics) -> float:
    cap = spread_cap(symbol, econ)
    sp = feat["spread_pct"]
    req = econ.entry_required_gross_edge_pct(sp, 0.0, 0.0)
    projected = feat["projected_gross"]
    surplus = projected - req
    cap_util = sp / cap if cap > 0 else 1.0
    reachability = projected / req if req > 0 else 0.0

    score = 0.0
    score += min(max(surplus, 0.0) * 12000.0, 5.0)
    score += min(reachability * 2.0, 2.0)
    score += min(feat["recent_range_pct"] * 500.0, 1.5)
    score += (1.0 - min(cap_util, 1.0)) * 2.0
    if feat["momentum_ok"]:
        score += 1.0
    if feat["breakout_ok"]:
        score += 1.0
    if surplus < MIN_SURPLUS_CUSHION:
        score -= 2.5
    score -= sp * 400.0
    return score


def simulate(
    aligned: list[int],
    series: dict[str, list[dict]],
    idx_by_ts: dict[str, dict[int, int]],
    econ: ScalpEconomics,
    stale_sec: int,
) -> dict:
    econ = replace(econ, stale_scalp_timeout_sec=stale_sec)
    open_pos = None
    trades: list[dict] = []

    for ts_ms in aligned:
        if open_pos:
            sym = open_pos["symbol"]
            idx = idx_by_ts[sym].get(ts_ms)
            if idx is None:
                continue
            bar = series[sym][idx]
            age = (ts_ms - open_pos["entry_ts_ms"]) / 1000.0
            entry = open_pos["entry_price"]
            rt = econ.roundtrip_cost_pct(open_pos["entry_spread"], 0.0, 0.0)
            target_px = entry * (1.0 + econ.net_profit_target_pct + rt)
            profit_hit = bar["high"] >= target_px
            stale = age >= stale_sec
            if profit_hit or stale:
                exit_px = target_px if profit_hit else bar["close"]
                net = (exit_px - entry) / entry - rt
                trades.append(
                    {
                        "symbol": sym,
                        "hold_sec": age,
                        "pnl_usd": NOTIONAL * net,
                        "exit_reason": "NET_PROFIT_TARGET" if profit_hit else "STALE_SCALP_TIMEOUT",
                        "win": net >= econ.net_profit_target_pct,
                        "entry_spread_pct": open_pos["entry_spread"],
                    }
                )
                open_pos = None
            continue

        candidates: list[tuple[float, str, dict]] = []
        for sym in SYMBOLS:
            idx = idx_by_ts[sym].get(ts_ms)
            if idx is None or idx < 6:
                continue
            feat = bar_features(series[sym], idx)
            if not entry_pass(sym, feat, econ):
                continue
            candidates.append((rank_score_v2(sym, feat, econ), sym, feat))
        if not candidates:
            continue
        candidates.sort(key=lambda x: -x[0])
        _, sym, feat = candidates[0]
        bar = series[sym][idx_by_ts[sym][ts_ms]]
        open_pos = {
            "symbol": sym,
            "entry_ts_ms": ts_ms,
            "entry_price": bar["open"],
            "entry_spread": feat["spread_pct"],
        }

    net = sum(t["pnl_usd"] for t in trades)
    return {
        "trades": len(trades),
        "wins": sum(1 for t in trades if t["win"]),
        "losses": sum(1 for t in trades if not t["win"]),
        "net_pnl_usd": round(net, 4),
        "profit_exits": sum(1 for t in trades if t["exit_reason"] == "NET_PROFIT_TARGET"),
        "stale_exits": sum(1 for t in trades if t["exit_reason"] == "STALE_SCALP_TIMEOUT"),
        "avg_hold_sec": round(statistics.mean(t["hold_sec"] for t in trades), 1) if trades else 0,
        "xrp_trades": [t for t in trades if t["symbol"] == "XRPUSDT"],
    }


def main() -> int:
    base = apply_profile(ScalpEconomics.from_env(), "moderate")
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    series, idx_by_ts = {}, {}
    for sym in SYMBOLS:
        series[sym] = fetch_klines(sym, start_ms, end_ms)
        idx_by_ts[sym] = {b["ts_ms"]: i for i, b in enumerate(series[sym])}
    aligned = sorted({ts for sym in SYMBOLS for ts in idx_by_ts[sym] if ts >= start_ms + 360_000})

    xrp_cap_compare: dict[str, dict] = {}
    for cap in XRP_CAPS:
        econ = replace(base, paper_spread_caps=caps_with_xrp(cap))
        r = simulate(aligned, series, idx_by_ts, econ, stale_sec=180)
        r["xrp_cap_pct"] = cap
        r["blocks_losing_xrp_0945"] = cap < 0.000945
        r["allows_winning_xrp_0603"] = cap >= 0.000603
        xrp_cap_compare[f"{cap * 100:.2f}%"] = r

    best_xrp_cap = max(
        XRP_CAPS,
        key=lambda c: xrp_cap_compare[f"{c * 100:.2f}%"]["net_pnl_usd"],
    )
    best_econ = replace(base, paper_spread_caps=caps_with_xrp(best_xrp_cap))

    stale_compare: dict[str, dict] = {}
    for stale in STALE_TIMEOUTS:
        stale_compare[f"{stale}s"] = simulate(aligned, series, idx_by_ts, best_econ, stale_sec=stale)
        stale_compare[f"{stale}s"]["stale_timeout_sec"] = stale

    best_stale = max(STALE_TIMEOUTS, key=lambda s: stale_compare[f"{s}s"]["net_pnl_usd"])
    final = simulate(aligned, series, idx_by_ts, best_econ, stale_sec=best_stale)

    out = {
        "window_hours": HOURS,
        "xrp_cap_comparison": xrp_cap_compare,
        "best_xrp_cap": best_xrp_cap,
        "stale_timeout_comparison": stale_compare,
        "best_stale_timeout_sec": best_stale,
        "recommended_config": {
            "profile": "moderate",
            "xrp_cap": best_xrp_cap,
            "stale_timeout_sec": best_stale,
            "caps": caps_with_xrp(best_xrp_cap),
        },
        "final_replay": final,
        "phase4_trade_cap_analysis": {
            "winning_xrp_spread_0.0603%": "passes all caps >= 0.07%",
            "losing_xrp_spread_0.0945%": "blocked at 0.09% and tighter",
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
