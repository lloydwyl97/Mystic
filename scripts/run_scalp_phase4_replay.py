#!/usr/bin/env python3
"""Phase 4 replay: configs A/B/C/D + stale-timeout comparison (paper-only)."""

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
from backend.services.binance_scalp.paper_spread_caps import (
    DEFAULT_PAPER_SPREAD_CAPS,
    parse_paper_spread_caps_json,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
HOURS = 6
STALE_TIMEOUTS = (180, 240, 300)
NOTIONAL = 25.0


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
        time.sleep(0.1)
    return bars


def spread_est(bar: dict) -> float:
    c = bar["close"]
    if c <= 0:
        return 1.0
    return (bar["high"] - bar["low"]) / c


def bar_features(bars: list[dict], idx: int) -> dict:
    b0 = bars[idx]
    window = bars[max(0, idx - 5) : idx + 1]
    highs = [b["high"] for b in window]
    lows = [b["low"] for b in window]
    hi = max(highs)
    lo = min(lows)
    close = b0["close"]
    recent_range = (hi - lo) / close if close > 0 else 0.0

    def chg(i: int, j: int) -> float:
        if j < 0 or bars[j]["close"] <= 0:
            return 0.0
        return (bars[i]["close"] - bars[j]["close"]) / bars[j]["close"]

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
        "mid15": mid15,
        "up_ticks": up,
    }


def spread_cap(symbol: str, econ: ScalpEconomics, *, per_symbol: bool) -> float:
    if per_symbol and econ.paper_spread_caps:
        return econ.paper_spread_caps.get(symbol, econ.spread_cap_pct)
    return econ.spread_cap_pct


def entry_pass(
    symbol: str,
    feat: dict,
    econ: ScalpEconomics,
    *,
    per_symbol_cap: bool,
) -> tuple[bool, str]:
    cap = spread_cap(symbol, econ, per_symbol=per_symbol_cap)
    sp = feat["spread_pct"]
    if sp > cap:
        return False, "SPREAD_TOO_WIDE"
    if not feat["breakout_ok"]:
        return False, "BREAKOUT_NOT_CONFIRMED"
    if not feat["momentum_ok"]:
        return False, "SCALP_NO_MOMENTUM_CONFIRMATION"
    req = econ.entry_required_gross_edge_pct(sp, 0.0, 0.0)
    projected = feat["projected_gross"]
    surplus = projected - req
    if projected < req:
        return False, "MOMENTUM_GROSS_BELOW_REQUIRED"
    if surplus < econ.min_projected_surplus_pct:
        return False, "PROJECTED_SURPLUS_TOO_SMALL"
    return True, "PASS"


def rank_score(symbol: str, feat: dict, econ: ScalpEconomics, *, per_symbol_cap: bool) -> float:
    cap = spread_cap(symbol, econ, per_symbol=per_symbol_cap)
    sp = feat["spread_pct"]
    req = econ.entry_required_gross_edge_pct(sp, 0.0, 0.0)
    surplus = feat["projected_gross"] - req
    score = 0.0
    if sp <= cap:
        score += 2.0
    score += min(max(surplus, 0.0) * 8000.0, 3.0)
    if feat["momentum_ok"]:
        score += 2.0
    if feat["breakout_ok"]:
        score += 1.5
    score += min(feat["recent_range_pct"] * 400.0, 1.0)
    score -= sp * 800.0
    return score


def simulate_config(
    aligned: list[int],
    series: dict[str, list[dict]],
    idx_by_ts: dict[str, dict[int, int]],
    econ: ScalpEconomics,
    *,
    per_symbol_cap: bool,
    stale_sec: int,
    label: str,
) -> dict:
    econ = replace(econ, stale_scalp_timeout_sec=stale_sec)
    open_pos: dict | None = None
    trades: list[dict] = []
    missed_profitable = 0
    false_entries = 0

    for ts_ms in aligned:
        if open_pos:
            sym = open_pos["symbol"]
            idx = idx_by_ts[sym].get(ts_ms)
            if idx is None:
                continue
            bar = series[sym][idx]
            age_sec = (ts_ms - open_pos["entry_ts_ms"]) / 1000.0
            entry = open_pos["entry_price"]
            rt = econ.roundtrip_cost_pct(open_pos["entry_spread"], 0.0, 0.0)
            target_px = entry * (1.0 + econ.net_profit_target_pct + rt)
            profit_hit = bar["high"] >= target_px
            stale = age_sec >= stale_sec
            if profit_hit or stale:
                exit_px = target_px if profit_hit else bar["close"]
                gross = (exit_px - entry) / entry
                net = gross - rt
                pnl_usd = NOTIONAL * net
                reason = "NET_PROFIT_TARGET" if profit_hit else "STALE_SCALP_TIMEOUT"
                if not profit_hit and net < 0:
                    false_entries += 1
                trades.append(
                    {
                        "symbol": sym,
                        "entry_ts_ms": open_pos["entry_ts_ms"],
                        "exit_ts_ms": ts_ms,
                        "hold_sec": age_sec,
                        "entry_price": entry,
                        "exit_price": exit_px,
                        "net_pct": net,
                        "pnl_usd": pnl_usd,
                        "exit_reason": reason,
                        "win": net >= econ.net_profit_target_pct,
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
            ok, _ = entry_pass(sym, feat, econ, per_symbol_cap=per_symbol_cap)
            if not ok:
                continue
            candidates.append((rank_score(sym, feat, econ, per_symbol_cap=per_symbol_cap), sym, feat))

        if not candidates:
            # missed window: any symbol had +0.15% move in next 5 bars?
            for sym in SYMBOLS:
                idx = idx_by_ts[sym].get(ts_ms)
                if idx is None or idx + 5 >= len(series[sym]):
                    continue
                entry_px = series[sym][idx]["open"]
                if entry_px <= 0:
                    continue
                max_hi = max(series[sym][j]["high"] for j in range(idx, idx + 5))
                fav = (max_hi - entry_px) / entry_px
                rt = econ.roundtrip_cost_pct(spread_est(series[sym][idx]), 0.0, 0.0)
                if fav - rt >= econ.net_profit_target_pct:
                    missed_profitable += 1
                    break
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

    by_sym: dict[str, dict] = {}
    for sym in SYMBOLS:
        sym_trades = [t for t in trades if t["symbol"] == sym]
        by_sym[sym] = {
            "trades": len(sym_trades),
            "wins": sum(1 for t in sym_trades if t["win"]),
            "losses": sum(1 for t in sym_trades if not t["win"]),
            "net_pnl_usd": round(sum(t["pnl_usd"] for t in sym_trades), 4),
            "profit_exits": sum(1 for t in sym_trades if t["exit_reason"] == "NET_PROFIT_TARGET"),
            "stale_exits": sum(1 for t in sym_trades if t["exit_reason"] == "STALE_SCALP_TIMEOUT"),
            "avg_hold_sec": round(statistics.mean(t["hold_sec"] for t in sym_trades) if sym_trades else 0, 1),
        }

    profit_exits = sum(1 for t in trades if t["exit_reason"] == "NET_PROFIT_TARGET")
    stale_exits = sum(1 for t in trades if t["exit_reason"] == "STALE_SCALP_TIMEOUT")
    net_pnl = sum(t["pnl_usd"] for t in trades)
    wins = sum(1 for t in trades if t["win"])
    losses = len(trades) - wins

    best_sym = max(SYMBOLS, key=lambda s: by_sym[s]["net_pnl_usd"])
    worst_sym = min(SYMBOLS, key=lambda s: by_sym[s]["net_pnl_usd"])

    return {
        "label": label,
        "stale_timeout_sec": stale_sec,
        "per_symbol_spread_caps": per_symbol_cap,
        "economics": {
            "profile": label,
            "net_profit_target_pct": econ.net_profit_target_pct,
            "entry_edge_buffer_pct": econ.entry_edge_buffer_pct,
            "min_projected_surplus_pct": econ.min_projected_surplus_pct,
            "paper_spread_caps": econ.paper_spread_caps,
        },
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round(100 * wins / len(trades), 1) if trades else 0,
        "net_pnl_usd": round(net_pnl, 4),
        "profit_exits": profit_exits,
        "stale_exits": stale_exits,
        "false_entries": false_entries,
        "missed_profitable_windows": missed_profitable,
        "best_symbol": best_sym,
        "worst_symbol": worst_sym,
        "by_symbol": by_sym,
    }


def main() -> int:
    base = ScalpEconomics.from_env()
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=HOURS)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    series: dict[str, list[dict]] = {}
    idx_by_ts: dict[str, dict[int, int]] = {}
    for sym in SYMBOLS:
        series[sym] = fetch_klines(sym, start_ms, end_ms)
        idx_by_ts[sym] = {b["ts_ms"]: i for i, b in enumerate(series[sym])}

    all_ts = sorted({ts for sym in SYMBOLS for ts in idx_by_ts[sym]})
    aligned = [ts for ts in all_ts if ts >= start_ms + 360_000]

    strict = apply_profile(base, "strict")
    moderate = apply_profile(base, "moderate")
    moderate_caps = replace(
        moderate,
        paper_spread_caps=parse_paper_spread_caps_json(),
    )

    configs = {
        "A_strict_uniform": simulate_config(
            aligned,
            series,
            idx_by_ts,
            strict,
            per_symbol_cap=False,
            stale_sec=300,
            label="A_strict_uniform",
        ),
        "B_moderate_uniform": simulate_config(
            aligned,
            series,
            idx_by_ts,
            moderate,
            per_symbol_cap=False,
            stale_sec=300,
            label="B_moderate_uniform",
        ),
        "C_moderate_per_symbol_caps": simulate_config(
            aligned,
            series,
            idx_by_ts,
            moderate_caps,
            per_symbol_cap=True,
            stale_sec=300,
            label="C_moderate_per_symbol_caps",
        ),
    }

    stale_compare: dict[str, dict] = {}
    for stale in STALE_TIMEOUTS:
        stale_compare[f"{stale}s"] = simulate_config(
            aligned,
            series,
            idx_by_ts,
            moderate_caps,
            per_symbol_cap=True,
            stale_sec=stale,
            label=f"C_stale_{stale}s",
        )

    best_stale = min(
        STALE_TIMEOUTS,
        key=lambda s: -stale_compare[f"{s}s"]["net_pnl_usd"],
    )
    configs["D_moderate_caps_best_stale"] = simulate_config(
        aligned,
        series,
        idx_by_ts,
        moderate_caps,
        per_symbol_cap=True,
        stale_sec=best_stale,
        label="D_moderate_caps_best_stale",
    )

    out = {
        "window_hours": HOURS,
        "start_utc": start.isoformat(),
        "end_utc": end.isoformat(),
        "paper_spread_caps": dict(DEFAULT_PAPER_SPREAD_CAPS),
        "configs": configs,
        "stale_timeout_comparison": stale_compare,
        "best_stale_timeout_sec": best_stale,
        "recommendation": {
            "best_config": max(configs, key=lambda k: configs[k]["net_pnl_usd"]),
            "best_stale_sec": best_stale,
        },
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
