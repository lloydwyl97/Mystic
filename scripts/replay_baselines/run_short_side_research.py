#!/usr/bin/env python3
"""
Research-only synthetic short replay harness (spot simulation).
NO live shorting, NO leverage, NO exchange execution.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.run_day_execution_replay import CACHE_DIR, SYMBOL_API, fetch_klines_cached

SCRIPT_NAME = "scripts/replay_baselines/run_short_side_research.py"
OUT_PATH = REPO / "scripts" / "replay_baselines" / "short_side_research_latest.json"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
SPAN_DAYS = int(os.getenv("SHORT_RESEARCH_SPAN_DAYS", "720"))

ONE_WAY_COST = float(os.getenv("LAB_TAKER_FEE", "0.0002")) + float(os.getenv("LAB_SLIPPAGE", "0.0002")) + float(os.getenv("LAB_HALF_SPREAD", "0.0002"))
ROUNDTRIP_COST = ONE_WAY_COST * 2.0
MAX_HOLD_HOURS = float(os.getenv("SHORT_RESEARCH_MAX_HOLD_H", "72"))

SHORT_STRATEGIES = [
    "BEAR_TREND_CONTINUATION",
    "FAILED_RECLAIM_SHORT",
    "BREAKDOWN_RETEST",
    "VWAP_REJECTION",
    "LOWER_HIGH_CONTINUATION",
    "VOL_EXPANSION_SHORT",
    "SCALP_SHORT_CONT",
]

REGIMES = [
    "bull_trend",
    "bear_trend",
    "range",
    "chop",
    "vol_expansion",
    "dump_reversal",
    "pump_continuation",
]


@dataclass
class Trade:
    symbol: str
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    pnl_pct_net: float
    hold_h: float
    regime: str
    strategy: str
    mae_pct: float = 0.0


def _ema(values, period: int) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not values:
        return []
    k = 2.0 / (period + 1)
    out = [float(values[0])]
    for v in values[1:]:
        out.append(float(v) * k + out[-1] * (1 - k))
    return out


def _rsi(closes, period: int = 14) -> float:
    if hasattr(closes, "tolist"):
        closes = closes.tolist()
    if len(closes) < period + 1:
        return 50.0
    gains = losses = 0.0
    for i in range(-period, 0):
        ch = closes[i] - closes[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    if losses == 0:
        return 100.0
    rs = (gains / period) / (losses / period)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 1:
        return 0.0
    trs = []
    for i in range(-period, 0):
        h, low, pc = bars[i]["high"], bars[i]["low"], bars[i - 1]["close"]
        trs.append(max(h - low, abs(h - pc), abs(low - pc)))
    return sum(trs) / len(trs)


def _adx(bars: list[dict], period: int = 14) -> float:
    if len(bars) < period + 2:
        return 0.0
    trs, pdm, mdm = [], [], []
    for i in range(-period - 1, 0):
        h, bar_low = bars[i]["high"], bars[i]["low"]
        ph, pl, pc = bars[i - 1]["high"], bars[i - 1]["low"], bars[i - 1]["close"]
        tr = max(h - bar_low, abs(h - pc), abs(bar_low - pc))
        up = h - ph
        dn = pl - bar_low
        trs.append(tr or 1e-9)
        pdm.append(up if (up > dn and up > 0) else 0.0)
        mdm.append(dn if (dn > up and dn > 0) else 0.0)
    atr = sum(trs)
    pdi = 100.0 * sum(pdm) / atr if atr else 0.0
    mdi = 100.0 * sum(mdm) / atr if atr else 0.0
    s = pdi + mdi
    return 100.0 * abs(pdi - mdi) / s if s else 0.0


def _to_internal(sym: str) -> str:
    s = sym.upper().replace("/", "")
    if s.endswith("USDT"):
        return f"{s[:-4]}/USDT"
    return sym


def _normalize_bars(raw: list[dict]) -> list[dict]:
    out = []
    for b in raw:
        ts = int(b.get("ts") or b.get("timestamp") or b.get("time") or 0)
        out.append(
            {
                "t": ts,
                "ts": ts,
                "o": float(b.get("open") or b.get("o") or 0),
                "h": float(b.get("high") or b.get("h") or 0),
                "l": float(b.get("low") or b.get("l") or 0),
                "c": float(b.get("close") or b.get("c") or 0),
                "high": float(b.get("high") or b.get("h") or 0),
                "low": float(b.get("low") or b.get("l") or 0),
                "close": float(b.get("close") or b.get("c") or 0),
            }
        )
    return out


def _cache_path(symbol: str, interval: str, start_ms: int, end_ms: int) -> Path:
    api = SYMBOL_API[_to_internal(symbol)]
    return CACHE_DIR / f"{api}_{interval}_{start_ms}_{end_ms}.json"


def _classify_regime(bars: list[dict]) -> str:
    if not bars:
        return "unknown"
    closes = np.array([b["c"] for b in bars[-200:] if b.get("c")])
    if len(closes) < 50:
        return "neutral"
    ema_fast = _ema(closes, 20)[-1]
    ema_slow = _ema(closes, 50)[-1]
    adx = _adx(bars[-200:], 14) if len(bars) >= 50 else 15.0
    vol = float(np.std(np.diff(np.log(closes[-50:])))) if len(closes) > 50 else 0.01

    if ema_fast > ema_slow and adx > 22:
        return "bull_trend"
    if ema_fast < ema_slow and adx > 22:
        return "bear_trend"
    if vol > 0.018:
        return "vol_expansion"
    if abs(ema_fast - ema_slow) / max(ema_slow, 1e-9) < 0.008:
        return "range"
    return "chop"


def _short_signal(bars: list[dict], regime: str) -> tuple[str | None, float]:
    if len(bars) < 60:
        return None, 0.0
    closes = np.array([b["c"] for b in bars])
    highs = np.array([b["h"] for b in bars])
    lows = np.array([b["l"] for b in bars])

    ema20 = _ema(closes, 20)[-1]
    ema50 = _ema(closes, 50)[-1]
    rsi = _rsi(closes, 14)
    atr = _atr(bars, 14)
    adx = _adx(bars, 14)

    c = closes[-1]
    h = highs[-1]
    bar_low = lows[-1]

    if regime in ("bear_trend", "dump_reversal") and c < ema20 < ema50 and adx > 25 and rsi < 55:
        return "BEAR_TREND_CONTINUATION", 0.55
    if regime in ("chop", "range", "vol_expansion") and c < ema20 and rsi < 48 and (highs[-5] - bar_low) > atr * 1.2:
        return "FAILED_RECLAIM_SHORT", 0.48
    if c < ema50 and (highs[-1] - c) < atr * 0.8 and adx > 20:
        return "BREAKDOWN_RETEST", 0.50
    if c < ema20 and (ema20 - c) > atr * 0.6 and rsi < 50:
        return "VWAP_REJECTION", 0.42
    if len(closes) > 30 and highs[-1] < highs[-8] and c < ema20:
        return "LOWER_HIGH_CONTINUATION", 0.40
    if adx > 32 and atr / max(c, 1e-9) > 0.014 and c < ema50:
        return "VOL_EXPANSION_SHORT", 0.52
    if regime in ("chop", "range") and rsi < 45 and (c - bar_low) / (h - bar_low + 1e-9) < 0.35:
        return "SCALP_SHORT_CONT", 0.35
    return None, 0.0


def _simulate_short(bars: list[dict], entry_idx: int, symbol: str, strategy: str, regime: str, *, roundtrip: float) -> Trade | None:
    entry = bars[entry_idx]["c"]
    entry_ts = bars[entry_idx]["t"]
    atr = _atr(bars[max(0, entry_idx - 30) : entry_idx + 1], 14) if entry_idx > 30 else entry * 0.01
    stop = entry + atr * 1.8
    target = entry - atr * 2.2
    max_hold_bars = int(MAX_HOLD_HOURS * 4)
    worst = 0.0

    for j in range(1, min(max_hold_bars, len(bars) - entry_idx)):
        b = bars[entry_idx + j]
        c = b["c"]
        ts = b["t"]
        adverse = (c - entry) / entry
        worst = max(worst, adverse)
        if c >= stop or c <= target or (ts - entry_ts) > MAX_HOLD_HOURS * 3600:
            pnl = (entry - c) / entry - roundtrip
            return Trade(
                symbol=symbol,
                entry_ts=entry_ts,
                exit_ts=ts,
                entry=entry,
                exit=c,
                pnl_pct_net=pnl,
                hold_h=(ts - entry_ts) / 3600.0,
                regime=regime,
                strategy=strategy,
                mae_pct=worst,
            )
    return None


def _metrics(trades: list[Trade], months: float) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "trades_per_month": 0.0,
            "monthly_pnl_on_25k": 0.0,
            "percent_per_month": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
            "expectancy_per_trade": 0.0,
            "max_drawdown": 0.0,
            "longest_hold_h": 0.0,
            "worst_mae": 0.0,
        }
    pnls = [t.pnl_pct_net for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    total_pct = sum(pnls)
    monthly_usd = (total_pct / max(months, 1.0)) * 25000.0 / 100.0
    cum = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum)
    dd = float(np.min(cum - running_max)) if len(cum) else 0.0
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "trades": len(trades),
        "trades_per_month": round(len(trades) / max(months, 1.0), 2),
        "monthly_pnl_on_25k": round(monthly_usd, 2),
        "percent_per_month": round((monthly_usd / 25000.0) * 100.0, 4),
        "win_rate": round(len(wins) / len(trades), 4),
        "avg_win": round(float(np.mean(wins)), 6) if wins else 0.0,
        "avg_loss": round(float(np.mean(losses)), 6) if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 4) if gross_loss > 0 else 99.0,
        "expectancy_per_trade": round(float(np.mean(pnls)), 6),
        "max_drawdown": round(dd, 6),
        "longest_hold_h": round(max(t.hold_h for t in trades), 2),
        "worst_mae": round(min(t.mae_pct for t in trades), 6),
    }


def _walk_forward(trades: list[Trade]) -> dict[str, Any]:
    if len(trades) < 20:
        return {"passed": False, "reason": "insufficient_trades", "train_trades": len(trades), "test_trades": 0}
    ordered = sorted(trades, key=lambda t: t.entry_ts)
    split = int(len(ordered) * 0.7)
    train = ordered[:split]
    test = ordered[split:]
    train_exp = float(np.mean([t.pnl_pct_net for t in train])) if train else 0.0
    test_exp = float(np.mean([t.pnl_pct_net for t in test])) if test else 0.0
    passed = test_exp > 0 and test_exp >= train_exp * 0.5
    return {
        "passed": passed,
        "train_trades": len(train),
        "test_trades": len(test),
        "train_expectancy": round(train_exp, 6),
        "test_expectancy": round(test_exp, 6),
    }


def _stress(trades: list[Trade], months: float) -> dict[str, Any]:
    stressed = []
    for t in trades:
        stressed.append(
            Trade(
                symbol=t.symbol,
                entry_ts=t.entry_ts,
                exit_ts=t.exit_ts,
                entry=t.entry,
                exit=t.exit,
                pnl_pct_net=t.pnl_pct_net - ONE_WAY_COST,
                hold_h=t.hold_h,
                regime=t.regime,
                strategy=t.strategy,
                mae_pct=t.mae_pct,
            )
        )
    m = _metrics(stressed, months)
    return {"passed": m["monthly_pnl_on_25k"] > 0, "monthly_pnl_on_25k": m["monthly_pnl_on_25k"], "profit_factor": m["profit_factor"]}


def _window(bars: list[dict], i: int, lookback: int = 200) -> list[dict]:
    start = max(0, i - lookback + 1)
    return bars[start : i + 1]


def run_symbol(symbol: str, span_days: int = SPAN_DAYS) -> dict[str, Any]:
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=span_days + 5)
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    internal = _to_internal(symbol)
    cache = _cache_path(symbol, "15m", start_ms, end_ms)

    raw = fetch_klines_cached(internal, "15m", start_ms, end_ms)
    bars = _normalize_bars(raw)
    if len(bars) < 200:
        return {
            "symbol": symbol,
            "error": "no_data",
            "bars": len(bars),
            "cache_path": str(cache),
            "cache_hit": cache.exists(),
        }

    trades: list[Trade] = []
    i = 150
    while i < len(bars) - 25:
        window = _window(bars, i, 200)
        regime = _classify_regime(window)
        sig, conf = _short_signal(_window(bars, i, 120), regime)
        if sig and conf >= 0.40:
            t = _simulate_short(bars, i, symbol, sig, regime, roundtrip=ROUNDTRIP_COST)
            if t:
                trades.append(t)
                i += max(8, int(t.hold_h * 3))
            else:
                i += 8
        else:
            i += 6

    span_days_actual = max(1.0, (bars[-1]["t"] - bars[0]["t"]) / 86400.0)
    months = max(span_days_actual / 30.4375, 1.0)
    overall = _metrics(trades, months)

    by_regime: dict[str, list[Trade]] = {r: [] for r in REGIMES}
    by_strategy: dict[str, list[Trade]] = {s: [] for s in SHORT_STRATEGIES}
    for t in trades:
        by_regime.setdefault(t.regime, []).append(t)
        by_strategy.setdefault(t.strategy, []).append(t)

    best_strat = None
    best_strat_monthly = -1e9
    for s, ts in by_strategy.items():
        if not ts:
            continue
        m = _metrics(ts, months)
        if m["monthly_pnl_on_25k"] > best_strat_monthly:
            best_strat_monthly = m["monthly_pnl_on_25k"]
            best_strat = s

    best_reg = None
    best_reg_monthly = -1e9
    for r, ts in by_regime.items():
        if not ts:
            continue
        m = _metrics(ts, months)
        if m["monthly_pnl_on_25k"] > best_reg_monthly:
            best_reg_monthly = m["monthly_pnl_on_25k"]
            best_reg = r

    return {
        "symbol": symbol,
        "internal_symbol": internal,
        "cache_path": str(cache),
        "cache_hit": cache.exists(),
        "date_range": {
            "start_ms": start_ms,
            "end_ms": end_ms,
            "start_iso": datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat(),
            "end_iso": datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat(),
        },
        "candle_count": len(bars),
        "span_days": round(span_days_actual, 2),
        "months": round(months, 2),
        "overall": overall,
        "walk_forward": _walk_forward(trades),
        "stress_2x_entry_cost": _stress(trades, months),
        "best_short_strategy": best_strat,
        "best_short_strategy_monthly_on_25k": round(best_strat_monthly, 2),
        "best_regime": best_reg,
        "best_regime_monthly_on_25k": round(best_reg_monthly, 2),
        "per_regime": {r: _metrics(ts, months) for r, ts in by_regime.items() if ts},
        "per_strategy": {s: _metrics(ts, months) for s, ts in by_strategy.items() if ts},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--span-days", type=int, default=SPAN_DAYS)
    parser.add_argument("--out", default=str(OUT_PATH))
    args = parser.parse_args()

    cmd = f"python3 {SCRIPT_NAME} --symbols {' '.join(args.symbols)} --span-days {args.span_days}"
    results: dict[str, Any] = {}
    all_trades_meta: list[dict] = []

    for sym in args.symbols:
        print(f"short research: {sym} ...", flush=True)
        r = run_symbol(sym, span_days=args.span_days)
        results[sym] = r
        if "overall" in r:
            all_trades_meta.append({"symbol": sym, **r["overall"]})

    # Portfolio aggregate (sum monthly across symbols — research view)
    total_monthly = sum(x.get("monthly_pnl_on_25k", 0) for x in all_trades_meta)
    best_sym = max(all_trades_meta, key=lambda x: x.get("monthly_pnl_on_25k", -1e9)) if all_trades_meta else None

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "script": SCRIPT_NAME,
        "command": cmd,
        "exit_code": 0,
        "partial_run": False,
        "run_type": "full_script",
        "symbols_tested": args.symbols,
        "span_days": args.span_days,
        "assumptions": {
            "one_way_cost": ONE_WAY_COST,
            "roundtrip_cost": ROUNDTRIP_COST,
            "max_hold_h": MAX_HOLD_HOURS,
            "no_live_short": True,
            "no_leverage": True,
        },
        "source_cache_dir": str(CACHE_DIR),
        "results": results,
        "aggregate": {
            "portfolio_monthly_on_25k_sum": round(total_monthly, 2),
            "best_symbol": best_sym["symbol"] if best_sym else None,
            "best_symbol_monthly_on_25k": best_sym.get("monthly_pnl_on_25k") if best_sym else None,
            "best_symbol_strategy": results[best_sym["symbol"]].get("best_short_strategy") if best_sym else None,
            "best_symbol_regime": results[best_sym["symbol"]].get("best_regime") if best_sym else None,
        },
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
