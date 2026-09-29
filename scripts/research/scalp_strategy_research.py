"""SCALP strategy-structure research on closed Binance.US 1m klines.

Read-only. Replays genuinely different long-only entry families, each paired
with a small set of coherent exit contracts, with executable costs:

  entry  = next bar open * (1 + half_spread + slip)       (taker at ask)
  target = trigger price * (1 - half_spread - slip)       (taker at bid)
  stop   = min(stop, bar open) * (1 - half_spread - stop_slip)
  fees   = 2.0 bps per side (measured on live SCALP fills), qty floored to venue step

Decisions use only bars closed at decision time; 5m/15m/1h bars are the last
fully completed clock bars. When a bar touches both stop and target the stop
fills first. One open position per symbol per family.

Folds: the sample is cut into 5 chronological blocks. Fold k fits the
(exit, threshold) choice on trades that closed before block k starts and
evaluates on trades entered inside block k.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
STEP = {"BTCUSDT": 1e-5, "ETHUSDT": 1e-4, "SOLUSDT": 1e-3, "XRPUSDT": 0.1}
MIN_NOTIONAL = 1.0
# Measured p75 spread (bps) from microstructure_feature_snapshots on Ocean.
SPREAD_BPS = {"BTCUSDT": 0.06, "ETHUSDT": 0.41, "SOLUSDT": 1.64, "XRPUSDT": 1.33}
FEE = 0.0002
SLIP = 0.0001
STOP_SLIP = 0.0002
NOTIONAL = 50.0
ATR_FLOOR = 0.0010
# The 15m scale keeps brackets several round-trip costs wide (~6.5 bps RT).
ATR15_FLOOR = 0.0025
N_BLOCKS = 5
MIN_FIT_TRADES = 30


# ── features ────────────────────────────────────────────────────────────────


def load_symbol(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop_duplicates("open_ms").set_index("open_ms").sort_index()
    full = np.arange(df.index[0], df.index[-1] + 60_000, 60_000)
    df = df.reindex(full)
    df["close"] = df["close"].ffill()
    for col in ("open", "high", "low"):
        df[col] = df[col].fillna(df["close"])
    for col in ("volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"):
        df[col] = df[col].fillna(0.0)
    df.index.name = "open_ms"
    return df


def _clock_bars(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    ms = minutes * 60_000
    key = (df.index.to_numpy() // ms) * ms
    g = df.groupby(key)
    bars = pd.DataFrame(
        {
            "open": g["open"].first(),
            "high": g["high"].max(),
            "low": g["low"].min(),
            "close": g["close"].last(),
            "n": g["close"].size(),
        }
    )
    return bars[bars["n"] == minutes]


def _atr_pct(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"], (bars["high"] - pc).abs(), (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean() / bars["close"]


def _completed(df: pd.DataFrame, bars: pd.DataFrame, minutes: int, cols: dict[str, pd.Series]) -> None:
    """Attach the value of the last clock bar fully closed at each 1m bar close."""
    ms = minutes * 60_000
    close_ms = df.index.to_numpy() + 60_000
    last_start = (close_ms // ms) * ms - ms
    for name, series in cols.items():
        df[name] = series.reindex(last_start).to_numpy()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    c, h, lo, v = df["close"], df["high"], df["low"], df["volume"]
    df["ret1"] = c.pct_change()
    df["ema20_1m"] = c.ewm(span=20, adjust=False).mean()
    df["hh20"] = h.shift(1).rolling(20).max()
    df["hh60"] = h.rolling(60).max()
    df["ll30"] = lo.shift(1).rolling(30).min()
    df["low10"] = lo.rolling(10).min()
    df["volx"] = v / v.shift(1).rolling(60).mean().replace(0, np.nan)
    df["taker"] = np.where(v > 0, df["taker_buy_base"] / v.replace(0, np.nan), 0.5)
    tb5 = df["taker_buy_base"].rolling(5).sum()
    v5 = v.rolling(5).sum()
    df["flow5"] = np.where(v5 > 0, tb5 / v5.replace(0, np.nan), 0.5)
    df["vwap60"] = df["quote_volume"].rolling(60).sum() / v.rolling(60).sum().replace(0, np.nan)
    df["vwap60"] = df["vwap60"].fillna(c)
    df["rv15"] = df["ret1"].rolling(15).std()
    df["rv240"] = df["ret1"].rolling(240).std()
    rng = (h - lo).replace(0, np.nan)
    df["close_pos"] = ((c - lo) / rng).fillna(0.5)

    b5 = _clock_bars(df, 5)
    b15 = _clock_bars(df, 15)
    b60 = _clock_bars(df, 60)
    _completed(df, b5, 5, {"atr5": _atr_pct(b5), "c5": b5["close"], "ema20_5m": b5["close"].ewm(span=20, adjust=False).mean()})
    _completed(
        df,
        b15,
        15,
        {
            "atr15": _atr_pct(b15),
            "c15": b15["close"],
            "ema20_15m": b15["close"].ewm(span=20, adjust=False).mean(),
            "ema50_15m": b15["close"].ewm(span=50, adjust=False).mean(),
        },
    )
    e60 = b60["close"].ewm(span=20, adjust=False).mean()
    _completed(df, b60, 60, {"ema20_1h_slope": e60.pct_change(3)})
    df["a"] = df["atr5"].clip(lower=ATR_FLOOR)
    df["a15"] = df["atr15"].clip(lower=ATR15_FLOOR)
    return df


# ── entry families ──────────────────────────────────────────────────────────

LEVELS = ("loose", "base", "strict")


def signals(df: pd.DataFrame, family: str, level: str) -> np.ndarray:
    i = LEVELS.index(level)
    c, o = df["close"], df["open"]
    trend15 = (df["ema20_15m"] > df["ema50_15m"]) & (df["c15"] > df["ema20_15m"])
    if family == "A_MOMENTUM_BREAKOUT":
        volx_min = (1.5, 2.0, 3.0)[i]
        sig = (c > df["hh20"]) & (df["volx"] >= volx_min) & (df["taker"] >= 0.6) & (df["close_pos"] >= 0.67) & (df["ret1"] > 0)
    elif family == "B_TREND_PULLBACK_RECLAIM":
        depth = (0.2, 0.35, 0.5)[i]
        pulled = df["low10"] <= df["ema20_1m"] * (1 - depth * df["a"])
        controlled = (df["hh60"] - df["low10"]) <= 2.0 * df["a"] * c
        reclaim = (c > df["ema20_1m"]) & (c > df["high"].shift(1)) & (df["taker"] >= 0.55)
        sig = trend15 & (df["c5"] > df["ema20_5m"]) & pulled & controlled & reclaim
    elif family in ("D_VOLATILITY_EXPANSION", "D2_VOL_EXPANSION_HTF_TREND"):
        ratio = (1.5, 2.0, 2.5)[i]
        expanded = (df["rv15"] / df["rv240"]) >= ratio
        directional = (c - c.shift(5)) >= 0.5 * df["a"] * c
        sig = expanded & directional & (c > df["vwap60"]) & (df["flow5"] >= 0.55)
        if family == "D2_VOL_EXPANSION_HTF_TREND":
            sig = sig & trend15 & (df["ema20_1h_slope"] > 0)
    elif family == "E_FAILED_BREAK_REVERSAL":
        flow_min = (0.5, 0.55, 0.6)[i]
        lvl = df["ll30"]
        swept = pd.Series(False, index=df.index)
        level_ref = pd.Series(np.nan, index=df.index)
        for k in (0, 1, 2):
            hit = df["low"].shift(k) < lvl.shift(k)
            depth_ok = (lvl.shift(k) - df["low"].shift(k)) <= 1.0 * df["a"].shift(k) * lvl.shift(k)
            m = hit & depth_ok & ~swept
            level_ref = level_ref.where(~m, lvl.shift(k))
            swept = swept | m
        sig = swept & (c > level_ref) & (c > o) & (df["taker"] >= flow_min)
    else:
        raise ValueError(family)
    ready = df["a"].notna() & df["a15"].notna() & df["ema50_15m"].notna() & df["rv240"].notna() & (df["volume"] > 0)
    return (sig & ready).fillna(False).to_numpy()


FAMILIES = ("A_MOMENTUM_BREAKOUT", "B_TREND_PULLBACK_RECLAIM", "D_VOLATILITY_EXPANSION", "D2_VOL_EXPANSION_HTF_TREND", "E_FAILED_BREAK_REVERSAL")


# ── exit contracts ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Contract:
    name: str
    stop_a: float | None
    target_a: float | None
    max_bars: int
    be_at_a: float | None = None
    trail_at_a: float | None = None
    trail_a: float | None = None
    scratch_bar: int | None = None
    scratch_mfe_a: float | None = None
    legacy: bool = False
    scale: str = "a"


CONTRACTS = (
    Contract("X1_ATR_BRACKET_1_1.5", stop_a=1.0, target_a=1.5, max_bars=60),
    Contract("X2_ATR_BE_TRAIL", stop_a=1.0, target_a=None, max_bars=120, be_at_a=1.0, trail_at_a=1.5, trail_a=0.75),
    Contract("X3_ATR_2R_SCRATCH", stop_a=1.0, target_a=2.0, max_bars=90, scratch_bar=15, scratch_mfe_a=0.25),
    Contract("W1_ATR15_BRACKET_1_2", stop_a=1.0, target_a=2.0, max_bars=240, scale="a15"),
    Contract("W2_ATR15_BE_TRAIL", stop_a=1.0, target_a=None, max_bars=480, be_at_a=1.0, trail_at_a=1.5, trail_a=0.75, scale="a15"),
    Contract("W3_ATR15_2R_SCRATCH", stop_a=1.0, target_a=2.0, max_bars=240, scratch_bar=30, scratch_mfe_a=0.3, scale="a15"),
)
LEGACY = Contract("X0_CURRENT_SCALP_V2", stop_a=None, target_a=None, max_bars=1440, legacy=True)


def _floor_qty(qty: float, step: float) -> float:
    return math.floor(qty / step + 1e-9) * step


def simulate(arr: dict[str, np.ndarray], i_sig: int, ct: Contract, sym: str) -> dict | None:
    """Enter at bar i_sig+1 open; return trade dict or None if not fillable."""
    n = len(arr["open"])
    ie = i_sig + 1
    if ie >= n:
        return None
    hs = SPREAD_BPS[sym] / 2e4
    entry = arr["open"][ie] * (1 + hs + SLIP)
    qty = _floor_qty(NOTIONAL / entry, STEP[sym])
    if qty * entry < MIN_NOTIONAL:
        return None
    a = float(arr[ct.scale][i_sig])
    cost_frac = 2 * FEE + 2 * (hs + SLIP)
    stop = entry * (1 - ct.stop_a * a) if ct.stop_a else entry * (1 - 0.015)
    target = entry * (1 + ct.target_a * a) if ct.target_a else None
    if ct.legacy:
        target = entry * (1 + 0.004 + cost_frac) / (1 - hs - SLIP)
    hwm = entry
    exit_px = None
    reason = ""
    j = ie
    last = min(n - 1, ie + ct.max_bars - 1)
    while j <= last:
        o, hi, lo, cl = arr["open"][j], arr["high"][j], arr["low"][j], arr["close"][j]
        if lo <= stop:
            exit_px = min(stop, o) * (1 - hs - STOP_SLIP)
            reason = "STOP" if exit_px < entry else "BE_OR_TRAIL"
            break
        if target is not None and hi >= target:
            exit_px = target * (1 - hs - SLIP)
            reason = "TARGET"
            break
        hwm = max(hwm, hi)
        mfe = hwm / entry - 1
        bars_held = j - ie + 1
        if ct.legacy:
            net_now = cl * (1 - hs - SLIP) / entry - 1 - 2 * FEE
            if bars_held >= 120 and net_now <= 0:
                exit_px = cl * (1 - hs - SLIP)
                reason = "TIME_STOP"
                break
        if ct.scratch_bar and bars_held == ct.scratch_bar and mfe < (ct.scratch_mfe_a or 0) * a:
            exit_px = cl * (1 - hs - SLIP)
            reason = "SCRATCH"
            break
        if ct.be_at_a and mfe >= ct.be_at_a * a:
            stop = max(stop, entry * (1 + cost_frac))
        if ct.trail_at_a and mfe >= ct.trail_at_a * a:
            stop = max(stop, hwm * (1 - (ct.trail_a or 0) * a))
        j += 1
    if exit_px is None:
        j = last
        exit_px = arr["close"][j] * (1 - hs - SLIP)
        reason = "MAX_HOLD"
    gross = qty * (exit_px - entry)
    fees = FEE * qty * (entry + exit_px)
    return {
        "symbol": sym,
        "i_sig": i_sig,
        "entry_ms": int(arr["t"][ie]),
        "exit_ms": int(arr["t"][j]) + 60_000,
        "exit_i": j,
        "net": gross - fees,
        "reason": reason,
        "a_bps": a * 1e4,
        "mfe_bps": (hwm / entry - 1) * 1e4,
    }


def run_config(data: dict[str, dict], family: str, level: str, ct: Contract) -> list[dict]:
    trades: list[dict] = []
    for sym, d in data.items():
        sig_idx = np.flatnonzero(d["sig"][(family, level)])
        free_at = -1
        for i in sig_idx:
            if i <= free_at:
                continue
            tr = simulate(d["arr"], int(i), ct, sym)
            if tr is None:
                continue
            trades.append(tr)
            free_at = tr["exit_i"]
    trades.sort(key=lambda t: t["exit_ms"])
    return trades


# ── metrics ─────────────────────────────────────────────────────────────────


def metrics(trades: list[dict]) -> dict:
    n = len(trades)
    if n == 0:
        return {"n": 0, "net": 0.0}
    nets = np.array([t["net"] for t in sorted(trades, key=lambda t: t["exit_ms"])])
    wins, losses = nets[nets > 0], nets[nets <= 0]
    gw, gl = wins.sum(), -losses.sum()
    eq = np.cumsum(nets)
    dd = float((eq - np.maximum.accumulate(np.concatenate([[0.0], eq]))[1:]).min())
    avg_w = float(wins.mean()) if len(wins) else 0.0
    avg_l = float(losses.mean()) if len(losses) else 0.0
    payoff = avg_w / -avg_l if avg_l < 0 else None
    return {
        "n": n,
        "wins": len(wins),
        "losses": len(losses),
        "net": round(float(nets.sum()), 4),
        "pf": round(gw / gl, 3) if gl > 0 else None,
        "expectancy": round(float(nets.mean()), 5),
        "win_rate": round(len(wins) / n, 3),
        "avg_win": round(avg_w, 4),
        "avg_loss": round(avg_l, 4),
        "payoff": round(payoff, 3) if payoff else None,
        "breakeven_wr": round(1 / (1 + payoff), 3) if payoff else None,
        "largest_loss": round(float(nets.min()), 4),
        "p95_loss": round(float(np.percentile(losses, 5)), 4) if len(losses) else 0.0,
        "wins_to_recover_avg_loss": round(-avg_l / avg_w, 2) if avg_w > 0 and avg_l < 0 else None,
        "max_dd": round(dd, 4),
    }


def by_key(trades: list[dict], key: str) -> dict:
    out: dict[str, list] = {}
    for t in trades:
        out.setdefault(str(t[key]), []).append(t)
    return {k: metrics(v) for k, v in sorted(out.items())}


def block_edges(t0: int, t1: int) -> list[int]:
    return [int(t0 + (t1 - t0) * k / N_BLOCKS) for k in range(N_BLOCKS + 1)]


def in_block(trades: list[dict], lo: int, hi: int) -> list[dict]:
    return [t for t in trades if lo <= t["entry_ms"] < hi]


def fit_window(trades: list[dict], lo: int, cut: int) -> list[dict]:
    """Trades usable for fitting a fold whose validation block starts at ``cut``."""
    return [t for t in trades if t["exit_ms"] <= cut and t["entry_ms"] >= lo]


# ── main ────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/scalp_research")
    ap.add_argument("--out", default="/tmp/scalp_research/strategy_research.json")
    ap.add_argument("--symbols", default=",".join(SYMBOLS))
    args = ap.parse_args()

    data: dict[str, dict] = {}
    t0, t1 = None, None
    for sym in args.symbols.split(","):
        df = build_features(load_symbol(Path(args.data) / f"{sym}_1m.csv"))
        arr = {k: df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close", "a", "a15")}
        arr["t"] = df.index.to_numpy()
        sig = {(f, lv): signals(df, f, lv) for f in FAMILIES for lv in LEVELS}
        data[sym] = {"arr": arr, "sig": sig}
        t0 = df.index[0] if t0 is None else max(t0, df.index[0])
        t1 = df.index[-1] if t1 is None else min(t1, df.index[-1])
    edges = block_edges(int(t0) + 240 * 60_000, int(t1))

    grid: dict[str, list[dict]] = {}
    for fam in FAMILIES:
        for lv in LEVELS:
            for ct in (*CONTRACTS, LEGACY):
                grid[f"{fam}|{lv}|{ct.name}"] = run_config(data, fam, lv, ct)

    fixed = {}
    for key, trades in grid.items():
        blocks = [metrics(in_block(trades, edges[k], edges[k + 1]))["net"] for k in range(N_BLOCKS)]
        fixed[key] = {"all": metrics(trades), "blocks_net": blocks}

    walk: dict[str, dict] = {}
    for fam in FAMILIES:
        oos: list[dict] = []
        folds = []
        for k in range(1, N_BLOCKS):
            best, best_net = None, -1e18
            for lv in LEVELS:
                for ct in CONTRACTS:
                    key = f"{fam}|{lv}|{ct.name}"
                    fit = fit_window(grid[key], edges[0], edges[k])
                    if len(fit) < MIN_FIT_TRADES:
                        continue
                    net = sum(t["net"] for t in fit)
                    if net > best_net:
                        best, best_net = key, net
            if best is None:
                folds.append({"fold": k, "chosen": None})
                continue
            ev = in_block(grid[best], edges[k], edges[k + 1])
            oos.extend(ev)
            folds.append({"fold": k, "chosen": best, "fit_net": round(best_net, 4), "oos": metrics(ev), "oos_by_symbol": by_key(ev, "symbol")})
        gated = [t for f in folds if f.get("chosen") and f["fit_net"] > 0 for t in in_block(grid[f["chosen"]], edges[f["fold"]], edges[f["fold"] + 1])]
        walk[fam] = {
            "folds": folds,
            "oos": metrics(oos),
            "oos_by_symbol": by_key(oos, "symbol"),
            "oos_by_reason": by_key(oos, "reason"),
            "oos_only_if_fit_positive": metrics(gated),
            "positive_folds": sum(1 for f in folds if f.get("oos", {}).get("net", 0) > 0),
        }

    report = {
        "window": {"start_ms": int(edges[0]), "end_ms": int(edges[-1]), "block_edges_ms": edges},
        "costs": {"fee_side": FEE, "slip_side": SLIP, "stop_slip": STOP_SLIP, "spread_bps_p75": SPREAD_BPS, "notional": NOTIONAL},
        "fixed": fixed,
        "walk_forward": walk,
    }
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    for fam, w in walk.items():
        o = w["oos"]
        print(fam, {k: o.get(k) for k in ("n", "net", "pf", "expectancy", "max_dd")}, [f.get("oos", {}).get("net") for f in w["folds"]])


if __name__ == "__main__":
    main()
