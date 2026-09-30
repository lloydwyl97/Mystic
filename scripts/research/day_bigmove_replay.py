"""DAY V2 exit-contract replay on 180 days of Binance.US 1m klines.

Read-only. Entries come from the LIVE ``_detect_setup`` (imported, not
copied) on closed 15m/1h/4h clock bars built from 1m klines, with the live
consumed-opportunity rule, 24h frequency caps and one DAY position per
symbol. Fill = first 1m open after the 15m close at the ask. Exits are
simulated minute by minute; within a minute losses are checked first.

Each contract shares the same loss side unless it says otherwise; contracts
differ in winner management. Folds: 5 chronological blocks; per-block results
for every fixed contract plus a walk-forward pick that uses only prior blocks.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.services.day_v2.live_signal import evaluate_entry_signal
from backend.services.day_v2.winner_contract import (
    DAY_EXIT_CONTRACT_RUNNER,
    RUNNER_ACTIVATION_ATR_1H,
    RUNNER_TIGHT_TRAIL_ATR_1H,
    RUNNER_TRAIL_ATR_1H,
    objective_level,
    structural_objective,
)
from scripts.research.scalp_strategy_research import SPREAD_BPS, STEP, block_edges, by_key, fit_window, in_block, load_symbol, metrics

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
FEE = 0.0002
SLIP = 0.0001
STOP_SLIP = 0.0002
LIVE_RT_COST = 0.0006
NOTIONAL = 60.0
N_BLOCKS = 5
PER_SYMBOL_24H = 2
TOTAL_24H = 8
SAFETY_MAX_MIN = 14_400


@dataclass(frozen=True)
class Contract:
    name: str
    legacy: bool = False
    activate_atr1h: float = 1.0
    trail_atr1h: float = 1.0
    tight_trail_atr1h: float = 0.5
    sell_at_objective: bool = False
    structural_on_closed_15m: bool = False
    max_hold_min: int = 1440


CONTRACTS = (
    Contract("V0_CURRENT_LIVE", legacy=True),
    Contract("N1_SETUP_OBJECTIVE_SELL", sell_at_objective=True),
    Contract("N2_RUNNER_ATR1H", trail_atr1h=1.0, tight_trail_atr1h=0.5),
    Contract("N3_RUNNER_WIDE", trail_atr1h=1.5, tight_trail_atr1h=0.75),
    Contract("N4_RUNNER_CLOSED15_STRUCT", trail_atr1h=1.0, tight_trail_atr1h=0.5, structural_on_closed_15m=True),
    Contract(
        DAY_EXIT_CONTRACT_RUNNER,
        activate_atr1h=RUNNER_ACTIVATION_ATR_1H,
        trail_atr1h=RUNNER_TRAIL_ATR_1H,
        tight_trail_atr1h=RUNNER_TIGHT_TRAIL_ATR_1H,
        max_hold_min=SAFETY_MAX_MIN,
    ),
)


def clock(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    ms = minutes * 60_000
    g = df.groupby((df.index.to_numpy() // ms) * ms)
    b = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last(), "n": g["close"].size()})
    return b[b["n"] == minutes]


def _atr(b: pd.DataFrame, n: int = 14) -> float:
    if len(b) < n + 1:
        return float("nan")
    pc = b["close"].shift(1)
    tr = pd.concat([b["high"] - b["low"], (b["high"] - pc).abs(), (b["low"] - pc).abs()], axis=1).max(axis=1)
    return float(tr.iloc[-n:].mean())


def _bars(b: pd.DataFrame) -> list[dict]:
    return [{"ts_epoch": int(t // 1000), "open": r[0], "high": r[1], "low": r[2], "close": r[3]} for t, r in zip(b.index, b[["open", "high", "low", "close"]].to_numpy(), strict=True)]


def objective_price(setup: str, entry: float, atr1h: float, b15: pd.DataFrame, b1h: pd.DataFrame, b4h: pd.DataFrame) -> float:
    """Live setup objective (backend.services.day_v2.winner_contract) on replay bars."""
    return objective_level(setup, entry, atr1h, structural_objective(setup, _bars(b15), _bars(b1h), _bars(b4h)))


def build_signals(df: pd.DataFrame, sym: str) -> list[dict]:
    b15, b1h, b4h = clock(df, 15), clock(df, 60), clock(df, 240)
    t1h = b1h.index.to_numpy() + 3_600_000
    t4h = b4h.index.to_numpy() + 14_400_000
    out = []
    for j in range(60, len(b15)):
        close_ms = int(b15.index[j]) + 900_000
        w15 = b15.iloc[j - 59 : j + 1]
        k1 = int(np.searchsorted(t1h, close_ms, side="right"))
        k4 = int(np.searchsorted(t4h, close_ms, side="right"))
        if k1 < 20 or k4 < 15:
            continue
        w1h, w4h = b1h.iloc[k1 - 20 : k1], b4h.iloc[k4 - 15 : k4]
        sig = evaluate_entry_signal(sym, _bars(w15), _bars(w1h), _bars(w4h))
        if sig is None:
            continue
        out.append(
            {
                "decision_ms": close_ms,
                "setup": sig.setup,
                "regime": sig.regime,
                "anchor": sig.structural_anchor,
                "target": sig.target_price,
                "atr15": sig.atr,
                "atr1h": _atr(w1h),
                "atr4h": _atr(w4h),
                "opp": sig.opportunity_id,
                "w15": w15,
                "w1h": w1h,
                "w4h": w4h,
            }
        )
    return out


def simulate(arr: dict[str, np.ndarray], ie: int, s: dict, ct: Contract, sym: str) -> dict | None:
    n = len(arr["open"])
    if ie >= n:
        return None
    hs = SPREAD_BPS[sym] / 2e4
    entry = arr["open"][ie] * (1 + hs + SLIP)
    step = STEP[sym]
    qty = math.floor(NOTIONAL / entry / step + 1e-9) * step
    if qty * entry < 1.0:
        return None
    atr15, atr1h = s["atr15"], s["atr1h"]
    if not (atr1h > 0):
        return None
    cat_px = entry * (1 - 3.0 * atr15 / entry)
    anchor = s["anchor"]
    obj = s["target"] if ct.legacy else objective_price(s["setup"], entry, atr1h, s["w15"], s["w1h"], s["w4h"])
    hwm = entry
    stop = 0.0
    obj_hit = False
    exit_px, reason, j = None, "", ie
    limit = min(n - 1, ie + (SAFETY_MAX_MIN if ct.legacy else ct.max_hold_min) - 1)
    while j <= limit:
        o, hi, lo, cl = arr["open"][j], arr["high"][j], arr["low"][j], arr["close"][j]
        held = j - ie + 1
        if lo <= cat_px:
            exit_px, reason = min(cat_px, o) * (1 - hs - STOP_SLIP), "CATASTROPHIC"
            break
        if held >= 45 and anchor > 0:
            if ct.structural_on_closed_15m:
                broke = (arr["t"][j] + 60_000) % 900_000 == 0 and cl < anchor
            else:
                broke = cl < anchor
            if broke:
                exit_px, reason = cl * (1 - hs - SLIP), "STRUCTURAL"
                break
        if ct.legacy:
            mfe = hwm / entry - 1
            if mfe >= 0.008:
                trail = max(0.005, 1.5 * atr15 / entry)
                trig = max(hwm * (1 - trail), entry * (1 + LIVE_RT_COST))
                if lo <= trig:
                    exit_px, reason = min(trig, o) * (1 - hs - SLIP), "WINNER_TRAIL"
                    break
            if obj > 0 and hi >= obj:
                exit_px, reason = obj * (1 - hs - SLIP), "OBJECTIVE"
                break
        else:
            if stop > 0 and lo <= stop:
                exit_px, reason = min(stop, o) * (1 - hs - STOP_SLIP), "RATCHET"
                break
            if ct.sell_at_objective and hi >= obj:
                exit_px, reason = obj * (1 - hs - SLIP), "OBJECTIVE"
                break
        hwm = max(hwm, hi)
        if not ct.legacy:
            if hi >= obj:
                obj_hit = True
            if hwm - entry >= ct.activate_atr1h * atr1h:
                dist = (ct.tight_trail_atr1h if obj_hit else ct.trail_atr1h) * atr1h
                stop = max(stop, entry * (1 + LIVE_RT_COST), hwm - dist)
        net_now = cl * (1 - hs - SLIP) / entry - 1 - 2 * FEE
        if held >= 300 and net_now <= 0 and (ct.legacy or stop <= 0):
            exit_px, reason = cl * (1 - hs - SLIP), "TIME_NO_DEVELOPMENT"
            break
        j += 1
    if exit_px is None:
        j = limit
        exit_px, reason = arr["close"][j] * (1 - hs - SLIP), "MAX_HOLD" if not ct.legacy else "DATA_END"
    net = qty * (exit_px - entry) - FEE * qty * (entry + exit_px)
    return {
        "symbol": sym,
        "setup": s["setup"],
        "regime": s["regime"],
        "entry_ms": int(arr["t"][ie]),
        "exit_ms": int(arr["t"][j]) + 60_000,
        "exit_i": j,
        "hold_min": j - ie + 1,
        "net": net,
        "net_bps": (net / (qty * entry)) * 1e4,
        "reason": reason,
        "mfe_bps": (hwm / entry - 1) * 1e4,
        "atr1h_bps": atr1h / entry * 1e4,
        "obj_bps": (obj / entry - 1) * 1e4,
    }


def run(data: dict[str, dict], ct: Contract) -> list[dict]:
    events = sorted((s["decision_ms"], sym, s) for sym, d in data.items() for s in d["signals"])
    free_at = dict.fromkeys(data, -1)
    used_opp: set[str] = set()
    fills: list[tuple[int, str]] = []
    trades = []
    for t, sym, s in events:
        arr = data[sym]["arr"]
        ie = int(np.searchsorted(arr["t"], t))
        if ie <= free_at[sym] or s["opp"] in used_opp:
            continue
        recent = [x for x in fills if x[0] > t - 86_400_000]
        if len(recent) >= TOTAL_24H or sum(1 for x in recent if x[1] == sym) >= PER_SYMBOL_24H:
            continue
        tr = simulate(arr, ie, s, ct, sym)
        if tr is None:
            continue
        used_opp.add(s["opp"])
        fills.append((t, sym))
        free_at[sym] = tr["exit_i"]
        trades.append(tr)
    trades.sort(key=lambda x: x["exit_ms"])
    return trades


def summary(trades: list[dict]) -> dict:
    m = metrics(trades)
    if trades:
        holds = [t["hold_min"] for t in trades]
        m["avg_hold_min"] = round(float(np.mean(holds)), 1)
        m["median_hold_min"] = float(np.median(holds))
        m["avg_net_bps"] = round(float(np.mean([t["net_bps"] for t in trades])), 2)
        w = [t for t in trades if t["net"] > 0]
        m["avg_win_bps"] = round(float(np.mean([t["net_bps"] for t in w])), 1) if w else 0.0
        lo = [t for t in trades if t["net"] <= 0]
        m["avg_loss_bps"] = round(float(np.mean([t["net_bps"] for t in lo])), 1) if lo else 0.0
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/scalp_research")
    ap.add_argument("--out", default="/tmp/scalp_research/day_bigmove_replay.json")
    args = ap.parse_args()
    data = {}
    t0, t1 = None, None
    for sym in SYMBOLS:
        df = load_symbol(Path(args.data) / f"{sym}_1m.csv")
        arr = {k: df[k].to_numpy().astype(float) for k in ("open", "high", "low", "close")}
        arr["t"] = df.index.to_numpy()
        data[sym] = {"arr": arr, "signals": build_signals(df, sym)}
        t0 = df.index[0] if t0 is None else max(t0, df.index[0])
        t1 = df.index[-1] if t1 is None else min(t1, df.index[-1])
        print(sym, "signals", len(data[sym]["signals"]), flush=True)
    edges = block_edges(int(t0) + 5 * 86_400_000, int(t1))
    grid = {ct.name: run(data, ct) for ct in CONTRACTS}
    fixed = {}
    for name, trades in grid.items():
        fixed[name] = {
            "all": summary(trades),
            "blocks": [summary(in_block(trades, edges[k], edges[k + 1])) for k in range(N_BLOCKS)],
            "by_setup": {k: summary([t for t in trades if t["setup"] == k]) for k in sorted({t["setup"] for t in trades})},
            "by_symbol": {k: summary([t for t in trades if t["symbol"] == k]) for k in SYMBOLS},
            "by_regime": {k: summary([t for t in trades if t["regime"] == k]) for k in sorted({t["regime"] for t in trades})},
            "by_reason": by_key(trades, "reason"),
        }
    walk, oos = [], []
    for k in range(1, N_BLOCKS):
        best = max(grid, key=lambda nm: sum(t["net"] for t in fit_window(grid[nm], edges[0], edges[k])))
        ev = in_block(grid[best], edges[k], edges[k + 1])
        oos.extend(ev)
        walk.append({"fold": k, "chosen": best, "oos": summary(ev)})
    report = {"block_edges_ms": edges, "fixed": fixed, "walk_forward": {"folds": walk, "oos": summary(oos)}}
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    for name, f in fixed.items():
        a = f["all"]
        print(name, {x: a.get(x) for x in ("n", "net", "pf", "win_rate", "avg_win", "avg_loss", "payoff", "max_dd", "median_hold_min")}, [b.get("net") for b in f["blocks"]])
    print("WALK", [(w["chosen"], w["oos"].get("net")) for w in walk], summary(oos).get("net"))


if __name__ == "__main__":
    main()
