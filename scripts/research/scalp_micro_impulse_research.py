"""SCALP microstructure-impulse research on recorded Ocean book snapshots.

Read-only. Input is an export of ``microstructure_feature_snapshots`` (~5.6s
cadence, recorded best bid/ask). Signals use only the snapshot at decision
time and trailing statistics; entry fills at the NEXT snapshot's best ask,
exits fill at the recorded best bid. No midpoint fills.

Folds: the window is cut into 5 chronological blocks; fold k fits the
(level, contract) choice on trades that closed before block k and evaluates
on trades entered inside block k.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.scalp_strategy_research import by_key, fit_window, in_block, metrics

FEE = 0.0002
SLIP = 0.0001
STOP_SLIP = 0.0002
NOTIONAL = 50.0
STEP = {"BTC": 1e-5, "ETH": 1e-4, "SOL": 1e-3, "XRP": 0.1}
MAX_ENTRY_GAP_SEC = 15.0
Z_WINDOW = 640
N_BLOCKS = 5
MIN_FIT_TRADES = 20
LEVELS = ("loose", "base", "strict")


def load(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(path)
    df = df[(df["best_bid"] > 0) & (df["best_ask"] >= df["best_bid"])]
    out = {}
    for sym, grp in df.groupby("symbol"):
        g = grp.sort_values("ts").reset_index(drop=True)
        roll = g["ofi_30s"].shift(1).rolling(Z_WINDOW, min_periods=200)
        g["ofi30_z"] = (g["ofi_30s"] - roll.mean()) / roll.std().replace(0, np.nan)
        mid = g["mid"]
        g["disp30_bps"] = (mid / mid.shift(5) - 1) * 1e4
        g["agg30_vol"] = g["agg_buy_vol_30s"].fillna(0) + g["agg_sell_vol_30s"].fillna(0)
        out[str(sym)] = g
    return out


def signals(g: pd.DataFrame, level: str) -> np.ndarray:
    i = LEVELS.index(level)
    z_min = (1.5, 2.0, 2.5)[i]
    obi_min = (0.2, 0.3, 0.4)[i]
    sig = (
        (g["ofi30_z"] >= z_min)
        & (g["obi_l5"] >= obi_min)
        & (g["agg30_vol"] > 0)
        & (g["agg_flow_imbalance_30s"] >= 0.3)
        & (g["microprice_pressure"] > 0)
        & (g["disp30_bps"] > 0)
        & (g["p_adverse_move"] <= 0.35)
        & (g["spread_pct"] <= 0.0002)
    )
    return sig.fillna(False).to_numpy()


@dataclass(frozen=True)
class Contract:
    name: str
    target_bps: float
    stop_bps: float
    max_sec: float
    adverse_flow_exit: bool = False


CONTRACTS = (
    Contract("M1_8_8_120s", 8.0, 8.0, 120.0),
    Contract("M2_12_8_300s_ADVFLOW", 12.0, 8.0, 300.0, adverse_flow_exit=True),
    Contract("M3_20_10_600s_ADVFLOW", 20.0, 10.0, 600.0, adverse_flow_exit=True),
)


def simulate(g: dict[str, np.ndarray], i_sig: int, ct: Contract, sym: str) -> dict | None:
    n = len(g["ts"])
    ie = i_sig + 1
    if ie >= n or g["ts"][ie] - g["ts"][i_sig] > MAX_ENTRY_GAP_SEC:
        return None
    entry = g["best_ask"][ie] * (1 + SLIP)
    qty = math.floor(NOTIONAL / entry / STEP[sym] + 1e-9) * STEP[sym]
    if qty * entry < 1.0:
        return None
    tgt = entry * (1 + ct.target_bps / 1e4)
    stp = entry * (1 - ct.stop_bps / 1e4)
    t_end = g["ts"][ie] + ct.max_sec
    j = ie + 1
    exit_px, reason = None, ""
    best = entry
    while j < n:
        bid = g["best_bid"][j]
        best = max(best, bid)
        if bid <= stp:
            exit_px, reason = bid * (1 - STOP_SLIP), "STOP"
            break
        if bid >= tgt:
            exit_px, reason = bid * (1 - SLIP), "TARGET"
            break
        if ct.adverse_flow_exit and g["agg_flow_imbalance_5s"][j] <= -0.5 and g["ofi_5s"][j] < 0:
            exit_px, reason = bid * (1 - SLIP), "ADVERSE_FLOW"
            break
        if g["ts"][j] >= t_end:
            exit_px, reason = bid * (1 - SLIP), "MAX_HOLD"
            break
        j += 1
    if exit_px is None:
        return None
    net = qty * (exit_px - entry) - FEE * qty * (entry + exit_px)
    return {
        "symbol": sym,
        "entry_ms": int(g["ts"][ie] * 1000),
        "exit_ms": int(g["ts"][j] * 1000),
        "exit_i": j,
        "net": net,
        "reason": reason,
        "mfe_bps": (best / entry - 1) * 1e4,
    }


def run(data: dict[str, dict], level: str, ct: Contract) -> list[dict]:
    trades = []
    for sym, d in data.items():
        free_at = -1
        for i in np.flatnonzero(d["sig"][level]):
            if i <= free_at:
                continue
            tr = simulate(d["arr"], int(i), ct, sym)
            if tr is None:
                continue
            trades.append(tr)
            free_at = tr["exit_i"]
    trades.sort(key=lambda t: t["exit_ms"])
    return trades


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/scalp_research/micro_export.csv")
    ap.add_argument("--out", default="/tmp/scalp_research/micro_impulse_research.json")
    args = ap.parse_args()
    frames = load(Path(args.data))
    data = {}
    t0, t1 = -1e18, 1e18
    for sym, g in frames.items():
        arr = {k: g[k].to_numpy().astype(float) for k in ("ts", "best_bid", "best_ask", "agg_flow_imbalance_5s", "ofi_5s")}
        arr["agg_flow_imbalance_5s"] = np.nan_to_num(arr["agg_flow_imbalance_5s"])
        arr["ofi_5s"] = np.nan_to_num(arr["ofi_5s"])
        data[sym] = {"arr": arr, "sig": {lv: signals(g, lv) for lv in LEVELS}}
        t0, t1 = max(t0, g["ts"].iloc[0]), min(t1, g["ts"].iloc[-1])
    edges = [int((t0 + (t1 - t0) * k / N_BLOCKS) * 1000) for k in range(N_BLOCKS + 1)]
    grid = {f"{lv}|{ct.name}": run(data, lv, ct) for lv in LEVELS for ct in CONTRACTS}
    fixed = {k: {"all": metrics(v), "by_reason": by_key(v, "reason")} for k, v in grid.items()}
    oos, folds = [], []
    for k in range(1, N_BLOCKS):
        best, best_net = None, -1e18
        for key, trades in grid.items():
            fit = fit_window(trades, edges[0], edges[k])
            if len(fit) >= MIN_FIT_TRADES and sum(t["net"] for t in fit) > best_net:
                best, best_net = key, sum(t["net"] for t in fit)
        if best is None:
            folds.append({"fold": k, "chosen": None})
            continue
        ev = in_block(grid[best], edges[k], edges[k + 1])
        oos.extend(ev)
        folds.append({"fold": k, "chosen": best, "fit_net": round(best_net, 4), "oos": metrics(ev)})
    report = {
        "window_sec": [t0, t1],
        "block_edges_ms": edges,
        "fixed": fixed,
        "walk_forward": {"folds": folds, "oos": metrics(oos), "oos_by_symbol": by_key(oos, "symbol"), "oos_by_reason": by_key(oos, "reason")},
    }
    Path(args.out).write_text(json.dumps(report, indent=1, default=str))
    for key, v in fixed.items():
        a = v["all"]
        print(key, {x: a.get(x) for x in ("n", "net", "pf", "win_rate", "avg_win", "avg_loss")})
    print("WALK", report["walk_forward"]["oos"])
    for f in folds:
        print(f.get("fold"), f.get("chosen"), f.get("oos", {}).get("n"), f.get("oos", {}).get("net"))


if __name__ == "__main__":
    main()
