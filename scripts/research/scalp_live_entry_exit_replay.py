"""Replay short-horizon SCALP exit contracts on the real live SCALP entries.

Entries are the venue-filled SCALP_V2 BUY prices and times. Exits are
simulated on Binance.US 1m klines starting the minute after the fill, with
the same costs as the kline research: 2 bps taker fee per side, half the
p75 spread and 1 bp slippage on market exits, 2 bps slippage on stops.
Stops are checked before targets inside a bar.

Usage:
    python -m scripts.research.scalp_live_entry_exit_replay \
        --trades /tmp/scalp_trades.json --data /tmp/scalp_research
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from scripts.research.scalp_strategy_research import SPREAD_BPS, load_symbol

FEE = 0.0002
SLIP = 0.0001
STOP_SLIP = 0.0002
N_FOLDS = 4


@dataclass(frozen=True)
class ScalpExit:
    name: str
    target_net: float
    stop_pct: float
    max_hold_min: int
    fail_after_min: int = 0
    fail_mfe_pct: float = 0.0
    time_only_if_negative: bool = False


CONTRACTS = (
    ScalpExit("S0_CURRENT_APPROX", target_net=0.004, stop_pct=0.015, max_hold_min=120, time_only_if_negative=True),
    ScalpExit("S1_TIGHT_30M", target_net=0.003, stop_pct=0.004, max_hold_min=30),
    ScalpExit("S2_FAIL_TO_DEVELOP_45M", target_net=0.004, stop_pct=0.006, max_hold_min=45, fail_after_min=15, fail_mfe_pct=0.001),
    ScalpExit("S3_SYMMETRIC_20M", target_net=0.0025, stop_pct=0.0025, max_hold_min=20),
)


def _epoch_ms(ts: str) -> int:
    return int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp() * 1000)


def simulate(arr: dict[str, np.ndarray], entry_ms: int, entry: float, sym: str, ct: ScalpExit) -> dict | None:
    t = arr["t"]
    i0 = int(np.searchsorted(t, (entry_ms // 60_000 + 1) * 60_000, side="left"))
    if i0 >= len(t):
        return None
    hs = SPREAD_BPS[sym] / 2e4
    target_px = entry * (1 + ct.target_net + 2 * FEE) / (1 - hs - SLIP)
    stop_px = entry * (1 - ct.stop_pct)
    hwm = entry
    last = min(len(t) - 1, i0 + ct.max_hold_min - 1)
    if last - i0 + 1 < ct.max_hold_min:
        return None
    exit_px, reason, j = None, "", i0
    for j in range(i0, last + 1):
        o, hi, lo, cl = arr["open"][j], arr["high"][j], arr["low"][j], arr["close"][j]
        held = j - i0 + 1
        if lo <= stop_px:
            exit_px, reason = min(stop_px, o) * (1 - hs - STOP_SLIP), "STOP"
            break
        if hi >= target_px:
            exit_px, reason = max(target_px, o) * (1 - hs - SLIP), "TARGET"
            break
        hwm = max(hwm, hi)
        if ct.fail_after_min and held == ct.fail_after_min and hwm / entry - 1 < ct.fail_mfe_pct:
            exit_px, reason = cl * (1 - hs - SLIP), "FAIL_TO_DEVELOP"
            break
    if exit_px is None:
        cl = arr["close"][last]
        px = cl * (1 - hs - SLIP)
        if ct.time_only_if_negative and px / entry - 1 - 2 * FEE > 0:
            exit_px, reason = px, "TIME_POSITIVE_HELD"
        else:
            exit_px, reason = px, "TIME"
    net = exit_px / entry - 1 - FEE * (1 + exit_px / entry)
    return {"net_bps": net * 1e4, "reason": reason, "hold_min": j - i0 + 1, "entry_ms": entry_ms}


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    x = np.array([r["net_bps"] for r in rows])
    wins, losses = x[x > 0], x[x <= 0]
    gl = float(-losses.sum())
    return {
        "n": len(rows),
        "net_bps_sum": round(float(x.sum()), 1),
        "avg_bps": round(float(x.mean()), 2),
        "pf": round(float(wins.sum()) / gl, 3) if gl > 0 else None,
        "win_rate": round(float((x > 0).mean()), 3),
        "avg_win_bps": round(float(wins.mean()), 1) if len(wins) else 0.0,
        "avg_loss_bps": round(float(losses.mean()), 1) if len(losses) else 0.0,
        "median_hold_min": float(np.median([r["hold_min"] for r in rows])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default="/tmp/scalp_trades.json")
    ap.add_argument("--data", default="/tmp/scalp_research")
    args = ap.parse_args()
    trades = [t for t in json.loads(Path(args.trades).read_text()) if t.get("has_buy") and t.get("entry", 0) > 0]
    syms = sorted({t["symbol"].replace("/", "").replace("-", "") for t in trades})
    data = {}
    for s in syms:
        df = load_symbol(Path(args.data) / f"{s}_1m.csv")
        data[s] = {"t": df.index.to_numpy(), **{k: df[k].to_numpy() for k in ("open", "high", "low", "close")}}
    live_bps = {}
    for t in trades:
        live_bps[_epoch_ms(t["entry_ts"])] = t["pnl"] / (t["qty"] * t["entry"]) * 1e4
    report = {}
    for ct in CONTRACTS:
        rows = []
        for t in trades:
            s = t["symbol"].replace("/", "").replace("-", "")
            r = simulate(data[s], _epoch_ms(t["entry_ts"]), float(t["entry"]), s, ct)
            if r:
                rows.append(r)
        rows.sort(key=lambda r: r["entry_ms"])
        folds = [summarize(list(f)) for f in np.array_split(np.array(rows, dtype=object), N_FOLDS)]
        report[ct.name] = {
            "all": summarize(rows),
            "folds_net_bps": [f.get("net_bps_sum") for f in folds],
            "positive_folds": sum(1 for f in folds if (f.get("net_bps_sum") or 0) > 0),
            "live_same_entries_avg_bps": round(float(np.mean([live_bps[r["entry_ms"]] for r in rows])), 2),
        }
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
