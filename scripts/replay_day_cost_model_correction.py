"""Replay the DAY entry cost-model veto against real forward prices.

`post_cost_economics_ev` (day_direct_path_ev_authority) vetoes a BUY when

    p_buy*efe - p_sell*|eae| - fees - slippage - spread <= 0

With the defaults stamped in `_augment_full_universe_candidates` the cost term is
22 bps (10 fees + 8 slippage + 4 spread). Measured Binance.US cost is ~5-6 bps.
This replays both cost terms over the recorded inference log and real 1m bars,
charging TRUE measured costs on every simulated fill in both arms, so the only
difference is which candidates were allowed through.

    sudo -u mystic venv/bin/python3 scripts/replay_day_cost_model_correction.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

DB = "/home/mystic/mystic/mystic_trading.db"
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")

# Stamped candidate defaults (portfolio_engine.py:14804-14805)
EFE = 0.012
EAE = 0.007

# Arm A: what production subtracts today (portfolio_engine.py:14806-14807 + 14454)
CURRENT_FEES = 0.0010
CURRENT_SLIP = 0.0008
CURRENT_SPREAD = 0.0004

# Arm B: evidence-based. Commission from 129 post-Apr-2026 Binance.US fills
# (2.000 bps/side taker, 0 bps maker). Slippage from 60 live exits
# (0.723 bps realized). Spread from 980k decision_book_tape samples.
TRUE_COMMISSION_RT = 0.0004
TRUE_SLIP_RT = 0.000144
REAL_SPREAD = {"BTCUSDT": 0.000059, "ETHUSDT": 0.000072, "SOLUSDT": 0.000219, "XRPUSDT": 0.000200}
SAFETY_BUFFER = 0.0001

# Engine constraints (backend/config/trading_economics.py)
MIN_NET_PROFIT_TO_SELL = 0.004
HORIZON_MIN = 180
MAX_SLOTS = 4
COOLDOWN_SEC = 2400


def true_roundtrip_cost(symbol: str) -> float:
    """Actual all-in round-trip cost charged on a fill, per measured evidence."""
    return TRUE_COMMISSION_RT + TRUE_SLIP_RT + REAL_SPREAD.get(symbol, 0.0002)


def gate_cost_current(symbol: str) -> float:
    return CURRENT_FEES + CURRENT_SLIP + CURRENT_SPREAD


def gate_cost_corrected(symbol: str) -> float:
    return TRUE_COMMISSION_RT + TRUE_SLIP_RT + REAL_SPREAD.get(symbol, 0.0002) + SAFETY_BUFFER


def econ_ev(p_buy: float, p_sell: float, p_hold: float, cost: float) -> float:
    total = p_buy + p_sell + p_hold
    if total > 0:
        p_buy = p_buy / total
        p_sell = p_sell / total
    return p_buy * EFE - p_sell * abs(EAE) - cost


def load_bars(conn: sqlite3.Connection) -> dict[str, list[tuple[int, float, float, float]]]:
    """1m bars per symbol as (epoch, high, low, close), ascending."""
    out: dict[str, list[tuple[int, float, float, float]]] = {}
    for sym in SYMBOLS:
        rows: list[tuple[Any, ...]] = []
        for name in (f"{sym[:-4]}-USDT", f"{sym[:-4]}/USDT", sym):
            rows = conn.execute(
                "SELECT ts, high, low, close FROM feature_ohlcv WHERE interval='1m' AND symbol=? ORDER BY ts ASC",
                (name,),
            ).fetchall()
            if rows:
                break
        bars: list[tuple[int, float, float, float]] = []
        for ts, h, low, c in rows:
            try:
                epoch = int(ts) if not isinstance(ts, str) else int(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
            except (TypeError, ValueError):
                continue
            bars.append((epoch, float(h or 0), float(low or 0), float(c or 0)))
        out[sym] = bars
    return out


def load_inferences(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT symbol, ts_utc, prob_buy, prob_hold, prob_sell
        FROM ai_inference_log
        WHERE strategy_id='day' OR strategy_id IS NULL OR strategy_id=''
        ORDER BY ts_utc ASC
        """
    ).fetchall()
    if not rows:
        rows = conn.execute("SELECT symbol, ts_utc, prob_buy, prob_hold, prob_sell FROM ai_inference_log ORDER BY ts_utc ASC").fetchall()
    out: list[dict[str, Any]] = []
    for sym, ts, pb, ph, ps in rows:
        s = str(sym or "").replace("/", "").replace("-", "").upper()
        if s not in SYMBOLS:
            continue
        try:
            epoch = int(datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp())
        except (TypeError, ValueError):
            continue
        out.append({"symbol": s, "epoch": epoch, "p_buy": float(pb or 0), "p_hold": float(ph or 0), "p_sell": float(ps or 0)})
    return out


def simulate_trade(bars: list[tuple[int, float, float, float]], entry_epoch: int, cost: float) -> tuple[float, int] | None:
    """Return (gross_return, exit_epoch) applying the DAY net-profit exit rule.

    Exit when gross >= MIN_NET_PROFIT_TO_SELL + cost (the engine's net gate),
    otherwise at horizon close. Costs are charged by the caller.
    """
    lo, hi = 0, len(bars) - 1
    idx = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if bars[mid][0] >= entry_epoch:
            idx = mid
            hi = mid - 1
        else:
            lo = mid + 1
    if idx is None or idx + 1 >= len(bars):
        return None
    entry_price = bars[idx][3]
    if entry_price <= 0:
        return None
    target = entry_price * (1.0 + MIN_NET_PROFIT_TO_SELL + cost)
    end = min(len(bars) - 1, idx + HORIZON_MIN)
    for j in range(idx + 1, end + 1):
        if bars[j][1] >= target:
            return (MIN_NET_PROFIT_TO_SELL + cost, bars[j][0])
    exit_price = bars[end][3]
    if exit_price <= 0:
        return None
    return ((exit_price - entry_price) / entry_price, bars[end][0])


@dataclass
class ArmResult:
    name: str
    accepted: int = 0
    rejected: int = 0
    trades: list[dict[str, Any]] = field(default_factory=list)


def run_arm(
    name: str,
    inferences: list[dict[str, Any]],
    bars: dict[str, list[tuple[int, float, float, float]]],
    gate_cost_fn,
    *,
    slot_constrained: bool,
) -> ArmResult:
    res = ArmResult(name=name)
    cooldown: dict[str, int] = defaultdict(int)
    held: set[str] = set()
    open_positions: list[tuple[int, str]] = []

    for inf in inferences:
        sym = inf["symbol"]
        ep = inf["epoch"]
        gate = gate_cost_fn(sym)
        ev = econ_ev(inf["p_buy"], inf["p_sell"], inf["p_hold"], gate)
        if ev <= 0.0:
            res.rejected += 1
            continue
        res.accepted += 1
        if slot_constrained:
            open_positions = [(t, s) for (t, s) in open_positions if t > ep]
            held = {s for (_, s) in open_positions}
            if len(open_positions) >= MAX_SLOTS or sym in held or ep < cooldown[sym]:
                continue
        sim = simulate_trade(bars.get(sym, []), ep, true_roundtrip_cost(sym))
        if sim is None:
            continue
        gross, exit_epoch = sim
        net = gross - true_roundtrip_cost(sym)
        res.trades.append({"symbol": sym, "entry": ep, "exit": exit_epoch, "gross": gross, "net": net})
        if slot_constrained:
            open_positions.append((exit_epoch, sym))
            cooldown[sym] = exit_epoch + COOLDOWN_SEC
    return res


def summarize(res: ArmResult) -> dict[str, Any]:
    t = res.trades
    n = len(t)
    if n == 0:
        return {"arm": res.name, "accepted_candidates": res.accepted, "rejected_candidates": res.rejected, "trades": 0}
    nets = [x["net"] for x in t]
    wins = [x for x in nets if x > 0]
    losses = [x for x in nets if x <= 0]
    gp = sum(wins)
    gl = abs(sum(losses))
    total_net = sum(nets)
    exposure_sec = sum(max(0, x["exit"] - x["entry"]) for x in t)
    span = (max(x["exit"] for x in t) - min(x["entry"] for x in t)) or 1
    peak = 0.0
    equity = 0.0
    max_dd = 0.0
    for x in sorted(t, key=lambda r: r["exit"]):
        equity += x["net"]
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "arm": res.name,
        "accepted_candidates": res.accepted,
        "rejected_candidates": res.rejected,
        "trades": n,
        "wins": len(wins),
        "win_rate_pct": round(100.0 * len(wins) / n, 2),
        "net_bps_per_trade": round(total_net / n * 1e4, 3),
        "total_net_bps": round(total_net * 1e4, 2),
        "profit_factor": round(gp / gl, 4) if gl > 0 else None,
        "capital_utilization_slot_frac": round(exposure_sec / (span * MAX_SLOTS), 4),
        "max_drawdown_bps": round(max_dd * 1e4, 2),
        "daily_return_on_deployed_bps": round(total_net * 1e4 / max(1.0, span / 86400.0) / MAX_SLOTS, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    conn = sqlite3.connect(DB)
    try:
        bars = load_bars(conn)
        inferences = load_inferences(conn)
    finally:
        conn.close()

    print(f"inferences={len(inferences)}  bars=" + ", ".join(f"{s}:{len(bars.get(s, []))}" for s in SYMBOLS))
    print()
    print("cost terms (bps):")
    for s in SYMBOLS:
        print(f"  {s:8s} gate_current={gate_cost_current(s) * 1e4:6.2f}  gate_corrected={gate_cost_corrected(s) * 1e4:6.2f}  true_charged={true_roundtrip_cost(s) * 1e4:6.2f}")
    print()

    cost_terms = {
        s: {
            "gate_current": gate_cost_current(s) * 1e4,
            "gate_corrected": gate_cost_corrected(s) * 1e4,
            "true_charged": true_roundtrip_cost(s) * 1e4,
        }
        for s in SYMBOLS
    }
    out: dict[str, Any] = {"cost_terms_bps": cost_terms}

    for mode, slot in (("candidate_level", False), ("slot_constrained", True)):
        print(f"===== {mode} =====")
        rows = []
        for nm, fn in (("A_current_22bps", gate_cost_current), ("B_corrected_evidence", gate_cost_corrected)):
            r = summarize(run_arm(nm, inferences, bars, fn, slot_constrained=slot))
            rows.append(r)
            print(json.dumps(r, indent=2, sort_keys=True))
        out[mode] = rows
        if len(rows) == 2 and rows[0].get("trades") and rows[1].get("trades"):
            a, b = rows
            print(f"  delta trades: {b['trades'] - a['trades']:+d}")
            print(f"  delta net_bps_per_trade: {b['net_bps_per_trade'] - a['net_bps_per_trade']:+.3f}")
            print(f"  delta total_net_bps: {b['total_net_bps'] - a['total_net_bps']:+.2f}")
        print()

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
