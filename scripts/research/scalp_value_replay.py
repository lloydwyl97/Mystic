"""Causal replay of the SCALP value model against the installed SCALP policy.

Research only; writes nothing. At every decision time the model holds only
labels whose horizon had elapsed (plus the bar lag) by then. Exits are checked
at 1m bar closes, and catastrophic protection fires at the stop price.

    venv/bin/python3 scripts/research/scalp_value_replay.py --db /tmp/scalp_replay.db \
        --start 2026-10-15T00:00:00Z --end 2026-10-15T16:00:00Z
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.services.scalp_v2 import continuation_value as cv
from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT


def _epoch(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class CausalModel:
    """Minute snapshots of the per-horizon coefficients, folded in label-time order."""

    def __init__(self, db: str, candidates: list[cv.Candidate], bars: dict[str, cv.Bars], start: float, end: float) -> None:
        self.start = start
        model = cv.ValueModel()
        before = [c for c in candidates if c.t0 < start]
        for index in range(0, len(before), 2000):
            model.fold(cv._labels_for(db, before[index : index + 2000], -math.inf, start, bars), start)
        recent = [c for c in candidates if c.t0 >= start - cv.SPAN_SEC]
        pending = sorted(cv._labels_for(db, recent, start, end, bars), key=lambda item: item[3])
        self.snapshots: list[dict[int, np.ndarray]] = [self._betas(model)]
        cursor = 0
        minute = start + 60.0
        while minute <= end:
            batch = []
            while cursor < len(pending) and pending[cursor][3] <= minute:
                batch.append(pending[cursor])
                cursor += 1
            if batch:
                model.fold(batch, minute)
            self.snapshots.append(self._betas(model))
            minute += 60.0

    @staticmethod
    def _betas(model: cv.ValueModel) -> dict[int, np.ndarray]:
        return {h: beta.copy() for h in cv.HORIZONS_SEC if (beta := model.beta(h)) is not None}

    def values(self, moment: float, x: np.ndarray) -> dict[int, float]:
        index = math.floor((moment - cv.LABEL_LAG_SEC - self.start) / 60.0)
        betas = self.snapshots[max(0, min(index, len(self.snapshots) - 1))]
        return {h: float(x @ beta) for h, beta in betas.items()}

    def advantage(self, moment: float, x: np.ndarray) -> float:
        per = self.values(moment, x)
        return max(per.values()) if per else 0.0


def simulate(model: CausalModel, lot: cv.Candidate, bars: cv.Bars, data_end: float) -> dict:
    """Hold ``lot`` (t0 = fill time, ref = fill price) under the learned value, checked each minute."""
    safety = lot.ref * (1.0 - SCALP_V2_CATASTROPHIC_PCT)
    first = int(np.searchsorted(bars.opened, lot.t0, side="left"))
    trace = []
    age = 0
    while True:
        moment = lot.t0 + 60.0 * age
        last = int(np.searchsorted(bars.closes_at, moment, side="right")) - 1
        if last >= first and float(bars.low[first : last + 1].min()) <= safety:
            hit = first + int(np.argmax(bars.low[first : last + 1] <= safety))
            return {"reason": "CATASTROPHIC", "net": -SCALP_V2_CATASTROPHIC_PCT - lot.cost, "hold_min": (float(bars.closes_at[hit]) - lot.t0) / 60.0, "trace": trace}
        if moment > data_end:
            gross = float(bars.close[last]) / lot.ref - 1.0
            return {"reason": "DATA_END", "net": gross - lot.cost, "hold_min": age, "trace": trace}
        found = cv.path_state(lot, bars, age)
        if found is not None:
            _last, _gross, state = found
            adv = model.advantage(moment, cv.vector(state, lot.entry, lot.symbol, lot.setup, lot.regime))
            trace.append((age, round(state["net"] * 1e4, 1), round(adv * 1e4, 1)))
            if adv < 0.0:
                return {"reason": "LEARNED_EXIT", "net": state["net"], "hold_min": age, "trace": trace}
        age += 1


def _stats(nets: list[float]) -> dict:
    wins = [n for n in nets if n > 0]
    losses = [n for n in nets if n <= 0]
    gross_loss = -sum(losses)
    return {
        "trades": len(nets),
        "wins": len(wins),
        "sum_bps": round(sum(nets) * 1e4, 1),
        "mean_bps": round(sum(nets) / len(nets) * 1e4, 2) if nets else None,
        "pf": round(sum(wins) / gross_loss, 2) if gross_loss > 0 else None,
    }


def _trades(db: str, start: float, end: float) -> list[dict]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    out = []
    for buy in conn.execute("SELECT trade_id, timestamp, symbol, price, quantity FROM paper_trades WHERE engine_id='SCALP_V2' AND UPPER(side)='BUY' ORDER BY timestamp"):
        at = _epoch(buy["timestamp"])
        if not start <= at < end:
            continue
        sym = str(buy["symbol"]).replace("/", "")
        cand = conn.execute(
            "SELECT * FROM adaptive_candidate_markouts WHERE REPLACE(symbol,'/','')=? AND filled=1 AND evaluated_at BETWEEN ? AND ? ORDER BY evaluated_at DESC LIMIT 1",
            (sym, at - 120, at + 5),
        ).fetchone()
        if cand is None:
            continue
        out.append({"buy": dict(buy), "at": at, "cand": dict(cand)})
    conn.close()
    return out


def run(db: str, start: float, end: float) -> dict:
    candidates = cv.load_candidates(db, 0.0, end)
    horizon_end = end + cv.SPAN_SEC
    bars = {sym: cv.load_bars(db, sym, candidates[0].t0 - 60.0, horizon_end + 60.0) for sym in {c.symbol for c in candidates}}
    data_end = max(float(b.closes_at[-1]) for b in bars.values() if b.opened.size)
    model = CausalModel(db, candidates, bars, start, min(horizon_end, data_end))
    by_id = {c.id: c for c in candidates}

    rows = []
    for item in _trades(db, start, end):
        cand = by_id.get(int(item["cand"]["id"]))
        if cand is None:
            continue
        econ = json.loads(item["cand"]["economic_json"] or "{}")
        x0 = cv.vector(cv.entry_position_state(cand.cost), cand.entry, cand.symbol, cand.setup, cand.regime)
        v0 = model.values(cand.t0, x0)
        final = (max(v0.values()) if v0 else 0.0) - cand.cost
        fill = cv.Candidate(cand.id, cand.symbol, cand.setup, cand.regime, item["at"], float(item["buy"]["price"]), cand.cost, cand.entry)
        held = simulate(model, fill, bars[cand.symbol], data_end)
        notional = float(item["buy"]["price"]) * float(item["buy"]["quantity"])
        installed = float(item["cand"]["realized_net"] or 0.0)
        rows.append(
            {
                "time": item["buy"]["timestamp"][11:19],
                "symbol": cand.symbol[:3],
                "notional": round(notional, 2),
                "legacy_final_bps": round(float(econ.get("policy_value") or 0.0) * 1e4, 1),
                "legacy_market_edge_bps": round(float(econ.get("market_edge") or 0.0) * 1e4, 1),
                "repaired_final_bps": round(final * 1e4, 1),
                "entry_value_by_horizon_bps": {h: round(v * 1e4, 1) for h, v in v0.items()},
                "admitted": final > 0.0,
                "installed_net_bps": round(installed * 1e4, 1),
                "continuation_on_actual_entry": {
                    "reason": held["reason"],
                    "hold_min": round(held["hold_min"], 1),
                    "net_bps": round(held["net"] * 1e4, 1),
                    "trace_head": held["trace"][:4],
                    "trace_tail": held["trace"][-3:],
                },
                "repaired_net_bps": round(held["net"] * 1e4, 1) if final > 0.0 else 0.0,
                "_usd": (installed * notional, held["net"] * notional, (held["net"] * notional) if final > 0.0 else 0.0),
            }
        )

    day = [c for c in candidates if start <= c.t0 < end]
    calib = []
    for cand in day:
        x0 = cv.vector(cv.entry_position_state(cand.cost), cand.entry, cand.symbol, cand.setup, cand.regime)
        predicted = model.advantage(cand.t0, x0) - cand.cost
        outcome = simulate(model, cand, bars[cand.symbol], data_end)
        calib.append((cand, predicted, outcome))
    busy_until: dict[str, float] = {}
    policy = []
    for cand, predicted, outcome in calib:
        if predicted <= 0.0 or busy_until.get(cand.symbol, 0.0) > cand.t0:
            continue
        busy_until[cand.symbol] = cand.t0 + 60.0 * outcome["hold_min"]
        policy.append(
            {
                "time": datetime.fromtimestamp(cand.t0, timezone.utc).strftime("%H:%M:%S"),
                "symbol": cand.symbol[:3],
                "setup": cand.setup,
                "predicted_bps": round(predicted * 1e4, 1),
                "reason": outcome["reason"],
                "hold_min": outcome["hold_min"],
                "net_bps": round(outcome["net"] * 1e4, 1),
            }
        )

    conn = sqlite3.connect(db)
    legacy = {int(r[0]): json.loads(r[1] or "{}") for r in conn.execute("SELECT id, economic_json FROM adaptive_candidate_markouts WHERE evaluated_at>=? AND evaluated_at<?", (start, end))}
    conn.close()
    pred = np.asarray([p for _c, p, _o in calib])
    real = np.asarray([o["net"] for _c, _p, o in calib])
    leg = np.asarray([float(legacy.get(c.id, {}).get("policy_value") or 0.0) for c, _p, _o in calib])
    order = np.argsort(pred)
    deciles = [
        {"predicted_bps": round(float(pred[chunk].mean()) * 1e4, 1), "realized_bps": round(float(real[chunk].mean()) * 1e4, 1), "n": len(chunk)} for chunk in np.array_split(order, 10) if len(chunk)
    ]
    markouts = {}
    for horizon in cv.HORIZONS_SEC:
        keep = [(i, g - c.cost) for i, (c, _p, _o) in enumerate(calib) if (g := cv.horizon_gross(c, bars[c.symbol], -1, c.t0 + horizon)) is not None]
        if len(keep) < 3:
            continue
        idx = np.asarray([i for i, _ in keep])
        net = np.asarray([v for _, v in keep])
        markouts[str(horizon)] = {
            "n": len(keep),
            "mean_net_bps": round(float(net.mean()) * 1e4, 2),
            "positive_share": round(float((net > 0).mean()), 3),
            "corr_repaired": round(float(np.corrcoef(pred[idx], net)[0, 1]), 3) if pred[idx].std() > 0 else None,
            "corr_legacy": round(float(np.corrcoef(leg[idx], net)[0, 1]), 3) if leg[idx].std() > 0 else None,
            "legacy_positive_mean_net_bps": round(float(net[leg[idx] > 0].mean()) * 1e4, 2) if (leg[idx] > 0).any() else None,
        }
    usd = np.asarray([r.pop("_usd") for r in rows]) if rows else np.zeros((0, 3))
    return {
        "window": [start, end],
        "trades": rows,
        "summary_21": {
            "installed": {**_stats([r["installed_net_bps"] / 1e4 for r in rows]), "usd": round(float(usd[:, 0].sum()), 3)},
            "repaired_continuation_on_installed_entries": {
                **_stats([r["continuation_on_actual_entry"]["net_bps"] / 1e4 for r in rows]),
                "usd": round(float(usd[:, 1].sum()), 3),
                "exits_at_entry_check": sum(1 for r in rows if r["continuation_on_actual_entry"]["hold_min"] == 0 and r["continuation_on_actual_entry"]["reason"] == "LEARNED_EXIT"),
                "catastrophic": sum(1 for r in rows if r["continuation_on_actual_entry"]["reason"] == "CATASTROPHIC"),
            },
            "repaired_entry_and_continuation": {
                **_stats([r["repaired_net_bps"] / 1e4 for r in rows if r["admitted"]]),
                "admitted": sum(1 for r in rows if r["admitted"]),
                "usd": round(float(usd[:, 2].sum()), 3),
            },
        },
        "day_policy": {
            **_stats([p["net_bps"] / 1e4 for p in policy]),
            "catastrophic": sum(1 for p in policy if p["reason"] == "CATASTROPHIC"),
            "exits_at_entry_check": sum(1 for p in policy if p["hold_min"] == 0 and p["reason"] == "LEARNED_EXIT"),
            "entries": policy,
        },
        "entry_calibration_all_day_candidates": {
            "n": len(calib),
            "repaired_mean_predicted_bps": round(float(pred.mean()) * 1e4, 2) if len(pred) else None,
            "legacy_mean_final_bps": round(float(leg.mean()) * 1e4, 2) if len(leg) else None,
            "mean_realized_under_repaired_continuation_bps": round(float(real.mean()) * 1e4, 2) if len(real) else None,
            "corr_repaired": round(float(np.corrcoef(pred, real)[0, 1]), 3) if len(pred) > 2 else None,
            "corr_legacy": round(float(np.corrcoef(leg, real)[0, 1]), 3) if len(leg) > 2 and leg.std() > 0 else None,
            "legacy_positive": int((leg > 0).sum()),
            "repaired_positive": int((pred > 0).sum()),
            "deciles_by_repaired_prediction": deciles,
            "forward_markouts_after_cost": markouts,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.db, _epoch(args.start), _epoch(args.end)), indent=1, default=str))


if __name__ == "__main__":
    main()
