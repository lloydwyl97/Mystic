"""Causal DAY V2 lifecycle label for a qualified candidate.

A fixed-horizon markout asks where price was N minutes after the decision. A
DAY position is not held for a fixed time: the live exit contract (catastrophic
protection, structural invalidation, structure runner) decides when it leaves.
This walks closed 1m bars after the decision through that same contract
(``evaluate_day_v2_exit`` with an explicit clock) and returns the net return a
fill at the decision ask would have realized after the canonical round-trip
cost. A position still open at ``DAY_LIFECYCLE_MAX_MIN`` is marked at that
bar's close and flagged censored.

Only bars opening at or after the decision are read, and a label is final only
once its exit or its horizon has passed, so it can never inform the decision
that produced it.
"""

from __future__ import annotations

import json
import math
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from typing import Any

DAY_LIFECYCLE_MAX_MIN = 720.0
# A horizon with missing bars is closed at the last available close once this
# much time has passed beyond it; with no bars at all it closes unlabeled.
DAY_LIFECYCLE_GRACE_SEC = 900.0

Bar = tuple[float, float, float, float, float]  # (open_epoch, open, high, low, close)


@dataclass(frozen=True)
class LifecycleParams:
    """Entry-time exit inputs, the same fields the live position is stamped with."""

    setup: str
    entry_price: float
    entry_time: float
    atr_15m: float
    structural_anchor: float
    target_price: float
    atr_1h: float = 0.0
    objective_structural: float = 0.0
    objective_atr_mult: float = 1.0
    structural_emphasis: float = 1.0
    runner_activation_mult: float = 1.0
    runner_trail_mult: float = 1.0
    runner_tighten_mult: float = 1.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str | None) -> LifecycleParams | None:
        try:
            data = json.loads(raw or "")
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict):
            return None
        names = {f.name for f in fields(cls)}
        try:
            return cls(**{k: v for k, v in data.items() if k in names})
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_signal(cls, signal: Any, *, entry_price: float, entry_time: float, adaptive: dict | None = None) -> LifecycleParams:
        adapt = adaptive if isinstance(adaptive, dict) else {}

        def mult(key: str) -> float:
            try:
                return float(adapt.get(key) or 1.0)
            except (TypeError, ValueError):
                return 1.0

        return cls(
            setup=str(getattr(signal, "setup", "") or ""),
            entry_price=float(entry_price),
            entry_time=float(entry_time),
            atr_15m=float(getattr(signal, "atr", 0.0) or 0.0),
            structural_anchor=float(getattr(signal, "structural_anchor", 0.0) or 0.0),
            target_price=float(getattr(signal, "target_price", 0.0) or 0.0),
            atr_1h=float(getattr(signal, "atr_1h", 0.0) or 0.0),
            objective_structural=float(getattr(signal, "objective_structural", 0.0) or 0.0),
            objective_atr_mult=mult("objective_atr_mult"),
            structural_emphasis=mult("structural_emphasis"),
            runner_activation_mult=mult("runner_activation_mult"),
            runner_trail_mult=mult("runner_trail_mult"),
            runner_tighten_mult=mult("runner_tighten_mult"),
        )


def simulate_lifecycle(
    params: LifecycleParams,
    bars_1m: Sequence[Bar],
    *,
    roundtrip_cost: float,
    max_minutes: float = DAY_LIFECYCLE_MAX_MIN,
    now: float | None = None,
) -> dict[str, Any]:
    """Run the live DAY exit contract over ``bars_1m`` (ascending by open time).

    Returns ``{"final": False}`` while the exit and the horizon are both still
    ahead. A final label carries ``net`` (None when no bar was available),
    the exit reason, time and price, MFE/MAE from the entry price, and
    ``censored`` when the horizon closed the position instead of an exit.
    """
    from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit

    entry = float(params.entry_price or 0.0)
    clock = float(now if now is not None else time.time())
    horizon_end = float(params.entry_time) + float(max_minutes) * 60.0
    if entry <= 0 or not math.isfinite(entry):
        return {"final": True, "net": None, "reason": "NO_ENTRY_PRICE"}
    low, high = math.inf, entry
    last_close: float | None = None
    last_t: float | None = None
    for bar_open_time, b_open, b_high, b_low, b_close in bars_1m:
        t_open = float(bar_open_time)
        if t_open < float(params.entry_time) - 1e-6:
            continue
        t_close = t_open + 60.0
        if t_close > horizon_end + 1e-6 or t_close > clock + 1e-6:
            break
        low = min(low, float(b_low))
        high = max(high, float(b_high))
        last_close, last_t = float(b_close), t_close
        decision = evaluate_day_v2_exit(
            engine_id="DAY_V2",
            entry_price=entry,
            current_price=float(b_close),
            bar_low=low,
            highest_price=high,
            atr_at_entry=float(params.atr_15m),
            structural_anchor=float(params.structural_anchor),
            target_price=float(params.target_price),
            entry_time=float(params.entry_time),
            estimated_roundtrip_cost=float(roundtrip_cost),
            setup=str(params.setup),
            atr_1h_at_entry=float(params.atr_1h),
            objective_structural=float(params.objective_structural),
            objective_atr_mult=float(params.objective_atr_mult),
            structural_emphasis=float(params.structural_emphasis),
            runner_activation_mult=float(params.runner_activation_mult),
            runner_trail_mult=float(params.runner_trail_mult),
            runner_tighten_mult=float(params.runner_tighten_mult),
            now=t_close,
        )
        if decision:
            exit_px = float(decision.get("exit_price_estimate") or b_close)
            if decision.get("reason") == "DAY_V2_CATASTROPHIC_PROTECTION":
                # A bar that opened through the threshold fills at its open.
                exit_px = min(exit_px, float(b_open))
            return _label(params, entry, exit_px, t_close, str(decision.get("reason") or ""), high, low, roundtrip_cost, censored=False)
    if last_t is not None and last_t >= horizon_end - 1e-6:
        return _label(params, entry, float(last_close), last_t, "HORIZON_MARK", high, low, roundtrip_cost, censored=True)
    if clock < horizon_end + DAY_LIFECYCLE_GRACE_SEC:
        return {"final": False}
    if last_close is None:
        return {"final": True, "net": None, "reason": "NO_BARS"}
    return _label(params, entry, float(last_close), float(last_t or horizon_end), "HORIZON_MARK_PARTIAL", high, low, roundtrip_cost, censored=True)


def _label(params: LifecycleParams, entry: float, exit_px: float, exit_t: float, reason: str, high: float, low: float, cost: float, *, censored: bool) -> dict[str, Any]:
    return {
        "final": True,
        "net": (exit_px - entry) / entry - float(cost),
        "gross": (exit_px - entry) / entry,
        "reason": reason,
        "exit_price": exit_px,
        "exit_time": exit_t,
        "minutes": max(0.0, (exit_t - float(params.entry_time)) / 60.0),
        "mfe": (high - entry) / entry,
        "mae": (min(low, entry) - entry) / entry if math.isfinite(low) else 0.0,
        "censored": censored,
    }


def ohlcv_bars_1m(db_path: str, symbol: str, start: float, end: float) -> list[Bar]:
    """1m bars opening in [floor(start to minute), end), ascending. Empty on any error."""
    raw = str(symbol or "").upper().replace("-", "").replace("/", "")
    lo = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(math.floor(float(start) / 60.0) * 60.0))
    hi = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(float(end)))
    from backend.services.candle_contract import _parse_ts

    try:
        conn = sqlite3.connect(db_path, timeout=5)
    except sqlite3.Error:
        return []
    try:
        for variant in (raw.replace("USDT", "-USDT"), raw, raw.replace("USDT", "/USDT")):
            rows = conn.execute(
                "SELECT ts, open, high, low, close FROM feature_ohlcv WHERE symbol=? AND interval='1m' AND ts>=? AND ts<? ORDER BY ts",
                (variant, lo, hi),
            ).fetchall()
            if rows:
                out: list[Bar] = []
                for ts, o, h, low, c in rows:
                    opened = _parse_ts(ts)
                    if opened is None or None in (o, h, low, c):
                        continue
                    out.append((float(opened), float(o), float(h), float(low), float(c)))
                return out
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    return []


__all__ = ["DAY_LIFECYCLE_MAX_MIN", "LifecycleParams", "ohlcv_bars_1m", "simulate_lifecycle"]
