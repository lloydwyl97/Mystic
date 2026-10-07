"""Causal alignment of a research label to the horizon it names.

The price at a requested horizon is the latest executable observation whose
timestamp is at or before that horizon and no older than the horizon's
tolerance. An observation after the horizon is never used. A 1-minute bar
close is known at ``open + 60``, not at the open, and 30s / 60s labels do
not accept a bar close at all: when no bid/ask/tick falls inside the
tolerance the label is ``MISSING``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

LABEL_VERSION = "HORIZON_LABELS_V2"
SHORT_HORIZONS: tuple[int, ...] = (30, 60)
SCALP_HORIZONS: tuple[int, ...] = (30, 60, 120, 300, 600, 1200)
DAY_HORIZONS: tuple[int, ...] = (900, 1800, 3600, 7200, 14400, 21600, 43200)
BAR_CLOSE_SOURCE = "bar_close"
# 30s and 60s are executable-only. Longer horizons may use a bar close whose
# known-at time (open + interval) sits inside the tolerance.
BAR_CLOSE_MIN_HORIZON = 120
# How early an observation may be and still count as the price at the horizon.
# Nothing later than the horizon is accepted (late tolerance is zero).
MAX_EARLY_SEC: dict[int, float] = {
    30: 5.0,
    60: 5.0,
    120: 15.0,
    300: 30.0,
    600: 30.0,
    1200: 60.0,
    900: 60.0,
    1800: 60.0,
    3600: 60.0,
    7200: 60.0,
    14400: 120.0,
    21600: 120.0,
    43200: 120.0,
}


def max_early_sec(horizon: float) -> float:
    h = int(horizon)
    if h in MAX_EARLY_SEC:
        return MAX_EARLY_SEC[h]
    return 60.0 if h >= 120 else 5.0


def bar_close_known_at(bar_open: float, interval_sec: float = 60.0) -> float:
    """When a bar's close price became knowable. Not the bar's open stamp."""
    return float(bar_open) + float(interval_sec)


def align_observation(points: Sequence[tuple[float, float, str]], target: float, horizon: float) -> dict[str, Any]:
    """Pick the observation for ``target`` or return status ``MISSING``.

    ``points`` are ``(obs_ts, price, source)``. ``obs_ts`` is when the price
    was knowable. The chosen point is the latest one with ``obs_ts <= target``
    and ``target - obs_ts <= max_early``. A later point is ignored, not used
    as a fallback.
    """
    early = max_early_sec(horizon)
    allow_bar = float(horizon) >= BAR_CLOSE_MIN_HORIZON
    best: tuple[float, float, str] | None = None
    for ts, price, source in points:
        if not math.isfinite(ts) or not math.isfinite(price) or price <= 0:
            continue
        if source == BAR_CLOSE_SOURCE and not allow_bar:
            continue
        if ts > target:
            continue
        if target - ts > early:
            continue
        if best is None or ts > best[0]:
            best = (float(ts), float(price), str(source))
    if best is None:
        return {"status": "MISSING", "horizon_sec": int(horizon), "target_ts": float(target), "obs_ts": None, "timing_error_sec": None, "source": None, "price": None, "version": LABEL_VERSION}
    return {
        "status": "OK",
        "horizon_sec": int(horizon),
        "target_ts": float(target),
        "obs_ts": best[0],
        "timing_error_sec": best[0] - float(target),
        "source": best[2],
        "price": best[1],
        "version": LABEL_VERSION,
    }


def executable_gross(entry_ask: float, exit_bid: float) -> float | None:
    """Long liquidation: bid at the horizon over the ask at the decision."""
    if entry_ask <= 0 or exit_bid <= 0:
        return None
    return float(exit_bid) / float(entry_ask) - 1.0


def rebuild_disposition(old_present: bool, new_status: str) -> str:
    """How a stored short-horizon label changes once it is realigned.

    A missing old label stays ``unchanged``. A present one is ``replaced``
    when an aligned executable price exists and ``removed`` when it does not.
    """
    if not old_present:
        return "unchanged"
    return "replaced" if new_status == "OK" else "removed"


__all__ = [
    "BAR_CLOSE_MIN_HORIZON",
    "BAR_CLOSE_SOURCE",
    "DAY_HORIZONS",
    "LABEL_VERSION",
    "MAX_EARLY_SEC",
    "SCALP_HORIZONS",
    "SHORT_HORIZONS",
    "align_observation",
    "bar_close_known_at",
    "executable_gross",
    "max_early_sec",
    "rebuild_disposition",
]
