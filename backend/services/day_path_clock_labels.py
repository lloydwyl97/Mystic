"""True clock-horizon research labels. Separate from production exits.

Primary target is expected executable net bps. HOLD is exactly 0.
A production exit is stored alongside clock markouts and is never treated
as the same target as a fixed-horizon counterfactual.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from backend.services.day_4h_entry_features import HOLD_SYMBOL
from backend.services.day_path_clock_features import ClockBar, clip_asof, close_at_or_before, normalize_bars, parse_as_of
from backend.services.day_path_clock_v2 import (
    CLOCK_LABEL_HORIZONS_SEC,
    EXECUTABLE_PRICE_METHOD,
    PRIMARY_TARGET,
    SCHEMA_VERSION,
)
from backend.services.day_path_input_validity import MAX_LAST_BAR_AGE_SEC


def _bps(start: float, end: float) -> float:
    return (end - start) / start * 1e4


def hold_clock_labels() -> dict[str, Any]:
    nets = dict.fromkeys(CLOCK_LABEL_HORIZONS_SEC, 0.0)
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": HOLD_SYMBOL,
        "target": PRIMARY_TARGET,
        "hold_value_bps": 0.0,
        "clock_gross_bps": dict(nets),
        "clock_net_bps": dict(nets),
        "label_horizon_seconds": dict(CLOCK_LABEL_HORIZONS_SEC),
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "slippage_bps": 0.0,
        "all_in_cost_bps": 0.0,
        "provenance": "hold_zero",
        "counterfactual": False,
        "production_exit_net_bps": 0.0,
        "production_exit_reason": None,
        "same_target_as_production_exit": False,
    }


def clock_markout_net(
    bars: Any,
    *,
    decision_ts: Any,
    horizon_sec: int,
    entry_px: float | None = None,
    cost_bps: float = 0.0,
    max_age_sec: float = MAX_LAST_BAR_AGE_SEC,
) -> dict[str, Any]:
    when = parse_as_of(decision_ts)
    if when is None or (entry_px is not None and entry_px <= 0):
        return {"gross_bps": None, "net_bps": None, "exit_px": None, "market_data_cutoff": None, "provenance": "unknown"}
    horizon_at = when + timedelta(seconds=int(horizon_sec))
    clipped = clip_asof(normalize_bars(bars), horizon_at)
    start = close_at_or_before(clipped, when, max_age_sec=max_age_sec)
    end = close_at_or_before(clipped, horizon_at, max_age_sec=max_age_sec)
    px0 = float(entry_px) if entry_px not in (None, "") else (start.close if start else None)
    if px0 is None or px0 <= 0 or end is None:
        return {
            "gross_bps": None,
            "net_bps": None,
            "exit_px": None,
            "market_data_cutoff": horizon_at.isoformat(),
            "provenance": "unknown",
        }
    # Future bars after the horizon must not be used; clip_asof already enforces that.
    if end.ts > horizon_at:
        return {"gross_bps": None, "net_bps": None, "exit_px": None, "market_data_cutoff": horizon_at.isoformat(), "provenance": "unknown"}
    gross = _bps(px0, end.close)
    return {
        "gross_bps": gross,
        "net_bps": gross - float(cost_bps),
        "exit_px": end.close,
        "entry_px": px0,
        "market_data_cutoff": horizon_at.isoformat(),
        "horizon_bar_ts": end.ts.isoformat(),
        "provenance": "reconstructed",
    }


def build_clock_labels(
    bars: Any,
    *,
    decision_ts: Any,
    symbol: str,
    cost_bps: float = 0.0,
    commission_bps: float = 0.0,
    spread_bps: float = 0.0,
    slippage_bps: float = 0.0,
    entry_px: float | None = None,
    production_exit_net_bps: float | None = None,
    production_exit_reason: str | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    if str(symbol or "").upper() == HOLD_SYMBOL:
        return hold_clock_labels()
    when = parse_as_of(decision_ts)
    clock_gross: dict[str, float | None] = {}
    clock_net: dict[str, float | None] = {}
    cutoffs: dict[str, str | None] = {}
    provenances: dict[str, str] = {}
    for name, sec in CLOCK_LABEL_HORIZONS_SEC.items():
        if now is not None and when is not None and now.timestamp() + 1e-9 < when.timestamp() + sec:
            clock_gross[name] = None
            clock_net[name] = None
            cutoffs[name] = None
            provenances[name] = "immature"
            continue
        row = clock_markout_net(bars, decision_ts=decision_ts, horizon_sec=sec, entry_px=entry_px, cost_bps=cost_bps)
        clock_gross[name] = row["gross_bps"]
        clock_net[name] = row["net_bps"]
        cutoffs[name] = row["market_data_cutoff"]
        provenances[name] = str(row["provenance"])
    return {
        "schema_version": SCHEMA_VERSION,
        "symbol": str(symbol or ""),
        "decision_timestamp": when.isoformat() if when else None,
        "target": PRIMARY_TARGET,
        "label_horizon_seconds": dict(CLOCK_LABEL_HORIZONS_SEC),
        "market_data_cutoff": cutoffs,
        "executable_price_method": EXECUTABLE_PRICE_METHOD,
        "commission_bps": float(commission_bps),
        "spread_bps": float(spread_bps),
        "slippage_bps": float(slippage_bps),
        "all_in_cost_bps": float(cost_bps),
        "clock_gross_bps": clock_gross,
        "clock_net_bps": clock_net,
        "horizon_provenance": provenances,
        "provenance": "reconstructed" if any(v is not None for v in clock_net.values()) else "unknown",
        "counterfactual": True,
        "production_exit_net_bps": production_exit_net_bps,
        "production_exit_reason": production_exit_reason,
        "same_target_as_production_exit": False,
        "primary_research_label": clock_net.get("4h"),
    }


def bars_have_future_leak(bars: list[ClockBar], cutoff: datetime) -> bool:
    return any(b.ts > cutoff for b in bars)
