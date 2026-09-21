"""Persist the controlling exit family and trail/stop evidence on every SELL."""

from __future__ import annotations

import time
from typing import Any


def build_exit_decision_evidence(
    *,
    controlling_exit_family: str,
    trail_info: dict[str, Any] | None = None,
    position: Any = None,
    executable_bid: float | None = None,
    submitted_price: float | None = None,
    fill_price: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    info = dict(trail_info or {})
    highest = float(info.get("high_water") or getattr(position, "highest_price", 0.0) or 0.0)
    raw_stop = float(info.get("ratchet_trail_price") or getattr(position, "trailing_stop_price", 0.0) or 0.0)
    cost_floor = float(info.get("cost_aware_floor") or 0.0)
    controlling_stop = info.get("executable_trailing_stop")
    activated = bool(info.get("trailing_stop_in_exit_authority") or getattr(position, "trail_activated", False))
    activation_ts = float(info.get("activation_ts") or getattr(position, "trail_activated_at", 0.0) or 0.0)
    activation_px = float(info.get("activation_price") or getattr(position, "trail_activation_price", 0.0) or 0.0)
    bid = float(executable_bid) if executable_bid not in (None, "") else None
    submitted = float(submitted_price) if submitted_price not in (None, "") else None
    filled = float(fill_price) if fill_price not in (None, "") else None
    gap = None
    if bid is not None and submitted is not None:
        gap = submitted - bid
    slip = None
    if filled is not None and submitted is not None:
        slip = submitted - filled
    return {
        "controlling_exit_family": str(controlling_exit_family or ""),
        "activation_state": "activated" if activated else "inactive",
        "activation_timestamp": activation_ts or None,
        "activation_price": activation_px or None,
        "high_water_source": str(info.get("high_water_source") or getattr(position, "trail_high_water_source", "") or "position.highest_price"),
        "high_water_price": highest or None,
        "raw_stop": raw_stop or None,
        "cost_aware_floor": cost_floor or None,
        "controlling_stop": float(controlling_stop) if controlling_stop else None,
        "executable_bid_at_decision": bid,
        "submitted_price": submitted,
        "fill_price": filled,
        "gap": gap,
        "slippage": slip,
        "evidence_ts": float(now_epoch if now_epoch is not None else time.time()),
    }


def stamp_trail_activation(position: Any, *, highest: float, now_epoch: float | None = None, source: str = "highest_price") -> None:
    if position is None:
        return
    if bool(getattr(position, "trail_activated", False)):
        return
    position.trail_activated = True
    position.trail_activated_at = float(now_epoch if now_epoch is not None else time.time())
    position.trail_activation_price = float(highest or 0.0)
    position.trail_high_water_source = source
