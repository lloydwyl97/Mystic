"""Classify existing DAY blockers. Does not change production gates.

SAME_4H_THESIS_SLOT_CAP counts currently open intact-4H names plus pending
buys. After a close the slot is free, so it is concurrency, not a same-4H
re-entry lock.

ENTRY_EXIT_THESIS_INVALID_AT_ENTRY mixes:
- deterministic: mark already through the live invalidation / 4H-break / stop
- subjective: EMA/VWAP/setup alignment opinions
"""

from __future__ import annotations

from typing import Any

from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
    _bundle_tf_align,
    floor_invalidation_level,
    htf_4h_rise_broken,
    intact_4h_slot_blocked,
)


def classify_same_4h_thesis_slot_cap() -> dict[str, Any]:
    return {
        "code": "SAME_4H_THESIS_SLOT_CAP",
        "class": "concurrency",
        "limits_open_economic_positions": True,
        "blocks_reentry_after_close": False,
        "notes": "Counts currently open intact-4H positions and pending buys only.",
    }


def same_4h_slot_is_concurrency(*, open_intact: int, closed_this_4h: int, candidate_intact: bool) -> bool:
    """True when the cap is about open names, ignoring already-closed lots."""
    blocked = intact_4h_slot_blocked(open_intact=open_intact, candidate_intact=candidate_intact)
    reentry_would_block = intact_4h_slot_blocked(open_intact=max(0, open_intact - closed_this_4h), candidate_intact=candidate_intact)
    return blocked and not (closed_this_4h > 0 and not reentry_would_block and open_intact - closed_this_4h < open_intact)


def classify_thesis_invalid_components(
    *,
    setup: str,
    mark: float,
    invalid_level: float,
    bundle: dict[str, Any] | None,
    entry_vwap: float = 0.0,
    entry_price: float = 0.0,
    atr_pct: float = 0.01,
    spread_pct: float = 0.0,
) -> dict[str, Any]:
    deterministic: list[str] = []
    subjective: list[str] = []
    if mark > 0 and invalid_level > 0:
        if entry_price > 0:
            eff = floor_invalidation_level(entry_price, invalid_level, atr_pct=atr_pct, spread_pct=spread_pct)
            if mark < eff:
                deterministic.append("price_through_invalidation_level")
        elif mark < invalid_level:
            deterministic.append("price_through_invalidation_level")
    if isinstance(bundle, dict) and setup == SETUP_BREAKOUT_CONTINUATION and htf_4h_rise_broken(bundle):
        deterministic.append("4h_break_already_live")
    if setup == SETUP_HTF_TREND_PULLBACK and isinstance(bundle, dict):
        h1 = _bundle_tf_align(bundle, "1h")
        h4 = _bundle_tf_align(bundle, "4h")
        if h1 is not None and h4 is not None and h1 < 0.38 and h4 < 0.40:
            subjective.append("ema_alignment_1h_4h")
    if setup == SETUP_VWAP_REVERSION:
        if entry_vwap > 0 and mark < entry_vwap * 0.993:
            subjective.append("vwap_extension")
        if isinstance(bundle, dict):
            m5 = _bundle_tf_align(bundle, "5m")
            if m5 is not None and m5 < 0.35:
                subjective.append("ema_alignment_5m")
    if setup == SETUP_BREAKOUT_CONTINUATION and isinstance(bundle, dict) and not htf_4h_rise_broken(bundle):
        m5 = _bundle_tf_align(bundle, "5m")
        m15 = _bundle_tf_align(bundle, "15m")
        if m5 is not None and m15 is not None and m5 < 0.42 and m15 < 0.45:
            subjective.append("ema_alignment_5m_15m")
    rank_delta = -0.04 * len(subjective)
    return {
        "deterministic": deterministic,
        "subjective": subjective,
        "keep_as_hard_block": bool(deterministic),
        "subjective_rank_delta": rank_delta if subjective and not deterministic else 0.0,
    }
