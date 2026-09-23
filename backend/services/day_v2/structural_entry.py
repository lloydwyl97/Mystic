"""DAY V2 structural entry zone evaluator.

For each supported setup, derives a price zone from the signal's structural
anchor and ATR. The zone defines the price band where a pullback entry is
structurally valid. The reclaim_level is the price that must be exceeded by
a 5m bar's close to confirm bullish intent.

Setup zone definitions
----------------------
HTF_TREND_PULLBACK:
    zone = [anchor, anchor + 1.5 * atr]
    reclaim = anchor + 1.5 * atr

BREAKOUT_CONTINUATION:
    zone = [anchor * 0.995, anchor * 1.005]
    reclaim = anchor

RANGE_BOUNCE:
    zone = [anchor, anchor + atr]
    reclaim = anchor + 0.5 * atr

VWAP_REVERSION:
    zone = [anchor, target * 0.998]
    reclaim = target * 0.998

EXHAUSTION_MR:
    zone = [anchor, anchor + 2 * atr]
    reclaim = anchor + atr

REVERSAL_BREAKOUT: UNSUPPORTED_NOT_IMPLEMENTED (constant-only; no zone).

Public API
----------
evaluate_structural_zone(signal: DayV2Signal) -> StructuralZone
price_in_structural_zone(current_price: float, zone: StructuralZone) -> bool
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.services.day_v2.live_signal import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_EXHAUSTION_MR,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_RANGE_BOUNCE,
    SETUP_VWAP_REVERSION,
    DayV2Signal,
)

# Sentinel returned when a signal has no computable structural entry level.
MISSING_STRUCTURAL_ENTRY_LEVEL: str = "MISSING_STRUCTURAL_ENTRY_LEVEL"


@dataclass(frozen=True)
class StructuralZone:
    """Price zone for a structural pullback entry.

    Attributes
    ----------
    valid:          False when the zone could not be computed (see reason).
    zone_low:       Lower boundary of the structural entry zone.
    zone_high:      Upper boundary of the structural entry zone.
    reclaim_level:  Close price above which 5m confirmation is valid.
    reason:         Human-readable explanation; MISSING_STRUCTURAL_ENTRY_LEVEL
                    or UNSUPPORTED_SETUP:<name> when valid=False.
    """

    valid: bool
    zone_low: float
    zone_high: float
    reclaim_level: float
    reason: str


def evaluate_structural_zone(signal: DayV2Signal) -> StructuralZone:
    """Compute the structural entry zone for a DAY V2 signal.

    Returns a StructuralZone with valid=False when the setup is unsupported,
    anchor or atr is missing/zero, or the signal cannot produce a meaningful
    zone.
    """
    setup = str(signal.setup or "")
    anchor = float(signal.structural_anchor or 0.0)
    atr = float(signal.atr or 0.0)
    target = float(signal.target_price or 0.0)

    if anchor <= 0.0:
        return StructuralZone(
            valid=False,
            zone_low=0.0,
            zone_high=0.0,
            reclaim_level=0.0,
            reason=MISSING_STRUCTURAL_ENTRY_LEVEL,
        )

    if setup == SETUP_HTF_TREND_PULLBACK:
        if atr <= 0.0:
            return StructuralZone(
                valid=False,
                zone_low=0.0,
                zone_high=0.0,
                reclaim_level=0.0,
                reason=MISSING_STRUCTURAL_ENTRY_LEVEL,
            )
        zone_low = anchor
        zone_high = anchor + 1.5 * atr
        reclaim = anchor + 1.5 * atr
        return StructuralZone(valid=True, zone_low=zone_low, zone_high=zone_high, reclaim_level=reclaim, reason="OK")

    if setup == SETUP_BREAKOUT_CONTINUATION:
        zone_low = anchor * 0.995
        zone_high = anchor * 1.005
        reclaim = anchor
        return StructuralZone(valid=True, zone_low=zone_low, zone_high=zone_high, reclaim_level=reclaim, reason="OK")

    if setup == SETUP_RANGE_BOUNCE:
        if atr <= 0.0:
            return StructuralZone(
                valid=False,
                zone_low=0.0,
                zone_high=0.0,
                reclaim_level=0.0,
                reason=MISSING_STRUCTURAL_ENTRY_LEVEL,
            )
        zone_low = anchor
        zone_high = anchor + atr
        reclaim = anchor + 0.5 * atr
        return StructuralZone(valid=True, zone_low=zone_low, zone_high=zone_high, reclaim_level=reclaim, reason="OK")

    if setup == SETUP_VWAP_REVERSION:
        if target <= 0.0:
            return StructuralZone(
                valid=False,
                zone_low=0.0,
                zone_high=0.0,
                reclaim_level=0.0,
                reason=MISSING_STRUCTURAL_ENTRY_LEVEL,
            )
        zone_low = anchor
        zone_high = target * 0.998
        reclaim = target * 0.998
        return StructuralZone(valid=True, zone_low=zone_low, zone_high=zone_high, reclaim_level=reclaim, reason="OK")

    if setup == SETUP_EXHAUSTION_MR:
        if atr <= 0.0:
            return StructuralZone(
                valid=False,
                zone_low=0.0,
                zone_high=0.0,
                reclaim_level=0.0,
                reason=MISSING_STRUCTURAL_ENTRY_LEVEL,
            )
        zone_low = anchor
        zone_high = anchor + 2.0 * atr
        reclaim = anchor + atr
        return StructuralZone(valid=True, zone_low=zone_low, zone_high=zone_high, reclaim_level=reclaim, reason="OK")

    # Unknown / unsupported setup (includes REVERSAL_BREAKOUT)
    return StructuralZone(
        valid=False,
        zone_low=0.0,
        zone_high=0.0,
        reclaim_level=0.0,
        reason=f"UNSUPPORTED_SETUP:{setup}",
    )


def price_in_structural_zone(current_price: float, zone: StructuralZone) -> bool:
    """Return True when current_price falls within the structural entry zone."""
    if not zone.valid or zone.zone_low <= 0.0 or zone.zone_high <= 0.0:
        return False
    return zone.zone_low <= current_price <= zone.zone_high
