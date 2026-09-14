"""Setup-discovery and intent-validity flags.

Thresholds are locked from Ocean trailing-buy evidence (318 intents,
2026-09-13 22:24 UTC -> 2026-09-14, 0 FILLED). ETH live arm 2533.20 /
ATR 32.06 made the 14 bp dip equal 0.11 ATR — market noise, not a pullback.
"""

from __future__ import annotations

import os
from typing import Final

from backend.config.day_entry_execution import trailing_buy_max_wait_seconds

EARLY_TREND = "EARLY_TREND_CONTINUATION"
STRUCTURED_PULLBACK = "STRUCTURED_PULLBACK_RECLAIM"
REJECT_EXTENDED = "REJECT_EXTENDED"
REJECT_NO_SETUP = "REJECT_NO_SETUP"

# Ocean 14 bp dip / ETH ATR 126 bp = 0.11. Require a real fraction of ATR.
PULLBACK_ATR_MULT: Final[float] = 0.40
RECLAIM_ATR_MULT: Final[float] = 0.12
EARLY_ROOM_ATR_MULT: Final[float] = 0.50
EXTENDED_1H_HIGH_ATR: Final[float] = 0.25
EXTENDED_4H_RANGE_PCT: Final[float] = 0.80
CONSOLIDATION_BARS: Final[int] = 45
BREAK_CONFIRM_BARS: Final[int] = 8
VOLUME_CONFIRM_MULT: Final[float] = 1.15
COST_PULLBACK_MULT: Final[float] = 2.0
# One intent lifetime, not two. The trailing-buy expiry is authoritative; a second
# independent constant here cancels as STALE_INTENT_MAX_AGE before expiry is reached.
MAX_INTENT_AGE_SEC: Final[int] = trailing_buy_max_wait_seconds()
# EARLY_TREND _range_break counts asof bars. STRUCTURE/4h windows are 240m + 8m prior.
EARLY_TREND_MIN_BARS: Final[int] = CONSOLIDATION_BARS + BREAK_CONFIRM_BARS + 2
STRUCTURE_LOOKBACK_MINUTES: Final[int] = 240 + BREAK_CONFIRM_BARS
# feature_ohlcv persist-now is ~2 rows/min; keep 1-row/min coverage too.
SETUP_DISCOVERY_LOOKBACK_BARS: Final[int] = max(EARLY_TREND_MIN_BARS, STRUCTURE_LOOKBACK_MINUTES * 3)

SHADOW_EARLY_ENV: Final[str] = "DAY_EARLY_TREND_SHADOW"
SHADOW_PULLBACK_ENV: Final[str] = "DAY_STRUCTURED_PULLBACK_SHADOW"
ROUTE_ENV: Final[str] = "DAY_SETUP_DISCOVERY_ROUTE"
ENFORCE_ENV: Final[str] = "DAY_SETUP_VALIDITY_ENFORCE"


def _flag(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, "true" if default else "false") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def early_trend_shadow() -> bool:
    return _flag(SHADOW_EARLY_ENV, True)


def structured_pullback_shadow() -> bool:
    return _flag(SHADOW_PULLBACK_ENV, True)


def setup_discovery_route() -> bool:
    """When true, only A/B setups may arm. Default off (shadow labels only)."""
    return _flag(ROUTE_ENV, False)


def setup_validity_enforced() -> bool:
    """When true, observe/arm cancel on extension/stale/broken structure."""
    return _flag(ENFORCE_ENV, False)
