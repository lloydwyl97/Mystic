"""DAY V2 live entry management.

Creates trailing-buy intents in day_trailing_buy_intents with
engine_id='DAY_V2'. Namespaced configuration; no SCALP V2 defaults
are inherited.

Calibrated trailing-buy values are from the qualifying replay
(scripts/research/day_v2_replay.py @ a88479a):
  - min_dip: 20 bps (multi-hour setup, more patience than 14-bps scalp)
  - rebound:  6 bps (confirm reversal before arming)
  - hold ceiling: 300 min (5 hours)

One intent per symbol — the day_trailing_buy_store enforces this.
If SCALP V2 already holds a symbol-level intent, DAY V2 will not arm
that symbol until the existing intent expires or fills.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from backend.services.day_v2.config import (
    DAY_STRUCTURAL_PULLBACK_V1,
    DAY_V2_ENABLED,
    DAY_V2_MAX_HOLD_MINUTES,
)
from backend.services.day_v2.live_signal import DayV2Signal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DAY V2 entry constants (calibrated from qualifying replay — do not
# silently inherit the 14/4 bps SCALP V2 values)
# ---------------------------------------------------------------------------

DAY_V2_ENGINE_ID: str = "DAY_V2"

# Opportunity lifetime: 60 minutes from signal detection.
# Replaced DAY_V2_MAX_HOLD_MINUTES (300 min) for structural-pullback intents.
STRUCTURAL_OPPORTUNITY_LIFETIME_SEC: float = 3600.0  # 60 minutes

# Trailing-buy parameters: wait for a MIN_DIP decline then confirm a
# REBOUND before entering. Multi-hour setups warrant more patience.
DAY_V2_MIN_DIP_BPS: float = 20.0  # minimum decline from arm price (bps)
DAY_V2_REBOUND_BPS: float = 6.0  # minimum rebound from dip low (bps)

# Notional cap for a single DAY V2 position.
# Sized so that both DAY V2 and SCALP V2 can hold up to MAX_OPEN_POSITIONS
# positions concurrently without exceeding the established account risk.
import os as _os

DAY_V2_MAX_NOTIONAL_USD: float = float(_os.environ.get("DAY_V2_MAX_NOTIONAL_USD", "0"))
# 0 means "use calculate_position_size" (preferred path). Non-zero caps it.

ROUNDTRIP_COST_BPS: float = 6.0  # matches production trading_economics
SPREAD_BPS: float = 1.0


def create_day_v2_intent(
    db_path: str,
    signal: DayV2Signal,
    ask_price: float,
    quantity: float,
    *,
    structural_zone: Any | None = None,
    reclaim_level: float = 0.0,
    db_symbol: str = "",
) -> dict | None:
    """Arm a DAY V2 trailing-buy intent for the given signal.

    Returns the created intent row (dict) or None if blocked (e.g. another
    intent already active for this symbol, or DAY_V2_ENABLED is False).

    Raises:
        RuntimeError: if DAY_V2_ENABLED is False.
    """
    if not DAY_V2_ENABLED:
        raise RuntimeError("DAY_V2_ENABLED is False — live entry disabled")

    from backend.services.day_trailing_buy_store import create_intent

    intent_id = str(uuid.uuid4())
    decision_id = str(uuid.uuid4())
    now = time.time()
    expires_at = now + STRUCTURAL_OPPORTUNITY_LIFETIME_SEC  # 60-minute structural window

    # payload carries DAY V2-specific metadata that doesn't have a dedicated
    # column in day_trailing_buy_intents (target level, opportunity linkage).
    payload: dict = {
        "thesis_target_level": signal.target_price,
        "day_opportunity_id": signal.opportunity_id,
        "setup": signal.setup,
        "regime": signal.regime,
        "h1_bullish": signal.h1_bullish,
        "signal_bar_ts": signal.signal_bar_ts,
        "atr_at_signal": signal.atr,
        # Structural-pullback fields for 5m confirmation gate
        "reclaim_level": reclaim_level,
        "db_symbol": db_symbol or signal.symbol,
    }

    # create_intent reads symbol and decision_id from the fields dict.
    # Derive structural_entry_level from zone if provided
    _structural_entry_level: float = 0.0
    if structural_zone is not None:
        _zone_low = getattr(structural_zone, "zone_low", None)
        if _zone_low is not None:
            _structural_entry_level = float(_zone_low)

    fields: dict = {
        # Required by create_intent
        "symbol": signal.symbol,
        "decision_id": decision_id,
        "intent_id": intent_id,
        # Identity
        "engine_id": DAY_V2_ENGINE_ID,
        # Entry policy
        "policy_version": DAY_STRUCTURAL_PULLBACK_V1,
        # Structural entry level (for telemetry and 5m confirmation gate)
        "structural_entry_level": _structural_entry_level,
        # Re-use scalp_opportunity_id column for the DAY opportunity ID.
        # The column name is a legacy artefact; the value here is the
        # canonical DAY V2 opportunity identifier.
        "scalp_opportunity_id": signal.opportunity_id,
        # Setup / thesis
        "setup": signal.setup,
        "thesis_invalid_level": signal.structural_anchor,
        "atr": signal.atr,
        # Pricing at arm time
        "arm_ts": now,
        "arm_ask": ask_price,
        "arm_bid": ask_price * 0.9999,
        "arm_midpoint": ask_price,
        # Trailing-buy calibration
        "min_dip_bps": DAY_V2_MIN_DIP_BPS,
        "rebound_bps": DAY_V2_REBOUND_BPS,
        "required_improvement_bps": DAY_V2_MIN_DIP_BPS + DAY_V2_REBOUND_BPS,
        # Cost model
        "round_trip_cost_bps": ROUNDTRIP_COST_BPS,
        "spread_bps": SPREAD_BPS,
        # Sizing
        "quantity": quantity,
        "notional_usd": ask_price * quantity,
        # Lifecycle
        "expires_at": expires_at,
        "bar_timestamp": signal.signal_bar_ts,
        # Scoring (DAY V2 does not use ML model score here)
        "decision_score": 1.0,
        "predicted_ev": 0.0,
        "confidence": 1.0,
        # Payload (carries thesis_target_level and other metadata)
        "payload": payload,
    }

    ok, reason, intent = create_intent(db_path, fields=fields)

    if not ok:
        logger.info(
            "DAY_V2_INTENT_NOT_ARMED symbol=%s reason=%s",
            signal.symbol,
            reason,
        )
        return None

    logger.warning(
        "DAY_V2_INTENT_ARMED symbol=%s setup=%s opp=%s anchor=%.6f target=%.6f ask=%.6f qty=%.8f min_dip=%.0f rebound=%.0f expires_in=%.0fs policy=%s intent_id=%s",
        signal.symbol,
        signal.setup,
        signal.opportunity_id,
        signal.structural_anchor,
        signal.target_price,
        ask_price,
        quantity,
        DAY_V2_MIN_DIP_BPS,
        DAY_V2_REBOUND_BPS,
        expires_at - now,
        DAY_STRUCTURAL_PULLBACK_V1,
        str(intent.get("intent_id") or intent_id),
    )
    return intent
