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

# Entry policy version for direct live entries (no trailing-buy wait).
# Distinct from DAY_STRUCTURAL_PULLBACK_V1 so forensics can separate the
# direct-entry book from the historical trailing-buy book.
DAY_DIRECT_ENTRY_V1: str = "DAY_DIRECT_ENTRY_V1"

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


async def submit_day_v2_direct_entry(
    engine: Any,
    *,
    signal: DayV2Signal,
    ask_price: float,
    quantity: float,
    stop_price: float = 0.0,
    structural_zone: Any | None = None,
    reclaim_level: float = 0.0,
    db_symbol: str = "",
    decision_id: str = "",
    sleeve: str = "",
) -> dict | None:
    """Submit a DAY V2 live BUY immediately after a qualified setup.

    No WAIT_DIP, no dip/rebound/retention bracket, no 5m confirmation, no
    entry TTL. Hard safety is identical to the trailing-submit path: the
    shared ``_pre_submit_safety`` gate runs unchanged, and the order flows
    through ``execute_buy_fifo`` with ``DAY_V2_CONFIRMED`` authority.

    Returns the fill result dict, or None when blocked/rejected (the
    engine's ``last_buy_reject_reason`` carries the durable reason).
    """
    if not DAY_V2_ENABLED:
        raise RuntimeError("DAY_V2_ENABLED is False — live entry disabled")

    import uuid as _uuid

    from backend.config.day_entry_execution import ENTRY_AUTHORITY_DAY_V2_CONFIRMED
    from backend.services.day_trailing_buy import _pre_submit_safety

    symbol = str(signal.symbol or "")
    ask = float(ask_price or 0.0)
    qty = float(quantity or 0.0)
    if ask <= 0 or qty <= 0:
        return None
    did = str(decision_id or _uuid.uuid4())

    # Same hard-safety gate as the trailing-submit path (kill switch, pause,
    # _can_open_position cash/slots, open position, pending buy). The pseudo
    # intent carries live pricing so cash/duplicate checks see real values.
    pseudo_intent: dict[str, Any] = {
        "symbol": symbol,
        "decision_id": did,
        "notional_usd": ask * qty,
        "quantity": qty,
        "stop_price": float(stop_price or 0.0),
        "atr": float(signal.atr or 0.0),
        "confidence": 1.0,
        "bar_timestamp": int(signal.signal_bar_ts or 0),
        "thesis_invalid_level": float(signal.structural_anchor or 0.0),
        "sleeve": str(sleeve or ""),
        "engine_id": DAY_V2_ENGINE_ID,
    }
    safe, safety_reason = await _pre_submit_safety(engine, pseudo_intent, ask)
    if not safe:
        engine.last_buy_outcome = f"DIRECT_SUBMIT_BLOCKED:{safety_reason}"
        logger.info("DAY_V2_DIRECT_SUBMIT_BLOCKED symbol=%s reason=%s", symbol, safety_reason)
        return None

    # Two-engine capital allocator: DAY deploys only within its remaining
    # engine budget; physical free USDT (net of both engines' reservations)
    # must cover the order. Grandfathered SCALP lots are never touched.
    from backend.services.two_engine_capital import check_engine_budget

    _budget_ok, _budget_reason, _ = check_engine_budget(
        str(getattr(engine, "db_path", "") or ""),
        DAY_V2_ENGINE_ID,
        float(ask * qty),
        float(getattr(engine, "_total_equity", 0) or 0),
        float(getattr(engine, "_available_balance", 0) or 0),
        getattr(engine, "open_positions", None) or {},
    )
    if not _budget_ok:
        engine.last_buy_outcome = f"DIRECT_SUBMIT_BLOCKED:{_budget_reason}"
        try:
            engine.last_buy_reject_reason = _budget_reason
        except Exception:
            pass
        logger.info("DAY_V2_DIRECT_SUBMIT_BLOCKED symbol=%s reason=%s", symbol, _budget_reason)
        return None

    from backend.services.portfolio_engine import TradeExplainability

    exp = TradeExplainability(
        trade_id="",
        symbol=symbol,
        side="BUY",
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    exp.setup_type = str(signal.setup or "")
    exp.entry_thesis = str(signal.setup or "")
    exp.thesis_invalid_level = float(signal.structural_anchor or 0.0)
    exp.thesis_target_level = float(signal.target_price or 0.0)
    exp.regime = str(signal.regime or "unknown")
    exp.decision_id = did
    exp.ai_confidence = 1.0
    exp.entry_provenance = {
        "entry_policy_version": DAY_DIRECT_ENTRY_V1,
        "model_version": "day_deterministic_v1",
        "selected_action": f"BUY_{symbol}",
        "selection_reason": "QUALIFIED_SETUP_DIRECT_ENTRY",
        "strategy": "day",
        "setup": str(signal.setup or ""),
        "regime": str(signal.regime or ""),
        "structural_anchor": float(signal.structural_anchor or 0.0),
        "thesis_target_level": float(signal.target_price or 0.0),
        "signal_bar_ts": int(signal.signal_bar_ts or 0),
        "reclaim_level": float(reclaim_level or 0.0),
        "db_symbol": str(db_symbol or symbol),
        "structural_zone_low": float(getattr(structural_zone, "zone_low", 0.0) or 0.0),
        "structural_zone_high": float(getattr(structural_zone, "zone_high", 0.0) or 0.0),
    }

    result = await engine.execute_buy_fifo(
        symbol=symbol,
        quantity=qty,
        price=ask,
        stop_price=float(stop_price or 0.0),
        atr=float(signal.atr or 0.0),
        confidence=1.0,
        bar_timestamp=int(signal.signal_bar_ts or 0),
        explainability=exp,
        decision_id=did,
        sleeve=str(sleeve or ""),
        entry_authority=ENTRY_AUTHORITY_DAY_V2_CONFIRMED,
        client_order_id=_uuid.uuid4().hex,
        trailing_buy_intent_id="",
        fill_engine_id=DAY_V2_ENGINE_ID,
        fill_opportunity_id=str(signal.opportunity_id or ""),
    )
    if not result:
        logger.info(
            "DAY_V2_DIRECT_SUBMIT_REJECTED symbol=%s setup=%s reject=%s",
            symbol,
            signal.setup,
            str(getattr(engine, "last_buy_reject_reason", "") or "UNSPECIFIED"),
        )
        return None

    # Mark the DAY V2 opportunity consumed so the same move cannot re-enter.
    try:
        from backend.services.day_v2.migrations import consume_opportunity

        consume_opportunity(
            str(getattr(engine, "db_path", "") or ""),
            str(signal.opportunity_id or ""),
            symbol,
            str(signal.setup or ""),
            trade_id=str(result.get("trade_id") or ""),
        )
    except Exception:
        logger.warning("DAY_V2_DIRECT_CONSUME_OPP_FAILED symbol=%s", symbol, exc_info=True)

    result = dict(result)
    result["entry_authority"] = ENTRY_AUTHORITY_DAY_V2_CONFIRMED
    result["entry_policy_version"] = DAY_DIRECT_ENTRY_V1
    result["decision_id"] = did
    engine.last_buy_outcome = "DAY_V2_DIRECT_FILLED"
    logger.warning(
        "DAY_V2_DIRECT_FILLED symbol=%s setup=%s opp=%s qty=%.8f price=%.6f order=%s policy=%s",
        symbol,
        signal.setup,
        signal.opportunity_id,
        float(result.get("quantity") or qty),
        float(result.get("price") or ask),
        str(result.get("order_id") or ""),
        DAY_DIRECT_ENTRY_V1,
    )
    return result
