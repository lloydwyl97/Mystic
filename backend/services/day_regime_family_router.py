"""
DAY regime-family paper router.

Selects the active sleeve for paper trading based on regime, without enabling live money.

Sleeves (expanded to generate trades + learnable outcomes instead of permanent flat):
- TREND_BREAKOUT_PULLBACK_SLEEVE : BREAKOUT / TREND_PULLBACK in bull (and limited neutral)
- NEUTRAL_VWAP_REVERSION_SLEEVE : VWAP reversion or RANGE_BOUNCE in range/neutral
- BEAR_REVERSAL_SLEEVE : FAILED_BREAKDOWN_REVERSAL (and limited other reversals) in bear/trend_down
- SCALP : separate, regime-filtered, always paper-only for now

Paper and (future) live use identical rules. No repair-add, no mixing with scalp ledger.
"""

from __future__ import annotations

from typing import Any

from backend.services.day_regime_router import (
    DAY_REGIME_BEAR,
    DAY_REGIME_BULL,
    DAY_REGIME_CHOP,
    DAY_REGIME_NEUTRAL,
    DAY_REGIME_RANGE,
    classify_day_regime,
    evaluate_day_entry_route,
)
from backend.services.day_trade_thesis import (
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_RANGE_BOUNCE,
    SETUP_VWAP_REVERSION,
)

SLEEVE_TREND_BREAKOUT = "TREND_BREAKOUT_PULLBACK_SLEEVE"
SLEEVE_NEUTRAL_VWAP = "NEUTRAL_VWAP_REVERSION_SLEEVE"
SLEEVE_BEAR_FLAT = "BEAR_FLAT_SLEEVE"
SLEEVE_NONE = "NO_ACTIVE_SLEEVE"

PAPER_FAMILY_ROUTER_ENABLED = True  # paper routing gate (live remains blocked elsewhere)


def _is_vwap_replay_proven_range(regime: str, decision_data: dict[str, Any]) -> bool:
    """Conservative: only allow VWAP sleeve where the router already permits it and adx low enough."""
    if regime not in (DAY_REGIME_RANGE, DAY_REGIME_NEUTRAL):
        return False
    adx = float(decision_data.get("adx") or 20.0)
    # The strict reclaim checks are in evaluate_day_entry_route; we just gate the sleeve here.
    return adx <= 28.0


def choose_paper_sleeve(
    *,
    decision_data: dict[str, Any],
    context_payload: dict[str, Any] | None = None,
    price_structure_regime: str = "unknown",
    chop_score: float = 0.5,
    atr_ratio: float = 0.0,
) -> dict[str, Any]:
    """
    Returns which sleeve should be active for this bar/symbol, and whether a trade is allowed under that sleeve.

    Detects pre-stamped regime (from AW adapter) and the actual setup (allweather_setup / setup_type / entry_thesis).
    Uses the real setup for route evaluation so reversal and bounce can be allowed in their regimes.
    """
    dd = decision_data or {}

    # Pre-stamped regime from AW adapter takes precedence for new reversal/bounce paths
    pre = str(dd.get("day_route_regime") or dd.get("allweather_regime") or "").lower()
    if "bear" in pre or "trend_down" in pre:
        regime = DAY_REGIME_BEAR
    elif "range" in pre:
        regime = DAY_REGIME_RANGE
    else:
        regime = classify_day_regime(
            dd,
            context_payload=context_payload,
            chop_score=chop_score,
            atr_ratio=atr_ratio,
            price_structure_regime=price_structure_regime,
        )

    # Effective setup from the actual signal (supports the new active setups)
    eff = str(dd.get("allweather_setup") or dd.get("setup_type") or dd.get("entry_thesis") or dd.get("setup") or "").strip().upper()

    is_reversal = "FAILED_BREAKDOWN_REVERSAL" in eff or "RECL" in eff
    is_bounce = "RANGE_BOUNCE" in eff

    if regime == DAY_REGIME_CHOP:
        return {
            "regime": regime,
            "sleeve": SLEEVE_NONE,
            "allowed": False,
            "reason": "REGIME_CHOP_NO_DAY",
            "notes": "Chop: no DAY entries; scalp separate and currently excluded from this paper router.",
        }

    if regime == DAY_REGIME_BEAR:
        # Use the actual reversal setup if provided; otherwise default to the reversal name for the sleeve
        setup_to_use = eff if is_reversal else SETUP_FAILED_BREAKDOWN_REVERSAL
        route = evaluate_day_entry_route(
            setup_type=setup_to_use,
            day_regime=regime,
            decision_data=dd,
            context_payload=context_payload,
            current_price=float(dd.get("price") or 0.0),
            thesis_score=float(dd.get("thesis_score") or 0.55),
            strategy_family="REVERSAL_IN_BEAR",
        )
        return {
            "regime": regime,
            "sleeve": "BEAR_REVERSAL_SLEEVE",
            "allowed": bool(route.get("allowed")),
            "reason": route.get("block_reason") or "BEAR_REVERSAL_EVALUATE",
            "route": route,
            "notes": "Bear: reversal longs only (failed breakdown / capitulation reclaim). Paper mirrors live.",
            "effective_setup": setup_to_use,
        }

    if regime == DAY_REGIME_BULL:
        # Trend sleeve (AW breakout/pullback mapped to bull)
        route = evaluate_day_entry_route(
            setup_type="HTF_TREND_PULLBACK",
            day_regime=regime,
            decision_data=dd,
            context_payload=context_payload,
            current_price=float(dd.get("price") or 0.0),
            thesis_score=float(dd.get("thesis_score") or 0.6),
            strategy_family=SLEEVE_TREND_BREAKOUT,
        )
        return {
            "regime": regime,
            "sleeve": SLEEVE_TREND_BREAKOUT,
            "allowed": bool(route.get("allowed")),
            "reason": route.get("block_reason") or "TREND_SLEEVE_EVALUATE",
            "route": route,
            "notes": "Trend sleeve active — breakout or trend-pullback only.",
        }

    if regime in (DAY_REGIME_RANGE, DAY_REGIME_NEUTRAL):
        # Choose the actual setup (bounce or vwap)
        setup_to_use = SETUP_RANGE_BOUNCE if is_bounce else SETUP_VWAP_REVERSION
        route = evaluate_day_entry_route(
            setup_type=setup_to_use,
            day_regime=regime,
            decision_data=dd,
            context_payload=context_payload,
            current_price=float(dd.get("price") or 0.0),
            thesis_score=float(dd.get("thesis_score") or 0.55),
            strategy_family=SLEEVE_NEUTRAL_VWAP,
        )
        allowed = bool(route.get("allowed"))
        return {
            "regime": regime,
            "sleeve": SLEEVE_NEUTRAL_VWAP,
            "allowed": allowed,
            "reason": route.get("block_reason") or "RANGE_VWAP_OR_BOUNCE_EVALUATE",
            "route": route,
            "notes": "Range/neutral: VWAP reversion or range bounce active for trading + learning (paper mirrors live).",
            "effective_setup": setup_to_use,
        }

    # default neutral-ish
    return {
        "regime": regime,
        "sleeve": SLEEVE_NONE,
        "allowed": False,
        "reason": "NO_SLEEVE_FOR_REGIME",
    }


def get_sleeve_definitions() -> dict[str, Any]:
    return {
        SLEEVE_TREND_BREAKOUT: {
            "description": "Breakout + trend pullback on 1h structure in uptrend/qualifying neutral. ATR brackets, 72h stop.",
            "regimes": ["bull", "qualifying_neutral"],
            "setups": ["BREAKOUT_CONTINUATION", "HTF_TREND_PULLBACK"],
            "from": "allweather_breakout_pullback_adapter (TREND_BREAKOUT_PULLBACK_SLEEVE)",
        },
        SLEEVE_NEUTRAL_VWAP: {
            "description": "Range/neutral: VWAP reversion or range bounce. Active to generate trades and learning data in non-trend markets. Same rules for paper and live.",
            "regimes": ["range", "neutral"],
            "setups": ["VWAP_REVERSION", "RANGE_BOUNCE"],
        },
        "BEAR_REVERSAL_SLEEVE": {
            "description": "Bear/trend_down: reversal longs (failed breakdown reclaim) only. Conservative ATR. Generates outcomes so the system can learn and improve instead of idling.",
            "regimes": ["bear"],
            "setups": ["FAILED_BREAKDOWN_REVERSAL"],
        },
    }
