"""
DAY entry regime router — uses existing signal/context fields only.

Routes top-four DAY longs by market structure before execution:
  bull   → trend pullback, breakout continuation
  range  → VWAP / range-low mean reversion only
  bear   → no normal longs; reversal breakout or exhaustion MR only
  chop   → no DAY entries (scalp engine separate)
"""

from __future__ import annotations

from typing import Any

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_FAILED_BREAKDOWN_REVERSAL,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_NO_CLEAR_THESIS,
    SETUP_RANGE_BOUNCE,
    SETUP_VWAP_REVERSION,
    _safe_float,
    _tf_align,
    parse_mtf_json,
)

STRATEGY_FAMILY_ALLWEATHER_BREAKOUT_PULLBACK = "ALLWEATHER_BREAKOUT_PULLBACK"

DAY_REGIME_BULL = "bull"
DAY_REGIME_RANGE = "range"
DAY_REGIME_BEAR = "bear"
DAY_REGIME_CHOP = "chop"
DAY_REGIME_NEUTRAL = "neutral"

ALL_DAY_REGIMES = (
    DAY_REGIME_BULL,
    DAY_REGIME_RANGE,
    DAY_REGIME_BEAR,
    DAY_REGIME_CHOP,
    DAY_REGIME_NEUTRAL,
)

# High ATR ratio → chop/volatility block for DAY (not scalp).
DAY_CHOP_ATR_RATIO = 0.032
DAY_CHOP_ADX_MAX = 17.0


def _mtf_bundle(decision_data: dict[str, Any], context_payload: dict[str, Any] | None) -> dict[str, Any]:
    mtf = parse_mtf_json(decision_data or {})
    if context_payload and isinstance(context_payload.get("mtf"), dict):
        mtf = {**mtf, **context_payload["mtf"]}
    return mtf


def classify_day_regime(
    decision_data: dict[str, Any],
    *,
    context_payload: dict[str, Any] | None = None,
    chop_score: float = 0.5,
    atr_ratio: float = 0.0,
    price_structure_regime: str = "unknown",
) -> str:
    """Classify DAY routing regime from existing HTF/ADX/vol/context fields."""
    dd = decision_data or {}
    mtf = _mtf_bundle(dd, context_payload)
    h1 = _tf_align(mtf, "1h") if isinstance(mtf.get("1h"), dict) else None
    h4 = _tf_align(mtf, "4h") if isinstance(mtf.get("4h"), dict) else None
    ema = _safe_float(dd.get("ema_alignment"), 0.5)
    adx = _safe_float(dd.get("adx"), 0.0)
    mr = str(dd.get("ctx_market_regime") or dd.get("market_regime") or "").strip().lower()

    if atr_ratio >= DAY_CHOP_ATR_RATIO:
        return DAY_REGIME_CHOP
    if adx > 0 and adx <= DAY_CHOP_ADX_MAX and float(chop_score or 0.5) >= 0.62:
        return DAY_REGIME_CHOP
    if adx > 0 and adx <= DAY_CHOP_ADX_MAX and price_structure_regime == "range_bound" and float(chop_score or 0.5) >= 0.58:
        return DAY_REGIME_CHOP

    if h1 is not None and h4 is not None and h1 >= 0.58 and h4 >= 0.52 and ema >= 0.55:
        return DAY_REGIME_BULL
    if "bear" in mr or "fear" in mr or "extreme fear" in mr:
        return DAY_REGIME_BEAR
    if h1 is not None and h4 is not None and h1 <= 0.42 and h4 <= 0.40:
        return DAY_REGIME_BEAR
    if adx > 0 and adx < 25 and price_structure_regime == "range_bound":
        return DAY_REGIME_RANGE
    if adx > 0 and adx < 22:
        return DAY_REGIME_RANGE
    return DAY_REGIME_NEUTRAL


def htf_allows_day_long(
    decision_data: dict[str, Any],
    *,
    setup_type: str,
    context_payload: dict[str, Any] | None = None,
    thesis_score: float = 0.0,
) -> tuple[bool, str]:
    """Higher-timeframe permission — 5m/15m bounce cannot override weak 1h/4h."""
    dd = decision_data or {}
    mtf = _mtf_bundle(dd, context_payload)
    h1 = _tf_align(mtf, "1h") if isinstance(mtf.get("1h"), dict) else None
    h4 = _tf_align(mtf, "4h") if isinstance(mtf.get("4h"), dict) else None
    m15 = _tf_align(mtf, "15m") if isinstance(mtf.get("15m"), dict) else None
    rsi = _safe_float(dd.get("rsi"), 50.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    score = float(thesis_score or 0.0)

    if h1 is not None and h1 >= 0.48:
        return True, "htf_1h_permission"
    if h4 is not None and h4 >= 0.50:
        return True, "htf_4h_permission"
    if setup_type == SETUP_BREAKOUT_CONTINUATION and score >= 0.68 and m15 is not None and m15 >= 0.55:
        return True, "htf_breakout_reversal_confirmed"
    if setup_type == SETUP_VWAP_REVERSION and rsi <= 35.0 and bb <= 0.28:
        return True, "htf_exhaustion_mr"
    if setup_type == SETUP_FAILED_BREAKDOWN_REVERSAL:
        # Reversal allowed on LTF reclaim signal even if HTF is weak (bear trap case)
        if (m15 is not None and m15 > 0.42) or rsi < 40:
            return True, "htf_reversal_ltf_reclaim"
        return True, "htf_reversal_bear_allowed"  # permit for activity + learning in down regimes
    if setup_type == SETUP_RANGE_BOUNCE and (bb <= 0.35 or rsi < 45):
        return True, "htf_range_bounce_ltf"
    return False, "htf_structure_denied"


def _range_vwap_reclaim_ok(decision_data: dict[str, Any], current_price: float) -> bool:
    dd = decision_data or {}
    vwap = _safe_float(dd.get("vwap"), 0.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    if vwap > 0 and current_price > 0:
        vwap_dist = (current_price - vwap) / vwap
        if vwap_dist <= -0.0008 and bb <= 0.55:
            return True
    return bool(bb <= 0.3 and _safe_float(dd.get("rsi"), 50.0) <= 38.0)


def _range_vwap_reclaim_strict(decision_data: dict[str, Any], current_price: float) -> bool:
    """Range VWAP: true range low + reclaim start, not mid-range chop."""
    dd = decision_data or {}
    vwap = _safe_float(dd.get("vwap"), 0.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    rsi = _safe_float(dd.get("rsi"), 50.0)
    mom = _safe_float(dd.get("price_momentum"), 0.0)
    if bb > 0.28 or rsi > 36.0:
        return False
    if mom > 0.025:
        return False
    if vwap > 0 and current_price > 0:
        vwap_dist = (current_price - vwap) / vwap
        if vwap_dist <= -0.0012 and bb <= 0.28:
            return True
    return bb <= 0.25 and rsi <= 34.0


def _vwap_expected_mfe_after_fees_ok(decision_data: dict[str, Any], current_price: float) -> bool:
    dd = decision_data or {}
    target = _safe_float(dd.get("thesis_target_level"), 0.0)
    if target <= 0 or current_price <= 0:
        return False
    gross_mfe = (target - current_price) / current_price
    return gross_mfe - ESTIMATED_ROUNDTRIP_COST >= MIN_NET_PROFIT_TO_SELL * 0.45


def _bear_reversal_breakout_ok(decision_data: dict[str, Any], thesis_score: float) -> bool:
    dd = decision_data or {}
    mtf = parse_mtf_json(dd)
    m15 = _tf_align(mtf, "15m") if isinstance(mtf.get("15m"), dict) else None
    mom = _safe_float(dd.get("price_momentum"), 0.0)
    return float(thesis_score or 0.0) >= 0.65 and m15 is not None and m15 >= 0.50 and mom > 0.04


def _bear_exhaustion_mr_ok(decision_data: dict[str, Any], current_price: float) -> bool:
    dd = decision_data or {}
    vwap = _safe_float(dd.get("vwap"), 0.0)
    rsi = _safe_float(dd.get("rsi"), 50.0)
    bb = _safe_float(dd.get("bb_position"), 0.5)
    if rsi > 38.0 or bb > 0.32:
        return False
    if vwap > 0 and current_price > 0:
        return (current_price - vwap) / vwap <= -0.002
    return rsi <= 32.0 and bb <= 0.25


def evaluate_day_entry_route(
    *,
    setup_type: str,
    day_regime: str,
    decision_data: dict[str, Any],
    context_payload: dict[str, Any] | None = None,
    current_price: float = 0.0,
    thesis_score: float = 0.0,
    xrp_churn_active: bool = False,
    strategy_family: str | None = None,
) -> dict[str, Any]:
    """
    Returns allowed, block_reason, rank_delta, size_factor, min_thesis_score.
    Hard blocks chop and mismatched setup/regime pairs.
    """
    setup = str(setup_type or SETUP_NO_CLEAR_THESIS)
    regime = str(day_regime or DAY_REGIME_NEUTRAL)
    score = float(thesis_score or 0.0)
    rank_delta = 0.0
    size_factor = 1.0
    min_thesis = 0.40
    family = str(strategy_family or decision_data.get("strategy_family") or "").strip().upper()

    if family == STRATEGY_FAMILY_ALLWEATHER_BREAKOUT_PULLBACK:
        if regime == DAY_REGIME_CHOP:
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_CHOP_NO_DAY",
                "day_route_regime": regime,
                "strategy_family": family,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if regime == DAY_REGIME_BULL:
            if setup not in (SETUP_HTF_TREND_PULLBACK, SETUP_BREAKOUT_CONTINUATION):
                return {
                    "allowed": False,
                    "block_reason": "ALLWEATHER_ROUTE_BULL_SETUP_MISMATCH",
                    "day_route_regime": regime,
                    "strategy_family": family,
                    "route_rank_delta": 0.0,
                    "route_size_factor": 0.0,
                    "route_min_thesis_score": 1.0,
                }
        elif regime in (DAY_REGIME_NEUTRAL, DAY_REGIME_RANGE):
            if setup not in (SETUP_BREAKOUT_CONTINUATION, SETUP_VWAP_REVERSION, SETUP_RANGE_BOUNCE):
                return {
                    "allowed": False,
                    "block_reason": "ALLWEATHER_ROUTE_NEUTRAL_RANGE_ONLY",
                    "day_route_regime": regime,
                    "strategy_family": family,
                    "route_rank_delta": 0.0,
                    "route_size_factor": 0.0,
                    "route_min_thesis_score": 1.0,
                }
        elif regime == DAY_REGIME_BEAR:
            # Profit policy: prefer RANGE/VWAP in bear — FBR is disabled for fills.
            if setup not in (SETUP_RANGE_BOUNCE, SETUP_VWAP_REVERSION):
                return {
                    "allowed": False,
                    "block_reason": "ALLWEATHER_ROUTE_BEAR_MR_ONLY",
                    "day_route_regime": regime,
                    "strategy_family": family,
                    "route_rank_delta": 0.0,
                    "route_size_factor": 0.0,
                    "route_min_thesis_score": 1.0,
                }
        else:
            return {
                "allowed": False,
                "block_reason": "ALLWEATHER_ROUTE_REGIME_BLOCKED",
                "day_route_regime": regime,
                "strategy_family": family,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        return {
            "allowed": True,
            "block_reason": "",
            "day_route_regime": regime,
            "strategy_family": family,
            "route_rank_delta": 0.0,
            "route_size_factor": 1.0,
            "route_min_thesis_score": 0.40,
        }

    if setup == SETUP_NO_CLEAR_THESIS:
        return {
            "allowed": False,
            "block_reason": "REGIME_ROUTE_NO_CLEAR_THESIS",
            "day_route_regime": regime,
            "route_rank_delta": 0.0,
            "route_size_factor": 0.0,
            "route_min_thesis_score": 1.0,
        }

    if regime == DAY_REGIME_CHOP:
        return {
            "allowed": False,
            "block_reason": "REGIME_ROUTE_CHOP_NO_DAY",
            "day_route_regime": regime,
            "route_rank_delta": 0.0,
            "route_size_factor": 0.0,
            "route_min_thesis_score": 1.0,
        }

    htf_ok, htf_reason = htf_allows_day_long(
        decision_data,
        setup_type=setup,
        context_payload=context_payload,
        thesis_score=score,
    )
    if not htf_ok:
        return {
            "allowed": False,
            "block_reason": "REGIME_ROUTE_HTF_DENIED",
            "day_route_regime": regime,
            "htf_route_detail": htf_reason,
            "route_rank_delta": 0.0,
            "route_size_factor": 0.0,
            "route_min_thesis_score": 1.0,
        }

    if regime == DAY_REGIME_BULL:
        if setup not in (SETUP_HTF_TREND_PULLBACK, SETUP_BREAKOUT_CONTINUATION):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_BULL_SETUP_MISMATCH",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }

    elif regime == DAY_REGIME_RANGE:
        if setup not in (SETUP_VWAP_REVERSION, SETUP_RANGE_BOUNCE):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_RANGE_MR_ONLY",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if setup == SETUP_RANGE_BOUNCE:
            adx = _safe_float(decision_data.get("adx"), 20.0)
            if adx > 30:
                return {"allowed": False, "block_reason": "RANGE_BOUNCE_ADX_TOO_HIGH", "day_route_regime": regime, "route_rank_delta": 0.0, "route_size_factor": 0.0, "route_min_thesis_score": 1.0}
            size_factor = min(size_factor, 0.70)
            rank_delta -= 0.02
            min_thesis = 0.48
            # Bounce is intentionally lighter — skip the full VWAP reclaim stricts
        else:
            # VWAP strict path
            adx = _safe_float(decision_data.get("adx"), 20.0)
            if adx > 24.0:
                return {
                    "allowed": False,
                    "block_reason": "REGIME_ROUTE_VWAP_ADX_TOO_HIGH",
                    "day_route_regime": regime,
                    "route_rank_delta": 0.0,
                    "route_size_factor": 0.0,
                    "route_min_thesis_score": 1.0,
                }
            if not _range_vwap_reclaim_strict(decision_data, current_price):
                return {"allowed": False, "block_reason": "REGIME_ROUTE_RANGE_NOT_AT_LOW", "day_route_regime": regime, "route_rank_delta": 0.0, "route_size_factor": 0.0, "route_min_thesis_score": 1.0}
            if not _vwap_expected_mfe_after_fees_ok(decision_data, current_price):
                return {
                    "allowed": False,
                    "block_reason": "REGIME_ROUTE_RANGE_VWAP_MFE_TOO_LOW",
                    "day_route_regime": regime,
                    "route_rank_delta": 0.0,
                    "route_size_factor": 0.0,
                    "route_min_thesis_score": 1.0,
                }
            min_thesis = max(min_thesis, 0.62)
            size_factor = min(size_factor, 0.55)
            rank_delta -= 0.06
        # Only apply VWAP-specific MFE for the VWAP setup (bounce uses its own lighter thesis levels)
        if setup == SETUP_VWAP_REVERSION and not _vwap_expected_mfe_after_fees_ok(decision_data, current_price):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_RANGE_VWAP_MFE_TOO_LOW",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if setup == SETUP_VWAP_REVERSION:
            min_thesis = max(min_thesis, 0.62)
            size_factor = min(size_factor, 0.55)
            rank_delta -= 0.06
        # For RANGE_BOUNCE we keep the lighter min_thesis (0.48) and size we set earlier

    elif regime == DAY_REGIME_BEAR:
        if setup == SETUP_HTF_TREND_PULLBACK:
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_BEAR_NO_TREND_PULLBACK",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if setup == SETUP_FAILED_BREAKDOWN_REVERSAL:
            # Allowed reversal in bear (paper mirrors live). Conservative size/min_thesis already applied upstream.
            rank_delta -= 0.03
            size_factor = 0.55
            min_thesis = 0.58
        if setup == SETUP_BREAKOUT_CONTINUATION:
            if not _bear_reversal_breakout_ok(decision_data, score):
                return {
                    "allowed": False,
                    "block_reason": "REGIME_ROUTE_BEAR_BREAKOUT_UNCONFIRMED",
                    "day_route_regime": regime,
                    "route_rank_delta": -0.06,
                    "route_size_factor": 0.45,
                    "route_min_thesis_score": 0.65,
                }
            rank_delta -= 0.04
            size_factor = 0.55
            min_thesis = 0.65
        elif setup == SETUP_VWAP_REVERSION:
            if not _bear_exhaustion_mr_ok(decision_data, current_price):
                return {
                    "allowed": False,
                    "block_reason": "REGIME_ROUTE_BEAR_MR_NOT_EXHAUSTED",
                    "day_route_regime": regime,
                    "route_rank_delta": -0.06,
                    "route_size_factor": 0.40,
                    "route_min_thesis_score": 0.62,
                }
            rank_delta -= 0.05
            size_factor = 0.42
            min_thesis = 0.62

    elif regime == DAY_REGIME_NEUTRAL:
        # Neutral/range: VWAP or range bounce (to generate activity + learning in non-trend).
        if setup not in (SETUP_VWAP_REVERSION, SETUP_RANGE_BOUNCE):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_NEUTRAL_MR_ONLY",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        adx = _safe_float(decision_data.get("adx"), 20.0)
        if adx > 28.0:
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_VWAP_ADX_TOO_HIGH",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if not _range_vwap_reclaim_ok(decision_data, current_price):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_NEUTRAL_VWAP_NOT_RECLAIM",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        if not _vwap_expected_mfe_after_fees_ok(decision_data, current_price):
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_NEUTRAL_VWAP_MFE_TOO_LOW",
                "day_route_regime": regime,
                "route_rank_delta": 0.0,
                "route_size_factor": 0.0,
                "route_min_thesis_score": 1.0,
            }
        size_factor = min(size_factor, 0.72)
        rank_delta -= 0.04
        min_thesis = max(min_thesis, 0.58)

    if score < min_thesis:
        return {
            "allowed": False,
            "block_reason": "REGIME_ROUTE_THESIS_SCORE_TOO_LOW",
            "day_route_regime": regime,
            "route_rank_delta": rank_delta,
            "route_size_factor": size_factor,
            "route_min_thesis_score": min_thesis,
        }

    if xrp_churn_active:
        min_thesis = max(min_thesis, 0.68)
        if score < min_thesis:
            return {
                "allowed": False,
                "block_reason": "REGIME_ROUTE_XRP_CHURN_CONFIRMATION",
                "day_route_regime": regime,
                "route_rank_delta": -0.12,
                "route_size_factor": 0.35,
                "route_min_thesis_score": min_thesis,
            }
        rank_delta -= 0.12
        size_factor = min(size_factor, 0.35)

    return {
        "allowed": True,
        "block_reason": "",
        "day_route_regime": regime,
        "htf_route_detail": htf_reason,
        "route_rank_delta": round(rank_delta, 4),
        "route_size_factor": round(size_factor, 4),
        "route_min_thesis_score": min_thesis,
    }


def compute_hist_expectancy_pct(
    *,
    win_rate: float,
    avg_win_usd: float,
    avg_loss_usd: float,
    equity: float,
) -> float:
    """win_rate * avg_win - loss_rate * avg_loss as fractional return vs equity."""
    eq = max(float(equity or 0.0), 1.0)
    wr = max(0.0, min(1.0, float(win_rate or 0.0)))
    aw = max(0.0, float(avg_win_usd or 0.0))
    al = max(0.0, float(avg_loss_usd or 0.0))
    return (wr * aw - (1.0 - wr) * al) / eq
