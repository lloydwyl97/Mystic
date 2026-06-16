"""DAY regime router — entry permission by structure."""

from __future__ import annotations

from backend.services.day_regime_router import (
    DAY_REGIME_BEAR,
    DAY_REGIME_BULL,
    DAY_REGIME_CHOP,
    DAY_REGIME_RANGE,
    classify_day_regime,
    evaluate_day_entry_route,
    htf_allows_day_long,
)
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    SETUP_HTF_TREND_PULLBACK,
    SETUP_VWAP_REVERSION,
)


def test_classify_bear_from_htf():
    dd = {"adx": 22, "ema_alignment": 0.35}
    ctx = {"mtf": {"1h": {"ema_align": 0.35}, "4h": {"ema_align": 0.32}}}
    assert classify_day_regime(dd, context_payload=ctx, chop_score=0.4) == DAY_REGIME_BEAR


def test_classify_chop_high_atr():
    dd = {"adx": 14}
    assert classify_day_regime(dd, chop_score=0.7, atr_ratio=0.04) == DAY_REGIME_CHOP


def test_bull_blocks_vwap_reversion():
    route = evaluate_day_entry_route(
        setup_type=SETUP_VWAP_REVERSION,
        day_regime=DAY_REGIME_BULL,
        decision_data={
            "vwap": 100,
            "bb_position": 0.3,
            "rsi": 40,
            "mtf_json": '{"1h":{"ema_align":0.62},"4h":{"ema_align":0.58}}',
        },
        current_price=99.5,
        thesis_score=0.7,
    )
    assert route["allowed"] is False
    assert route["block_reason"] == "REGIME_ROUTE_BULL_SETUP_MISMATCH"


def test_bear_blocks_trend_pullback():
    route = evaluate_day_entry_route(
        setup_type=SETUP_HTF_TREND_PULLBACK,
        day_regime=DAY_REGIME_BEAR,
        decision_data={"mtf_json": '{"1h":{"ema_align":0.7}}'},
        thesis_score=0.72,
    )
    assert route["allowed"] is False


def test_range_requires_vwap_reclaim():
    route = evaluate_day_entry_route(
        setup_type=SETUP_VWAP_REVERSION,
        day_regime=DAY_REGIME_RANGE,
        decision_data={
            "vwap": 100,
            "bb_position": 0.8,
            "rsi": 55,
            "mtf_json": '{"1h":{"ema_align":0.52},"4h":{"ema_align":0.51}}',
        },
        current_price=101.0,
        thesis_score=0.65,
    )
    assert route["allowed"] is False
    assert route["block_reason"] == "REGIME_ROUTE_RANGE_NOT_AT_LOW"


def test_htf_denies_ltf_bounce():
    dd = {"mtf_json": '{"5m":{"ema_align":0.62},"15m":{"ema_align":0.58},"1h":{"ema_align":0.35},"4h":{"ema_align":0.33}}'}
    ok, reason = htf_allows_day_long(dd, setup_type=SETUP_HTF_TREND_PULLBACK, thesis_score=0.6)
    assert ok is False
    assert reason == "htf_structure_denied"


def test_bull_allows_pullback_with_htf():
    dd = {"mtf_json": '{"1h":{"ema_align":0.62},"4h":{"ema_align":0.58}}'}
    route = evaluate_day_entry_route(
        setup_type=SETUP_HTF_TREND_PULLBACK,
        day_regime=DAY_REGIME_BULL,
        decision_data=dd,
        thesis_score=0.72,
    )
    assert route["allowed"] is True


def test_xrp_churn_requires_higher_thesis():
    route = evaluate_day_entry_route(
        setup_type=SETUP_BREAKOUT_CONTINUATION,
        day_regime=DAY_REGIME_BEAR,
        decision_data={
            "mtf_json": '{"15m":{"ema_align":0.58},"1h":{"ema_align":0.52},"4h":{"ema_align":0.51}}',
            "price_momentum": 0.06,
        },
        thesis_score=0.66,
        xrp_churn_active=True,
    )
    assert route["allowed"] is False
    assert route["block_reason"] == "REGIME_ROUTE_XRP_CHURN_CONFIRMATION"
