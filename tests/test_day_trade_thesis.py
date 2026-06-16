"""Controlled paper proof for trade thesis classification and position management."""

from __future__ import annotations

import json

from backend.config.trading_economics import ESTIMATED_ROUNDTRIP_COST, MIN_NET_PROFIT_TO_SELL
from backend.services.day_trade_thesis import (
    SETUP_HTF_TREND_PULLBACK,
    SETUP_NO_CLEAR_THESIS,
    SETUP_VWAP_REVERSION,
    apply_trade_thesis_to_candidate_fields,
    classify_buy_thesis,
    evaluate_thesis_exit,
    scalp_strategy_to_thesis,
)


def _htf_pullback_dd() -> dict:
    mtf = {
        "15m": {"ema_align": 0.82, "trend": 0.82},
        "30m": {"ema_align": 0.78, "trend": 0.78},
        "1h": {"ema_align": 0.80, "trend": 0.80},
        "4h": {"ema_align": 0.76, "trend": 0.76},
        "1m": {"ema_align": 0.48, "trend": 0.48},
        "5m": {"ema_align": 0.52, "trend": 0.52},
    }
    return {
        "ema_alignment": 0.74,
        "price_momentum": 0.05,
        "adx": 24.0,
        "rsi": 46.0,
        "bb_position": 0.42,
        "vwap": 99.5,
        "relative_volume": 1.3,
        "volume_ratio": 1.2,
        "ctx_rs_btc": 0.18,
        "mtf_json": json.dumps(mtf),
        "price_structure_regime": "trending",
    }


def test_day_htf_pullback_classifies_and_stores_levels():
    dd = apply_trade_thesis_to_candidate_fields(
        _htf_pullback_dd(),
        symbol="BTC/USDT",
        current_price=100.0,
        atr=1.2,
        strategy_id="day",
        price_structure_regime="trending",
    )
    assert dd["setup_type"] == SETUP_HTF_TREND_PULLBACK
    assert dd["entry_thesis"] == SETUP_HTF_TREND_PULLBACK
    assert dd["thesis_score"] >= 0.4
    assert dd["thesis_invalid_level"] > 0
    assert dd["thesis_target_level"] > 100.0
    assert dd["thesis_trend_tf"] in ("15m", "30m", "1h", "4h")
    assert dd["thesis_rank_delta"] > 0
    assert dd["thesis_size_factor"] >= 0.8


def test_scalp_vwap_reversion_maps_strategy():
    thesis = scalp_strategy_to_thesis(
        "vwap_ema_reclaim",
        {"vwap": 2.45, "prior_low": 2.40, "ema_fast": 2.46, "ema_slow": 2.44},
    )
    assert thesis["setup_type"] == SETUP_VWAP_REVERSION
    assert thesis["entry_thesis"] == SETUP_VWAP_REVERSION
    assert thesis["entry_vwap"] == 2.45
    assert thesis["thesis_invalid_level"] > 0


def test_position_management_reads_thesis_after_buy_fields():
    dd = apply_trade_thesis_to_candidate_fields(
        _htf_pullback_dd(),
        symbol="ETH/USDT",
        current_price=2500.0,
        atr=18.0,
        strategy_id="day",
    )
    position_like = {
        "entry_thesis": dd["entry_thesis"],
        "thesis_score": dd["thesis_score"],
        "thesis_invalid_level": dd["thesis_invalid_level"],
        "thesis_target_level": dd["thesis_target_level"],
        "entry_vwap": dd["entry_vwap"],
        "entry_price": 2500.0,
    }
    assert position_like["entry_thesis"] == SETUP_HTF_TREND_PULLBACK
    assert position_like["thesis_score"] > 0


def test_noise_red_hold_when_thesis_valid():
    eval_hold = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.72,
        thesis_invalid_level=98.0,
        thesis_target_level=102.5,
        entry_vwap=0.0,
        entry_price=100.0,
        mark=99.2,
        bundle={"1h": {"ema_align": 0.7}, "4h": {"ema_align": 0.68}},
    )
    assert eval_hold["action"] == "hold"
    assert eval_hold["reason"] == "THESIS_HOLD_NOISE"


def test_real_invalidation_warns_only_no_red_sell():
    eval_cut = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.72,
        thesis_invalid_level=98.0,
        thesis_target_level=102.5,
        entry_vwap=0.0,
        entry_price=100.0,
        mark=96.0,
        bundle={"1h": {"ema_align": 0.7}, "4h": {"ema_align": 0.68}},
    )
    assert eval_cut["action"] == "warn"
    assert "THESIS_INVALIDATION_WARNING_ONLY" in str(eval_cut["reason"])

    eval_weak = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.72,
        thesis_invalid_level=99.5,
        thesis_target_level=102.5,
        entry_vwap=0.0,
        entry_price=100.0,
        mark=99.2,
        bundle={"1h": {"ema_align": 0.30}, "4h": {"ema_align": 0.32}},
    )
    assert eval_weak["action"] == "warn"
    assert "THESIS_INVALIDATION_WARNING_ONLY" in str(eval_weak["reason"])


def test_profit_near_target_triggers_net_profit_exit():
    target = 101.2
    mark = target
    entry = 100.0
    net = (mark - entry) / entry - ESTIMATED_ROUNDTRIP_COST
    assert net >= MIN_NET_PROFIT_TO_SELL * 0.45
    eval_tp = evaluate_thesis_exit(
        entry_thesis=SETUP_HTF_TREND_PULLBACK,
        thesis_score=0.72,
        thesis_invalid_level=98.5,
        thesis_target_level=target,
        entry_vwap=0.0,
        entry_price=entry,
        mark=mark,
        bundle={"1h": {"ema_align": 0.7}, "4h": {"ema_align": 0.68}},
    )
    assert eval_tp["action"] == "sell"
    assert eval_tp["reason"] == "NET_PROFIT_EXIT"


def test_no_clear_thesis_ranks_poorly_and_sizes_tiny():
    dd = classify_buy_thesis(
        {"ema_alignment": 0.51, "adx": 14, "rsi": 50, "bb_position": 0.5, "price_momentum": 0.0},
        symbol="XRP/USDT",
        current_price=1.0,
        atr=0.01,
        strategy_id="day",
    )
    assert dd["setup_type"] == SETUP_NO_CLEAR_THESIS
    assert dd["thesis_rank_delta"] < 0
    assert dd["thesis_size_factor"] <= 0.25
    assert dd["thesis_ev_factor"] <= 0.55


def test_leaderboard_row_shape_from_classification():
    dd = apply_trade_thesis_to_candidate_fields(
        _htf_pullback_dd(),
        symbol="SOL/USDT",
        current_price=150.0,
        atr=2.0,
        strategy_id="day",
    )
    row = {
        "symbol": "SOL/USDT",
        "setup_type": dd["setup_type"],
        "thesis_score": dd["thesis_score"],
        "selected_net_expected_value": 0.0018 * dd["thesis_ev_factor"],
        "thesis_size_factor": dd["thesis_size_factor"],
        "symbol_trust_score": 0.12,
        "rank_score": 0.71 + dd["thesis_rank_delta"],
    }
    assert row["setup_type"] == SETUP_HTF_TREND_PULLBACK
    assert row["thesis_score"] > 0
    assert "selected_net_expected_value" in row
