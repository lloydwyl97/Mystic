"""DAY V2 structure-runner exit contract and DAY/SCALP lifecycle separation."""

from __future__ import annotations

import inspect
import time
from itertools import pairwise
from types import SimpleNamespace

import pytest

from backend.services.day_v2.config import DAY_V2_MAX_HOLD_MINUTES, DAY_V2_UNIVERSE
from backend.services.day_v2.live_exit_evaluator import evaluate_day_v2_exit
from backend.services.day_v2.winner_contract import (
    DAY_EXIT_CONTRACT_RUNNER,
    LEGACY_ATR_1H_PER_ATR_15M,
    RUNNER_ACTIVATION_ATR_1H,
    RUNNER_TIGHT_TRAIL_ATR_1H,
    RUNNER_TRAIL_ATR_1H,
    atr_from_bars,
    move_potential,
    objective_level,
    runner_stop,
    structural_objective,
)

ENTRY = 100.0
ATR15 = 0.4
ATR1H = 1.0
COST = 0.0006


def _exit(**kw):
    base = {
        "engine_id": "DAY_V2",
        "entry_price": ENTRY,
        "current_price": ENTRY,
        "bar_low": ENTRY,
        "highest_price": ENTRY,
        "atr_at_entry": ATR15,
        "structural_anchor": 98.5,
        "target_price": 100.9,
        "entry_time": time.time() - 60 * 60,
        "estimated_roundtrip_cost": COST,
        "setup": "HTF_TREND_PULLBACK",
        "atr_1h_at_entry": ATR1H,
        "objective_structural": 103.0,
    }
    base.update(kw)
    return evaluate_day_v2_exit(**base)


def _bar(o, h, lo, c, ts=0):
    return {"open": o, "high": h, "low": lo, "close": c, "ts_epoch": ts}


# --- no scalp-like small exit -------------------------------------------------


@pytest.mark.parametrize("gain", [0.003, 0.005, 0.009, 0.0099])
def test_day_does_not_exit_at_old_small_target_or_small_trail(gain):
    high = ENTRY * (1 + gain)
    assert _exit(current_price=high, highest_price=high) is None
    assert _exit(current_price=high * 0.994, highest_price=high) is None


def test_old_entry_target_is_not_a_sell_level():
    assert _exit(current_price=101.0, highest_price=101.0, target_price=100.9) is None


# --- setup objectives are structure / volatility based ------------------------


def test_structural_objective_per_setup():
    b15 = [_bar(100, 100 + i * 0.1, 99 - i * 0.05, 100) for i in range(21)]
    b1h = [_bar(100, 101, 99, 95.0 + i * 0.5) for i in range(20)]
    b4h = [_bar(100, 104.0 + (i % 3), 99, 100) for i in range(15)]
    high20 = max(b["high"] for b in b15[-20:])
    prior_high = max(b["high"] for b in b15[:-1])
    prior_low = min(b["low"] for b in b15[:-1])
    assert structural_objective("HTF_TREND_PULLBACK", b15, b1h, b4h) == pytest.approx(max(b["high"] for b in b4h[-6:]))
    assert structural_objective("BREAKOUT_CONTINUATION", b15, b1h, b4h) == pytest.approx(prior_high + (prior_high - prior_low))
    assert structural_objective("RANGE_BOUNCE", b15, b1h, b4h) == pytest.approx(high20)
    mean1h = sum(b["close"] for b in b1h) / 20
    assert structural_objective("VWAP_REVERSION", b15, b1h, b4h) == pytest.approx(mean1h)
    assert structural_objective("EXHAUSTION_MR", b15, b1h, b4h) == pytest.approx(mean1h)


def test_objective_is_floored_by_atr_not_by_fixed_percent():
    assert objective_level("HTF_TREND_PULLBACK", 100.0, 1.0, 101.0) == pytest.approx(102.0)
    assert objective_level("RANGE_BOUNCE", 100.0, 1.0, 101.0) == pytest.approx(101.5)
    assert objective_level("RANGE_BOUNCE", 100.0, 1.0, 104.0) == pytest.approx(104.0)
    assert objective_level("RANGE_BOUNCE", 100.0, 3.0, 0.0) == pytest.approx(104.5)


def test_atr_from_bars_uses_true_range():
    bars = [_bar(100, 101, 99, 100)] + [_bar(100, 100.5, 99.5, 100)] * 14
    assert atr_from_bars(bars) == pytest.approx(1.0)
    assert atr_from_bars(bars[:5]) == 0.0


def test_move_potential_is_in_atr_units():
    assert move_potential("HTF_TREND_PULLBACK", 100.0, 1.0, 105.0) == pytest.approx(5.0)
    assert move_potential("HTF_TREND_PULLBACK", 100.0, 0.0, 105.0) == 0.0


def test_move_potential_is_not_an_entry_gate():
    from backend.services import portfolio_engine_integration
    from backend.services.day_v2 import live_entry

    assert "move_potential" not in inspect.getsource(portfolio_engine_integration)
    src = inspect.getsource(live_entry.submit_day_v2_direct_entry)
    lines = [ln for ln in src.splitlines() if "move_potential" in ln]
    assert lines and all('"move_potential_atr_1h":' in ln for ln in lines)


# --- winner ratchet -----------------------------------------------------------


def test_ratchet_arms_only_when_trail_locks_profit():
    below = runner_stop(entry_price=ENTRY, highest_price=ENTRY + 0.99 * ATR1H, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    at_activation = runner_stop(entry_price=ENTRY, highest_price=ENTRY + RUNNER_ACTIVATION_ATR_1H * ATR1H, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    assert below["activated"] is False and below["stop"] == 0.0
    # Activation distance is 1.0 ATR and the trail is 1.5 ATR, so the first
    # arming high still computes a trail below entry. That must not sell.
    assert at_activation["activated"] is False and at_activation["stop"] == 0.0
    hwm = ENTRY * (1 + COST) + RUNNER_TRAIL_ATR_1H * ATR1H + 0.01
    armed = runner_stop(entry_price=ENTRY, highest_price=hwm, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    assert armed["activated"] is True
    assert armed["stop"] == pytest.approx(hwm - RUNNER_TRAIL_ATR_1H * ATR1H)
    assert armed["stop"] > ENTRY * (1 + COST)


def test_ratchet_only_moves_up():
    stops = []
    for hwm in [100.5, 101.0, 101.3, 101.8, 102.4, 102.99, 103.0, 103.5, 104.2, 104.2]:
        stops.append(runner_stop(entry_price=ENTRY, highest_price=hwm, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)["stop"])
    assert all(b >= a for a, b in pairwise(stops))


def test_ratchet_tightens_after_objective():
    pre = runner_stop(entry_price=ENTRY, highest_price=102.5, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    post = runner_stop(entry_price=ENTRY, highest_price=103.2, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    assert pre["trail_atr_1h"] == RUNNER_TRAIL_ATR_1H
    assert post["trail_atr_1h"] == RUNNER_TIGHT_TRAIL_ATR_1H
    assert post["stop"] == pytest.approx(103.2 - RUNNER_TIGHT_TRAIL_ATR_1H * ATR1H)


def test_ratchet_levels_do_not_sell_and_a_worse_terminal_does():
    hi = 102.5
    stop = hi - RUNNER_TRAIL_ATR_1H * ATR1H
    assert _exit(highest_price=hi, current_price=stop - 0.01) is None
    sold = _exit(highest_price=hi, current_price=101.0, expected_terminal_net=-0.01)
    assert sold["reason"] == "DAY_V2_LEARNED_CONTINUATION"


def test_winner_protection_does_not_substitute_break_even():
    hi = ENTRY + 1.05 * ATR1H
    state = runner_stop(entry_price=ENTRY, highest_price=hi, atr_1h=ATR1H, objective=103.0, estimated_roundtrip_cost=COST)
    assert state["activated"] is False and state["stop"] == 0.0
    assert _exit(highest_price=hi, current_price=ENTRY * (1 + COST) - 0.001) is None


def test_pre_runner_position_does_not_scratch_at_break_even():
    atr1h = LEGACY_ATR_1H_PER_ATR_15M * ATR15
    hi = ENTRY + 1.01 * atr1h
    dec = _exit(atr_1h_at_entry=0.0, objective_structural=0.0, highest_price=hi, current_price=hi - RUNNER_TRAIL_ATR_1H * atr1h - 0.01)
    assert dec is None


# --- loss protection unchanged -----------------------------------------------


def test_catastrophic_stop_still_fires_outside_the_anchor():
    # Anchor 98.5 plus one 15m ATR (0.4) is farther than 3x ATR (1.2).
    outside = ENTRY - ((ENTRY - 98.5) + ATR15) - 0.01
    inside = ENTRY - 3.0 * ATR15 - 0.01
    assert _exit(bar_low=outside, current_price=outside)["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"
    assert _exit(bar_low=inside, current_price=99.0) is None


def test_price_under_the_anchor_holds_until_the_learned_terminal_or_catastrophe():
    assert _exit(current_price=98.4, bar_low=98.9) is None
    sold = _exit(current_price=98.4, bar_low=98.9, expected_terminal_net=-0.05)
    assert sold["reason"] == "DAY_V2_LEARNED_CONTINUATION"


# --- time stop is not a scalp time stop ---------------------------------------


def test_time_stop_waits_a_day_trader_hold():
    assert DAY_V2_MAX_HOLD_MINUTES >= 300
    early = _exit(current_price=99.9, entry_time=time.time() - 200 * 60)
    assert early is None


def test_clock_cannot_sell_a_structurally_valid_day_trade():
    late = time.time() - (DAY_V2_MAX_HOLD_MINUTES + 5) * 60
    assert _exit(current_price=99.9, entry_time=late) is None
    assert _exit(current_price=100.3, entry_time=late) is None
    assert _exit(current_price=99.9, bar_low=99.5, highest_price=100.2, entry_time=time.time() - 900 * 60, structural_anchor=98.5) is None


# --- universe / slots ---------------------------------------------------------


def test_four_coins_and_slots_unchanged():
    from backend.services import portfolio_engine as pe
    from backend.services import two_engine_claim

    assert set(DAY_V2_UNIVERSE) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}
    assert pe.DAY_MAX_OPEN_POSITIONS == 4 and pe.SCALP_MAX_OPEN_POSITIONS == 4
    assert pe.COMBINED_ENGINE_MAX_POSITIONS == 8
    assert two_engine_claim.engine_cap("DAY_V2") == 4 and two_engine_claim.engine_cap("SCALP_V2") == 4


# --- persistence --------------------------------------------------------------


def test_runner_fields_round_trip_through_thesis_json():
    from backend.services.day_inventory_recovery import thesis_json_for_position

    pos = SimpleNamespace(day_exit_contract=DAY_EXIT_CONTRACT_RUNNER, day_atr_1h_at_entry=12.5, day_objective_structural=61000.0)
    payload = thesis_json_for_position(pos)
    assert payload["day_exit_contract"] == DAY_EXIT_CONTRACT_RUNNER
    assert payload["day_atr_1h_at_entry"] == 12.5
    assert payload["day_objective_structural"] == 61000.0


def test_signal_carries_runner_inputs():
    from backend.services.day_v2.live_signal import DayV2Signal

    fields = DayV2Signal.__dataclass_fields__
    assert {"atr_1h", "objective_structural", "move_potential"} <= set(fields)


# --- SCALP stays a scalper ----------------------------------------------------


def _scalp_pos(**kw):
    base = {"engine_id": "SCALP_V2", "entry_price": 100.0, "highest_price": 100.0, "lowest_price": 100.0, "symbol": "BTCUSDT"}
    base.update(kw)
    return SimpleNamespace(**base)


def test_scalp_age_does_not_sell():
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN, evaluate_scalp_v2_exit

    assert SCALP_V2_TIME_STOP_MIN < DAY_V2_MAX_HOLD_MINUTES
    dec = evaluate_scalp_v2_exit(position=_scalp_pos(), current_price=99.8, net_pnl_pct=-0.0026, hold_minutes=SCALP_V2_TIME_STOP_MIN + 1, bar_low=99.8)
    assert dec.get("action") == "hold"
    sold = evaluate_scalp_v2_exit(position=_scalp_pos(), current_price=99.8, net_pnl_pct=-0.0026, hold_minutes=3, bar_low=99.8, expected_terminal_net=-0.02)
    assert sold.get("reason") == "SCALP_V2_LEARNED_CONTINUATION"


def test_scalp_loss_containment():
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_CATASTROPHIC_PCT, evaluate_scalp_v2_exit

    low = 100.0 * (1 - SCALP_V2_CATASTROPHIC_PCT) - 0.01
    dec = evaluate_scalp_v2_exit(position=_scalp_pos(lowest_price=low), current_price=low, net_pnl_pct=-0.016, hold_minutes=5, bar_low=low)
    assert dec.get("action") == "sell"


def test_engines_never_cross_evaluate():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    assert _exit(engine_id="SCALP_V2", bar_low=90.0, current_price=90.0) is None
    assert evaluate_scalp_v2_exit(position=_scalp_pos(engine_id="DAY_V2"), current_price=90.0, net_pnl_pct=-0.1, hold_minutes=500, bar_low=90.0) == {}


def test_scalp_net_edge_safety_is_a_hard_block():
    from backend.services.binance_scalp import protected_preflight

    src = inspect.getsource(protected_preflight)
    assert protected_preflight.NET_EDGE_BELOW_MIN == "NET_EDGE_BELOW_MIN"
    assert "expected_net <= 0 or expected_net < econ.min_net_edge_pct" in src


def test_observed_day_trades_do_not_scratch_at_break_even():
    cost = 0.00066
    sol = runner_stop(entry_price=117.67, highest_price=119.59, atr_1h=1.4986, objective=120.67, estimated_roundtrip_cost=cost, trail_mult=0.949, activation_mult=1.25)
    eth_2702 = runner_stop(entry_price=2682.21, highest_price=2698.795, atr_1h=11.476, objective=2737.68, estimated_roundtrip_cost=cost)
    assert sol["activated"] is False and sol["stop"] == 0.0
    assert eth_2702["activated"] is False and eth_2702["stop"] == 0.0
    eth_2696 = runner_stop(entry_price=2699.05, highest_price=2721.9, atr_1h=13.6686, objective=2726.39, estimated_roundtrip_cost=cost)
    trail_2696 = 2721.9 - 1.5 * 13.6686
    assert trail_2696 > 2699.05 * (1 + cost)
    assert eth_2696["activated"] is True
    assert eth_2696["stop"] == pytest.approx(trail_2696)
    btc = runner_stop(entry_price=83567.12, highest_price=84098.2, atr_1h=296.11, objective=84159.0, estimated_roundtrip_cost=cost, trail_mult=0.806, activation_mult=1.25)
    trail_2704 = 84098.2 - 1.5 * 0.806 * 296.11
    assert btc["activated"] is True
    assert btc["stop"] == pytest.approx(trail_2704)
    assert btc["stop"] > 83567.12 * (1 + cost)


def test_adaptive_multipliers_cannot_arm_a_below_cost_trail():
    scratch = runner_stop(entry_price=100.0, highest_price=101.0, atr_1h=1.0, objective=110.0, estimated_roundtrip_cost=0.00066, activation_mult=0.80, trail_mult=1.20)
    assert scratch["activated"] is False
    hwm = 100.0 * 1.00066 + 1.5 * 0.80 + 0.01
    locked = runner_stop(entry_price=100.0, highest_price=hwm, atr_1h=1.0, objective=110.0, estimated_roundtrip_cost=0.00066, activation_mult=1.25, trail_mult=0.80)
    assert locked["activated"] is True
    assert locked["stop"] > 100.0 * 1.00066
    later = runner_stop(entry_price=100.0, highest_price=hwm + 1.0, atr_1h=1.0, objective=110.0, estimated_roundtrip_cost=0.00066, activation_mult=1.25, trail_mult=0.80)
    assert later["stop"] >= locked["stop"]


def test_quiet_15m_atr_cannot_front_run_the_structural_anchor():
    from backend.services.day_v2.live_exit_evaluator import catastrophic_threshold_price

    threshold = catastrophic_threshold_price(1.505, 0.00380, 1.4939925)
    assert threshold < 1.4939925
    assert threshold == pytest.approx(1.505 - ((1.505 - 1.4939925) + 0.00380))
    quiet = _exit(
        entry_price=1.505,
        current_price=1.4932,
        bar_low=1.4932,
        highest_price=1.50995,
        atr_at_entry=0.00380,
        structural_anchor=1.4939925,
        atr_1h_at_entry=0.009586,
        entry_time=time.time() - 60 * 60,
    )
    assert quiet is None
    hard = _exit(
        entry_price=1.505,
        current_price=1.4900,
        bar_low=1.4900,
        highest_price=1.50995,
        atr_at_entry=0.00380,
        structural_anchor=1.4939925,
        atr_1h_at_entry=0.009586,
        entry_time=time.time() - 60 * 60,
    )
    assert hard["reason"] == "DAY_V2_CATASTROPHIC_PROTECTION"
    assert catastrophic_threshold_price(100.0, 0.4, 0.0) == pytest.approx(98.8)


def test_scalp_soft_opinions_do_not_veto_and_history_is_market_data():
    from backend.services.binance_scalp.momentum_tracker import MomentumDiagnostics
    from backend.services.binance_scalp.scalp_arm_blocker import arm_blocked
    from backend.services.binance_scalp.scalp_candidate_ranking import HARD_REJECT_REASONS
    from backend.services.scalp_v2.decision_log import classify_scalp_candidate

    soft = {"NOT_NEAR_SUPPORT", "NO_REJECTION_WICK", "WEAK_REJECTION_WICK", "MOMENTUM_NOT_FLIPPED", "RANGE_TOO_WIDE", "REGIME_BLOCKED"}
    assert soft.isdisjoint(HARD_REJECT_REASONS)
    hard = {"SPREAD_TOO_WIDE", "DEPTH_OR_IMPACT_FAIL", "NO_EXECUTABLE_NET_EDGE", "NO_EXECUTABLE_EDGE_ESTIMATE", "INSUFFICIENT_BARS", "INSUFFICIENT_HISTORY", "STALE_DATA"}
    assert hard <= HARD_REJECT_REASONS
    assert classify_scalp_candidate({"entry_eligible": True, "hard_block": None, "soft_reason": "NOT_NEAR_SUPPORT", "executable_edge": {"final_executable_edge_pct": 0.0004}})[0] == "ARMED"
    assert classify_scalp_candidate({"entry_eligible": False, "hard_block": "NO_EXECUTABLE_NET_EDGE"})[0] == "REJECTED:NO_EXECUTABLE_NET_EDGE"
    assert classify_scalp_candidate({"entry_eligible": False, "hard_block": "INSUFFICIENT_HISTORY"})[0] == "REJECTED:INSUFFICIENT_HISTORY"
    assert "insufficient_windows" in inspect.getsource(MomentumDiagnostics)
    blocked, _reason, stats = arm_blocked("ZZZUSDT", "range_bounce_scalp", db_path=":memory:")
    assert blocked is False
    assert int(stats.get("n") or 0) == 0
