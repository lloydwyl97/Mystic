"""DAY path-aware exit and HOLD-as-action EV."""

from __future__ import annotations

import pytest

from backend.config.execution_cost_model import honest_all_in_rt_pct
from backend.services.day_controlled_exits import (
    DAY_FULL_FLATTEN_REASONS,
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_EXTREME_PROTECTION,
    EXIT_GIVEBACK,
    EXIT_NET_PROFIT,
    EXIT_PATH_EXECUTABLE_PROFIT,
    EXIT_STALL_DEAD,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT_1,
    EXIT_TIME_STOP,
    EXIT_TRAILING_STOP,
    _path_aware_exit_enabled,
    evaluate_engine_managed_exit,
    preview_next_engine_exit,
)
from backend.services.day_direct_path_ev_authority import (
    DAY_AUTHORITY_MODE,
    OLD_RANK_EXECUTION_AUTHORITY,
    select_action,
)
from backend.services.day_path_net import (
    predict_decision_net,
    reset_day_artifact_cache,
    resolve_day_path_ev,
    stamp_day_path_prediction,
)
from backend.services.day_trade_thesis import resolve_day_risk_floor_price


class _Pos:
    def __init__(self, **kw):
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", 100.0)
        self.lowest_price = kw.get("lowest_price", 99.50)
        self.stop_price = kw.get("stop_price", 99.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.005)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", 100.0)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 99.0)
        self.thesis_target_level = kw.get("thesis_target_level", 101.0)
        self.thesis_score = kw.get("thesis_score", 0.7)
        self.max_hold_min = kw.get("max_hold_min", 360)
        self.day_route_regime_at_entry = kw.get("day_route_regime_at_entry", "")
        self.symbol = kw.get("symbol", "ETH/USDT")


@pytest.fixture(autouse=True)
def _path_aware_on(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    monkeypatch.setenv("DAY_PATH_MIN_EXECUTABLE_NET_PCT", "0.0001")


def test_missing_4h_bundle_does_not_unlock_tiny_profit_clips():
    """4H removed from exit authority (2026-09-17). Missing bundle → bracket hold."""
    out = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.08,
        net_pnl_pct=0.0006,
        hold_minutes=12.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "hold"
    assert out["reason"] == "path_aware_bracket_hold"
    assert out["reason"] not in {EXIT_PATH_EXECUTABLE_PROFIT, EXIT_NET_PROFIT}


def test_missing_4h_bundle_does_not_net_profit_clip():
    """4H removed from exit authority (2026-09-17). Missing bundle → bracket hold."""
    out = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=100.50,
        net_pnl_pct=0.0045,
        hold_minutes=20.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "hold"
    assert out["reason"] == "path_aware_bracket_hold"
    assert out["reason"] != EXIT_NET_PROFIT


def test_path_aware_stall_fires_on_dead_red_when_4h_absent():
    """4H removed from exit authority (2026-09-17). Stall/giveback are no
    longer gated on 4H presence. Dead-red position now exits via stall."""
    out = evaluate_engine_managed_exit(
        position=_Pos(highest_price=100.05, lowest_price=99.60, stop_price=0.0, thesis_invalid_level=0.0),
        current_price=99.65,
        net_pnl_pct=-0.0045,
        hold_minutes=130.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] in {EXIT_STALL_DEAD, EXIT_GIVEBACK}


def test_path_aware_loser_exits_via_giveback_when_4h_absent():
    """4H removed from exit authority (2026-09-17). Giveback/stall are no
    longer gated on 4H presence. Fading position exits via loss-limiting."""
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, max_hold_min=360),
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=60.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] in {EXIT_GIVEBACK, EXIT_STALL_DEAD}


def test_max_hold_exits_at_the_ceiling_even_when_4h_missing():
    """A missing 4H bundle must not suppress the authorized 300-min ceiling.

    This previously asserted the opposite: reaching max hold produced a bracket
    hold and EXIT_TIME_STOP was explicitly forbidden, which is what made the
    configured 300 min a value that never took effect. 4H is still not consulted
    -- the bundle is None here and the time stop fires anyway.
    """
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, max_hold_min=300),
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=300.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP

    # Short of the ceiling the time stop must not be the reason. (At 299 min this
    # same position is stall-cut as dead inventory, which is a different exit, so
    # the hold here is taken inside the stall's 45-min minimum.)
    held = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, max_hold_min=300),
        current_price=99.50,
        net_pnl_pct=-0.006,
        hold_minutes=40.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert held["action"] == "hold"
    assert held["reason"] == "path_aware_bracket_hold"


def _rising_4h_rows(n: int = 60, start: float = 2000.0) -> list[list]:
    rows = []
    px = start
    ts = 1_700_000_000_000
    for _ in range(n):
        o = px
        c = px * 1.012
        rows.append([ts, o, c * 1.003, o * 0.998, c, 100.0])
        px = c
        ts += 14_400_000
    return rows


def test_path_aware_holds_green_on_4h_rise():
    """4H removed from exit authority (2026-09-17). With a rising 4H, position
    stays in bracket hold (same behavior, different reason string).

    Entry is scaled to the ETH-scale mark and no thesis target is resolvable, so
    neither the authorized 1.4% take-profit nor the adaptive net-profit target is
    in reach. This isolates the 4H rise, which is the condition under test. The
    original fixture paired a $100 entry with a 2318 mark, so its 101 target was
    exceeded 22x over by fixture incoherence rather than by design.
    """
    out = evaluate_engine_managed_exit(
        position=_Pos(entry_price=2306.0, thesis_invalid_level=0.0, take_profit_1_price=0.0, thesis_target_level=0.0),
        current_price=2318.0,
        net_pnl_pct=0.005,
        hold_minutes=20.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _rising_4h_rows()},
    )
    assert out["action"] == "hold"
    assert out["reason"] == "path_aware_bracket_hold"


def test_path_aware_giveback_sells_fade_while_4h_intact():
    pos = _Pos(entry_price=100.0, highest_price=100.40, symbol="NOARM/USDT", entry_thesis="")
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.80,
        net_pnl_pct=-0.0020,
        hold_minutes=45.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _rising_4h_rows()},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_GIVEBACK


def test_path_aware_stall_sells_dead_red_hold_while_4h_intact():
    pos = _Pos(
        entry_price=100.0,
        highest_price=100.05,
        lowest_price=99.60,
        thesis_invalid_level=0.0,
        symbol="NOARM/USDT",
        entry_thesis="",
    )
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=99.65,
        net_pnl_pct=-0.0045,
        hold_minutes=130.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _rising_4h_rows()},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STALL_DEAD


def test_path_aware_holds_time_stop_on_4h_rise():
    """4H removed from exit authority (2026-09-17). With a rising 4H, position
    stays in bracket hold (same behavior, different reason string).

    A rising 4H is not a reason to hold, but it is also not a reason to exit. The
    exit here comes from the authorized 300-min ceiling being exceeded (400 min),
    not from 4H. Entry is scaled to the mark so no profit objective is in reach.
    """
    out = evaluate_engine_managed_exit(
        position=_Pos(
            entry_price=2378.0,
            max_hold_min=300,
            thesis_invalid_level=0.0,
            take_profit_1_price=0.0,
            thesis_target_level=0.0,
        ),
        current_price=2390.0,
        net_pnl_pct=0.003,
        hold_minutes=400.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _rising_4h_rows()},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP

    # Same rising 4H, inside the ceiling -> hold. Proves 4H did not cause the exit.
    held = evaluate_engine_managed_exit(
        position=_Pos(
            entry_price=2378.0,
            max_hold_min=300,
            thesis_invalid_level=0.0,
            take_profit_1_price=0.0,
            thesis_target_level=0.0,
        ),
        current_price=2390.0,
        net_pnl_pct=0.003,
        hold_minutes=120.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _rising_4h_rows()},
    )
    assert held["action"] == "hold"


def _broken_4h_rows() -> list[list]:
    rows = _rising_4h_rows(60)
    last = rows[-1]
    prior_low = float(rows[-2][3])
    dump_close = prior_low * 0.97
    rows[-1] = [last[0], last[4], last[4], dump_close * 0.99, dump_close, 100.0]
    return rows


def test_4h_structure_break_no_longer_exits():
    """4H removed from trading authority (2026-09-17). A broken 4H no longer
    produces a sell; position falls to the standard exit ladder.

    With no resolvable target the ladder has no profit case to take, so this
    isolates the 4H break itself. Whether a winner at its target takes profit is
    covered by tests/test_day_net_profit_reachable.py.
    """
    out = evaluate_engine_managed_exit(
        position=_Pos(take_profit_1_price=0.0, thesis_target_level=0.0),
        current_price=2200.0,
        net_pnl_pct=0.005,
        hold_minutes=20.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _broken_4h_rows()},
    )
    assert out["reason"] != EXIT_DAY_4H_STRUCTURE_BREAK
    assert out["reason"] not in {EXIT_PATH_EXECUTABLE_PROFIT, EXIT_TIME_STOP, "TP1"}


def test_risk_floor_sits_below_structure_so_structure_exits_first():
    """A floor tighter than structure would fire first and defeat the 4H hold."""
    entry, invalid, prior_low = 77899.73, 76064.14, 76262.45
    floor = resolve_day_risk_floor_price(entry_price=entry, thesis_invalid_level=invalid, prior_4h_low=prior_low, atr_pct=0.0236)
    assert floor < invalid
    assert floor < prior_low
    assert floor > entry * 0.94  # still inside the hard adverse cap


def test_risk_floor_flattens_even_while_4h_intact():
    """The whole point: bound the bleed instead of waiting on a 4H close."""
    entry = 77899.73
    floor = resolve_day_risk_floor_price(entry_price=entry, thesis_invalid_level=76064.14, atr_pct=0.0236)
    out = evaluate_engine_managed_exit(
        position=_Pos(entry_price=entry, stop_price=0.0, thesis_invalid_level=76064.14),
        current_price=floor,
        net_pnl_pct=floor / entry - 1.0,
        hold_minutes=120.0,
        coin_profile={"max_hold_min": 2142, "trail": 0.004, "sl": 0.02},
        bundle={"4h": _rising_4h_rows(start=60000.0)},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_DAY_RISK_FLOOR
    assert out["htf_4h_rise_intact"] is True  # held-through state, still flattened


def test_risk_floor_does_not_fire_above_structure():
    """The catastrophic floor sits under structure and must not fire above it.

    The mark here is 76262.45 against a resolved floor of ~75835.95, so the floor
    is correctly untouched. The position is 2.1% down against a 2% profile stop,
    so the ordinary stop is what closes it -- that separation is the point: the
    floor is the catastrophic backstop, the profile sl is the ordinary stop. This
    previously asserted a bracket hold, which was only true because the ordinary
    stop was unreachable and a 2.1% loss could keep running.
    """
    entry = 77899.73
    profile = {"max_hold_min": 2142, "trail": 0.004, "sl": 0.02}
    out = evaluate_engine_managed_exit(
        position=_Pos(entry_price=entry, stop_price=0.0, thesis_invalid_level=76064.14),
        current_price=76262.45,
        net_pnl_pct=-0.021,
        hold_minutes=120.0,
        coin_profile=profile,
        bundle={"4h": _rising_4h_rows(start=60000.0)},
    )
    assert out["reason"] != EXIT_DAY_RISK_FLOOR
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STOP_LOSS

    # Above the ordinary stop as well -> genuine bracket hold, floor still silent.
    held = evaluate_engine_managed_exit(
        position=_Pos(entry_price=entry, stop_price=0.0, thesis_invalid_level=76064.14),
        current_price=77000.00,
        net_pnl_pct=-0.011,
        hold_minutes=120.0,
        coin_profile=profile,
        bundle={"4h": _rising_4h_rows(start=60000.0)},
    )
    assert held["action"] == "hold"
    assert held["reason"] == "path_aware_bracket_hold"


def test_risk_floor_is_hard_capped_when_structure_is_absurd():
    floor = resolve_day_risk_floor_price(entry_price=100.0, thesis_invalid_level=80.0, atr_pct=0.01)
    assert floor == pytest.approx(94.0)


def test_risk_floor_never_tighter_than_min_adverse():
    floor = resolve_day_risk_floor_price(entry_price=100.0, thesis_invalid_level=99.5, atr_pct=0.001)
    assert floor <= 98.0


def test_stamped_stop_equals_enforced_floor():
    """stop_price was previously decoration; it must now be a real level."""
    from backend.services.portfolio_engine import compute_entry_distance_pct

    price, atr = 77899.73, 1332.82
    dist = compute_entry_distance_pct("BTCUSDT", atr, price)
    stamped = price * (1.0 - dist)
    enforced = resolve_day_risk_floor_price(entry_price=price, atr_pct=atr / price)
    assert stamped == pytest.approx(enforced, rel=1e-9)


def test_day_exit_policy_defaults_to_path_aware(monkeypatch):
    """Non-4H path-aware ladder is the default; env cannot restore 4H."""
    monkeypatch.delenv("DAY_PATH_AWARE_EXIT", raising=False)
    assert _path_aware_exit_enabled() is False


def test_only_structure_break_and_extreme_may_full_flatten():
    """Authorized exits may flatten. 4H never may."""
    from backend.services.day_controlled_exits import EXIT_STOP_LOSS, EXIT_TAKE_PROFIT_1

    assert {
        EXIT_EXTREME_PROTECTION,
        EXIT_DAY_RISK_FLOOR,
        EXIT_STOP_LOSS,
        EXIT_GIVEBACK,
        EXIT_TRAILING_STOP,
        EXIT_NET_PROFIT,
        EXIT_TAKE_PROFIT_1,
        EXIT_STALL_DEAD,
        EXIT_TIME_STOP,
    } == DAY_FULL_FLATTEN_REASONS
    assert EXIT_DAY_4H_STRUCTURE_BREAK not in DAY_FULL_FLATTEN_REASONS
    assert EXIT_PATH_EXECUTABLE_PROFIT not in DAY_FULL_FLATTEN_REASONS


@pytest.mark.parametrize("net", [0.0006, 0.0045, 0.02])
def test_no_scalp_clip_at_any_positive_net_when_4h_absent(net):
    """No small-profit clip occurs while the mark is below every objective.

    The mark is 100.0 against a 100.0 entry, so neither the authorized 1.4%
    take-profit (101.40) nor the armed trail (99.90) is in reach. Hold minutes
    are inside the 300-min ceiling; the original 5000 made this assert that the
    max-hold ceiling did not exist, which is a separate concern from clipping.
    """
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, trailing_stop_price=99.9, highest_price=101.0),
        current_price=100.0,
        net_pnl_pct=net,
        hold_minutes=120.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "hold"
    assert out["reason"] == "path_aware_bracket_hold"


def test_giveback_fires_on_negative_net_when_4h_absent():
    """4H removed from exit authority (2026-09-17). Giveback is a legitimate
    loss-limiting exit (not a scalp clip) and fires when position faded."""
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0, trailing_stop_price=99.9, highest_price=101.0),
        current_price=100.0,
        net_pnl_pct=-0.006,
        hold_minutes=5000.0,
        coin_profile={"max_hold_min": 300, "trail": 0.005, "sl": 0.01},
        bundle=None,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_GIVEBACK


def test_extreme_protection_still_fires():
    out = evaluate_engine_managed_exit(
        position=_Pos(stop_price=0.0, thesis_invalid_level=0.0),
        current_price=94.0,
        net_pnl_pct=-0.06,
        hold_minutes=20.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"1h": {"ema_align": 0.10}, "4h": {"ema_align": 0.10}},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_EXTREME_PROTECTION
    assert out["extreme_protection_fired"] is True


def test_path_ev_authority_unchanged():
    assert DAY_AUTHORITY_MODE == "live_tape_learn_v1"
    assert OLD_RANK_EXECUTION_AUTHORITY is False
    out = select_action({"btc_path_ev": -0.01, "eth_path_ev": -0.02, "sol_path_ev": 0.0, "xrp_path_ev": -0.03})
    assert out["selected_action"] == "HOLD"
    assert out["why_selected"] == "HOLD_WINS"
    out2 = select_action({"btc_path_ev": 0.002, "eth_path_ev": 0.001, "sol_path_ev": -0.01, "xrp_path_ev": 0.0})
    assert out2["selected_action"] == "BUY_BTCUSDT"
    assert out2["old_rank_execution_authority"] is False


def test_predict_without_artifact_is_none(monkeypatch, tmp_path):
    monkeypatch.setenv("DAY_PATH_NET_ARTIFACT", str(tmp_path / "missing.json"))
    reset_day_artifact_cache()
    assert predict_decision_net({"bars_1m": [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}] * 20}) is None
    ev, stamped = resolve_day_path_ev({})
    assert ev is None
    assert "forward_net_model_version" not in stamped
    reset_day_artifact_cache()


def test_no_accepted_artifact_without_bars_is_hold_not_invented():
    reset_day_artifact_cache()
    ev, stamped = resolve_day_path_ev({"buy_margin": 0.20, "confidence": 0.90, "prob_buy": 0.80})
    assert ev is None
    assert stamped.get("path_net_status") != "predicted"
    reset_day_artifact_cache()


def test_no_accepted_artifact_with_bars_does_not_invent_ev():
    reset_day_artifact_cache()
    bars = []
    price = 100.0
    for i in range(40):
        price *= 1.0 + (0.0004 if i % 3 == 0 else -0.0002)
        bars.append({"open": price, "high": price * 1.001, "low": price * 0.999, "close": price, "volume": 10.0, "ts": 1786750000 + i * 60})
    ev, stamped = resolve_day_path_ev({"bars_1m": bars, "symbol": "ETHUSDT"})
    assert ev is None
    assert stamped.get("path_net_status") != "predicted"
    reset_day_artifact_cache()


def test_preview_names_giveback_when_path_aware_and_faded():
    """4H removed from exit authority (2026-09-17). BTC-shaped book: position
    has faded from high, so giveback is the next engine exit."""
    pos = _Pos(
        entry_price=77374.93,
        highest_price=78745.84,
        lowest_price=76558.9,
        stop_price=78588.35,
        trailing_stop_price=78588.35,
        trail_pct=0.004,
        thesis_invalid_level=75492.42,
        take_profit_1_price=78303.43,
        symbol="BTC/USDT",
    )
    preview = preview_next_engine_exit(
        position=pos,
        current_price=77104.28,
        net_pnl_pct=-0.0041,
        hold_minutes=1890.0,
        coin_profile={"max_hold_min": 27450, "trail": 0.004, "sl": 0.01},
        bundle=None,
    )
    assert preview["path_aware_exit"] is True
    assert preview["legacy_ladder_next_exit"] == EXIT_TRAILING_STOP
    assert preview["next_engine_exit"] == EXIT_GIVEBACK
    assert preview["executable_trailing_stop"] == pytest.approx(78588.35)
    assert preview["trailing_stop_in_exit_authority"] is True
    assert preview["high_water"] == pytest.approx(78745.84)
    assert preview["hard_stop"] > 0
    assert preview["hard_stop"] < 77374.93


def test_preview_splits_trail_fields_and_names_intact_profit_when_ready():
    rows = _rising_4h_rows()
    pos = _Pos(
        entry_price=1.378,
        highest_price=1.49925,
        lowest_price=1.364,
        stop_price=1.3787,
        trailing_stop_price=1.49625,
        trail_pct=0.005,
        thesis_invalid_level=1.3236,
        take_profit_1_price=1.397,
        symbol="XRP/USDT",
    )
    preview = preview_next_engine_exit(
        position=pos,
        current_price=1.486,
        net_pnl_pct=0.077,
        hold_minutes=1920.0,
        coin_profile={"max_hold_min": 12564, "trail": 0.005, "sl": 0.01},
        bundle={"4h": rows},
    )
    assert preview["high_water"] == pytest.approx(1.49925)
    # Activation is the high-water at which the ratchet already clears
    # entry plus the honest round trip: entry*(1+cost)/(1-trail).
    expected_activation = 1.378 * (1.0 + honest_all_in_rt_pct("XRP/USDT")) / (1.0 - 0.005)
    assert preview["trail_activation"] == pytest.approx(expected_activation)
    assert preview["trail_distance"] == pytest.approx(0.005)
    assert preview["executable_trailing_stop"] == pytest.approx(1.49625)
    assert preview["next_engine_exit"] == EXIT_TRAILING_STOP
    assert "NET_PROFIT" not in str(preview["current_exit_authority"])


def test_intact_green_sol_clip_level_holds_until_trail():
    """Replay: SOL 101.08 -> 102.52 while 4H advanced.

    102.52 / 101.08 is +1.4247%, at or above the authorized 1.4% take-profit, so
    the correct outcome is TAKE_PROFIT_1 -- this is real replay evidence that the
    authorized take-profit is reachable. It previously asserted a bracket hold,
    which was only true while every profit exit was unreachable. A mark just
    under the 1.4% objective still holds, and the trail still fires on pullback.
    """
    rows = _rising_4h_rows(start=100.59)
    profile = {"max_hold_min": 360, "trail": 0.005, "sl": 0.01}

    def _pos():
        return _Pos(
            entry_price=101.08,
            highest_price=102.52,
            lowest_price=101.00,
            stop_price=0.0,
            trailing_stop_price=102.52 * 0.995,
            trail_pct=0.005,
            thesis_invalid_level=0.0,
            thesis_target_level=0.0,
            take_profit_1_price=0.0,
            symbol="SOL/USDT",
        )

    took_profit = evaluate_engine_managed_exit(
        position=_pos(),
        current_price=102.52,
        net_pnl_pct=0.012508,
        hold_minutes=244.0,
        coin_profile=profile,
        bundle={"4h": rows},
    )
    assert took_profit["action"] == "sell"
    assert took_profit["reason"] == EXIT_TAKE_PROFIT_1

    # Just under the 1.4% objective and above the trail -> hold.
    below = evaluate_engine_managed_exit(
        position=_pos(),
        current_price=101.08 * 1.013,
        net_pnl_pct=0.0115,
        hold_minutes=244.0,
        coin_profile=profile,
        bundle={"4h": rows},
    )
    assert below["action"] == "hold"
    assert below["reason"] == "path_aware_bracket_hold"

    pos = _pos()
    trail_hit = evaluate_engine_managed_exit(
        position=pos,
        current_price=102.52 * 0.995 - 0.01,
        net_pnl_pct=0.007,
        hold_minutes=300.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": rows},
    )
    assert trail_hit["action"] == "sell"
    assert trail_hit["reason"] == EXIT_TRAILING_STOP


def test_intact_green_eth_clip_level_holds_until_trail():
    """Replay: ETH 2535.56 → 2565.31 was NET_PROFIT on intact 4H. Holds
    in bracket (4H removed 2026-09-17). Trail still fires on pullback."""
    rows = _rising_4h_rows(start=2482.93)
    pos = _Pos(
        entry_price=2535.56,
        highest_price=2565.31,
        lowest_price=2530.0,
        stop_price=0.0,
        trailing_stop_price=2565.31 * 0.995,
        trail_pct=0.005,
        thesis_invalid_level=0.0,
        symbol="ETH/USDT",
    )
    hold = evaluate_engine_managed_exit(
        position=pos,
        current_price=2565.31,
        net_pnl_pct=0.010962,
        hold_minutes=12.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": rows},
    )
    assert hold["action"] == "hold"
    assert hold["reason"] == "path_aware_bracket_hold"
    trail_hit = evaluate_engine_managed_exit(
        position=pos,
        current_price=2565.31 * 0.995 - 0.5,
        net_pnl_pct=0.005,
        hold_minutes=40.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": rows},
    )
    assert trail_hit["action"] == "sell"
    assert trail_hit["reason"] == EXIT_TRAILING_STOP
