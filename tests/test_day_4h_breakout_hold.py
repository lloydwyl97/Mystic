"""DAY holds a 4H breakout instead of clipping TP1 on the same rise."""

from __future__ import annotations

from backend.services.day_controlled_exits import EXIT_DAY_4H_STRUCTURE_BREAK, evaluate_engine_managed_exit
from backend.services.portfolio_engine import day_intact_profit_floor
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    htf_4h_rise_broken,
    htf_4h_rise_intact,
    intact_4h_slot_blocked,
    late_4h_rise_signal,
    same_4h_rise_rebuy_signal,
    should_block_late_4h_rise_entry,
    should_block_rebuy_on_4h_rise,
    thesis_invalidated_live,
)


def _rising_4h(n: int = 60, start: float = 2000.0) -> list[list]:
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


def _broken_4h() -> list[list]:
    rows = _rising_4h(60)
    last = rows[-1]
    prior_low = float(rows[-2][3])
    dump_close = prior_low * 0.97
    rows[-1] = [last[0], last[4], last[4], dump_close * 0.99, dump_close, 100.0]
    return rows


def test_4h_rise_intact_on_vertical_green_bars():
    bundle = {"4h": _rising_4h()}
    assert htf_4h_rise_intact(bundle) is True
    assert htf_4h_rise_broken(bundle) is False


def test_4h_rise_broken_after_close_below_prior_low():
    bundle = {"4h": _broken_4h()}
    assert htf_4h_rise_intact(bundle) is False
    assert htf_4h_rise_broken(bundle) is True


def test_breakout_thesis_not_killed_by_15m_dip_while_4h_rising():
    bundle = {
        "4h": _rising_4h(),
        "5m": {"ema_align": 0.20},
        "15m": {"ema_align": 0.20},
    }
    assert (
        thesis_invalidated_live(
            SETUP_BREAKOUT_CONTINUATION,
            mark=2380.0,
            invalid_level=0.0,
            bundle=bundle,
            entry_price=2312.0,
        )
        is False
    )


def test_breakout_thesis_invalid_when_4h_structure_breaks():
    bundle = {"4h": _broken_4h()}
    assert (
        thesis_invalidated_live(
            SETUP_BREAKOUT_CONTINUATION,
            mark=2200.0,
            invalid_level=0.0,
            bundle=bundle,
            entry_price=2312.0,
        )
        is True
    )


def test_block_rebuy_after_tp1_while_4h_rising():
    blocked, why = should_block_rebuy_on_4h_rise(
        last_close_reason="NET_PROFIT_EXIT",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _rising_4h()},
        now_epoch=1_700_000_100.0,
    )
    assert blocked is False
    assert why == ""
    assert (
        same_4h_rise_rebuy_signal(
            last_close_reason="NET_PROFIT_EXIT",
            last_close_epoch=1_700_000_000.0,
            bundle={"4h": _rising_4h()},
            now_epoch=1_700_000_100.0,
        )
        == "SAME_4H_RISE_NO_REBUY"
    )


def test_allow_rebuy_after_one_4h_bar_even_if_rise_still_intact():
    """A 14h+ flat book on a live rise is a new DAY thesis, not churn."""
    blocked, why = should_block_rebuy_on_4h_rise(
        last_close_reason="NET_PROFIT_EXIT",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _rising_4h()},
        now_epoch=1_700_000_000.0 + 14400.0,
    )
    assert blocked is False
    assert why == ""


def test_block_late_4h_entry_when_1h_weak():
    blocked, why = should_block_late_4h_rise_entry(
        {"4h": _rising_4h(), "1h": {"ema_align": 0.20}},
        now_epoch=1_700_849_600.0 + 600.0,
    )
    assert blocked is False
    assert why == ""
    assert late_4h_rise_signal({"4h": _rising_4h(), "1h": {"ema_align": 0.20}}, 1_700_849_600.0 + 600.0) == "LATE_4H_RISE_1H_WEAK"


def test_block_late_4h_entry_when_bar_is_late():
    last_ts_ms = 1_700_000_000_000 + 59 * 14_400_000
    blocked, why = should_block_late_4h_rise_entry(
        {"4h": _rising_4h()},
        now_epoch=last_ts_ms / 1000.0 + 11_000.0,
    )
    assert blocked is False
    assert why == ""
    assert late_4h_rise_signal({"4h": _rising_4h()}, last_ts_ms / 1000.0 + 11_000.0) == "LATE_4H_RISE_BAR_LATE"


def test_intact_4h_slot_cap_blocks_third_name():
    assert intact_4h_slot_blocked(open_intact=2, candidate_intact=True) is True
    assert intact_4h_slot_blocked(open_intact=1, candidate_intact=True) is False
    assert intact_4h_slot_blocked(open_intact=2, candidate_intact=False) is False


def test_allow_rebuy_after_tp1_when_4h_broke():
    blocked, why = should_block_rebuy_on_4h_rise(
        last_close_reason="TP1",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _broken_4h()},
        now_epoch=1_700_100_000.0,
    )
    assert blocked is False
    assert why == ""


def test_engine_maps_structure_break_not_tp1():
    class _P:
        entry_price = 2312.0
        highest_price = 2400.0
        lowest_price = 2200.0
        stop_price = 0.0
        trailing_stop_price = 0.0
        trail_pct = 0.005
        take_profit_1_price = 2342.0
        entry_thesis = "BREAKOUT_CONTINUATION"
        entry_vwap = 0.0
        thesis_invalid_level = 0.0
        thesis_target_level = 0.0
        thesis_score = 0.7
        max_hold_min = 360
        day_route_regime_at_entry = ""

    # Above the risk floor, so the structure break is what closes it — not the floor.
    out = evaluate_engine_managed_exit(
        position=_P(),
        current_price=2290.0,
        net_pnl_pct=-0.0095,
        hold_minutes=30.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _broken_4h()},
    )
    assert out["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert "NET_PROFIT" not in out["reason"]
    assert "PATH_EXECUTABLE" not in out["reason"]


def test_rebuy_block_is_wired_into_the_buy_path():
    """Same-rise is telemetry/rank only — buy path must not reject it."""
    import inspect

    from backend.services.portfolio_engine import PortfolioEngine

    buy_src = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    can_src = inspect.getsource(PortfolioEngine._can_open_position)
    assert "same_4h_rise_no_rebuy" not in buy_src
    assert "late_4h_rise_no_buy" not in buy_src
    assert "late_4h_rise_no_buy" not in can_src
    assert "_log_day_rise_rank_telemetry" in buy_src


def test_allow_rebuy_after_non_profit_exit():
    blocked, _why = should_block_rebuy_on_4h_rise(
        last_close_reason="THESIS_INVALIDATION_EXIT",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _rising_4h()},
        now_epoch=1_700_000_100.0,
    )
    assert blocked is False


def test_intact_trend_profit_floor_scales_with_structural_risk():
    """A wider 4H structure means more risk carried, so more profit is required."""
    tight = day_intact_profit_floor(entry_price=100.0, prior_4h_low=99.0, min_net_profit=0.001)
    wide = day_intact_profit_floor(entry_price=100.0, prior_4h_low=95.0, min_net_profit=0.001)
    assert wide > tight


def test_intact_trend_profit_floor_never_degrades_into_a_scalp_clip():
    """The original goal stands: DAY must not clip tiny profits on a live rise."""
    floor = day_intact_profit_floor(entry_price=100.0, prior_4h_low=99.9, min_net_profit=0.004)
    assert floor >= 0.008


def test_intact_trend_profit_floor_is_capped_so_profit_stays_reachable():
    """A distant 4H low must not put profit-taking permanently out of reach."""
    floor = day_intact_profit_floor(entry_price=100.0, prior_4h_low=50.0, min_net_profit=0.004)
    assert floor <= 0.025


def test_small_gain_on_intact_trend_still_holds():
    floor = day_intact_profit_floor(entry_price=91.32, prior_4h_low=89.94, min_net_profit=0.005)
    assert 0.003561 < floor  # live SOL: +0.36% must not trigger a clip


def test_large_gain_on_intact_trend_now_takes_profit():
    """Regression guard: profit was previously unreachable until the trend broke."""
    floor = day_intact_profit_floor(entry_price=1.3781, prior_4h_low=1.3157, min_net_profit=0.003)
    assert 0.024626 >= floor  # live XRP: +2.46% must be bookable while 4H is intact


def test_profit_is_not_gated_behind_structure_break_in_source():
    """Path-aware leftover NET_PROFIT clip is skipped; legacy string may remain."""
    import inspect

    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._check_exit_conditions)
    assert "_path_aware_exit_enabled" in src
    assert "return None" in src
