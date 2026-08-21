"""DAY holds a 4H breakout instead of clipping TP1 on the same rise."""

from __future__ import annotations

from backend.services.day_controlled_exits import EXIT_DAY_4H_STRUCTURE_BREAK, evaluate_engine_managed_exit
from backend.services.day_trade_thesis import (
    SETUP_BREAKOUT_CONTINUATION,
    htf_4h_rise_broken,
    htf_4h_rise_intact,
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
    assert blocked is True
    assert why == "SAME_4H_RISE_NO_REBUY"


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

    out = evaluate_engine_managed_exit(
        position=_P(),
        current_price=2200.0,
        net_pnl_pct=0.008,
        hold_minutes=30.0,
        coin_profile={"max_hold_min": 360, "trail": 0.005, "sl": 0.01},
        bundle={"4h": _broken_4h()},
    )
    assert out["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert "NET_PROFIT" not in out["reason"]
    assert "PATH_EXECUTABLE" not in out["reason"]


def test_allow_rebuy_after_non_profit_exit():
    blocked, _why = should_block_rebuy_on_4h_rise(
        last_close_reason="THESIS_INVALIDATION_EXIT",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _rising_4h()},
        now_epoch=1_700_000_100.0,
    )
    assert blocked is False
