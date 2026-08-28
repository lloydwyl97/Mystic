"""CONSTANT coin-profile DAY trail: distance does not tighten with MFE."""

from __future__ import annotations

import inspect

import pytest

from backend.services.day_controlled_exits import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_TRAILING_STOP,
    apply_break_even_and_mfe_trail,
    evaluate_engine_managed_exit,
    refresh_trailing_stop,
)
from backend.services.day_trade_thesis import (
    late_4h_rise_signal,
    should_block_late_4h_rise_entry,
    should_block_rebuy_on_4h_rise,
)
from backend.services.portfolio_engine import COIN_PROFILES, PortfolioEngine, get_coin_profile


class _Pos:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "ETH/USDT")
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", 100.0)
        self.lowest_price = kw.get("lowest_price", 99.50)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.005)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", 100.0)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 0.0)
        self.thesis_target_level = kw.get("thesis_target_level", 0.0)
        self.thesis_score = kw.get("thesis_score", 0.7)
        self.max_hold_min = kw.get("max_hold_min", 360)
        self.day_route_regime_at_entry = kw.get("day_route_regime_at_entry", "")


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


def test_coin_profile_trail_distances_locked():
    assert get_coin_profile("BTCUSDT")["trail"] == pytest.approx(0.0040)
    assert get_coin_profile("ETHUSDT")["trail"] == pytest.approx(0.0045)
    assert get_coin_profile("SOLUSDT")["trail"] == pytest.approx(0.0055)
    assert get_coin_profile("XRPUSDT")["trail"] == pytest.approx(0.0050)
    assert COIN_PROFILES["BTCUSDT"]["trail"] == 0.0040
    assert COIN_PROFILES["ETHUSDT"]["trail"] == 0.0045
    assert COIN_PROFILES["SOLUSDT"]["trail"] == 0.0055
    assert COIN_PROFILES["XRPUSDT"]["trail"] == 0.0050


@pytest.mark.parametrize(
    "symbol,entry,trail_pct",
    [
        ("BTC/USDT", 80000.0, 0.0040),
        ("ETH/USDT", 2500.0, 0.0045),
        ("SOL/USDT", 100.0, 0.0055),
        ("XRP/USDT", 1.50, 0.0050),
    ],
)
def test_profile_distance_holds_after_half_and_one_pct_mfe(symbol, entry, trail_pct):
    profile = get_coin_profile(symbol)
    assert profile["trail"] == pytest.approx(trail_pct)

    for mfe in (0.005, 0.010, 0.015, 0.020, 0.050):
        high = entry * (1.0 + mfe)
        pos = _Pos(
            symbol=symbol,
            entry_price=entry,
            highest_price=high,
            trail_pct=trail_pct,
            trailing_stop_price=0.0,
            stop_price=entry * (1.0 - profile["sl"]),
            day_route_regime_at_entry="bull",
        )
        refresh_trailing_stop(pos, high, profile)
        profile_trail = high * (1.0 - trail_pct)
        be_floor = entry * 1.0005
        expected = max(profile_trail, be_floor)
        assert pos.trailing_stop_price == pytest.approx(expected, rel=1e-6)
        assert pos.trailing_stop_price != pytest.approx(high * (1.0 - 0.0030), rel=1e-4)
        assert pos.trailing_stop_price != pytest.approx(high * (1.0 - 0.0020), rel=1e-4)
        widened = high * (1.0 - min(trail_pct * 2.0, 0.025))
        if abs(widened - expected) > 1e-6:
            assert pos.trailing_stop_price != pytest.approx(widened, rel=1e-4)


def test_high_water_ratchet_rises_with_new_highs():
    pos = _Pos(symbol="SOL/USDT", entry_price=100.0, highest_price=100.60, trail_pct=0.0055)
    profile = get_coin_profile("SOLUSDT")
    refresh_trailing_stop(pos, 100.60, profile)
    first = pos.trailing_stop_price
    pos.highest_price = 102.00
    refresh_trailing_stop(pos, 102.00, profile)
    assert pos.trailing_stop_price > first
    assert pos.trailing_stop_price == pytest.approx(102.00 * (1.0 - 0.0055), rel=1e-6)


def test_pullback_through_constant_trail_exits():
    pos = _Pos(
        symbol="BTC/USDT",
        entry_price=80000.0,
        highest_price=80800.0,
        trail_pct=0.0040,
        trailing_stop_price=80800.0 * (1.0 - 0.0040),
        thesis_invalid_level=0.0,
        stop_price=0.0,
    )
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=80800.0 * (1.0 - 0.0040) - 1.0,
        net_pnl_pct=0.005,
        hold_minutes=40.0,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle={"4h": _rising_4h(start=79000.0)},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TRAILING_STOP


def test_fourh_break_still_exits():
    pos = _Pos(entry_price=2312.0, highest_price=2400.0, trailing_stop_price=0.0, trail_pct=0.0045)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=2290.0,
        net_pnl_pct=-0.0095,
        hold_minutes=30.0,
        coin_profile=get_coin_profile("ETHUSDT"),
        bundle={"4h": _broken_4h()},
    )
    assert out["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK


def test_risk_floor_still_exits():
    pos = _Pos(
        entry_price=100.0,
        highest_price=100.2,
        thesis_invalid_level=97.5,
        trailing_stop_price=0.0,
    )
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=97.0,
        net_pnl_pct=-0.027,
        hold_minutes=20.0,
        coin_profile=get_coin_profile("SOLUSDT"),
        bundle={"4h": _rising_4h(start=99.0)},
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_DAY_RISK_FLOOR


def test_intact_profit_clip_skipped_when_path_aware():
    src = inspect.getsource(PortfolioEngine._check_exit_conditions)
    assert "_path_aware_exit_enabled" in src
    assert "DAY_4H_INTACT_PROFIT_TAKE" in src
    buy = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    assert "same_4h_rise_no_rebuy" not in buy
    assert "late_4h_rise_no_buy" not in buy


def test_rebuy_and_late_rise_are_not_permission():
    blocked, why = should_block_rebuy_on_4h_rise(
        last_close_reason="NET_PROFIT_EXIT",
        last_close_epoch=1_700_000_000.0,
        bundle={"4h": _rising_4h()},
        now_epoch=1_700_000_100.0,
    )
    assert blocked is False
    assert why == ""
    late, late_why = should_block_late_4h_rise_entry(
        {"4h": _rising_4h(), "1h": {"ema_align": 0.20}},
        now_epoch=1_700_849_600.0 + 600.0,
    )
    assert late is False
    assert late_why == ""
    assert late_4h_rise_signal({"4h": _rising_4h(), "1h": {"ema_align": 0.20}}, 1_700_849_600.0 + 600.0)


def test_be_lift_still_fires_without_tightening():
    pos = _Pos(entry_price=100.0, highest_price=100.32, stop_price=99.0, trailing_stop_price=99.0)
    assert apply_break_even_and_mfe_trail(pos, 100.32) is True
    assert pos.stop_price == pytest.approx(100.05, rel=1e-6)
