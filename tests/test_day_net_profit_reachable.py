"""NET_PROFIT must stay reachable while path-aware exit is on.

The 2026-09-17 commits that removed 4H from exit authority also flipped
DAY_PATH_AWARE_EXIT to default true. evaluate_engine_managed_exit returns
_evaluate_path_aware_exit's result directly, and portfolio_engine then returns
None before the ladder, so every net-profit check became unreachable. Production
recorded 106 NET_PROFIT_EXIT fills worth +$1,110.64 (100% winners, avg +0.68%)
and then none at all after 2026-09-17 17:03, while the trail that replaced it
earned +$14.48 over 129 fires.

These tests pin the target_hit case as reachable and pin the precedence that
keeps loss protection and the armed trail ahead of it.
"""

from __future__ import annotations

import pytest

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL, min_net_profit_for_symbol
from backend.services.day_controlled_exits import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_NET_PROFIT,
    EXIT_STOP_LOSS,
    EXIT_TRAILING_STOP,
    evaluate_engine_managed_exit,
)

FOUR_COINS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")


class _Pos:
    """Mirrors tests/test_day_path_aware.py::_Pos so precedence is comparable."""

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


PROFILE = {"max_hold_min": 360, "trail": 0.005, "sl": 0.01}


@pytest.fixture(autouse=True)
def _path_aware_on(monkeypatch):
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")
    monkeypatch.setenv("DAY_PATH_MIN_EXECUTABLE_NET_PCT", "0.0001")


def _call(pos, price, net, hold=30.0, bundle=None):
    return evaluate_engine_managed_exit(
        position=pos,
        current_price=price,
        net_pnl_pct=net,
        hold_minutes=hold,
        coin_profile=PROFILE,
        bundle=bundle,
    )


def test_reaching_the_target_takes_profit():
    """Mark at/through the resolved target with net above floor*0.45 -> sell."""
    out = _call(_Pos(thesis_target_level=100.80), 100.85, 0.0075)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_NET_PROFIT
    assert out["detail"] == "target_hit"


def test_net_profit_is_reachable_for_all_four_coins():
    """Equal coins: none may be structurally barred from taking profit."""
    for sym in FOUR_COINS:
        out = _call(_Pos(symbol=sym, thesis_target_level=100.80), 100.85, 0.0075)
        assert out["action"] == "sell", f"{sym} could not take profit"
        assert out["reason"] == EXIT_NET_PROFIT, f"{sym} -> {out['reason']}"


def test_profit_floor_is_equal_across_the_four_coins():
    floors = {s: float(min_net_profit_for_symbol(s)) for s in FOUR_COINS}
    assert len(set(floors.values())) == 1, floors
    assert floors["BTC/USDT"] == pytest.approx(float(MIN_NET_PROFIT_TO_SELL))


def test_target_hit_but_below_the_scaled_floor_still_holds():
    """floor*0.45 still gates the clip, exactly as the ladder did."""
    floor = float(min_net_profit_for_symbol("ETH/USDT"))
    out = _call(_Pos(thesis_target_level=100.10), 100.15, floor * 0.40)
    assert out["action"] == "hold"
    assert out["reason"] != EXIT_NET_PROFIT


def test_unreached_target_is_not_clipped():
    """The 'let winners trail' contract: green but target unreached -> hold.

    This is the case test_missing_4h_bundle_does_not_net_profit_clip pins. The
    bare-floor form of the ladder's net-profit check is deliberately not restored.
    """
    out = _call(_Pos(thesis_target_level=101.0), 100.50, 0.0045)
    assert out["action"] == "hold"
    assert out["reason"] != EXIT_NET_PROFIT


def test_armed_trail_still_outranks_taking_profit():
    """A pullback through an armed ratchet is deterioration and keeps priority."""
    pos = _Pos(
        highest_price=101.20,
        trailing_stop_price=100.70,
        thesis_target_level=100.60,
    )
    out = _call(pos, 100.65, 0.0060)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TRAILING_STOP


def test_loss_protection_outranks_taking_profit():
    """Loss protection must never be pre-empted by a profit check.

    Two distinct adverse mechanisms, in precedence order: the catastrophic
    DAY_RISK_FLOOR (DAY_RISK_FLOOR_MIN_ADVERSE_PCT, 2%..6%) and the ordinary
    coin-profile stop (1.0%). The ordinary stop is the tighter of the two, so in
    live trading it is the one that fires; the floor is the gap backstop.
    """
    pos = _Pos(entry_price=100.0, thesis_invalid_level=99.50, thesis_target_level=100.10)

    # 1% down breaches the ordinary stop.
    out = _call(pos, 99.00, -0.0105)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STOP_LOSS

    # A gap straight through to 2.5% down is caught by the catastrophic floor,
    # which outranks the ordinary stop.
    out = _call(pos, 97.50, -0.0255)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_DAY_RISK_FLOOR


def test_taking_profit_never_reintroduces_4h_authority():
    """Restoring the profit exit must not resurrect the 4H structure break."""
    for price, net in ((100.85, 0.0075), (100.50, 0.0045), (99.00, -0.0105)):
        out = _call(_Pos(thesis_target_level=100.80), price, net)
        assert out["reason"] != EXIT_DAY_4H_STRUCTURE_BREAK


def test_a_losing_position_never_takes_profit():
    out = _call(_Pos(thesis_target_level=100.10), 100.15, -0.0020)
    assert out["reason"] != EXIT_NET_PROFIT


def test_zero_target_cannot_fire_the_profit_exit():
    """No resolvable target means no target_hit, never a divide-by-zero clip."""
    pos = _Pos(take_profit_1_price=0.0, thesis_target_level=0.0)
    out = _call(pos, 100.85, 0.0075)
    assert out["reason"] != EXIT_NET_PROFIT
