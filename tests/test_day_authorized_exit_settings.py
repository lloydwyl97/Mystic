"""The authorized shared DAY exit settings must be the effective ones.

Authorized values, identical for BTC, ETH, SOL and XRP:

    take profit        1.4%   (coin_profile["tp"])
    ordinary stop      1.0%   (coin_profile["sl"])
    trailing distance  0.25%  (coin_profile["trail"])
    maximum hold       300 min (coin_profile["max_hold_min"])

Every test here asserts an *effective* value produced by the live exit path, not
a configured constant. Each one corresponds to a contradiction found in the
audit where the configured value was not what production actually enforced.
"""

from __future__ import annotations

import pytest

from backend.config.trading_economics import MIN_NET_PROFIT_TO_SELL, min_net_profit_for_symbol
from backend.services.day_controlled_exits import (
    EXIT_DAY_RISK_FLOOR,
    EXIT_NET_PROFIT,
    EXIT_STOP_LOSS,
    EXIT_TAKE_PROFIT_1,
    EXIT_TIME_STOP,
    EXIT_TRAILING_STOP,
    effective_max_hold_min,
    evaluate_engine_managed_exit,
    refresh_trailing_stop,
)
from backend.services.portfolio_engine import COIN_PROFILES, DEFAULT_COIN_PROFILE, get_coin_profile

SYMBOLS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")

AUTHORIZED_TP = 0.014
AUTHORIZED_SL = 0.010
AUTHORIZED_TRAIL = 0.0025
AUTHORIZED_MAX_HOLD_MIN = 300

# Realistic per-symbol entry prices so no test relies on a $100 entry paired with
# a four-figure mark. Fixture incoherence of exactly that kind is what let the
# old suite pass while the profit exits were unreachable.
ENTRIES = {"BTC/USDT": 95000.0, "ETH/USDT": 2300.0, "SOL/USDT": 101.0, "XRP/USDT": 0.62}


class _Pos:
    """Minimal position stand-in matching what the exit path reads."""

    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "BTC/USDT")
        self.entry_price = kw.get("entry_price", 100.0)
        self.highest_price = kw.get("highest_price", self.entry_price)
        self.lowest_price = kw.get("lowest_price", self.entry_price)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", AUTHORIZED_TRAIL)
        self.thesis_invalid_level = kw.get("thesis_invalid_level", 0.0)
        self.thesis_target_level = kw.get("thesis_target_level", 0.0)
        self.take_profit_1_price = kw.get("take_profit_1_price", 0.0)
        self.take_profit_2_price = kw.get("take_profit_2_price", 0.0)
        self.max_hold_min = kw.get("max_hold_min", AUTHORIZED_MAX_HOLD_MIN)
        self.quantity = kw.get("quantity", 1.0)
        self.day_route_regime_at_entry = kw.get("day_route_regime_at_entry", "")
        self.tp1_hit = kw.get("tp1_hit", False)


@pytest.fixture(autouse=True)
def _path_aware_on(monkeypatch):
    """Production default. Every assertion here is about the live path."""
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "true")


def _profile(symbol: str) -> dict:
    return dict(get_coin_profile(symbol))


def _call(symbol: str, position, price: float, net: float, hold: float = 60.0, bundle=None) -> dict:
    return evaluate_engine_managed_exit(
        position=position,
        current_price=price,
        net_pnl_pct=net,
        hold_minutes=hold,
        coin_profile=_profile(symbol),
        bundle=bundle,
    )


# --------------------------------------------------------------------------
# Configured values, and that the four coins are identical
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_authorized_value_is_configured_per_symbol(symbol):
    p = _profile(symbol)
    assert p["tp"] == pytest.approx(AUTHORIZED_TP)
    assert p["sl"] == pytest.approx(AUTHORIZED_SL)
    assert p["trail"] == pytest.approx(AUTHORIZED_TRAIL)
    assert int(p["max_hold_min"]) == AUTHORIZED_MAX_HOLD_MIN


def test_all_four_coins_have_byte_identical_settings():
    """No per-coin override may exist for any of the four authorized values."""
    profiles = [_profile(s) for s in SYMBOLS]
    for key in ("tp", "sl", "trail", "max_hold_min"):
        values = {p[key] for p in profiles}
        assert len(values) == 1, f"{key} differs across the four symbols: {values}"


def test_default_profile_matches_the_authorized_values():
    """A symbol resolving to the default must not silently get other settings."""
    assert DEFAULT_COIN_PROFILE["tp"] == pytest.approx(AUTHORIZED_TP)
    assert DEFAULT_COIN_PROFILE["sl"] == pytest.approx(AUTHORIZED_SL)
    assert DEFAULT_COIN_PROFILE["trail"] == pytest.approx(AUTHORIZED_TRAIL)
    assert int(DEFAULT_COIN_PROFILE["max_hold_min"]) == AUTHORIZED_MAX_HOLD_MIN


def test_no_fifth_symbol_is_configured():
    """The universe is the four authorized symbols, equally."""
    assert set(COIN_PROFILES) == {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"}


# --------------------------------------------------------------------------
# 1.4% take profit -- effective, reachable, and distinct from net-profit
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_take_profit_fires_at_exactly_1_4_percent(symbol):
    entry = ENTRIES[symbol]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry)

    out = _call(symbol, pos, entry * (1.0 + AUTHORIZED_TP), 0.012)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TAKE_PROFIT_1

    # A hair under the objective must not take profit.
    below = _call(symbol, _Pos(symbol=symbol, entry_price=entry, highest_price=entry), entry * 1.0139, 0.012)
    assert below["action"] == "hold"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_take_profit_is_not_clipped_down_to_a_tighter_thesis_target(symbol):
    """stamp_position_exit_metadata used to min() TP1 down to the thesis target.

    That silently replaced the authorized 1.4% with whatever the thesis produced.
    TP1 must stay at entry * 1.014 even when a much tighter target exists.
    """
    from backend.services.day_controlled_exits import stamp_open_position_exit_metadata

    entry = ENTRIES[symbol]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry)
    tight_target = entry * 1.004  # far tighter than the authorized 1.4%

    stamp_open_position_exit_metadata(
        pos,
        fill_price=entry,
        stop_price=entry * 0.99,
        tp1_price=entry * (1.0 + AUTHORIZED_TP),
        tp2_price=entry * (1.0 + AUTHORIZED_TP * 2),
        coin_profile=_profile(symbol),
        thesis_invalid_level=entry * 0.99,
        thesis_target_level=tight_target,
    )

    assert pos.take_profit_1_price == pytest.approx(entry * (1.0 + AUTHORIZED_TP))
    # The tighter target is preserved separately for the net-profit mechanism.
    assert pos.thesis_target_level == pytest.approx(tight_target)


def test_net_profit_floor_is_a_separate_mechanism_not_the_take_profit():
    """0.4% is the net floor gating EXIT_NET_PROFIT, not a 1.4% replacement.

    Proven by having both fire under different conditions with different reasons:
    a tight resolved target reached at +0.5% is EXIT_NET_PROFIT, and 1.4% from
    entry with no resolvable target is EXIT_TAKE_PROFIT_1.
    """
    assert pytest.approx(0.004) == MIN_NET_PROFIT_TO_SELL
    assert pytest.approx(AUTHORIZED_TP) != MIN_NET_PROFIT_TO_SELL

    symbol, entry = "BTC/USDT", ENTRIES["BTC/USDT"]
    tight_target = entry * 1.005

    net_profit = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry, thesis_target_level=tight_target),
        tight_target,
        0.005,
    )
    assert net_profit["action"] == "sell"
    assert net_profit["reason"] == EXIT_NET_PROFIT

    take_profit = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry),
        entry * (1.0 + AUTHORIZED_TP),
        0.014,
    )
    assert take_profit["reason"] == EXIT_TAKE_PROFIT_1
    assert net_profit["reason"] != take_profit["reason"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_net_profit_floor_is_equal_across_the_four_coins(symbol):
    assert min_net_profit_for_symbol(symbol) == pytest.approx(MIN_NET_PROFIT_TO_SELL)


# --------------------------------------------------------------------------
# 1.0% ordinary stop -- effective and reachable, distinct from the 2% floor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_ordinary_stop_fires_at_exactly_1_0_percent(symbol):
    entry = ENTRIES[symbol]
    out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * (1.0 - AUTHORIZED_SL), -0.0105)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_STOP_LOSS
    assert f"sl_pct={AUTHORIZED_SL:.4f}" in out["detail"]


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_the_stop_is_not_a_hidden_two_percent_substitute(symbol):
    """The tightest reachable adverse exit must be 1.0%, not the 2% risk floor.

    Before the repair the ordinary stop was unreachable under path-aware, so the
    first adverse exit was DAY_RISK_FLOOR_MIN_ADVERSE_PCT at 2% -- double the
    authorized stop. Scan downward and assert where the first sell appears.
    """
    entry = ENTRIES[symbol]
    first_sell_pct = None
    for tenth_bp in range(1, 300):  # 0.01% .. 2.99%
        drop = tenth_bp / 10000.0
        out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * (1.0 - drop), -drop)
        if out["action"] == "sell":
            first_sell_pct = drop
            assert out["reason"] == EXIT_STOP_LOSS
            break
    assert first_sell_pct is not None, "no adverse exit is reachable at all"
    assert first_sell_pct == pytest.approx(AUTHORIZED_SL, abs=1e-4)


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_catastrophic_floor_is_preserved_as_a_separate_deeper_mechanism(symbol):
    """A gap straight through the 1% stop must still be caught by the floor."""
    entry = ENTRIES[symbol]
    out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * 0.95, -0.05)
    assert out["action"] == "sell"
    assert out["reason"] in (EXIT_DAY_RISK_FLOOR, "EXTREME_PROTECTION_EXIT")
    assert out["reason"] != EXIT_STOP_LOSS


# --------------------------------------------------------------------------
# 0.25% trailing distance -- effective from the high-water mark
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_trail_distance_is_exactly_0_25_percent_from_the_high_water_mark(symbol):
    entry = ENTRIES[symbol]
    profile = _profile(symbol)
    high = entry * 1.02  # comfortably past the activation threshold
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=high, trail_pct=profile["trail"])

    assert refresh_trailing_stop(pos, high, profile) is True
    assert pos.trailing_stop_price == pytest.approx(high * (1.0 - AUTHORIZED_TRAIL))


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_trail_arms_at_0_25_percent_above_entry(symbol):
    """The high-water ratchet activates at entry * 1.0025, not before.

    Asserted on the resulting level rather than the return value, because
    refresh_trailing_stop also runs the break-even ratchet (trigger 0.15% MFE),
    which is a separate mechanism and would otherwise mask the activation point.
    """
    entry = ENTRIES[symbol]
    profile = _profile(symbol)

    # Just under the activation threshold: no high-water ratchet.
    low = _Pos(symbol=symbol, entry_price=entry, highest_price=entry * 1.0024)
    refresh_trailing_stop(low, entry * 1.0024, profile)
    assert low.trailing_stop_price != pytest.approx(entry * 1.0024 * (1.0 - AUTHORIZED_TRAIL))

    # At the threshold: the ratchet arms at the 0.25% distance.
    at_high = entry * (1.0 + AUTHORIZED_TRAIL)
    at = _Pos(symbol=symbol, entry_price=entry, highest_price=at_high)
    assert refresh_trailing_stop(at, at_high, profile) is True
    assert at.trailing_stop_price >= at_high * (1.0 - AUTHORIZED_TRAIL) - 1e-9


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_trail_breach_sells_and_never_widens_or_tightens(symbol):
    entry = ENTRIES[symbol]
    profile = _profile(symbol)
    high = entry * 1.012
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=high, trail_pct=profile["trail"])
    refresh_trailing_stop(pos, high, profile)
    trail = pos.trailing_stop_price

    out = _call(symbol, pos, trail * 0.9999, 0.008)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TRAILING_STOP

    # The distance must not change as price falls back; the level never lowers.
    assert refresh_trailing_stop(pos, trail * 0.9999, profile) is False
    assert pos.trailing_stop_price == pytest.approx(trail)


def test_bull_trail_widening_is_gone():
    """DAY_BULL_TRAIL_MULTIPLIER must not exist as a trail-distance override."""
    import backend.services.day_controlled_exits as dce

    assert not hasattr(dce, "_bull_trail_multiplier")
    assert not hasattr(dce, "_bull_trail_mfe_threshold")


def test_adaptive_trail_width_override_has_no_consumer():
    """day_adaptive_trail exposes a per-arm width, but nothing may consume it."""
    import subprocess

    out = subprocess.run(
        ["git", "grep", "-l", "adaptive_trail_pct_for_arm", "--", "backend/"],
        capture_output=True,
        text=True,
    ).stdout.split()
    assert out == ["backend/services/day_adaptive_trail.py"], f"unexpected consumers: {out}"


# --------------------------------------------------------------------------
# 300-minute maximum hold -- effective and reachable
# --------------------------------------------------------------------------


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_effective_max_hold_is_exactly_300(symbol):
    assert effective_max_hold_min(_Pos(symbol=symbol), _profile(symbol)) == AUTHORIZED_MAX_HOLD_MIN


@pytest.mark.parametrize("regime", ["bull", "BULL", "bear", "neutral", ""])
def test_no_regime_extends_the_ceiling_past_300(regime):
    """The bull extension made the real ceiling 342 min. It must be gone."""
    pos = _Pos(day_route_regime_at_entry=regime, max_hold_min=AUTHORIZED_MAX_HOLD_MIN)
    assert effective_max_hold_min(pos, _profile("BTC/USDT")) == AUTHORIZED_MAX_HOLD_MIN


def test_a_longer_stamped_hold_cannot_outlive_the_authorized_ceiling():
    """A position stamped 342 (or 2142) under an older profile must clamp to 300."""
    for stamped in (342, 360, 2142):
        pos = _Pos(max_hold_min=stamped, day_route_regime_at_entry="bull")
        assert effective_max_hold_min(pos, _profile("ETH/USDT")) == AUTHORIZED_MAX_HOLD_MIN


def test_bull_hold_extension_helper_is_gone():
    import backend.services.day_controlled_exits as dce

    assert not hasattr(dce, "_bull_hold_extension_min")


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_time_stop_is_reachable_at_the_300_minute_ceiling(symbol):
    """The time stop must fire on its own, with no 4H bundle present."""
    entry = ENTRIES[symbol]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry * 1.001)

    out = _call(symbol, pos, entry * 1.0005, 0.0005, hold=float(AUTHORIZED_MAX_HOLD_MIN))
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP
    assert f"max_hold_min={AUTHORIZED_MAX_HOLD_MIN}" in out["detail"]


def test_time_stop_fires_even_when_the_position_is_profitable_but_short_of_target():
    """The legacy ladder only timed out unprofitable positions, so a position
    above the net floor but below its target could outlive the ceiling."""
    symbol, entry = "SOL/USDT", ENTRIES["SOL/USDT"]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry * 1.006)
    out = _call(symbol, pos, entry * 1.006, 0.006, hold=301.0)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TIME_STOP


def test_profit_objectives_outrank_the_time_stop():
    """At the ceiling with the objective reached, book the profit, not a timeout."""
    symbol, entry = "ETH/USDT", ENTRIES["ETH/USDT"]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry * (1.0 + AUTHORIZED_TP))
    out = _call(symbol, pos, entry * (1.0 + AUTHORIZED_TP), 0.014, hold=400.0)
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_TAKE_PROFIT_1


# --------------------------------------------------------------------------
# Precedence: the exact order reported in the audit
# --------------------------------------------------------------------------


def test_documented_exit_precedence_order():
    """Assert the emit order of the live path-aware ladder.

    1 catastrophic protection, 2 catastrophic risk floor, 3 ordinary 1% stop,
    4 break-even/giveback, 5 0.25% trail, 6 net-profit target, 7 1.4% take
    profit, 8 stall, 9 300-min maximum hold.
    """
    import inspect
    import re

    from backend.services.day_controlled_exits import _evaluate_path_aware_exit

    src = inspect.getsource(_evaluate_path_aware_exit)
    emitted = list(re.findall(r"EXIT_[A-Z_0-9]+", src))
    # Collapse the repeated net-profit symbol (condition + return share the name).
    deduped = [x for i, x in enumerate(emitted) if i == 0 or x != emitted[i - 1]]

    assert deduped == [
        "EXIT_EXTREME_PROTECTION",
        "EXIT_DAY_RISK_FLOOR",
        "EXIT_STOP_LOSS",
        "EXIT_GIVEBACK",
        "EXIT_TRAILING_STOP",
        "EXIT_NET_PROFIT",
        "EXIT_TAKE_PROFIT_1",
        "EXIT_STALL_DEAD",
        "EXIT_TIME_STOP",
    ]


def test_armed_trail_outranks_both_profit_exits():
    symbol, entry = "BTC/USDT", ENTRIES["BTC/USDT"]
    profile = _profile(symbol)
    # The high must clear TP1 by more than the trail distance for the trail level
    # itself to sit above the 1.4% objective.
    high = entry * 1.025
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=high, trail_pct=profile["trail"])
    refresh_trailing_stop(pos, high, profile)

    # Mark is above the 1.4% objective and simultaneously at/below the trail.
    mark = pos.trailing_stop_price
    assert mark > entry * (1.0 + AUTHORIZED_TP)
    out = _call(symbol, pos, mark, 0.014)
    assert out["reason"] == EXIT_TRAILING_STOP


def test_ordinary_stop_outranks_giveback_and_stall():
    symbol, entry = "XRP/USDT", ENTRIES["XRP/USDT"]
    pos = _Pos(symbol=symbol, entry_price=entry, highest_price=entry * 1.004)
    out = _call(symbol, pos, entry * (1.0 - AUTHORIZED_SL), -0.011, hold=250.0)
    assert out["reason"] == EXIT_STOP_LOSS


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_every_authorized_exit_is_reachable_for_every_symbol(symbol):
    """No enabled exit may be structurally unreachable for any of the four."""
    entry = ENTRIES[symbol]
    profile = _profile(symbol)
    reached: dict[str, str] = {}

    out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * (1.0 - AUTHORIZED_SL), -0.011)
    reached[out["reason"]] = "stop"

    out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * 0.95, -0.05)
    reached[out["reason"]] = "floor"

    high = entry * 1.012
    trail_pos = _Pos(symbol=symbol, entry_price=entry, highest_price=high, trail_pct=profile["trail"])
    refresh_trailing_stop(trail_pos, high, profile)
    out = _call(symbol, trail_pos, trail_pos.trailing_stop_price * 0.9999, 0.008)
    reached[out["reason"]] = "trail"

    target = entry * 1.005
    out = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry, thesis_target_level=target),
        target,
        0.005,
    )
    reached[out["reason"]] = "net_profit"

    out = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry),
        entry * (1.0 + AUTHORIZED_TP),
        0.014,
    )
    reached[out["reason"]] = "take_profit"

    out = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry * 1.001),
        entry * 1.0005,
        0.0005,
        hold=float(AUTHORIZED_MAX_HOLD_MIN),
    )
    reached[out["reason"]] = "time_stop"

    for required in (
        EXIT_STOP_LOSS,
        EXIT_DAY_RISK_FLOOR,
        EXIT_TRAILING_STOP,
        EXIT_NET_PROFIT,
        EXIT_TAKE_PROFIT_1,
        EXIT_TIME_STOP,
    ):
        assert required in reached, f"{required} unreachable for {symbol}; reached={reached}"


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_a_profitable_position_closes_through_a_profit_exit_not_a_loss_exit(symbol):
    """A winner at its objective must not have to wait for an unrelated loss exit."""
    entry = ENTRIES[symbol]
    out = _call(
        symbol,
        _Pos(symbol=symbol, entry_price=entry, highest_price=entry * (1.0 + AUTHORIZED_TP)),
        entry * (1.0 + AUTHORIZED_TP),
        0.014,
        hold=120.0,
    )
    assert out["action"] == "sell"
    assert out["reason"] in (EXIT_TAKE_PROFIT_1, EXIT_NET_PROFIT, EXIT_TRAILING_STOP)
    assert out["reason"] not in (EXIT_STOP_LOSS, EXIT_DAY_RISK_FLOOR, EXIT_TIME_STOP, "STALL_EXIT_DEAD_NO_MFE")


@pytest.mark.parametrize("symbol", SYMBOLS)
def test_a_loss_cannot_run_past_the_ordinary_stop(symbol):
    """Every mark at or below the 1% stop must produce a sell, for every symbol.

    This is the invariant that was broken: with the ordinary stop unreachable,
    marks between -1% and -2% produced a hold and the position kept bleeding.
    """
    entry = ENTRIES[symbol]
    for tenth_bp in range(100, 200):  # -1.00% .. -1.99%
        drop = tenth_bp / 10000.0
        out = _call(symbol, _Pos(symbol=symbol, entry_price=entry), entry * (1.0 - drop), -drop)
        assert out["action"] == "sell", f"{symbol} held at -{drop * 100:.2f}%"
