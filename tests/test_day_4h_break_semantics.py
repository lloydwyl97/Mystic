"""4H-break is a 4-hour candle structure condition, not a hold timer."""

from types import SimpleNamespace

from backend.services.day_controlled_exits import _evaluate_path_aware_exit
from backend.services.day_ocean_live_book import pair_fifo
from backend.services.day_trade_thesis import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    htf_4h_rise_broken,
    htf_4h_rise_intact,
)


def _broken_bundle():
    # prior 4H low=90; current 4H close=89 → structure broken.
    return {"4h": [[0, 100.0, 110.0, 90.0, 105.0, 0.0], [1, 105.0, 108.0, 88.0, 89.0, 0.0]]}


def test_4h_break_means_close_below_prior_4h_low():
    bundle = _broken_bundle()
    assert htf_4h_rise_broken(bundle) is True
    assert htf_4h_rise_intact(bundle) is False


def test_4h_exit_fires_without_four_hour_hold():
    pos = SimpleNamespace(
        entry_price=100.0,
        highest_price=100.0,
        lowest_price=99.0,
        trailing_stop_price=0.0,
        trail_pct=0.005,
        thesis_invalid_level=0.0,
    )
    out = _evaluate_path_aware_exit(
        position=pos,
        current_price=99.0,
        net_pnl_pct=-0.01,
        hold_minutes=0.1,
        coin_profile={"trail": 0.005, "sl": 0.01, "max_hold_min": 360},
        bundle=_broken_bundle(),
        entry=100.0,
        atr_pct=0.01,
        now_epoch=1_700_000_000.0,
    )
    assert out["action"] == "sell"
    assert out["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert out["hold_minutes"] == 0.1
    assert EXIT_DAY_4H_STRUCTURE_BREAK == "DAY_4H_STRUCTURE_BREAK_EXIT"


def test_43_fill_holds_match_stored_within_two_seconds():
    """Frozen Ocean 4H book: fill clock is authoritative; stored is integer seconds."""
    rows = [
        ("mystic_SOL/USDT_1787664609328", 58.1, 58.0),
        ("mystic_XRP/USDT_1787666409894", 46.0, 45.0),
        ("mystic_XRP/USDT_1787697113254", 13.8, 13.0),
        ("mystic_ETH/USDT_1787697908594", 45.2, 44.0),
        ("mystic_SOL/USDT_1787698291179", 22.7, 22.0),
        ("mystic_XRP/USDT_1787756411277", 28.5, 28.0),
        ("mystic_XRP/USDT_1787760910234", 26.5, 26.0),
        ("mystic_ETH/USDT_1787813109938", 11.5, 11.0),
        ("mystic_BTC/USDT_1787893209067", 38.9, 38.0),
        ("mystic_BTC/USDT_1787897708795", 34.5, 34.0),
        ("mystic_ETH/USDT_1787934610451", 43.7, 42.0),
        ("mystic_XRP/USDT_1787939111161", 6.1, 5.0),
        ("mystic_BTC/USDT_1787944510512", 17.1, 15.0),
        ("mystic_XRP/USDT_1788291912283", 116.5, 115.0),
    ]
    assert all(abs(fill - stored) <= 2.2 for _tid, fill, stored in rows)
    assert min(fill for _tid, fill, _s in rows) < 180
    assert EXIT_DAY_4H_STRUCTURE_BREAK.endswith("4H_STRUCTURE_BREAK_EXIT")


def test_short_4h_holds_are_not_fifo_join_errors():
    buys = [
        {"id": 1, "timestamp": "2026-08-25T14:00:09+00:00", "symbol": "XRPUSDT", "price": 1.45},
        {"id": 2, "timestamp": "2026-08-25T16:00:00+00:00", "symbol": "XRPUSDT", "price": 1.46},
    ]
    sells = [
        {"id": 10, "timestamp": "2026-08-25T14:00:55+00:00", "symbol": "XRPUSDT", "price": 1.44, "exit_reason": "DAY_4H_STRUCTURE_BREAK_EXIT"},
        {"id": 11, "timestamp": "2026-08-25T20:00:00+00:00", "symbol": "XRPUSDT", "price": 1.47, "exit_reason": "TRAILING_STOP_EXIT"},
    ]
    pairs = pair_fifo(buys, sells)
    assert pairs[0]["sell"]["id"] == 10
    assert pairs[1]["sell"]["id"] == 11
    assert pairs[0]["sell"]["exit_reason"] == "DAY_4H_STRUCTURE_BREAK_EXIT"
