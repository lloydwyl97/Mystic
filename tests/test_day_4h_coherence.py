"""DAY 4H entry/exit coherence: fresh forming close + pre-buy consistency."""

from __future__ import annotations

from datetime import datetime, timezone

from backend.services.day_controlled_exits import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    evaluate_engine_managed_exit,
    evaluate_pre_buy_exit_consistency,
)
from backend.services.day_trade_thesis import (
    current_utc_4h_open_ms,
    day_4h_structure_snapshot,
    fourh_requires_boundary_refresh,
    htf_4h_rise_broken,
    resolve_day_4h_structure_bundle,
)
from backend.services.portfolio_engine import get_coin_profile


DAY_4H_MS = 4 * 3600 * 1000


class _Pos:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "BTC/USDT")
        self.entry_price = kw.get("entry_price", 79637.25)
        self.highest_price = kw.get("highest_price", self.entry_price)
        self.lowest_price = kw.get("lowest_price", self.entry_price)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.004)
        self.take_profit_1_price = 0.0
        self.entry_thesis = kw.get("entry_thesis", "HTF_TREND_PULLBACK")
        self.entry_vwap = kw.get("entry_vwap", self.entry_price)
        self.thesis_invalid_level = 0.0
        self.thesis_target_level = 0.0
        self.thesis_score = 0.7
        self.max_hold_min = 360
        self.day_route_regime_at_entry = "bull"


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _bar(open_iso: str, o: float, h: float, l: float, c: float) -> list:
    return [int(_ts(open_iso) * 1000), o, h, l, c, 100.0]


def _rising_prefix(n: int, end_open_ms: int, start: float = 79000.0) -> list[list]:
    rows = []
    px = start
    ot = end_open_ms - n * DAY_4H_MS
    for _ in range(n):
        o = px
        c = px * 1.004
        rows.append([ot, o, c * 1.001, o * 0.999, c, 100.0])
        px = c
        ot += DAY_4H_MS
    return rows


def _btc_bundle(*, forming_close: float, prior_low: float = 79764.12) -> dict:
    """04:00 forming bar + 00:00 prior with given low. Rising history for align=1.0."""
    now_open = int(_ts("2026-08-28T04:00:00Z") * 1000)
    prior_open = now_open - DAY_4H_MS
    prefix = _rising_prefix(60, prior_open)
    prior = [prior_open, 80272.15, 81439.52, prior_low, 79829.70, 100.0]
    forming = [now_open, 79834.93, 80000.0, 79570.32, forming_close, 100.0]
    return {"4h": prefix + [prior, forming]}


def _pre_buy(*, entry: float, bundle: dict, now_epoch: float | None = None) -> dict:
    return evaluate_pre_buy_exit_consistency(
        setup="HTF_TREND_PULLBACK",
        entry_price=entry,
        stop_price=entry * 0.98,
        thesis_invalid_level=0.0,
        thesis_target_level=entry * 1.02,
        entry_vwap=entry,
        entry_ts=now_epoch or _ts("2026-08-28T05:00:09Z"),
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        spread_pct=0.0004,
        day_regime="bull",
        decision_data={"thesis_score": 0.8},
        thesis_score=0.8,
        bar_ts=now_epoch,
    )


def test_same_fresh_state_entry_and_exit_agree_broken():
    now = _ts("2026-08-28T05:00:09Z")
    bundle = _btc_bundle(forming_close=79850.0)
    mark = 79650.0
    snap_e = day_4h_structure_snapshot(bundle, current_price=mark, now_epoch=now)
    snap_x = day_4h_structure_snapshot(bundle, current_price=mark, now_epoch=now)
    assert snap_e["htf_4h_rise_broken"] is True
    assert snap_x["htf_4h_rise_broken"] is True
    assert snap_e["htf_4h_rise_broken"] == snap_x["htf_4h_rise_broken"]
    assert snap_e["current_4h_close"] == snap_x["current_4h_close"] == mark
    assert snap_e["prior_4h_low"] == snap_x["prior_4h_low"] == 79764.12


def test_fresh_mark_overrides_stale_forming_close_not_broken():
    """BTC #5: cached 79694, live ~79900 > prior 79764 → not broken."""
    now = _ts("2026-08-28T06:15:08Z")
    bundle = _btc_bundle(forming_close=79694.83)
    snap = day_4h_structure_snapshot(bundle, current_price=79900.0, now_epoch=now)
    assert snap["prior_4h_low"] == 79764.12
    assert snap["current_4h_close"] == 79900.0
    assert snap["htf_4h_rise_broken"] is False
    assert snap["forming_close_source"] == "canonical_mark"


def test_fresh_mark_overrides_stale_forming_close_broken():
    now = _ts("2026-08-28T05:00:09Z")
    bundle = _btc_bundle(forming_close=79850.0)
    snap = day_4h_structure_snapshot(bundle, current_price=79650.0, now_epoch=now)
    assert snap["prior_4h_low"] == 79764.12
    assert snap["current_4h_close"] == 79650.0
    assert snap["htf_4h_rise_broken"] is True


def test_pre_buy_honors_immediate_4h_structure_break():
    """BTC #4: same state already requires sell → no buy."""
    now = _ts("2026-08-28T05:00:09Z")
    entry = 79637.25
    bundle = _btc_bundle(forming_close=79672.42)
    result = _pre_buy(entry=entry, bundle=bundle, now_epoch=now)
    assert result["allowed"] is False
    assert "DAY_4H_STRUCTURE_BREAK" in result["block_reason"]
    assert result["immediate_exit_reason"] == EXIT_DAY_4H_STRUCTURE_BREAK


def test_btc4_cannot_buy_then_sell_same_structure():
    now = _ts("2026-08-28T05:00:09Z")
    entry = 79637.25
    bundle = _btc_bundle(forming_close=79672.42)
    pre = _pre_buy(entry=entry, bundle=bundle, now_epoch=now)
    pos = _Pos(entry_price=entry, highest_price=entry)
    managed = evaluate_engine_managed_exit(
        position=pos,
        current_price=entry,
        net_pnl_pct=-0.0008,
        hold_minutes=0.6,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        now_epoch=now,
    )
    assert pre["allowed"] is False
    assert managed["action"] == "sell"
    assert managed["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert managed["htf_4h_rise_broken"] is True


def test_btc5_stale_close_does_not_false_exit():
    now = _ts("2026-08-28T06:15:43Z")
    entry = 79903.71
    bundle = _btc_bundle(forming_close=79694.83)
    pos = _Pos(entry_price=entry, highest_price=entry)
    managed = evaluate_engine_managed_exit(
        position=pos,
        current_price=79899.89,
        net_pnl_pct=-0.0001,
        hold_minutes=0.6,
        coin_profile=get_coin_profile("BTCUSDT"),
        bundle=bundle,
        now_epoch=now,
    )
    assert managed["htf_4h_rise_broken"] is False
    assert managed.get("reason") != EXIT_DAY_4H_STRUCTURE_BREAK
    assert managed["action"] == "hold"


def test_sol1_post_entry_break_still_exits():
    now_entry = _ts("2026-08-28T01:45:12Z")
    now_exit = _ts("2026-08-28T02:01:06Z")
    bar0 = int(_ts("2026-08-28T00:00:00Z") * 1000)
    prefix = _rising_prefix(60, bar0, start=100.0)
    prior = _bar("2026-08-27T20:00:00Z", 109.19, 110.54, 108.28, 109.15)
    forming = [bar0, 109.17, 110.0, 108.30, 108.56, 100.0]
    bundle = {"4h": prefix[:-1] + [prior, forming]}
    pos = _Pos(symbol="SOL/USDT", entry_price=108.40, highest_price=108.56, trail_pct=0.0055)
    at_entry = evaluate_engine_managed_exit(
        position=pos,
        current_price=108.36,
        net_pnl_pct=-0.0004,
        hold_minutes=0.0,
        coin_profile=get_coin_profile("SOLUSDT"),
        bundle=bundle,
        now_epoch=now_entry,
    )
    assert at_entry["htf_4h_rise_broken"] is False
    at_exit = evaluate_engine_managed_exit(
        position=pos,
        current_price=107.77,
        net_pnl_pct=-0.0058,
        hold_minutes=15.9,
        coin_profile=get_coin_profile("SOLUSDT"),
        bundle=bundle,
        now_epoch=now_exit,
    )
    assert at_exit["action"] == "sell"
    assert at_exit["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert day_4h_structure_snapshot(bundle, current_price=108.36, now_epoch=now_entry)["htf_4h_rise_broken"] is False
    assert day_4h_structure_snapshot(bundle, current_price=107.77, now_epoch=now_exit)["htf_4h_rise_broken"] is True


def test_sol3_new_bar_prior_low_rollover_exits():
    """Bundle still has 04:00 as last bar after 08:00 UTC; prior rolls to 04:00 low."""
    now_exit = _ts("2026-08-28T10:12:06Z")
    bar04 = int(_ts("2026-08-28T04:00:00Z") * 1000)
    prefix = _rising_prefix(60, bar04, start=100.0)
    prior_00 = _bar("2026-08-28T00:00:00Z", 109.17, 110.0, 105.92, 106.82)
    forming_04 = [bar04, 106.88, 107.80, 106.30, 106.37, 100.0]
    bundle = {"4h": prefix[:-1] + [prior_00, forming_04]}
    assert fourh_requires_boundary_refresh(bundle["4h"], now_exit) is True
    snap = day_4h_structure_snapshot(bundle, current_price=105.06, now_epoch=now_exit)
    assert snap["prior_4h_low"] == 106.30
    assert snap["current_4h_close"] == 105.06
    assert snap["htf_4h_rise_broken"] is True
    pos = _Pos(symbol="SOL/USDT", entry_price=107.32, highest_price=107.80, trail_pct=0.0055)
    out = evaluate_engine_managed_exit(
        position=pos,
        current_price=105.80,
        net_pnl_pct=-0.014,
        hold_minutes=356.9,
        coin_profile=get_coin_profile("SOLUSDT"),
        bundle=bundle,
        now_epoch=now_exit,
    )
    assert out["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK


def test_utc_4h_boundary_forces_identity_refresh():
    now = _ts("2026-08-28T04:00:30Z")
    prev = int(_ts("2026-08-28T00:00:00Z") * 1000)
    rows = [[prev - i * DAY_4H_MS, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(5)]
    rows.reverse()
    assert fourh_requires_boundary_refresh(rows, now) is True
    rows[-1][0] = current_utc_4h_open_ms(now)
    assert fourh_requires_boundary_refresh(rows, now) is False


def test_no_lookahead_future_4h_bar():
    now = _ts("2026-08-28T02:00:00Z")
    bar00 = int(_ts("2026-08-28T00:00:00Z") * 1000)
    future = int(_ts("2026-08-28T04:00:00Z") * 1000)
    prefix = _rising_prefix(60, bar00, start=100.0)
    prior = _bar("2026-08-27T20:00:00Z", 109.19, 110.54, 108.28, 109.15)
    forming = [bar00, 109.17, 110.0, 108.30, 108.40, 100.0]
    future_bar = [future, 106.88, 107.8, 106.3, 105.00, 100.0]
    bundle = {"4h": prefix[:-1] + [prior, forming, future_bar]}
    resolved = resolve_day_4h_structure_bundle(bundle, current_price=108.40, now_epoch=now)
    last_ot = resolved["4h"][-1][0]
    assert last_ot == bar00
    assert day_4h_structure_snapshot(bundle, current_price=108.40, now_epoch=now)["current_4h_close"] == 108.40
    assert htf_4h_rise_broken(bundle, current_price=108.40, now_epoch=now) is False


def test_1m_close_used_when_mark_absent():
    now = _ts("2026-08-28T06:15:08Z")
    bundle = _btc_bundle(forming_close=79694.83)
    m1_open = int(_ts("2026-08-28T06:15:00Z") * 1000)
    bundle["1m"] = [[m1_open, 79903.71, 79903.71, 79884.92, 79899.89, 1.0]]
    snap = day_4h_structure_snapshot(bundle, now_epoch=now)
    assert snap["current_4h_close"] == 79899.89
    assert snap["htf_4h_rise_broken"] is False
    assert snap["forming_close_source"] == "1m_close"


def test_mtf_align_only_bundle_misses_already_broken_4h():
    """Production hole: pre-buy used ranking MTF dicts, not OHLCV."""
    now = _ts("2026-08-28T16:30:10Z")
    mtf_only = {"4h": {"ema_align": 1.0, "trend": 1.0, "rsi": 60.0}}
    snap = day_4h_structure_snapshot(mtf_only, current_price=2434.04, now_epoch=now)
    assert snap["4h_bundle_missing"] is True
    assert snap["htf_4h_rise_broken"] is False
    allowed = _pre_buy(entry=2434.04, bundle=mtf_only, now_epoch=now)
    assert allowed["allowed"] is True


def test_clean_window_eth_xrp_btc_already_below_prior_low_blocked():
    """Ocean 2026-08-28 clean-window BUY→seconds-later 4H-break cases."""
    cases = [
        ("2026-08-28T16:30:10Z", 2434.04, 2470.0, 2432.71),
        ("2026-08-28T17:45:11Z", 1.3865, 1.3879, 1.38625),
        ("2026-08-28T19:15:10Z", 77664.23, 78329.04, 77663.765),
    ]
    for iso, entry, prior_low, mark in cases:
        now = _ts(iso)
        open_ms = current_utc_4h_open_ms(now)
        prior_ms = open_ms - DAY_4H_MS
        prefix = _rising_prefix(60, prior_ms, start=prior_low * 0.95)
        prior = [prior_ms, prior_low * 1.01, prior_low * 1.02, prior_low, prior_low * 1.005, 100.0]
        forming = [open_ms, prior_low * 0.99, prior_low * 1.001, prior_low * 0.98, mark, 100.0]
        bundle = {"4h": prefix + [prior, forming]}
        snap = day_4h_structure_snapshot(bundle, current_price=entry, now_epoch=now)
        assert snap["prior_4h_low"] == prior_low
        assert snap["htf_4h_rise_broken"] is True
        pre = _pre_buy(entry=entry, bundle=bundle, now_epoch=now)
        assert pre["allowed"] is False, iso
        assert pre["immediate_exit_reason"] == EXIT_DAY_4H_STRUCTURE_BREAK


def test_resolve_pre_buy_prefers_ohlcv_cache(monkeypatch):
    from backend.services.day_active_market_bundle import resolve_pre_buy_day_structure_bundle

    ohlcv = {"4h": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], "1h": []}
    mtf = {"4h": {"ema_align": 1.0, "trend": 1.0}}
    monkeypatch.setattr(
        "backend.services.day_active_market_bundle.read_cached_day_active_bundle_sync",
        lambda _sym: ohlcv,
    )
    got = resolve_pre_buy_day_structure_bundle("ETH/USDT", mtf)
    assert got["4h"][1][3] == 10
    monkeypatch.setattr(
        "backend.services.day_active_market_bundle.read_cached_day_active_bundle_sync",
        lambda _sym: None,
    )
    got = resolve_pre_buy_day_structure_bundle("ETH/USDT", mtf)
    assert got["4h"]["ema_align"] == 1.0
