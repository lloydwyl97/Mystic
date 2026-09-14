"""Setup discovery and intent validity. No 24h-green entry."""

from __future__ import annotations

import inspect

import pytest

from backend.config.day_entry_execution import VALID_ENTRY_MODES
from backend.config.day_setup_discovery import (
    EARLY_TREND,
    REJECT_EXTENDED,
    STRUCTURED_PULLBACK,
)
from backend.services.day_setup_discovery import (
    classify_setup,
    intent_invalid_reason,
    live_intent_validity,
    may_arm_setup,
)
from backend.services.day_trailing_buy import (
    CANCELED,
    DAY_TRADE_SYMBOLS,
    observe_book,
)
from backend.services.portfolio_engine import PortfolioEngine


def _bars(*, start: int, n: int, px: float, step: float = 0.0, vol: float = 10.0, bounce: bool = False) -> list:
    out = []
    price = px
    for i in range(n):
        ts = start + i * 60
        o = price
        if bounce and i > n - 8:
            price = price + abs(step)
        else:
            price = price + step
        h = max(o, price) + 0.05
        lo = min(o, price) - 0.05
        out.append((ts, o, h, lo, price, vol))
    return out


def test_early_trend_eligible_before_day_high():
    start = 1_700_000_000
    prior = _bars(start=start, n=80, px=103.5, step=-0.04, vol=9.0)
    consol = _bars(start=start + 80 * 60, n=50, px=100.0, step=0.0, vol=8.0)
    brk = _bars(start=start + 130 * 60, n=10, px=100.10, step=0.08, vol=20.0)
    bars = prior + consol + brk
    last = bars[-1]
    out = classify_setup(bars, symbol="ETHUSDT", ts=last[0], atr=2.0, ask=last[4])
    assert out["setup_class"] == EARLY_TREND
    assert last[4] < 104.0
    assert out["prior_high_240"] > last[4]


def test_extended_move_rejected():
    start = 1_700_000_000
    run = _bars(start=start, n=240, px=100.0, step=0.08, vol=12.0)
    last = run[-1]
    out = classify_setup(run, symbol="ETHUSDT", ts=last[0], atr=2.0, ask=last[4])
    assert out["setup_class"] == REJECT_EXTENDED


def test_noise_dip_does_not_qualify_as_structured_pullback():
    start = 1_700_000_000
    up = _bars(start=start, n=80, px=100.0, step=0.05, vol=10.0)
    # 14 bp dip only: 100.05 * 0.0014 ≈ 0.14
    tip = up[-1][4]
    noise = [
        (up[-1][0] + 60, tip, tip, tip * 0.9986, tip * 0.9986, 10.0),
        (up[-1][0] + 120, tip * 0.9986, tip * 0.9990, tip * 0.9986, tip * 0.9988, 10.0),
    ]
    bars = up + noise
    last = bars[-1]
    out = classify_setup(bars, symbol="ETHUSDT", ts=last[0], atr=2.0, ask=last[4])
    assert out["setup_class"] != STRUCTURED_PULLBACK


def test_structured_pullback_reclaim_qualifies():
    start = 1_700_000_000
    up = _bars(start=start, n=80, px=100.0, step=0.04, vol=10.0)
    peak = up[-1][4]
    pull = []
    ts = up[-1][0]
    px = peak
    for i in range(16):
        ts += 60
        px = px - 0.09 if i < 13 else px + 0.05
        pull.append((ts, px + 0.02, px + 0.04, px - 0.02, px, 11.0))
    bars = up + pull
    last = bars[-1]
    out = classify_setup(bars, symbol="BTCUSDT", ts=last[0], atr=2.0, ask=last[4])
    assert out["setup_class"] == STRUCTURED_PULLBACK


def test_stale_intent_cancelled():
    intent = {"arm_ts": 1000.0, "arm_ask": 100.0, "atr": 2.0, "status": "WAIT_DIP"}
    assert intent_invalid_reason(intent, ask=100.0, now=1000.0 + 901) == "STALE_INTENT_MAX_AGE"
    d = observe_book(
        {**intent, "min_dip_bps": 14, "rebound_bps": 4, "required_improvement_bps": 10, "expires_at": 1000 + 5000, "lowest_ask": 0},
        ask=100.0,
        now=1001.0,
        book_fresh=True,
        validity_reason="STALE_INTENT_MAX_AGE",
    )
    assert d.action == "cancel"
    assert d.status == CANCELED
    assert d.reason == "STALE_INTENT_MAX_AGE"


def test_broken_structure_invalidates_intent():
    intent = {"arm_ts": 1000.0, "arm_ask": 100.0, "atr": 2.0, "thesis_invalid_level": 99.5}
    assert intent_invalid_reason(intent, ask=99.4, now=1100.0) == "STRUCTURE_BROKEN"


def test_validity_shadow_does_not_cancel_until_enforced(monkeypatch):
    monkeypatch.setenv("DAY_SETUP_VALIDITY_ENFORCE", "false")
    intent = {"arm_ts": 1000.0, "arm_ask": 100.0, "atr": 2.0}
    assert live_intent_validity(intent, ask=100.0, now=1000.0 + 901) == ""
    monkeypatch.setenv("DAY_SETUP_VALIDITY_ENFORCE", "true")
    assert live_intent_validity(intent, ask=100.0, now=1000.0 + 901) == "STALE_INTENT_MAX_AGE"


def test_no_immediate_legacy_buy_mode():
    assert frozenset({"trailing_buy"}) == VALID_ENTRY_MODES
    src = inspect.getsource(PortfolioEngine.process_bar_candidates)
    assert "await self.execute_buy_fifo" not in src


def test_all_four_symbols_remain_eligible():
    assert DAY_TRADE_SYMBOLS == ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    start = 1_700_000_000
    px0 = {"BTCUSDT": 78000.0, "ETHUSDT": 2500.0, "SOLUSDT": 100.0, "XRPUSDT": 1.40}
    atr0 = {"BTCUSDT": 650.0, "ETHUSDT": 30.0, "SOLUSDT": 1.5, "XRPUSDT": 0.02}
    for sym in DAY_TRADE_SYMBOLS:
        p = px0[sym]
        step = p * 0.0008
        prior = _bars(start=start, n=80, px=p * 1.03, step=-step, vol=9.0)
        consol = _bars(start=start + 80 * 60, n=50, px=p, step=0.0, vol=8.0)
        brk = _bars(start=start + 130 * 60, n=10, px=p * 1.001, step=step, vol=20.0)
        bars = prior + consol + brk
        last = bars[-1]
        out = classify_setup(bars, symbol=sym, ts=last[0], atr=atr0[sym], ask=last[4])
        assert out["setup_class"] in {EARLY_TREND, STRUCTURED_PULLBACK}, (sym, out["setup_class"], out["reason"])
        assert may_arm_setup(out) is True
