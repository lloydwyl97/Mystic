"""Captured XRP tbfbaa25b3736745b6 4H-consistency and dust regressions.

Values are from mystic-prod 2026-09-15 22:36 UTC. Test-only IDs are not
production exchange identifiers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.day_controlled_exits import (
    COMPLETED_4H_ALREADY_INVALID,
    EXIT_DAY_4H_STRUCTURE_BREAK,
    evaluate_completed_4h_buy_hard_safety,
    evaluate_engine_managed_exit,
)
from backend.services.day_trade_thesis import day_4h_structure_snapshot
from backend.services.day_trailing_buy import HARD_SAFETY_REJECTED, _pre_submit_safety, classify_trailing_submit_outcome
from backend.services.live_fill_economics import (
    LiveCommission,
    apply_live_buy_economics,
    extract_live_commission,
    sellable_and_residual_qty,
)
from backend.services.portfolio_engine import OpenPosition, PortfolioEngine, Sleeve, get_coin_profile

DAY_4H_MS = 4 * 3600 * 1000
XRP_INTENT = "tbfbaa25b3736745b6"
XRP_PRIOR_4H_LOW = 1.2892
XRP_EXIT_4H_CLOSE = 1.28225
XRP_TRIGGER_ASK = 1.2817
XRP_SUBMITTED_QTY = Decimal("44.400000")
XRP_EXECUTED_QTY = Decimal("44.400000")
XRP_COMMISSION_XRP = Decimal("0.008880")
XRP_COMMISSION_USD = Decimal("0.01138150")
XRP_NET_CREDITED = Decimal("44.391120")
XRP_STEP = Decimal("0.1")
XRP_SOLD = Decimal("44.300000")
XRP_RESIDUAL = Decimal("0.091120")
XRP_RESIDUAL_VALUE = XRP_RESIDUAL * Decimal("1.2817")


def _ts(iso: str) -> float:
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def _rising_prefix(n: int, end_open_ms: int, start: float = 1.20) -> list[list]:
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


def _xrp_bundle(*, forming_close: float, prior_low: float = XRP_PRIOR_4H_LOW) -> dict:
    now_open = int(_ts("2026-09-15T20:00:00Z") * 1000)
    prior_open = now_open - DAY_4H_MS
    prefix = _rising_prefix(60, prior_open)
    prior = [prior_open, 1.2950, 1.3100, prior_low, 1.2930, 100.0]
    forming = [now_open, 1.2930, 1.2960, 1.2800, forming_close, 100.0]
    return {"4h": [*prefix, prior, forming]}


class _Pos:
    def __init__(self, **kw):
        self.symbol = kw.get("symbol", "XRP/USDT")
        self.entry_price = kw.get("entry_price", XRP_TRIGGER_ASK)
        self.highest_price = kw.get("highest_price", self.entry_price)
        self.lowest_price = kw.get("lowest_price", self.entry_price)
        self.stop_price = kw.get("stop_price", 0.0)
        self.trailing_stop_price = kw.get("trailing_stop_price", 0.0)
        self.trail_pct = kw.get("trail_pct", 0.004)
        self.take_profit_1_price = 0.0
        self.entry_thesis = kw.get("entry_thesis", "")
        self.entry_vwap = kw.get("entry_vwap", self.entry_price)
        self.thesis_invalid_level = 0.0
        self.thesis_target_level = 0.0
        self.thesis_score = 0.0
        self.max_hold_min = 360
        self.day_route_regime_at_entry = ""


def test_captured_xrp_4h_already_invalid_at_submit():
    now = _ts("2026-09-15T22:36:28Z")
    bundle = _xrp_bundle(forming_close=XRP_EXIT_4H_CLOSE)
    snap_entry = day_4h_structure_snapshot(bundle, current_price=XRP_TRIGGER_ASK, now_epoch=now)
    snap_exit = day_4h_structure_snapshot(bundle, current_price=XRP_EXIT_4H_CLOSE, now_epoch=now)
    assert snap_entry["htf_4h_rise_broken"] is True
    assert snap_exit["htf_4h_rise_broken"] is True
    assert snap_entry["prior_4h_low"] == snap_exit["prior_4h_low"] == XRP_PRIOR_4H_LOW
    hard = evaluate_completed_4h_buy_hard_safety(mark=XRP_TRIGGER_ASK, bundle=bundle, now_epoch=now)
    assert hard["allowed"] is False
    assert hard["block_reason"] == COMPLETED_4H_ALREADY_INVALID
    assert hard["immediate_exit_reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    managed = evaluate_engine_managed_exit(
        position=_Pos(),
        current_price=XRP_EXIT_4H_CLOSE,
        net_pnl_pct=-0.00017,
        hold_minutes=0.2,
        coin_profile=get_coin_profile("XRPUSDT"),
        bundle=bundle,
        now_epoch=now,
    )
    assert managed["action"] == "sell"
    assert managed["reason"] == EXIT_DAY_4H_STRUCTURE_BREAK
    assert managed["htf_4h_rise_broken"] is True


def test_intact_completed_4h_still_allows_buy():
    now = _ts("2026-09-15T22:36:28Z")
    bundle = _xrp_bundle(forming_close=1.2950, prior_low=1.2700)
    hard = evaluate_completed_4h_buy_hard_safety(mark=1.2940, bundle=bundle, now_epoch=now)
    assert hard["allowed"] is True
    assert hard["htf_4h_rise_broken"] is False
    assert hard["block_reason"] == ""


@pytest.mark.asyncio
async def test_pre_submit_rejects_captured_xrp_before_exchange():
    now = _ts("2026-09-15T22:36:28Z")
    bundle = _xrp_bundle(forming_close=XRP_EXIT_4H_CLOSE)
    engine = SimpleNamespace(
        _check_kill_switch_buy=lambda: (True, ""),
        _trading_paused=False,
        _can_open_position=AsyncMock(return_value=(True, "")),
        open_positions={},
        _pending_buy_order_symbols=lambda: [],
    )
    intent = {"symbol": "XRP/USDT", "intent_id": XRP_INTENT, "notional_usd": 57.0, "decision_id": "day_XRPUSDT_1789511708854"}
    with (
        patch(
            "backend.services.day_active_market_bundle.resolve_pre_buy_day_structure_bundle",
            return_value=bundle,
        ),
        patch("backend.services.day_trailing_buy.time.time", return_value=now),
    ):
        ok, reason = await _pre_submit_safety(engine, intent, XRP_TRIGGER_ASK)
    assert ok is False
    assert reason == COMPLETED_4H_ALREADY_INVALID
    outcome, retryable = classify_trailing_submit_outcome(reason)
    assert outcome.startswith(HARD_SAFETY_REJECTED)
    assert COMPLETED_4H_ALREADY_INVALID in outcome
    assert retryable is False


def test_xrp_commission_asset_and_net_credited():
    order = {
        "filled": float(XRP_EXECUTED_QTY),
        "average": XRP_TRIGGER_ASK,
        "info": {"fills": [{"commission": str(XRP_COMMISSION_XRP), "commissionAsset": "XRP"}]},
    }
    comm = extract_live_commission(order, symbol="XRP/USDT", fill_price=XRP_TRIGGER_ASK)
    assert comm.fee_from_exchange is True
    assert comm.base_qty_reduction == pytest.approx(float(XRP_COMMISSION_XRP))
    received, fee, _cash = apply_live_buy_economics(
        filled_qty=float(XRP_EXECUTED_QTY),
        fill_price=XRP_TRIGGER_ASK,
        modeled_fee=0.01,
        commission=comm,
    )
    assert Decimal(str(received)) == XRP_NET_CREDITED
    assert abs(Decimal(str(fee)) - XRP_COMMISSION_USD) < Decimal("0.00001")
    sellable, residual = sellable_and_residual_qty(credited_qty=received, qty_step=XRP_STEP)
    assert sellable == XRP_SOLD
    assert residual == XRP_RESIDUAL
    assert residual * Decimal(str(XRP_TRIGGER_ASK)) == XRP_RESIDUAL_VALUE


def test_exit_sells_max_step_and_preserves_residual():
    eng = PortfolioEngine(principal=228.0, test_mode=True)
    eng._symbol_constraints["XRP/USDT"] = {"qty_step": 0.1, "min_qty": 0.1, "min_notional": 10.0}
    pos = OpenPosition(
        symbol="XRP/USDT",
        quantity=float(XRP_NET_CREDITED),
        entry_price=XRP_TRIGGER_ASK,
        entry_time=_ts("2026-09-15T22:36:31Z"),
        trade_id="TEST_ONLY_XRP_BUY",
        stop_price=1.268883,
        take_profit_1_price=1.2996438,
        take_profit_2_price=0.0,
        highest_price=XRP_TRIGGER_ASK,
        lowest_price=XRP_TRIGGER_ASK,
        sleeve=Sleeve.ACTIVE.value,
        status="ACTIVE",
    )
    rem = eng._apply_confirmed_sell_qty_to_memory(
        pos,
        float(XRP_SOLD),
        fill_price=1.2819,
        qty_step=0.1,
        min_qty=0.1,
        min_notional=10.0,
    )
    assert abs(rem - float(XRP_RESIDUAL)) < 1e-9
    assert pos.status == "DUST_PENDING"
    assert abs(pos.dust_qty_canonical - float(XRP_RESIDUAL)) < 1e-9


@pytest.mark.asyncio
async def test_real_residual_is_not_written_off_as_loss():
    eng = PortfolioEngine(principal=228.0, test_mode=True)
    pos = OpenPosition(
        symbol="XRP/USDT",
        quantity=float(XRP_RESIDUAL),
        entry_price=XRP_TRIGGER_ASK,
        entry_time=_ts("2026-09-15T22:36:31Z"),
        trade_id="TEST_ONLY_XRP_BUY",
        stop_price=1.26,
        take_profit_1_price=1.30,
        take_profit_2_price=0.0,
        highest_price=XRP_TRIGGER_ASK,
        lowest_price=XRP_TRIGGER_ASK,
        sleeve=Sleeve.ACTIVE.value,
        status="ACTIVE",
    )
    eng.open_positions["XRP/USDT"] = pos
    eng._persist_position_to_sqlite = AsyncMock()
    await eng._remove_dust_position_canonical_cleanup("XRP/USDT", pos)
    assert "XRP/USDT" in eng.open_positions
    assert eng.open_positions["XRP/USDT"].status == "DUST_PENDING"
    assert abs(eng.open_positions["XRP/USDT"].quantity - float(XRP_RESIDUAL)) < 1e-9


def test_classify_completed_4h_is_terminal_hard_safety():
    outcome, retryable = classify_trailing_submit_outcome(COMPLETED_4H_ALREADY_INVALID)
    assert outcome == f"{HARD_SAFETY_REJECTED}:{COMPLETED_4H_ALREADY_INVALID}"
    assert retryable is False
