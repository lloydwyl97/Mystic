"""Full-precision ETH/XRP trailing-buy cash and reservation repair."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest

from backend.services.day_entry_spendable import (
    INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION,
    cash_covers,
    is_terminal_buy_cash_reason,
    money,
    plan_executable_buy,
    spendable_quote,
)
from backend.services.day_trailing_buy import (
    _pre_submit_safety,
    _submit_claimed,
    observe_book,
    rebound_trigger_ask,
    retained_improvement_ceiling_ask,
)
from backend.services.day_trailing_buy_store import (
    CANCELED,
    EXPIRED,
    SUBMITTING,
    TRAIL_LOW,
    claim_submitting,
    create_intent,
    load_intent,
    mark_order_accepted,
    release_submitting_for_retry,
    update_watch,
)
from backend.services.portfolio_engine import PortfolioEngine


@pytest.fixture(autouse=True)
def _intact_4h_for_cash_tests():
    with patch(
        "backend.services.day_active_market_bundle.resolve_pre_buy_day_structure_bundle",
        return_value={},
    ):
        yield


# Captured ETH tbace6b0a92e3745d8 / XRP tbb226f475c96e428e (Ocean SQLite).
ETH_ACCOUNT_CASH = Decimal("227.12229294")
ETH_OWN = Decimal("62.291423")
ETH_SOL_OTHER = Decimal("62.339475")
ETH_XRP_OTHER = Decimal("62.39835000000001")
ETH_BTC_OTHER = Decimal("40.09304494")
ETH_OTHER = ETH_SOL_OTHER + ETH_XRP_OTHER + ETH_BTC_OTHER
ETH_QTY = Decimal("0.0251")
ETH_ARM_ASK = Decimal("2481.73")
ETH_SUBMIT_ASK = Decimal("2475.46")
ETH_STEP = Decimal("0.0001")
ETH_MIN_QTY = Decimal("0.0001")
ETH_MIN_NOTIONAL = Decimal("1.0")
ETH_TAKER = Decimal("0.0002")
ETH_SLIP = Decimal("0.0001")
ETH_DECISION = "day_ETHUSDT_1789338869283"

XRP_QTY = Decimal("46.5")
XRP_ARM_ASK = Decimal("1.3419")
XRP_SUBMIT_ASK = Decimal("1.3397")
XRP_OWN = Decimal("62.39835000000001")
XRP_STEP = Decimal("0.1")
XRP_MIN_QTY = Decimal("0.1")
XRP_MIN_NOTIONAL = Decimal("1.0")
XRP_DECISION = "day_XRPUSDT_1789338880557"


def _arm_and_claim(db, row):
    update_watch(db, row["intent_id"], status=TRAIL_LOW, lowest_ask=1.0, lowest_ask_ts=time.time())
    return claim_submitting(db, row["intent_id"])


def test_eth_other_reservations_subtracted_once():
    leftover = spendable_quote(
        account_cash=ETH_ACCOUNT_CASH,
        other_reservations=ETH_OTHER,
        open_order_commitment=0,
        own_reservation=ETH_OWN,
        include_own_reservation=True,
    )
    assert leftover == ETH_ACCOUNT_CASH - ETH_OTHER
    plan = plan_executable_buy(
        requested_qty=ETH_QTY,
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=leftover,
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
        allocation=ETH_OWN,
    )
    assert plan.ok
    assert plan.total_cost <= leftover


def test_eth_own_reservation_is_not_double_counted():
    blocked = spendable_quote(
        account_cash=ETH_ACCOUNT_CASH,
        other_reservations=ETH_OTHER,
        own_reservation=ETH_OWN,
        include_own_reservation=False,
    )
    assert blocked == ETH_ACCOUNT_CASH - ETH_OTHER - ETH_OWN
    assert blocked <= Decimal("0")
    assert not cash_covers(ETH_OWN, blocked)


def test_eth_commission_included_before_sizing():
    plan = plan_executable_buy(
        requested_qty=ETH_QTY,
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=ETH_OWN,
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
        allocation=ETH_OWN,
    )
    assert plan.ok
    assert plan.quantity == ETH_QTY
    assert plan.commission > 0
    assert plan.total_cost <= ETH_OWN
    assert plan.notional * (Decimal("1") + ETH_TAKER) == plan.total_cost


def test_eth_equality_float_reject_now_accepts():
    need = float(ETH_OWN)
    have = float(ETH_ACCOUNT_CASH) - float(ETH_SOL_OTHER) - float(ETH_XRP_OTHER) - float(ETH_BTC_OTHER)
    assert f"{need:.2f}" == f"{have:.2f}" == "62.29"
    leftover = money(ETH_ACCOUNT_CASH) - money(ETH_OTHER)
    plan = plan_executable_buy(
        requested_qty=ETH_QTY,
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=leftover,
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
        allocation=ETH_OWN,
    )
    assert plan.ok
    assert cash_covers(plan.total_cost, leftover)
    engine = PortfolioEngine(principal=float(ETH_ACCOUNT_CASH), test_mode=True)
    engine._available_balance = float(ETH_ACCOUNT_CASH)
    engine.cash_balance = float(ETH_ACCOUNT_CASH)
    engine.open_positions = {}
    engine._entry_reservations = {
        "SOL/USDT": {"notional": float(ETH_SOL_OTHER), "decision_id": "d-sol", "sleeve": "ACTIVE"},
        "ETH/USDT": {"notional": float(ETH_OWN), "decision_id": ETH_DECISION, "sleeve": "ACTIVE"},
        "XRP/USDT": {"notional": float(ETH_XRP_OTHER), "decision_id": XRP_DECISION, "sleeve": "ACTIVE"},
        "BTC/USDT": {"notional": float(ETH_BTC_OTHER), "decision_id": "d-btc", "sleeve": "ACTIVE"},
    }
    pending = engine._pending_buy_notional(exclude_decision_id=ETH_DECISION)
    assert money(pending) == ETH_SOL_OTHER + ETH_XRP_OTHER + ETH_BTC_OTHER
    assert money(pending) < ETH_ACCOUNT_CASH - Decimal("1")


@pytest.mark.asyncio
async def test_eth_can_open_spends_own_reservation():
    engine = PortfolioEngine(principal=float(ETH_ACCOUNT_CASH), test_mode=True)
    engine._available_balance = float(ETH_ACCOUNT_CASH)
    engine.cash_balance = float(ETH_ACCOUNT_CASH)
    engine.open_positions = {}
    engine._entry_reservations = {
        "SOL/USDT": {"notional": float(ETH_SOL_OTHER), "decision_id": "d-sol", "sleeve": "ACTIVE"},
        "ETH/USDT": {"notional": float(ETH_OWN), "decision_id": ETH_DECISION, "sleeve": "ACTIVE"},
        "XRP/USDT": {"notional": float(ETH_XRP_OTHER), "decision_id": XRP_DECISION, "sleeve": "ACTIVE"},
        "BTC/USDT": {"notional": float(ETH_BTC_OTHER), "decision_id": "d-btc", "sleeve": "ACTIVE"},
    }
    engine._pending_orders = {}
    engine._ensure_symbol_constraints = AsyncMock()
    allowed, reason = await engine._can_open_position("ETH/USDT", float(ETH_OWN), decision_id=ETH_DECISION)
    assert allowed is True, reason
    assert reason != "" or allowed is True


def test_requested_qty_shrinks_when_fee_overshoots():
    tight = ETH_OWN
    fat_qty = Decimal("0.0252")
    plan = plan_executable_buy(
        requested_qty=fat_qty,
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=tight,
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
        allocation=tight,
    )
    assert plan.ok
    assert plan.quantity < fat_qty
    assert plan.shrunk is True
    assert plan.total_cost <= tight
    assert (plan.quantity * ETH_SUBMIT_ASK * (Decimal("1") + ETH_TAKER)) <= tight


def test_quantity_never_exceeds_spendable():
    plan = plan_executable_buy(
        requested_qty=Decimal("10"),
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=ETH_OWN,
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
    )
    assert plan.ok
    assert plan.total_cost <= ETH_OWN


def test_min_notional_failure_is_terminal_and_explicit():
    plan = plan_executable_buy(
        requested_qty=Decimal("0.0001"),
        price=ETH_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=Decimal("0.50"),
        qty_step=ETH_STEP,
        min_qty=ETH_MIN_QTY,
        min_notional=ETH_MIN_NOTIONAL,
    )
    assert plan.ok is False
    assert plan.reason == INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION
    assert is_terminal_buy_cash_reason(plan.reason)


def test_deterministic_cash_failure_does_not_retry():
    assert is_terminal_buy_cash_reason("INSUFFICIENT_CASH: need 62.291423 have 62.291423")
    assert is_terminal_buy_cash_reason(INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION)
    assert is_terminal_buy_cash_reason("INSUFFICIENT_CASH_WITH_PENDING: need 1 free=0")
    assert not is_terminal_buy_cash_reason("PROTECTED_LIMIT_BUY_NOT_FILLED")
    assert not is_terminal_buy_cash_reason("LIVE_BUY_ERROR: timeout")


def test_xrp_plan_accepts_captured_rebound():
    plan = plan_executable_buy(
        requested_qty=XRP_QTY,
        price=XRP_SUBMIT_ASK,
        commission_rate=ETH_TAKER,
        spendable=XRP_OWN,
        qty_step=XRP_STEP,
        min_qty=XRP_MIN_QTY,
        min_notional=XRP_MIN_NOTIONAL,
        allocation=XRP_OWN,
    )
    assert plan.ok
    assert plan.quantity <= XRP_QTY
    assert plan.total_cost <= XRP_OWN


def test_eth_and_xrp_captured_triggers_submit():
    eth = {
        "status": TRAIL_LOW,
        "arm_ask": float(ETH_ARM_ASK),
        "min_dip_bps": 14.0,
        "rebound_bps": 4.0,
        "required_improvement_bps": 10.0,
        "lowest_ask": 2474.01,
        "lowest_ask_ts": 1.0,
        "expires_at": 9_999_999.0,
    }
    trigger = rebound_trigger_ask(2474.01, 4.0)
    ceiling = retained_improvement_ceiling_ask(float(ETH_ARM_ASK), 10.0)
    assert trigger == pytest.approx(2474.9996, abs=1e-5)
    assert ceiling == pytest.approx(2479.24827, abs=1e-5)
    assert float(ETH_SUBMIT_ASK) >= trigger
    assert float(ETH_SUBMIT_ASK) <= ceiling
    d = observe_book(eth, ask=float(ETH_SUBMIT_ASK), now=10.0, book_fresh=True)
    assert d.action == "submit"

    xrp = {
        "status": TRAIL_LOW,
        "arm_ask": float(XRP_ARM_ASK),
        "min_dip_bps": 16.408884,
        "rebound_bps": 5.962584,
        "required_improvement_bps": 10.4463,
        "lowest_ask": 1.3383,
        "lowest_ask_ts": 1.0,
        "expires_at": 9_999_999.0,
    }
    xd = observe_book(xrp, ask=float(XRP_SUBMIT_ASK), now=10.0, book_fresh=True)
    assert xd.action == "submit"


def test_timeout_and_genuine_jump_unchanged():
    intent = {
        "status": TRAIL_LOW,
        "arm_ask": 100.0,
        "min_dip_bps": 14.0,
        "rebound_bps": 4.0,
        "required_improvement_bps": 10.0,
        "lowest_ask": 99.80,
        "lowest_ask_ts": 1.0,
        "expires_at": 50.0,
    }
    expired = observe_book(intent, ask=99.85, now=51.0, book_fresh=True)
    assert expired.action == "expire"
    assert expired.reason == "TIMEOUT"
    jumped = observe_book({**intent, "expires_at": 9_999.0}, ask=100.0, now=10.0, book_fresh=True)
    assert jumped.action == "expire"
    assert jumped.reason == "IMPROVEMENT_LOST"


@pytest.mark.asyncio
async def test_eth_trigger_reaches_execute_buy_fifo(tmp_path):
    db = tmp_path / "eth.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": ETH_DECISION,
            "symbol": "ETH/USDT",
            "arm_ask": float(ETH_ARM_ASK),
            "arm_bid": 2481.57,
            "arm_midpoint": 2481.65,
            "round_trip_cost_bps": 6.1697,
            "spread_bps": 0.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
            "quantity": float(ETH_QTY),
            "notional_usd": float(ETH_OWN),
            "client_order_id": "tbace6b0a92e3745d8",
        },
    )
    claimed, intent = _arm_and_claim(db, row)
    assert claimed
    called = {}

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = ""
        _entry_reservations = {"ETH/USDT": {"decision_id": ETH_DECISION, "notional": float(ETH_OWN), "reservation_id": "res_d6"}}
        _symbol_constraints = {"ETH/USDT": {"qty_step": 0.0001, "min_qty": 0.0001, "min_notional": 1.0}}
        _available_balance = float(ETH_ACCOUNT_CASH)
        open_positions = {}

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _own_entry_reservation(self, symbol, decision_id=""):
            return self._entry_reservations["ETH/USDT"], "ETH/USDT"

        def _pending_buy_notional(self, **k):
            return float(ETH_OTHER)

        def _release_entry_reservation(self, *a, **k):
            self._entry_reservations.pop("ETH/USDT", None)

        async def execute_buy_fifo(self, **kwargs):
            called.update(kwargs)
            self._release_entry_reservation("ETH/USDT", decision_id=ETH_DECISION, reason="ORDER_ACCEPTED")
            return {"order_id": "1", "price": float(ETH_SUBMIT_ASK), "trade_id": "t1"}

    engine = _Eng()
    engine._trading_paused = False
    out = await _submit_claimed(engine, intent, float(ETH_SUBMIT_ASK))
    assert out is not None
    assert called["symbol"] == "ETH/USDT"
    assert called["quantity"] == float(ETH_QTY)
    assert called["client_order_id"] == "tbace6b0a92e3745d8"
    assert "ETH/USDT" not in engine._entry_reservations


@pytest.mark.asyncio
async def test_xrp_trigger_reaches_execute_and_terminal_cash_does_not_retry(tmp_path):
    db = tmp_path / "xrp.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": XRP_DECISION,
            "symbol": "XRP/USDT",
            "arm_ask": float(XRP_ARM_ASK),
            "arm_bid": 1.3415,
            "arm_midpoint": 1.3417,
            "round_trip_cost_bps": 7.4463,
            "spread_bps": 0.0,
            "required_improvement_bps": 10.4463,
            "rebound_bps": 5.962584,
            "min_dip_bps": 16.408884,
            "expires_at": time.time() + 900,
            "quantity": float(XRP_QTY),
            "notional_usd": float(XRP_OWN),
            "client_order_id": "tbb226f475c96e428e",
        },
    )
    claimed, intent = _arm_and_claim(db, row)
    assert claimed
    calls = []

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION
        _entry_reservations = {"XRP/USDT": {"decision_id": XRP_DECISION, "notional": float(XRP_OWN)}}
        open_positions = {}
        _trading_paused = False

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _release_entry_reservation(self, *a, **k):
            self._entry_reservations.pop("XRP/USDT", None)

        async def execute_buy_fifo(self, **kwargs):
            calls.append(kwargs)

    out = await _submit_claimed(_Eng(), intent, float(XRP_SUBMIT_ASK))
    assert out is None
    assert len(calls) == 1
    done = load_intent(db, row["intent_id"])
    assert done["status"] == CANCELED
    assert done["cancel_reason"] == INSUFFICIENT_EXECUTABLE_CASH_AFTER_QUANTIZATION
    assert release_submitting_for_retry(db, row["intent_id"]) is False


@pytest.mark.asyncio
async def test_transient_failure_remains_retryable(tmp_path):
    db = tmp_path / "tmp.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "d-tmp",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
            "quantity": 0.001,
            "notional_usd": 10.0,
        },
    )
    claimed, intent = _arm_and_claim(db, row)
    assert claimed

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = "LIVE_BUY_ERROR: timeout"
        _entry_reservations = {"BTC/USDT": {"decision_id": "d-tmp", "notional": 10.0}}
        open_positions = {}
        _trading_paused = False

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _release_entry_reservation(self, *a, **k):
            return None

        async def execute_buy_fifo(self, **kwargs):
            return None

    await _submit_claimed(_Eng(), intent, 99.0)
    again = load_intent(db, row["intent_id"])
    assert again["status"] == TRAIL_LOW
    assert again["order_accepted"] == 0


@pytest.mark.asyncio
async def test_reservation_converts_once_and_terminal_releases_once(tmp_path):
    releases = []
    db = tmp_path / "res.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "d-res",
            "symbol": "SOL/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 7.63,
            "spread_bps": 2.0,
            "required_improvement_bps": 10.63,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.63,
            "expires_at": time.time() + 900,
            "quantity": 0.5,
            "notional_usd": 50.0,
        },
    )
    _claimed, intent = _arm_and_claim(db, row)
    assert _claimed

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = ""
        _entry_reservations = {"SOL/USDT": {"decision_id": "d-res", "notional": 50.0}, **{"ETH/USDT": {"decision_id": "keep-me", "notional": 60.0}}}
        open_positions = {}
        _trading_paused = False

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _release_entry_reservation(self, symbol, decision_id="", reason=""):
            releases.append((symbol, decision_id, reason))
            self._entry_reservations.pop(symbol, None)

        async def execute_buy_fifo(self, **kwargs):
            return {"order_id": "9", "price": 100.0, "trade_id": "ok"}

    engine = _Eng()
    await _submit_claimed(engine, intent, 99.9)
    assert len(releases) == 0
    assert "SOL/USDT" not in engine._entry_reservations
    assert "ETH/USDT" in engine._entry_reservations


@pytest.mark.asyncio
async def test_duplicate_cycle_cannot_duplicate_orders(tmp_path):
    db = tmp_path / "dup.db"
    _, _, row = create_intent(
        db,
        fields={
            "decision_id": "d-dup",
            "symbol": "BTC/USDT",
            "arm_ask": 100.0,
            "arm_bid": 99.9,
            "arm_midpoint": 99.95,
            "round_trip_cost_bps": 6.04,
            "spread_bps": 1.0,
            "required_improvement_bps": 10.0,
            "rebound_bps": 4.0,
            "min_dip_bps": 14.0,
            "expires_at": time.time() + 900,
            "quantity": 0.001,
        },
    )
    first, _ = _arm_and_claim(db, row)
    second, _ = claim_submitting(db, row["intent_id"])
    assert first is True
    assert second is False
    mark_order_accepted(db, row["intent_id"], order_id="10")
    assert release_submitting_for_retry(db, row["intent_id"]) is False
    loaded = load_intent(db, row["intent_id"])
    assert loaded["status"] == SUBMITTING
    assert loaded["order_accepted"] == 1


@pytest.mark.asyncio
async def test_pre_submit_does_not_cancel_shrinkable_eth_cash():
    class _Eng:
        _trading_paused = False
        open_positions = {}
        _available_balance = float(ETH_ACCOUNT_CASH)
        _symbol_constraints = {"ETH/USDT": {"qty_step": 0.0001, "min_qty": 0.0001, "min_notional": 1.0}}
        _entry_reservations = {
            "ETH/USDT": {"decision_id": ETH_DECISION, "notional": float(ETH_OWN)},
            "SOL/USDT": {"decision_id": "d-sol", "notional": float(ETH_SOL_OTHER)},
        }

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return False, "INSUFFICIENT_CASH: need 62.291423 have 62.291423"

        def _pending_buy_order_symbols(self):
            return set()

        def _own_entry_reservation(self, symbol, decision_id=""):
            return self._entry_reservations["ETH/USDT"], "ETH/USDT"

        def _pending_buy_notional(self, **k):
            return float(ETH_OTHER)

    ok, reason = await _pre_submit_safety(
        _Eng(),
        {
            "symbol": "ETH/USDT",
            "notional_usd": float(ETH_OWN),
            "decision_id": ETH_DECISION,
            "quantity": float(ETH_QTY),
        },
        float(ETH_SUBMIT_ASK),
    )
    assert ok is True
    assert reason == ""


def test_no_threshold_or_exit_source_changes():
    import inspect

    from backend.services import day_trailing_buy as tb

    src = inspect.getsource(tb.observe_book)
    assert "required_improvement_bps" in src
    assert "IMPROVEMENT_LOST" in src
    engine_src = inspect.getsource(PortfolioEngine.monitor_all_positions)
    assert "_check_exit_conditions" in engine_src
    assert inspect.getsource(tb.required_improvement_bps).count("return max(10.0") == 1
