"""Captured SOL tbd6819896080b484c trailing-buy submit regression.

Values are from mystic-prod day_trailing_buy_intents / portfolio_engine_rejects
on 2026-09-15. Mocked orders and fills exist only in this test process/DB and
are labeled TEST_ONLY — they are not production exchange identifiers.
"""

from __future__ import annotations

import inspect
import time
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from backend.config.day_entry_execution import (
    ENTRY_AUTHORITY_TRAILING_BUY,
    is_trailing_buy_confirmed,
)
from backend.services.ai_artifact_contract_gate import evaluate_explainability_artifact_contract
from backend.services.day_entry_spendable import money, plan_executable_buy, spendable_quote
from backend.services.day_trailing_buy import (
    AMBIGUOUS_RECONCILE,
    ENTRY_AUTHORITY,
    EXCHANGE_REJECTED,
    FILL_ADOPTED,
    HARD_SAFETY_REJECTED,
    TRANSIENT_RETRY,
    _submit_claimed,
    classify_trailing_submit_outcome,
    observe_book,
    rebound_trigger_ask,
    recover_submitting_intent,
    retained_improvement_ceiling_ask,
)
from backend.services.day_trailing_buy_store import (
    CANCELED,
    FILLED,
    SUBMITTING,
    TRAIL_LOW,
    claim_submitting,
    create_intent,
    load_intent,
    mark_order_accepted,
    release_submitting_for_retry,
    update_watch,
)
from backend.services.portfolio_engine import PortfolioEngine, TradeExplainability

# Captured Ocean SOL intent tbd6819896080b484c
SOL_INTENT_ID = "tbd6819896080b484c"
SOL_DECISION = "day_SOLUSDT_1789488244790"
SOL_CLIENT_ORDER_ID = "tbd6819896080b484c"
SOL_RESERVATION_ID = "res_ad252b28913a4c0d"
SOL_ARM_ASK = Decimal("99.3")
SOL_ARM_BID = Decimal("99.29")
SOL_LOWEST_ASK = Decimal("98.92")
SOL_TRIGGER_ASK = Decimal("99.13")
SOL_MIN_DIP_BPS = Decimal("14.6317")
SOL_REBOUND_BPS = Decimal("4.0")
SOL_REQUIRED_IMPROVEMENT_BPS = Decimal("10.6317")
SOL_QTY = Decimal("0.5741998125799246")
SOL_NOTIONAL = Decimal("57.01804138918651")
SOL_ACCOUNT_CASH = Decimal("228.06746265")
SOL_OTHER = Decimal("171.04942126081349")
SOL_STEP = Decimal("0.001")
SOL_MIN_QTY = Decimal("0.001")
SOL_MIN_NOTIONAL = Decimal("10.0")
SOL_TAKER = Decimal("0.0002")
TEST_ONLY_ORDER_ID = "TEST_ONLY_SOL_ORDER"
TEST_ONLY_FILL_ID = "TEST_ONLY_SOL_FILL"
TEST_ONLY_TRADE_ID = "TEST_ONLY_SOL_TRADE"


def _sol_intent_fields(**extra):
    fields = {
        "intent_id": SOL_INTENT_ID,
        "decision_id": SOL_DECISION,
        "symbol": "SOL/USDT",
        "arm_ask": float(SOL_ARM_ASK),
        "arm_bid": float(SOL_ARM_BID),
        "arm_midpoint": 99.295,
        "round_trip_cost_bps": 7.6317,
        "spread_bps": 1.0071,
        "required_improvement_bps": float(SOL_REQUIRED_IMPROVEMENT_BPS),
        "rebound_bps": float(SOL_REBOUND_BPS),
        "min_dip_bps": float(SOL_MIN_DIP_BPS),
        "expires_at": time.time() + 900,
        "quantity": float(SOL_QTY),
        "notional_usd": float(SOL_NOTIONAL),
        "client_order_id": SOL_CLIENT_ORDER_ID,
        "reservation_id": SOL_RESERVATION_ID,
        "stop_price": 97.08158528023893,
        "atr": 1.5022764798407129,
        "confidence": 0.6890169030474266,
        "bar_timestamp": 1789488300,
        "sleeve": "ACTIVE",
        "thesis_invalid_level": 97.25072799,
        "payload": {
            "explainability": {
                "symbol": "SOL/USDT",
                "live_ai_strategy": "day",
                "feature_version": 0,
                "feature_dim": 0,
                "artifact_path": "",
                "artifact_sha256": "",
                "regime": "bear",
                "setup_type": "FAILED_BREAKDOWN_REVERSAL",
            },
            "entry_authority": ENTRY_AUTHORITY,
        },
    }
    fields.update(extra)
    return fields


def _arm_and_claim(db, row):
    update_watch(
        db,
        row["intent_id"],
        status=TRAIL_LOW,
        lowest_ask=float(SOL_LOWEST_ASK),
        lowest_ask_ts=1789488887.3214917,
    )
    return claim_submitting(db, row["intent_id"])


def test_captured_sol_reaches_rebound_confirmed():
    dip_got = (SOL_ARM_ASK - SOL_LOWEST_ASK) / SOL_ARM_ASK * Decimal("10000")
    assert dip_got > SOL_MIN_DIP_BPS
    trigger = rebound_trigger_ask(float(SOL_LOWEST_ASK), float(SOL_REBOUND_BPS))
    ceiling = retained_improvement_ceiling_ask(float(SOL_ARM_ASK), float(SOL_REQUIRED_IMPROVEMENT_BPS))
    assert trigger == pytest.approx(98.959568, abs=1e-6)
    assert ceiling == pytest.approx(99.19442722, abs=1e-6)
    assert float(SOL_TRIGGER_ASK) >= trigger
    assert float(SOL_TRIGGER_ASK) <= ceiling
    decision = observe_book(
        {
            "status": TRAIL_LOW,
            "arm_ask": float(SOL_ARM_ASK),
            "min_dip_bps": float(SOL_MIN_DIP_BPS),
            "rebound_bps": float(SOL_REBOUND_BPS),
            "required_improvement_bps": float(SOL_REQUIRED_IMPROVEMENT_BPS),
            "lowest_ask": float(SOL_LOWEST_ASK),
            "lowest_ask_ts": 1789488887.3214917,
            "expires_at": 1789489209.224047,
        },
        ask=float(SOL_TRIGGER_ASK),
        now=1789489180.0,
        book_fresh=True,
        thesis_invalid=False,
        validity_reason="",
    )
    assert decision.action == "submit"
    assert decision.reason == "REBOUND_CONFIRMED"


def test_captured_sol_artifact_contract_would_fail_and_is_advisory_only():
    exp = TradeExplainability(
        trade_id="",
        symbol="SOL/USDT",
        side="BUY",
        timestamp="2026-09-15T16:05:09.190160+00:00",
        live_ai_strategy="day",
        feature_version=0,
        feature_dim=0,
        artifact_path="",
        artifact_sha256="",
    )
    ok, code, _detail = evaluate_explainability_artifact_contract(exp)
    assert ok is False
    assert code == "ARTIFACT_CONTRACT_AMBIGUOUS_VERSION_DIM"
    assert is_trailing_buy_confirmed(ENTRY_AUTHORITY)
    assert ENTRY_AUTHORITY == "DAY_TRAILING_BUY_CONFIRMED"
    src = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    assert "TELEMETRY_ARTIFACT_CONTRACT" in src
    assert "DAY_TRAILING_BUY_CONFIRMED — not enforced" in src
    assert "TELEMETRY_ENTRY_CONTEXT" in src
    assert "TELEMETRY_ENTRY_EXIT_CONSISTENCY" in src


def test_captured_sol_own_reservation_is_spendable():
    spendable = spendable_quote(
        account_cash=SOL_ACCOUNT_CASH,
        other_reservations=SOL_OTHER,
        own_reservation=SOL_NOTIONAL,
        include_own_reservation=True,
    )
    assert spendable == SOL_ACCOUNT_CASH - SOL_OTHER
    plan = plan_executable_buy(
        requested_qty=SOL_QTY,
        price=SOL_TRIGGER_ASK,
        commission_rate=SOL_TAKER,
        spendable=spendable,
        qty_step=SOL_STEP,
        min_qty=SOL_MIN_QTY,
        min_notional=SOL_MIN_NOTIONAL,
        allocation=SOL_NOTIONAL,
    )
    assert plan.ok, plan.reason
    assert plan.quantity > 0
    assert plan.quantity == (plan.quantity / SOL_STEP).to_integral_value() * SOL_STEP
    assert plan.total_cost <= money(SOL_NOTIONAL)
    assert plan.total_cost <= spendable


def test_classify_does_not_retry_deterministic_or_empty():
    outcome, retryable = classify_trailing_submit_outcome("ARTIFACT_CONTRACT_AMBIGUOUS_VERSION_DIM")
    assert outcome.startswith(HARD_SAFETY_REJECTED)
    assert retryable is False
    empty, empty_retry = classify_trailing_submit_outcome("")
    assert empty.startswith(AMBIGUOUS_RECONCILE)
    assert empty_retry is False
    exch, exch_retry = classify_trailing_submit_outcome("EXCHANGE_REJECTED:-2010:insufficient balance")
    assert exch.startswith(EXCHANGE_REJECTED)
    assert exch_retry is False
    transient, transient_retry = classify_trailing_submit_outcome("LIVE_BUY_ERROR: timeout")
    assert transient.startswith(TRANSIENT_RETRY)
    assert transient_retry is True


@pytest.mark.asyncio
async def test_sol_trailing_authority_reaches_execute_and_skips_soft_verdict(tmp_path):
    db = tmp_path / "sol_submit.db"
    _, _, row = create_intent(db, fields=_sol_intent_fields())
    claimed, intent = _arm_and_claim(db, row)
    assert claimed
    called = {}
    adapter_calls = []

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = ""
        last_buy_outcome = ""
        _entry_reservations = {
            "SOL/USDT": {
                "decision_id": SOL_DECISION,
                "notional": float(SOL_NOTIONAL),
                "reservation_id": SOL_RESERVATION_ID,
            },
            "ETH/USDT": {"decision_id": "keep-eth", "notional": 57.0},
        }
        _symbol_constraints = {"SOL/USDT": {"qty_step": 0.001, "min_qty": 0.001, "min_notional": 10.0}}
        _available_balance = float(SOL_ACCOUNT_CASH)
        open_positions = {}
        _trading_paused = False

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _own_entry_reservation(self, symbol, decision_id=""):
            return self._entry_reservations["SOL/USDT"], "SOL/USDT"

        def _pending_buy_notional(self, **k):
            return float(SOL_OTHER)

        def _release_entry_reservation(self, *a, **k):
            self._entry_reservations.pop("SOL/USDT", None)

        async def execute_buy_fifo(self, **kwargs):
            called.update(kwargs)
            assert kwargs["entry_authority"] == ENTRY_AUTHORITY_TRAILING_BUY
            adapter_calls.append(
                {
                    "symbol": "SOLUSDT",
                    "side": "BUY",
                    "type": "LIMIT",
                    "quantity": kwargs["quantity"],
                    "client_order_id": kwargs["client_order_id"],
                    "label": "TEST_ONLY_ADAPTER",
                }
            )
            return {
                "order_id": TEST_ONLY_ORDER_ID,
                "fill_id": TEST_ONLY_FILL_ID,
                "trade_id": TEST_ONLY_TRADE_ID,
                "price": float(SOL_TRIGGER_ASK),
                "filled": float(SOL_QTY),
            }

    out = await _submit_claimed(_Eng(), intent, float(SOL_TRIGGER_ASK))
    assert out is not None
    assert out["outcome"] == FILL_ADOPTED
    assert called["entry_authority"] == ENTRY_AUTHORITY
    assert called["client_order_id"] == SOL_CLIENT_ORDER_ID
    assert called["trailing_buy_intent_id"] == SOL_INTENT_ID
    assert called["symbol"] == "SOL/USDT"
    assert len(adapter_calls) == 1
    done = load_intent(db, row["intent_id"])
    assert done["status"] == FILLED
    assert done["order_id"] == TEST_ONLY_ORDER_ID
    assert done["order_accepted"] == 1


@pytest.mark.asyncio
async def test_sol_duplicate_cycle_does_not_resubmit(tmp_path):
    db = tmp_path / "sol_dup.db"
    _, _, row = create_intent(db, fields=_sol_intent_fields())
    first, _ = _arm_and_claim(db, row)
    second, _ = claim_submitting(db, row["intent_id"])
    assert first is True
    assert second is False
    mark_order_accepted(db, row["intent_id"], order_id=TEST_ONLY_ORDER_ID)
    assert release_submitting_for_retry(db, row["intent_id"]) is False


@pytest.mark.asyncio
async def test_sol_exchange_rejection_is_terminal_with_reason(tmp_path):
    db = tmp_path / "sol_ex.db"
    _, _, row = create_intent(db, fields=_sol_intent_fields())
    claimed, intent = _arm_and_claim(db, row)
    assert claimed
    calls = []

    class _Eng:
        db_path = str(db)
        last_buy_reject_reason = "EXCHANGE_REJECTED:-2010:insufficient balance"
        last_buy_outcome = ""
        _entry_reservations = {"SOL/USDT": {"decision_id": SOL_DECISION, "notional": float(SOL_NOTIONAL)}}
        open_positions = {}
        _trading_paused = False

        def _check_kill_switch_buy(self):
            return True, ""

        async def _can_open_position(self, *a, **k):
            return True, ""

        def _pending_buy_order_symbols(self):
            return set()

        def _release_entry_reservation(self, *a, **k):
            self._entry_reservations.pop("SOL/USDT", None)

        async def execute_buy_fifo(self, **kwargs):
            calls.append(kwargs)

    engine = _Eng()
    out = await _submit_claimed(engine, intent, float(SOL_TRIGGER_ASK))
    assert out is None
    assert len(calls) == 1
    assert engine.last_buy_outcome.startswith(EXCHANGE_REJECTED)
    done = load_intent(db, row["intent_id"])
    assert done["status"] == CANCELED
    assert "insufficient balance" in str(done["cancel_reason"])
    assert release_submitting_for_retry(db, row["intent_id"]) is False


@pytest.mark.asyncio
async def test_sol_restart_adopts_existing_client_order(tmp_path):
    db = tmp_path / "sol_rec.db"
    _, _, row = create_intent(db, fields=_sol_intent_fields())
    _arm_and_claim(db, row)
    import sqlite3

    conn = sqlite3.connect(str(db))
    conn.execute(
        """
        CREATE TABLE paper_trades (
            trade_id TEXT, decision_id TEXT, symbol TEXT, side TEXT, price REAL, explainability_json TEXT
        )
        """
    )
    conn.execute(
        "INSERT INTO paper_trades VALUES (?,?,?,?,?,?)",
        (TEST_ONLY_TRADE_ID, SOL_DECISION, "SOL/USDT", "BUY", float(SOL_TRIGGER_ASK), '{"test_only":true}'),
    )
    conn.commit()
    conn.close()

    class _Eng:
        db_path = str(db)
        _live_service = None

        def _release_entry_reservation(self, *a, **k):
            return None

    await recover_submitting_intent(_Eng(), load_intent(db, row["intent_id"]))
    recovered = load_intent(db, row["intent_id"])
    assert recovered["status"] == FILLED
    assert recovered["trade_id"] == TEST_ONLY_TRADE_ID
    assert recovered["order_accepted"] == 1


@pytest.mark.asyncio
async def test_sol_locked_path_skips_artifact_and_reaches_hard_gates(tmp_path, monkeypatch):
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.last_buy_reject_reason = ""
    engine.last_buy_outcome = ""
    engine.db_path = str(tmp_path / "sol_locked.db")
    engine.cash_balance = float(SOL_ACCOUNT_CASH)
    engine._positions_value = 0.0
    engine._total_equity = float(SOL_ACCOUNT_CASH)
    engine._exit_mark_price_source_stale = False
    engine._check_kill_switch_buy = lambda: (True, "")

    def _hydrate(*_args, **_kwargs):
        return None

    def _ctx(_symbol):
        return {}, 0.0

    engine._hydrate_explainability_entry_context_from_redis_if_missing = _hydrate
    engine._get_context_payload = _ctx
    rejects = []

    async def _reject(_symbol, _side, reason, *_a, **_k):
        engine.last_buy_reject_reason = str(reason or "")
        rejects.append(reason)

    engine._record_reject = _reject
    engine._update_pipeline_decision = AsyncMock()

    async def _reached(*_a, **_k):
        raise RuntimeError("TEST_REACHED_HARD_GATES")

    engine._is_symbol_quarantined = _reached
    exp = TradeExplainability(
        trade_id="",
        symbol="SOL/USDT",
        side="BUY",
        timestamp="2026-09-15T16:17:00+00:00",
        live_ai_strategy="day",
        feature_version=0,
        feature_dim=0,
        artifact_path="",
        artifact_sha256="",
        regime="bear",
        ctx_ts_utc="",
        ctx_age_sec=-1.0,
        context_fresh_flag="",
    )
    with pytest.raises(RuntimeError, match="TEST_REACHED_HARD_GATES"):
        await PortfolioEngine._execute_buy_fifo_locked(
            engine,
            "SOL/USDT",
            float(SOL_QTY),
            float(SOL_TRIGGER_ASK),
            97.08158528023893,
            1.5022764798407129,
            0.6890169030474266,
            1789488300,
            exp,
            decision_id=SOL_DECISION,
            sleeve="ACTIVE",
            entry_authority=ENTRY_AUTHORITY,
            client_order_id=SOL_CLIENT_ORDER_ID,
            trailing_buy_intent_id=SOL_INTENT_ID,
        )
    assert rejects == []
    assert engine.last_buy_reject_reason == ""


@pytest.mark.asyncio
async def test_sol_locked_path_without_authority_still_records_artifact(tmp_path, monkeypatch):
    monkeypatch.delenv("DAY_ENTRY_EXECUTION_MODE", raising=False)
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.last_buy_reject_reason = ""
    engine.last_buy_outcome = ""
    engine.db_path = str(tmp_path / "sol_legacy.db")
    engine.cash_balance = float(SOL_ACCOUNT_CASH)
    engine._positions_value = 0.0
    engine._total_equity = float(SOL_ACCOUNT_CASH)
    engine._exit_mark_price_source_stale = False
    engine._check_kill_switch_buy = lambda: (True, "")

    def _hydrate(*_args, **_kwargs):
        return None

    engine._hydrate_explainability_entry_context_from_redis_if_missing = _hydrate
    rejects = []

    async def _reject(_symbol, _side, reason, *_a, **_k):
        engine.last_buy_reject_reason = str(reason or "")
        rejects.append(reason)

    engine._record_reject = _reject
    engine._update_pipeline_decision = AsyncMock()
    exp = TradeExplainability(
        trade_id="",
        symbol="SOL/USDT",
        side="BUY",
        timestamp="2026-09-15T16:17:00+00:00",
        live_ai_strategy="day",
        feature_version=0,
        feature_dim=0,
        artifact_path="",
        artifact_sha256="",
    )
    out = await PortfolioEngine._execute_buy_fifo_locked(
        engine,
        "SOL/USDT",
        float(SOL_QTY),
        float(SOL_TRIGGER_ASK),
        97.08,
        1.5,
        0.68,
        1789488300,
        exp,
        decision_id=SOL_DECISION,
        sleeve="ACTIVE",
        entry_authority="",
        client_order_id=SOL_CLIENT_ORDER_ID,
        trailing_buy_intent_id=SOL_INTENT_ID,
    )
    assert out is None
    assert rejects
    assert "ARTIFACT_CONTRACT" in str(engine.last_buy_reject_reason)


def test_no_strategy_or_exit_changes():
    src = inspect.getsource(observe_book)
    assert "REBOUND_CONFIRMED" in src
    assert "required_improvement_bps" in src
    engine_src = inspect.getsource(PortfolioEngine.monitor_all_positions)
    assert "_check_exit_conditions" in engine_src
    assert inspect.getsource(__import__("backend.services.day_trailing_buy", fromlist=["required_improvement_bps"]).required_improvement_bps).count("return max(10.0") == 1
