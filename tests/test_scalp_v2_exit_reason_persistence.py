"""SCALP V2 exit reasons must persist accurately without changing sell gating inputs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.services.day_trade_thesis import canonical_day_exit_reason
from backend.services.portfolio_engine import ExitType, OpenPosition, PortfolioEngine, Sleeve
from backend.services.scalp_v2.exit_evaluator import (
    SCALP_V2_EXIT_CATASTROPHIC,
    SCALP_V2_EXIT_GIVEBACK,
    SCALP_V2_EXIT_NET_PROFIT,
    SCALP_V2_EXIT_STALL,
    SCALP_V2_EXIT_TIME_STOP,
    scalp_v2_recorded_exit_reason,
)
from tests.test_sell_cash_credit import _allowed_sell_eval, _init_test_db, _seed_btc_position

QTY = 0.001
ENTRY = 60_000.0
SELL = 60_300.0


async def _run_sell(tmp_path: Path, *, exit_type: ExitType, trigger: str, engine_id: str) -> SimpleNamespace:
    from backend.services import day_mandatory_exit_execution as mandatory

    db_path = tmp_path / "sell.db"
    trade_id = "scalp_v2_BTCUSDT_1"
    _init_test_db(db_path, cash=1_000.0)
    _seed_btc_position(db_path, trade_id=trade_id, qty=QTY, entry=ENTRY, cash_after_buy=940.0)

    engine = PortfolioEngine(db_path=str(db_path), principal=1_000.0, test_mode=True)
    await engine.initialize_from_db()
    engine.open_positions["BTC/USDT"] = OpenPosition(
        symbol="BTC/USDT",
        quantity=QTY,
        entry_price=ENTRY,
        entry_time=time.time() - 600,
        trade_id=trade_id,
        stop_price=0.0,
        take_profit_1_price=0.0,
        take_profit_2_price=0.0,
        highest_price=SELL,
        lowest_price=ENTRY,
        sleeve=Sleeve.ACTIVE.value,
        engine_id=engine_id,
    )
    engine._positions_value = QTY * SELL
    engine._total_equity = engine.cash_balance + engine._positions_value
    engine._positions_initialized = True
    engine._live_execution_enabled = False
    engine._exit_in_progress = set()
    engine._paper_service = MagicMock(paper_run_id="test-run")
    engine._check_kill_switch_sell = MagicMock(return_value=(True, ""))
    engine._record_reject = AsyncMock()
    engine._delete_position_from_sqlite = AsyncMock()
    engine._clear_all_quarantines = AsyncMock()
    engine._update_coin_performance = AsyncMock()
    engine._record_thesis_regime_outcome = MagicMock()
    engine._record_day_bucket_outcome = MagicMock()
    engine._record_position_close_ledger = AsyncMock()
    engine._persist_quality_cooldowns = AsyncMock()
    engine.record_sell_cooldown = MagicMock()
    engine._get_loss_hold_until = AsyncMock(return_value=None)
    engine._set_loss_hold_until = AsyncMock()
    engine.get_rolling_24h_risk_metrics = AsyncMock(return_value=(0.0, 0))
    engine._validate_invariants = AsyncMock(return_value=True)
    engine._record_audit = AsyncMock()
    engine._entry_ensure_constraints = AsyncMock()
    engine._ensure_symbol_constraints = AsyncMock(return_value=None)
    engine._symbol_constraints["BTC/USDT"] = {"qty_step": 0.00001, "min_qty": 0.00001, "min_notional": 10.0}
    engine._normalize_order_amount = MagicMock(return_value=(QTY, "ok", QTY))
    engine._dust_check = MagicMock(return_value=(False, QTY, "", QTY * SELL))
    engine._floor_to_step = lambda q, _s: q

    gate = AsyncMock(return_value=_allowed_sell_eval(mark_price=SELL, avg_entry_price=ENTRY))
    preflight = MagicMock(passed=True, expected_avg_fill=SELL)
    preflight.to_audit_dict = MagicMock(return_value={"passed": True})
    flatten = MagicMock(wraps=mandatory.is_mandatory_day_flatten)
    residual = MagicMock(wraps=mandatory.mark_exit_residual_pending)
    idempotency = MagicMock(wraps=engine._sell_idempotency_duplicate_sync)
    live_order = AsyncMock()
    with (
        patch.object(engine, "_evaluate_sell_profitability", gate),
        patch.object(engine, "_sell_idempotency_duplicate_sync", idempotency),
        patch.object(mandatory, "is_mandatory_day_flatten", flatten),
        patch.object(mandatory, "mark_exit_residual_pending", residual),
        patch("backend.services.protected_limit_execution.execute_protected_limit_live", live_order),
        patch("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=preflight)),
        patch("backend.services.protected_limit_execution.USE_PROTECTED_LIMIT_EXECUTION", True),
        patch("backend.services.paper_trading_service.get_paper_trading_service", return_value=engine._paper_service),
    ):
        result = await engine.execute_sell_fifo("BTC/USDT", QTY, SELL, exit_type, trigger, force_sell=True)

    return SimpleNamespace(
        db_path=str(db_path),
        engine=engine,
        result=result,
        gate=gate,
        flatten=flatten,
        residual=residual,
        idempotency=idempotency,
        live_order=live_order,
    )


async def _sell(tmp_path: Path, *, exit_type: ExitType, trigger: str, engine_id: str) -> tuple[tuple, dict, str]:
    run = await _run_sell(tmp_path, exit_type=exit_type, trigger=trigger, engine_id=engine_id)
    assert run.result is not None
    with sqlite3.connect(run.db_path) as conn:
        row = conn.execute("SELECT exit_reason, exit_type, explainability_json FROM paper_trades WHERE side='SELL' ORDER BY id DESC LIMIT 1").fetchone()
    return row[:2], json.loads(row[2] or "{}"), str(run.gate.await_args.kwargs["exit_trigger"])


@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        (SCALP_V2_EXIT_NET_PROFIT, "NET_PROFIT_EXIT"),
        (SCALP_V2_EXIT_TIME_STOP, "TIME_STOP_EXIT"),
        (SCALP_V2_EXIT_CATASTROPHIC, "STOP_LOSS_EXIT"),
        (SCALP_V2_EXIT_GIVEBACK, "GIVEBACK_EXIT"),
        (SCALP_V2_EXIT_STALL, "STALL_EXIT"),
    ],
)
def test_every_scalp_v2_reason_has_a_reporting_label(trigger, expected):
    assert scalp_v2_recorded_exit_reason(trigger) == expected


def test_unknown_scalp_reason_has_no_label():
    assert scalp_v2_recorded_exit_reason("SCALP_V2_EXIT") == ""
    assert scalp_v2_recorded_exit_reason("") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "expected"),
    [
        (SCALP_V2_EXIT_NET_PROFIT, "NET_PROFIT_EXIT"),
        (SCALP_V2_EXIT_TIME_STOP, "TIME_STOP_EXIT"),
        (SCALP_V2_EXIT_CATASTROPHIC, "STOP_LOSS_EXIT"),
    ],
)
async def test_scalp_v2_sell_persists_actual_reason_not_manual(tmp_path, trigger, expected):
    (exit_reason, exit_type), explain, gate_trigger = await _sell(tmp_path, exit_type=ExitType.MANUAL, trigger=trigger, engine_id="SCALP_V2")

    assert exit_reason == expected
    assert exit_type == expected
    assert explain["raw_exit_reason"] == trigger
    assert explain["canonical_exit_reason"] == expected
    assert explain["exit_trigger"] == expected
    assert gate_trigger == canonical_day_exit_reason(trigger, exit_type_name="MANUAL") == "MANUAL_EXIT"


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ["NET_PROFIT_EXIT", "TRAILING_STOP_EXIT"])
async def test_canonical_reasons_on_scalp_positions_stay_unchanged(tmp_path, trigger):
    (exit_reason, exit_type), _explain, gate_trigger = await _sell(tmp_path, exit_type=ExitType.MANUAL, trigger=trigger, engine_id="SCALP_V2")

    assert exit_reason == trigger
    assert exit_type == trigger
    assert gate_trigger == trigger


@pytest.mark.asyncio
async def test_unknown_scalp_reason_keeps_the_existing_manual_fallback(tmp_path):
    (exit_reason, exit_type), explain, gate_trigger = await _sell(tmp_path, exit_type=ExitType.MANUAL, trigger="SCALP_V2_EXIT", engine_id="SCALP_V2")

    assert (exit_reason, exit_type) == ("MANUAL_EXIT", "MANUAL")
    assert explain["raw_exit_reason"] == "SCALP_V2_EXIT"
    assert gate_trigger == "MANUAL_EXIT"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("exit_type", "trigger", "stored"),
    [
        (ExitType.MANUAL, "DAY_4H_STRUCTURE_BREAK_EXIT", ("DAY_4H_STRUCTURE_BREAK_EXIT", "DAY_4H_STRUCTURE_BREAK_EXIT")),
        (ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT", ("NET_PROFIT_EXIT", "TP1")),
        (ExitType.MANUAL, "DAY_V2_TIME_EXPIRATION", ("TIME_STOP_EXIT", "TIME_STOP_EXIT")),
    ],
)
async def test_day_sell_labels_unchanged(tmp_path, exit_type, trigger, stored):
    (exit_reason, stored_type), _explain, gate_trigger = await _sell(tmp_path, exit_type=exit_type, trigger=trigger, engine_id="DAY_V2")

    assert (exit_reason, stored_type) == stored
    assert gate_trigger == canonical_day_exit_reason(trigger, exit_type_name=exit_type.name)


def test_day_canonical_mapping_does_not_know_scalp_reasons():
    for trigger in (SCALP_V2_EXIT_NET_PROFIT, SCALP_V2_EXIT_CATASTROPHIC):
        assert canonical_day_exit_reason(trigger, exit_type_name="MANUAL") == "MANUAL_EXIT"
