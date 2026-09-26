"""DAY V2 sells must persist the canonical exit label; execution inputs stay identical."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.services.day_trade_thesis import canonical_day_exit_reason
from backend.services.day_v2.live_exit_evaluator import day_v2_recorded_exit_reason
from backend.services.portfolio_engine import ExitType
from tests.test_scalp_v2_exit_reason_persistence import _run_sell, _sell
from tests.test_scalp_v2_learning_exit_labels import _execution_inputs, _learning_row, _pattern_reason

FIVE = [
    ("DAY_V2_CATASTROPHIC_PROTECTION", "STOP_LOSS_EXIT"),
    ("DAY_V2_STRUCTURAL_INVALIDATION", "THESIS_INVALIDATION_EXIT"),
    ("DAY_V2_WINNER_PROTECTION", "TRAILING_STOP_EXIT"),
    ("DAY_V2_OBJECTIVE_COMPLETE", "NET_PROFIT_EXIT"),
    ("DAY_V2_TIME_EXPIRATION", "TIME_STOP_EXIT"),
]


@pytest.mark.parametrize(("trigger", "expected"), FIVE)
def test_every_day_v2_reason_has_a_reporting_label(trigger, expected):
    assert day_v2_recorded_exit_reason(trigger) == expected


def test_unknown_day_v2_reason_has_no_label():
    assert day_v2_recorded_exit_reason("DAY_V2_EXIT") == ""
    assert day_v2_recorded_exit_reason("SCALP_V2_TIME_STOP") == ""
    assert day_v2_recorded_exit_reason("") == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "expected"), FIVE)
async def test_day_v2_sell_persists_actual_reason_not_manual(tmp_path, trigger, expected):
    (exit_reason, exit_type), explain, gate_trigger = await _sell(tmp_path, exit_type=ExitType.MANUAL, trigger=trigger, engine_id="DAY_V2")

    assert exit_reason == expected
    assert exit_type == expected
    assert explain["raw_exit_reason"] == trigger
    assert explain["canonical_exit_reason"] == expected
    assert explain["exit_trigger"] == expected
    assert gate_trigger == canonical_day_exit_reason(trigger, exit_type_name="MANUAL") == "MANUAL_EXIT"


@pytest.mark.asyncio
async def test_unknown_day_v2_reason_keeps_the_existing_manual_fallback(tmp_path):
    (exit_reason, exit_type), explain, gate_trigger = await _sell(tmp_path, exit_type=ExitType.MANUAL, trigger="DAY_V2_EXIT", engine_id="DAY_V2")

    assert (exit_reason, exit_type) == ("MANUAL_EXIT", "MANUAL")
    assert explain["raw_exit_reason"] == "DAY_V2_EXIT"
    assert gate_trigger == "MANUAL_EXIT"


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "expected"), FIVE)
async def test_day_v2_learning_outcome_uses_canonical_label_and_keeps_raw(tmp_path, trigger, expected):
    run = await _run_sell(tmp_path, exit_type=ExitType.MANUAL, trigger=trigger, engine_id="DAY_V2")
    assert run.result is not None

    row = _learning_row(run.db_path)
    assert row["close_reason"] == expected
    assert row["decision_reason"] == f"engine_close:{expected}"
    assert row["extra"]["exit_reason"] == expected
    assert row["extra"]["raw_exit_reason"] == trigger
    assert row["extra"]["canonical_exit_reason"] == expected
    assert _pattern_reason(run.db_path) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "_expected"), FIVE)
async def test_day_v2_label_fix_leaves_every_execution_input_unchanged(tmp_path, trigger, _expected):
    """A known DAY V2 reason and the unmapped fallback must drive the sell identically."""
    (tmp_path / "known").mkdir()
    (tmp_path / "fallback").mkdir()
    known = await _run_sell(tmp_path / "known", exit_type=ExitType.MANUAL, trigger=trigger, engine_id="DAY_V2")
    fallback = await _run_sell(tmp_path / "fallback", exit_type=ExitType.MANUAL, trigger="DAY_V2_EXIT", engine_id="DAY_V2")

    got = _execution_inputs(known)
    assert got == _execution_inputs(fallback)
    assert got["sold"] is True
    assert got["gate_trigger"] == "MANUAL_EXIT"
    assert got["gate_force_sell"] is True
    assert got["idempotency_trigger"] == "MANUAL_EXIT"
    assert got["flatten_args"][0] == "MANUAL_EXIT"
    assert got["live_orders"] == 0
    ledger_reason = known.engine._record_position_close_ledger.await_args.kwargs["close_reason"]
    assert ledger_reason == known.engine._resolve_learning_close_reason(ExitType.MANUAL, trigger, force_sell=True)


@pytest.mark.asyncio
async def test_day_v2_sell_quantities_and_prices_unchanged(tmp_path):
    run = await _run_sell(tmp_path, exit_type=ExitType.MANUAL, trigger="DAY_V2_TIME_EXPIRATION", engine_id="DAY_V2")
    with sqlite3.connect(run.db_path) as conn:
        row = conn.execute("SELECT quantity, price, pnl, engine_id, explainability_json FROM paper_trades WHERE side='SELL'").fetchone()
    fallback_dir = tmp_path / "fb"
    fallback_dir.mkdir()
    fb = await _run_sell(fallback_dir, exit_type=ExitType.MANUAL, trigger="DAY_V2_EXIT", engine_id="DAY_V2")
    with sqlite3.connect(fb.db_path) as conn:
        fb_row = conn.execute("SELECT quantity, price, pnl, engine_id FROM paper_trades WHERE side='SELL'").fetchone()

    assert row[:4] == fb_row
    assert json.loads(row[4])["exit_reason_raw"] == "DAY_V2_TIME_EXPIRATION"
