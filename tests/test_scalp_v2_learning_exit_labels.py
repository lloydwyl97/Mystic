"""SCALP V2 learning outcomes must carry the canonical exit label; execution inputs stay identical."""

from __future__ import annotations

import json
import sqlite3

import pytest

from backend.services.portfolio_engine import ExitType
from backend.services.scalp_v2.exit_evaluator import (
    SCALP_V2_EXIT_CATASTROPHIC,
    SCALP_V2_EXIT_GIVEBACK,
    SCALP_V2_EXIT_NET_PROFIT,
    SCALP_V2_EXIT_STALL,
    SCALP_V2_EXIT_TIME_STOP,
)
from tests.test_scalp_v2_exit_reason_persistence import QTY, SELL, _run_sell

FIVE = [
    (SCALP_V2_EXIT_NET_PROFIT, "NET_PROFIT_EXIT"),
    (SCALP_V2_EXIT_TIME_STOP, "TIME_STOP_EXIT"),
    (SCALP_V2_EXIT_CATASTROPHIC, "STOP_LOSS_EXIT"),
    (SCALP_V2_EXIT_GIVEBACK, "GIVEBACK_EXIT"),
    (SCALP_V2_EXIT_STALL, "STALL_EXIT"),
]


def _learning_row(db_path: str) -> dict:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT close_reason, decision_reason, extra_json FROM trade_learning_outcomes ORDER BY id DESC LIMIT 1").fetchone()
    assert row is not None
    return {"close_reason": row["close_reason"], "decision_reason": row["decision_reason"], "extra": json.loads(row["extra_json"] or "{}")}


def _pattern_reason(db_path: str) -> str | None:
    with sqlite3.connect(db_path) as conn:
        for table in ("ai_good_trade_patterns", "ai_bad_trade_patterns"):
            try:
                row = conn.execute(f"SELECT reason FROM {table} ORDER BY rowid DESC LIMIT 1").fetchone()
            except sqlite3.OperationalError:
                continue
            if row:
                return str(row[0])
    return None


def _execution_inputs(run) -> dict:
    with sqlite3.connect(run.db_path) as conn:
        sell = conn.execute("SELECT quantity, price, timestamp FROM paper_trades WHERE side='SELL' ORDER BY id DESC LIMIT 1").fetchone()
        reservations = conn.execute("SELECT COUNT(*) FROM day_entry_reservations").fetchone()[0] if _has_table(conn, "day_entry_reservations") else 0
    return {
        "sold": run.result is not None,
        "quantity": sell[0],
        "price": sell[1],
        "has_timestamp": bool(sell[2]),
        "gate_trigger": run.gate.await_args.kwargs["exit_trigger"],
        "gate_exit_type": run.gate.await_args.kwargs["exit_type"],
        "gate_force_sell": run.gate.await_args.kwargs["force_sell"],
        "flatten_args": run.flatten.call_args.args,
        "flatten_kwargs": run.flatten.call_args.kwargs,
        "idempotency_trigger": run.idempotency.call_args.args[3],
        "residual_calls": run.residual.call_count,
        "live_orders": run.live_order.await_count,
        "reservations": reservations,
    }


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone() is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "expected"), FIVE)
async def test_learning_outcome_uses_canonical_scalp_label_and_keeps_raw(tmp_path, trigger, expected):
    run = await _run_sell(tmp_path, exit_type=ExitType.MANUAL, trigger=trigger, engine_id="SCALP_V2")
    assert run.result is not None

    row = _learning_row(run.db_path)
    assert row["close_reason"] == expected
    assert row["decision_reason"] == f"engine_close:{expected}"
    assert row["extra"]["exit_reason"] == expected
    assert row["extra"]["raw_exit_reason"] == trigger
    assert row["extra"]["canonical_exit_reason"] == expected
    assert _pattern_reason(run.db_path) == expected


@pytest.mark.asyncio
async def test_unknown_scalp_reason_keeps_existing_learning_fallback(tmp_path):
    run = await _run_sell(tmp_path, exit_type=ExitType.MANUAL, trigger="SCALP_V2_EXIT", engine_id="SCALP_V2")

    row = _learning_row(run.db_path)
    assert row["close_reason"] == "MANUAL_EXIT"
    assert row["extra"]["raw_exit_reason"] == "SCALP_V2_EXIT"
    assert _pattern_reason(run.db_path) == "MANUAL_EXIT"


@pytest.mark.asyncio
@pytest.mark.parametrize(("trigger", "_expected"), FIVE)
async def test_learning_label_fix_leaves_every_execution_input_unchanged(tmp_path, trigger, _expected):
    """A known SCALP reason and the unmapped fallback must drive the sell identically."""
    (tmp_path / "known").mkdir()
    (tmp_path / "fallback").mkdir()
    known = await _run_sell(tmp_path / "known", exit_type=ExitType.MANUAL, trigger=trigger, engine_id="SCALP_V2")
    fallback = await _run_sell(tmp_path / "fallback", exit_type=ExitType.MANUAL, trigger="SCALP_V2_EXIT", engine_id="SCALP_V2")

    got = _execution_inputs(known)
    assert got == _execution_inputs(fallback)
    assert got["sold"] is True
    assert got["quantity"] == pytest.approx(QTY)
    assert got["price"] == pytest.approx(SELL)
    assert got["has_timestamp"] is True
    assert got["gate_trigger"] == "MANUAL_EXIT"
    assert got["idempotency_trigger"] == "MANUAL_EXIT"
    assert got["flatten_args"][0] == "MANUAL_EXIT"
    assert got["residual_calls"] == 0
    assert got["live_orders"] == 0
    ledger_reason = known.engine._record_position_close_ledger.await_args.kwargs["close_reason"]
    assert ledger_reason == known.engine._resolve_learning_close_reason(ExitType.MANUAL, trigger, force_sell=True)


@pytest.mark.asyncio
async def test_day_v2_learning_label_uses_day_mapping(tmp_path):
    run = await _run_sell(tmp_path, exit_type=ExitType.MANUAL, trigger="DAY_V2_TIME_EXPIRATION", engine_id="DAY_V2")

    assert _learning_row(run.db_path)["close_reason"] == "TIME_STOP_EXIT"
