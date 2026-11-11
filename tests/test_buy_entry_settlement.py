"""A flat lot is not a finished trade while its buy can still reveal base."""

from __future__ import annotations

import sqlite3

import pytest

from backend.services.ai_canonical_storage import strategy_terminal_outcome_rows
from backend.services.buy_entry_settlement import (
    completed_trade_row,
    entry_is_settled,
    flat_buy_needs_recovery,
    merge_terminal_learn,
    peek_unsettled_learns,
    remember_unsettled_learn,
    unsettled_buy_blocks_engine,
)
from backend.services.portfolio_engine import PortfolioEngine
from backend.services.strategy_order_intent import allocate_intent, apply_venue_order
from backend.services.trade_learning_writer import TradeLearningRecord, consume_setup_outcomes_for_ranking, record_trade_outcome


def _buy(db: str, *, status: str, filled: float = 0.25, venue: float = 0.0, qty: float = 1.0) -> str:
    cid = allocate_intent(
        db,
        engine_id="SCALP_V2",
        trade_id="scalp-btc",
        decision_id="buy-1",
        symbol="BTC/USDT",
        side="BUY",
        quantity=qty,
        limit_price=100,
    )
    with sqlite3.connect(db) as conn:
        conn.execute(
            "UPDATE strategy_order_intents SET status=?, filled_qty=?, venue_executed_qty=? WHERE client_order_id=?",
            (status, filled, venue, cid),
        )
    return cid


def test_flat_position_does_not_settle_a_partial_or_fee_pending_buy(tmp_path):
    db = str(tmp_path / "settle.db")
    _buy(db, status="PARTIAL", filled=0.25, venue=0.25)
    assert entry_is_settled(db, "scalp-btc") is False
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET status='FEE_PENDING', venue_executed_qty=0.55")
    assert entry_is_settled(db, "scalp-btc") is False
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET status='FILLED'")
    assert entry_is_settled(db, "scalp-btc") is True
    assert entry_is_settled(db, "no-such-trade") is True


def test_unsettled_buy_blocks_only_its_own_engine(tmp_path):
    db = str(tmp_path / "block.db")
    _buy(db, status="FEE_PENDING", filled=0.25, venue=0.55)
    assert unsettled_buy_blocks_engine(db, "BTC/USDT", "SCALP_V2") is True
    assert unsettled_buy_blocks_engine(db, "BTC/USDT", "DAY_V2") is False
    assert unsettled_buy_blocks_engine(db, "ETH/USDT", "SCALP_V2") is False


@pytest.mark.asyncio
async def test_can_open_position_rejects_same_engine_while_entry_is_unsettled(tmp_path):
    db = str(tmp_path / "open.db")
    _buy(db, status="FEE_PENDING", filled=0.25, venue=0.55)
    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db
    allowed, reason = await engine._can_open_position("BTC/USDT", 100.0, engine_id="SCALP_V2")
    assert allowed is False
    assert reason == "ENTRY_UNSETTLED_SAME_ENGINE"


def test_terminal_gap_needs_recovery_and_a_live_remainder_does_too_once_flat():
    assert flat_buy_needs_recovery({"status": "FEE_PENDING", "filled_qty": 0.25, "venue_executed_qty": 0.55, "quantity": 1, "booked_qty": 0.25}, 0) is True
    assert flat_buy_needs_recovery({"status": "FEE_PENDING", "filled_qty": 0.25, "venue_executed_qty": 0.25, "quantity": 1, "booked_qty": 0.25}, 0) is False
    assert flat_buy_needs_recovery({"status": "PARTIAL", "filled_qty": 0.25, "venue_executed_qty": 0.25, "quantity": 1, "booked_qty": 0.25}, 0) is True
    assert flat_buy_needs_recovery({"status": "PARTIAL", "filled_qty": 0.25, "venue_executed_qty": 0.25, "quantity": 1, "booked_qty": 0.25}, 0.25) is False
    assert flat_buy_needs_recovery({"status": "OPEN", "filled_qty": 0, "venue_executed_qty": 0, "quantity": 1}, 0) is False


def test_incomplete_terminal_buy_records_gross_qty_without_booking_it(tmp_path):
    db = str(tmp_path / "gross.db")
    cid = _buy(db, status="PARTIAL", filled=0.25, venue=0.0)
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE strategy_order_intents SET fill_ids_json='[\"leg-1\"]' WHERE client_order_id=?", (cid,))
    apply_venue_order(
        db,
        {
            "id": "94",
            "clientOrderId": cid,
            "status": "expired",
            "filled": 0.55,
            "average": 98,
            "side": "buy",
            "info": {"status": "EXPIRED_IN_MATCH", "executedQty": "0.55"},
            "trades": [{"qty": 0.25, "price": 100, "commission": 0, "commission_asset": "USDT", "trade_id": "leg-1"}],
        },
    )
    with sqlite3.connect(db) as conn:
        status, filled, venue = conn.execute("SELECT status, filled_qty, venue_executed_qty FROM strategy_order_intents").fetchone()
    assert status == "FEE_PENDING"
    assert filled == pytest.approx(0.25)
    assert venue == pytest.approx(0.55)


def test_terminal_label_waits_and_then_combines_both_sell_chunks(tmp_path):
    db = str(tmp_path / "learn.db")
    _buy(db, status="PARTIAL", filled=0.25, venue=0.25)
    first = {
        "learn": {"engine": "SCALP_V2", "symbol": "BTC/USDT", "net_pct": -0.02, "close_key": "scalp-btc", "entered_at": 1000.0, "exit_reason": "SCALP_V2_CATASTROPHIC_STOP"},
        "facts": {"net_profit_usd": -2.0, "entry_price": 100.0, "quantity": 0.25, "highest_price": 101.0, "lowest_price": 90.0, "entry_timestamp": 1000.0, "exit_timestamp": 1300.0},
    }
    remember_unsettled_learn(db, "scalp-btc", first["learn"], first["facts"])
    assert entry_is_settled(db, "scalp-btc") is False
    assert len(peek_unsettled_learns(db, "scalp-btc")) == 1
    second = {
        "learn": {"engine": "SCALP_V2", "symbol": "BTC/USDT", "net_pct": -0.01, "close_key": "scalp-btc", "entered_at": 1000.0, "exit_reason": "SCALP_V2_CATASTROPHIC_STOP"},
        "facts": {"net_profit_usd": -1.0, "entry_price": 96.0, "quantity": 0.30, "highest_price": 101.0, "lowest_price": 90.0, "entry_timestamp": 1000.0, "exit_timestamp": 1600.0},
    }
    merged = merge_terminal_learn([first, second])
    cost = 100 * 0.25 + 96 * 0.30
    assert merged["net_pct"] == pytest.approx((-2.0 + -1.0) / cost)
    assert merged["entered_at"] == pytest.approx(1000.0)
    assert merged["hold_min"] == pytest.approx(600 / 60)
    assert merged["mfe_pct"] == pytest.approx((101 - 100) / 100)
    assert merged["mae_pct"] == pytest.approx((100 - 90) / 100)
    assert merged["close_key"] == "scalp-btc"


def _outcome(db: str, *, pnl: float, settled: bool | None, terminal: bool | None, setup: str = "HTF_PULLBACK") -> None:
    extra: dict = {"setup_type": setup, "engine_id": "SCALP_V2"}
    if settled is not None:
        extra["entry_settled"] = settled
    if terminal is not None:
        extra["strategy_terminal"] = terminal
    record_trade_outcome(
        TradeLearningRecord(
            symbol="BTC/USDT",
            exit_price=100.0,
            net_profit_usd=pnl,
            net_profit_pct=pnl / 100.0,
            close_reason="SCALP_V2_CATASTROPHIC_STOP",
            event_key=f"row-{pnl}-{settled}-{terminal}",
            extra=extra,
        ),
        db_path=db,
    )


def test_setup_score_ignores_the_unsettled_loser_and_counts_the_final_winner(tmp_path):
    db = str(tmp_path / "rank.db")
    _outcome(db, pnl=-5.0, settled=False, terminal=False)
    before = consume_setup_outcomes_for_ranking(db, "HTF_PULLBACK", engine_id=None)
    assert before["n"] == 0
    assert completed_trade_row({"entry_settled": False}) is False
    assert completed_trade_row({}) is True
    _outcome(db, pnl=7.0, settled=True, terminal=True)
    after = consume_setup_outcomes_for_ranking(db, "HTF_PULLBACK", engine_id=None)
    assert after["n"] == 1
    assert after["wins"] == 1
    assert after["expectancy_usd"] == pytest.approx(7.0)


def test_ambiguous_duplicate_training_rows_are_not_strategy_samples():
    lone = {"symbol": "BTC/USDT", "strategy_id": "day", "opened_at_utc": "a", "outcome_label": "1", "features_json": "[1]", "context_json": "{}"}
    ambiguous = [
        {"symbol": "ETH/USDT", "strategy_id": "day", "opened_at_utc": "b", "outcome_label": "1", "features_json": "[1]", "context_json": "{}"},
        {"symbol": "ETH/USDT", "strategy_id": "day", "opened_at_utc": "b", "outcome_label": "0", "features_json": "[1]", "context_json": "{}"},
    ]
    same = [
        {"symbol": "SOL/USDT", "strategy_id": "scalp", "opened_at_utc": "c", "outcome_label": "1", "features_json": "[9]", "context_json": '{"trade_id":"t-1"}'},
        {"symbol": "SOL/USDT", "strategy_id": "scalp", "opened_at_utc": "c", "outcome_label": "1", "features_json": "[9]", "context_json": '{"trade_id":"t-1"}'},
    ]
    kept = strategy_terminal_outcome_rows([lone, *ambiguous, *same])
    assert [row["symbol"] for row in kept] == ["BTC/USDT", "SOL/USDT"]


def test_three_execution_rows_are_one_completed_trade(tmp_path):
    db = str(tmp_path / "three.db")
    for pnl in (-2.0, 4.0, 5.0):
        _outcome(db, pnl=pnl, settled=False, terminal=False)
    _outcome(db, pnl=7.0, settled=True, terminal=True)
    ranked = consume_setup_outcomes_for_ranking(db, "HTF_PULLBACK")
    assert ranked["n"] == 1
    assert ranked["expectancy_usd"] == pytest.approx(7.0)
