"""Structured HOLD categories have no order authority and are distinct."""

from __future__ import annotations

from backend.services.day_decision_state import (
    CAPITAL_OR_SLOT_BLOCK,
    DATA_REPAIR_REQUIRED,
    HARD_SAFETY_BLOCK,
    MODEL_HOLD_TELEMETRY,
    NO_RANKED_CANDIDATE,
    OPERATOR_CONTROL_BLOCK,
    TRAILING_LOW,
    WAITING_FOR_DIP,
    WAITING_FOR_REBOUND,
    build_hold_record,
    classify_hold_category,
    hold_blocks_live_execution,
    load_hold_records,
    persist_hold_record,
)
from backend.services.day_trailing_buy import ObserveDecision, observe_book
from backend.services.day_trailing_buy_store import CANCELED, TRAIL_LOW, WAIT_DIP


def test_model_hold_does_not_block_live():
    cat = classify_hold_category(model_side="HOLD")
    assert cat == MODEL_HOLD_TELEMETRY
    assert hold_blocks_live_execution(cat) is False


def test_no_ranked_candidate_is_distinct():
    assert classify_hold_category(ranked=False) == NO_RANKED_CANDIDATE


def test_trailing_states():
    assert classify_hold_category(trailing_status=WAIT_DIP) == WAITING_FOR_DIP
    assert classify_hold_category(trailing_status=TRAIL_LOW, observe_reason="NEW_LOW") == TRAILING_LOW
    assert classify_hold_category(trailing_status=TRAIL_LOW, observe_reason="") == WAITING_FOR_REBOUND


def test_safety_and_operator_block():
    assert classify_hold_category(reject_reason="INSUFFICIENT_CASH") == CAPITAL_OR_SLOT_BLOCK
    assert hold_blocks_live_execution(CAPITAL_OR_SLOT_BLOCK) is True
    assert classify_hold_category(reject_reason="KILL_SWITCH") == OPERATOR_CONTROL_BLOCK
    assert classify_hold_category(observe_action="cancel", observe_reason="STALE_MARKET_BOOK") == DATA_REPAIR_REQUIRED
    assert classify_hold_category(reject_reason="LIVE_EXECUTION_UNAVAILABLE") == HARD_SAFETY_BLOCK


def test_observe_stale_book_is_data_repair_not_model_hold():
    d = observe_book(
        {
            "status": WAIT_DIP,
            "arm_ask": 100.0,
            "min_dip_bps": 14,
            "rebound_bps": 4,
            "required_improvement_bps": 10,
            "expires_at": 9_999_999_999,
        },
        ask=99.0,
        now=1_700_000_100,
        book_fresh=False,
    )
    assert d.action == "cancel"
    assert d.reason == "STALE_MARKET_BOOK"
    cat = classify_hold_category(trailing_status=CANCELED, observe_action=d.action, observe_reason=d.reason)
    assert cat == DATA_REPAIR_REQUIRED
    rec = build_hold_record(
        symbol="BTCUSDT",
        category=cat,
        reason=d.reason,
        authority="observe_book",
        observed={"ask": 99.0},
    )
    assert rec["blocks_live_execution"] is True
    assert rec["category"] != MODEL_HOLD_TELEMETRY


def test_try_reserve_entry_accepts_trailing_ttl():
    import inspect

    from backend.services.portfolio_engine import PortfolioEngine

    params = inspect.signature(PortfolioEngine._try_reserve_entry).parameters
    assert "ttl_sec" in params


def test_persist_roundtrip(tmp_path):
    db = tmp_path / "holds.db"
    db.write_text("")
    import sqlite3

    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE operational_state (key TEXT PRIMARY KEY, value_json TEXT NOT NULL, updated_ts INTEGER NOT NULL)")
    conn.commit()
    conn.close()
    rec = build_hold_record(symbol="ETHUSDT", category=WAITING_FOR_DIP, reason="watch", authority="observe_book", intent_id="i1")
    persist_hold_record(str(db), rec)
    rows = load_hold_records(str(db))
    assert len(rows) == 1
    assert rows[0]["symbol"] == "ETHUSDT"
    assert rows[0]["category"] == WAITING_FOR_DIP
    assert rows[0]["blocks_live_execution"] is False
