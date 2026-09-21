"""Every no-action cycle must classify into one of the HOLD categories."""

from __future__ import annotations

from backend.services.day_decision_state import (
    CAPITAL_OR_SLOT_BLOCK,
    COOLDOWN_ACTIVE,
    DATA_REPAIR_REQUIRED,
    HARD_SAFETY_BLOCK,
    HOLD_CATEGORIES,
    MODEL_HOLD_TELEMETRY,
    NO_RANKED_CANDIDATE,
    OPEN_POSITION_HOLD,
    OPERATOR_CONTROL_BLOCK,
    ORDER_PENDING,
    TRAILING_BUY_TERMINAL,
    TRAILING_LOW,
    WAITING_FOR_DIP,
    WAITING_FOR_REBOUND,
    build_hold_record,
    classify_hold_category,
    hold_blocks_live_execution,
    load_hold_episodes,
    persist_hold_record,
)


def test_all_hold_categories_exist():
    assert HOLD_CATEGORIES == (
        MODEL_HOLD_TELEMETRY,
        NO_RANKED_CANDIDATE,
        WAITING_FOR_DIP,
        TRAILING_LOW,
        WAITING_FOR_REBOUND,
        OPEN_POSITION_HOLD,
        HARD_SAFETY_BLOCK,
        CAPITAL_OR_SLOT_BLOCK,
        ORDER_PENDING,
        DATA_REPAIR_REQUIRED,
        OPERATOR_CONTROL_BLOCK,
        COOLDOWN_ACTIVE,
        TRAILING_BUY_TERMINAL,
    )


def test_classifier_covers_every_named_category():
    cases = {
        MODEL_HOLD_TELEMETRY: {"model_side": "HOLD"},
        NO_RANKED_CANDIDATE: {"ranked": False},
        WAITING_FOR_DIP: {"trailing_status": "WAIT_DIP"},
        TRAILING_LOW: {"trailing_status": "TRAIL_LOW", "observe_reason": "NEW_LOW"},
        WAITING_FOR_REBOUND: {"trailing_status": "TRAIL_LOW", "observe_reason": "REBOUND_ABOVE_IMPROVEMENT"},
        OPEN_POSITION_HOLD: {"open_position": True},
        HARD_SAFETY_BLOCK: {"reject_reason": "SYMBOL_NOT_EXECUTABLE"},
        CAPITAL_OR_SLOT_BLOCK: {"reject_reason": "NO_REMAINING_SLOT_CASH"},
        ORDER_PENDING: {"trailing_status": "SUBMITTING"},
        DATA_REPAIR_REQUIRED: {"reject_reason": "STALE_OR_MISSING_BOOK"},
        OPERATOR_CONTROL_BLOCK: {"reject_reason": "KILL_SWITCH_HALT"},
        COOLDOWN_ACTIVE: {"reject_reason": "COOLDOWN_ACTIVE_UNTIL_1"},
        TRAILING_BUY_TERMINAL: {"trailing_status": "EXPIRED", "observe_reason": "TIMEOUT"},
    }
    for category, kwargs in cases.items():
        assert classify_hold_category(**kwargs) == category


def test_timeout_and_improvement_lost_are_not_hard_safety():
    for reason in ("TIMEOUT", "IMPROVEMENT_LOST", "INTENT_EXPIRED"):
        cat = classify_hold_category(trailing_status="EXPIRED", observe_reason=reason)
        assert cat == TRAILING_BUY_TERMINAL
        assert hold_blocks_live_execution(cat) is False
    assert classify_hold_category(reject_reason="SYMBOL_NOT_EXECUTABLE") == HARD_SAFETY_BLOCK


def test_path_ev_hold_is_telemetry_and_does_not_block():
    rec = build_hold_record(
        symbol="BTC/USDT",
        category=MODEL_HOLD_TELEMETRY,
        reason="DAY_PATH_EV_HOLD",
        authority="process_bar_candidates",
    )
    assert rec["blocks_live_execution"] is False
    assert hold_blocks_live_execution(MODEL_HOLD_TELEMETRY) is False


def test_record_has_every_required_field():
    rec = build_hold_record(
        symbol="ETH/USDT",
        category=WAITING_FOR_DIP,
        reason="WAIT_DIP",
        authority="observe_book",
        decision_id="dec-1",
        intent_id="int-1",
        observed={"ask": 2300.0},
        required={"min_dip_bps": 8.0},
    )
    for key in (
        "symbol",
        "decision_id",
        "intent_id",
        "controlling_authority",
        "exact_reason",
        "observed",
        "required",
        "blocks_live_execution",
        "next_reevaluation",
        "category",
    ):
        assert key in rec
        if key not in {"blocks_live_execution"}:
            assert rec[key] not in (None,)


def test_every_category_persists(tmp_path):
    db = str(tmp_path / "holds.db")
    for i, category in enumerate(HOLD_CATEGORIES):
        persist_hold_record(
            db,
            build_hold_record(
                symbol="SOL/USDT",
                category=category,
                reason=category,
                authority="test",
                decision_id=f"d{i}",
                intent_id=f"i{i}",
                required={"x": 1},
            ),
        )
    episodes = load_hold_episodes(db)
    seen = {e.get("category") for e in episodes}
    assert seen == set(HOLD_CATEGORIES)
