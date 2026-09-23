"""Engine structure/risk exits must not persist as MANUAL_EXIT."""

from __future__ import annotations

from backend.services.day_trade_thesis import (
    EXIT_DAY_4H_STRUCTURE_BREAK,
    EXIT_DAY_RISK_FLOOR,
    EXIT_EMERGENCY_FLATTEN,
    EXIT_MANUAL,
    EXIT_RESTART_FLATTEN,
    canonical_day_exit_reason,
)
from backend.services.portfolio_engine import ExitType, paper_trades_exit_type_label


def test_4h_structure_break_persists_canonical_not_manual():
    assert paper_trades_exit_type_label(ExitType.MANUAL, EXIT_DAY_4H_STRUCTURE_BREAK) == EXIT_DAY_4H_STRUCTURE_BREAK
    assert canonical_day_exit_reason(EXIT_DAY_4H_STRUCTURE_BREAK, exit_type_name="MANUAL") == EXIT_DAY_4H_STRUCTURE_BREAK


def test_risk_floor_persists_canonical_not_manual():
    assert paper_trades_exit_type_label(ExitType.MANUAL, EXIT_DAY_RISK_FLOOR) == EXIT_DAY_RISK_FLOOR
    assert canonical_day_exit_reason(EXIT_DAY_RISK_FLOOR, exit_type_name="MANUAL") == EXIT_DAY_RISK_FLOOR


def test_operator_manual_and_flatten_labels_stay_distinct():
    assert paper_trades_exit_type_label(ExitType.MANUAL, "MANUAL") == ExitType.MANUAL.value
    assert paper_trades_exit_type_label(ExitType.MANUAL, "EMERGENCY_FLATTEN") == EXIT_EMERGENCY_FLATTEN
    assert paper_trades_exit_type_label(ExitType.MANUAL, "RESTART_FLATTEN") == EXIT_RESTART_FLATTEN


def test_profit_exit_label_unchanged():
    assert paper_trades_exit_type_label(ExitType.TAKE_PROFIT_1, "NET_PROFIT_EXIT") == ExitType.TAKE_PROFIT_1.value


def test_day_v2_exit_reasons_pass_through_unchanged():
    """DAY_V2_* reasons must not collapse to MANUAL_EXIT."""
    for reason in (
        "DAY_V2_CATASTROPHIC_PROTECTION",
        "DAY_V2_STRUCTURAL_INVALIDATION",
        "DAY_V2_WINNER_PROTECTION",
        "DAY_V2_OBJECTIVE_COMPLETE",
        "DAY_V2_TIME_EXPIRATION",
    ):
        result = paper_trades_exit_type_label(ExitType.MANUAL, reason)
        assert result == reason, f"expected {reason!r}, got {result!r}"
        assert result != ExitType.MANUAL.value, f"{reason!r} collapsed to MANUAL"
