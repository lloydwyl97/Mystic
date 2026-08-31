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
