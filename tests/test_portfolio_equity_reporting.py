"""Equity invariant reporting. Reporting only — no accounting or gate behaviour.

The ledger publishes `positions_value` as an authoritative scalar and has no
`positions` collection. Summing the absent key returned 0.0 and raised a false
EQUITY INVARIANT BROKEN for every account holding a position.
"""

from __future__ import annotations

import pytest

from backend.endpoints.portfolio_engine_endpoints import _authoritative_positions_value


def test_authoritative_scalar_is_used():
    # Real Ocean ledger shape at 2026-09-05T22:15Z.
    ledger = {
        "principal": 233.34227851,
        "cash_balance": 132.49898256,
        "positions_value": 101.27437350000002,
        "total_equity": 233.77335606000003,
    }
    value = _authoritative_positions_value(ledger)
    assert value == pytest.approx(101.27437350000002)
    assert value + ledger["cash_balance"] == pytest.approx(ledger["total_equity"], abs=1e-6)


def test_invariant_holds_for_real_ledger_shape():
    ledger = {"cash_balance": 132.49898256, "positions_value": 101.27437350000002, "total_equity": 233.77335606000003}
    expected = float(ledger["cash_balance"]) + _authoritative_positions_value(ledger)
    assert abs(expected - float(ledger["total_equity"])) < 0.01


def test_old_behaviour_would_have_raised_a_false_alarm():
    """Regression: the previous expression summed a key the ledger never has."""
    ledger = {"cash_balance": 132.49898256, "positions_value": 101.27437350000002, "total_equity": 233.77335606000003}
    old = sum(pos.get("current_value", 0) for pos in ledger.get("positions", []))
    assert old == 0
    assert abs((ledger["cash_balance"] + old) - ledger["total_equity"]) >= 0.01
    assert abs((ledger["cash_balance"] + _authoritative_positions_value(ledger)) - ledger["total_equity"]) < 0.01


def test_falls_back_to_positions_collection_when_present():
    ledger = {
        "cash_balance": 10.0,
        "total_equity": 40.0,
        "positions": [
            {"current_value": 20.0},
            {"quantity": 2.0, "current_price": 5.0},
        ],
    }
    assert _authoritative_positions_value(ledger) == pytest.approx(30.0)


def test_flat_account_reports_zero_not_an_alarm():
    ledger = {"cash_balance": 233.0, "positions_value": 0.0, "total_equity": 233.0}
    assert _authoritative_positions_value(ledger) == 0.0


@pytest.mark.parametrize("ledger", [{}, None, {"positions_value": None}, {"positions_value": ""}, {"positions": []}])
def test_degenerate_inputs_do_not_raise(ledger):
    assert _authoritative_positions_value(ledger or {}) == 0.0


def test_non_numeric_positions_value_falls_through_safely():
    assert _authoritative_positions_value({"positions_value": "abc"}) == 0.0
    assert _authoritative_positions_value({"positions_value": "12.5"}) == pytest.approx(12.5)


def test_malformed_position_rows_are_skipped():
    ledger = {"positions": ["junk", {"current_value": "5.0"}, {"quantity": None, "current_price": 3.0}]}
    assert _authoritative_positions_value(ledger) == pytest.approx(5.0)
