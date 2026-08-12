"""Item p21: dynamic execution-style selection (MARKET vs LIMIT_IOC) for SCALP."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp import scalp_execution_selector as ses


def test_urgent_exit_always_market_regardless_of_spread():
    choice = ses.choose_execution_style(is_urgent_exit=True, spread_pct=0.01, adverse_selection_risk=0.0)
    assert choice.order_type == "MARKET"
    assert choice.reason == "urgent_exit_guaranteed_fill"


def test_high_adverse_selection_forces_market():
    choice = ses.choose_execution_style(is_urgent_exit=False, spread_pct=0.001, adverse_selection_risk=0.9)
    assert choice.order_type == "MARKET"
    assert choice.reason == "high_adverse_selection_prefer_certainty"


def test_wide_spread_low_risk_prefers_limit_ioc():
    choice = ses.choose_execution_style(is_urgent_exit=False, spread_pct=0.002, adverse_selection_risk=0.1)
    assert choice.order_type == "LIMIT_IOC"


def test_tight_spread_defaults_to_market():
    choice = ses.choose_execution_style(is_urgent_exit=False, spread_pct=0.0001, adverse_selection_risk=0.0)
    assert choice.order_type == "MARKET"
    assert choice.reason == "tight_spread_default_market"


def test_adverse_selection_risk_clamped_out_of_range():
    choice_high = ses.choose_execution_style(is_urgent_exit=False, spread_pct=0.0001, adverse_selection_risk=5.0)
    assert choice_high.order_type == "MARKET"
    choice_neg = ses.choose_execution_style(is_urgent_exit=False, spread_pct=0.002, adverse_selection_risk=-1.0)
    assert choice_neg.order_type == "LIMIT_IOC"


def test_resolve_order_type_dynamic_default(monkeypatch):
    monkeypatch.delenv("SCALP_ORDER_TYPE_OVERRIDE", raising=False)
    assert ses.resolve_order_type(is_urgent_exit=True, spread_pct=0.01) == "MARKET"
    assert ses.resolve_order_type(is_urgent_exit=False, spread_pct=0.002, adverse_selection_risk=0.0) == "LIMIT_IOC"


def test_resolve_order_type_operator_override_wins(monkeypatch):
    monkeypatch.setenv("SCALP_ORDER_TYPE_OVERRIDE", "LIMIT_IOC")
    # Even an urgent exit is overridden if operator forces LIMIT_IOC explicitly.
    assert ses.resolve_order_type(is_urgent_exit=True, spread_pct=0.0) == "LIMIT_IOC"


def test_resolve_order_type_invalid_override_falls_through(monkeypatch):
    monkeypatch.setenv("SCALP_ORDER_TYPE_OVERRIDE", "GARBAGE")
    assert ses.resolve_order_type(is_urgent_exit=True, spread_pct=0.0) == "MARKET"
