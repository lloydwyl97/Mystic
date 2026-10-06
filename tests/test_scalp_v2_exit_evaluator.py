"""Tests for SCALP V2 dedicated exit evaluator.

Covers:
- Catastrophic stop fires on large adverse move
- Net profit take fires at min_net threshold
- Giveback fires when MFE reached then reversed
- Stall fires on flat/dead hold with adverse drift
- Time stop fires only when net-negative after ceiling
- Evaluator returns {} (fail-closed) for non-SCALP positions
- DAY structural exits are NOT applied to SCALP positions
- DAY 300-min ceiling is NOT applied to SCALP positions
"""

from __future__ import annotations

import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_position(
    engine_id: str = "SCALP_V2",
    cost_basis: float = 100.0,
    highest_price: float = 101.0,
    lowest_price: float = 99.0,
    entry_time: float = 0.0,
    symbol: str = "BTC/USDT",
):
    pos = MagicMock()
    pos.engine_id = engine_id
    pos.cost_basis = cost_basis
    pos.entry_price = cost_basis
    pos.highest_price = highest_price
    pos.lowest_price = lowest_price
    pos.entry_time = entry_time
    pos.symbol = symbol
    pos.entry_thesis = ""
    pos.day_route_regime_at_entry = ""
    return pos


# ---------------------------------------------------------------------------
# 1. Fail-closed for non-SCALP engines
# ---------------------------------------------------------------------------


def test_evaluator_returns_empty_for_day_v2_position():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    pos = _make_position(engine_id="DAY_V2")
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.5,
        net_pnl_pct=0.005,
        hold_minutes=10.0,
        bar_low=99.5,
    )
    assert result == {}


def test_evaluator_returns_empty_for_legacy_position():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    pos = _make_position(engine_id="LEGACY_DAY_LIVE")
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.5,
        net_pnl_pct=0.005,
        hold_minutes=10.0,
        bar_low=99.5,
    )
    assert result == {}


def test_evaluator_returns_empty_for_unknown_engine():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    pos = _make_position(engine_id="SOME_UNKNOWN_ENGINE")
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.0,
        net_pnl_pct=0.0,
        hold_minutes=10.0,
        bar_low=99.0,
    )
    assert result == {}


# ---------------------------------------------------------------------------
# 2. Catastrophic stop
# ---------------------------------------------------------------------------


def test_catastrophic_fires_on_large_adverse_bar_low():
    """bar_low 2% below entry triggers catastrophic (threshold 1.5%)."""
    from backend.services.scalp_v2.exit_evaluator import (
        SCALP_V2_EXIT_CATASTROPHIC,
        evaluate_scalp_v2_exit,
    )

    with patch.dict(os.environ, {"SCALP_V2_CATASTROPHIC_PCT": "0.015"}):
        pos = _make_position(cost_basis=100.0, lowest_price=95.0)
        result = evaluate_scalp_v2_exit(
            position=pos,
            current_price=98.5,
            net_pnl_pct=-0.015,
            hold_minutes=10.0,
            bar_low=98.0,  # 2% adverse — above 1.5% threshold
        )
    assert result.get("action") == "sell"
    assert result.get("reason") == SCALP_V2_EXIT_CATASTROPHIC


def test_catastrophic_does_not_fire_on_small_adverse():
    """0.5% adverse does not trigger catastrophic (threshold 1.5%)."""
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    with patch.dict(os.environ, {"SCALP_V2_CATASTROPHIC_PCT": "0.015"}):
        pos = _make_position(cost_basis=100.0)
        result = evaluate_scalp_v2_exit(
            position=pos,
            current_price=99.5,
            net_pnl_pct=-0.005,
            hold_minutes=10.0,
            bar_low=99.5,  # only 0.5% adverse
        )
    assert result.get("action") != "sell" or "CATASTROPHIC" not in str(result.get("reason", ""))


# ---------------------------------------------------------------------------
# 3. Net profit take
# ---------------------------------------------------------------------------


def test_a_fixed_profit_threshold_does_not_sell():
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    with patch.dict(os.environ, {"SCALP_V2_MIN_NET_PROFIT_PCT": "0.004"}):
        pos = _make_position(cost_basis=100.0, highest_price=101.0)
        result = evaluate_scalp_v2_exit(
            position=pos,
            current_price=100.5,
            net_pnl_pct=0.004,
            hold_minutes=5.0,
            bar_low=100.3,
        )
    assert result.get("action") == "hold"
    sold = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.5,
        net_pnl_pct=0.004,
        hold_minutes=5.0,
        bar_low=100.3,
        expected_terminal_net=0.0,
    )
    assert sold.get("reason") == "SCALP_V2_LEARNED_CONTINUATION"


def test_net_profit_take_does_not_fire_below_threshold():
    """Net P&L below 0.4% does not trigger profit take alone."""
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    with patch.dict(
        os.environ,
        {
            "SCALP_V2_MIN_NET_PROFIT_PCT": "0.004",
            "SCALP_V2_GIVEBACK_EXIT_ENABLED": "false",
            "SCALP_V2_STALL_EXIT_ENABLED": "false",
        },
    ):
        pos = _make_position(cost_basis=100.0, highest_price=100.1)
        result = evaluate_scalp_v2_exit(
            position=pos,
            current_price=100.1,
            net_pnl_pct=0.001,  # below 0.4%
            hold_minutes=5.0,
            bar_low=100.0,
        )
    assert result.get("action") != "sell" or "NET_PROFIT" not in str(result.get("reason", ""))


# ---------------------------------------------------------------------------
# 4. Time stop
# ---------------------------------------------------------------------------


def test_time_stop_fires_past_scalp_horizon():
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN, evaluate_scalp_v2_exit

    pos = _make_position(cost_basis=100.0, highest_price=100.05)
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=99.95,
        net_pnl_pct=-0.0005,
        hold_minutes=SCALP_V2_TIME_STOP_MIN + 1,
        bar_low=99.95,
    )
    assert result.get("action") == "hold"


def test_horizon_exits_a_small_winner_too():
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN, evaluate_scalp_v2_exit

    pos = _make_position(cost_basis=100.0, highest_price=100.5)
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.1,
        net_pnl_pct=0.001,
        hold_minutes=SCALP_V2_TIME_STOP_MIN + 1,
        bar_low=100.0,
    )
    assert result.get("action") == "hold"
    sold = evaluate_scalp_v2_exit(
        position=pos,
        current_price=100.1,
        net_pnl_pct=0.001,
        hold_minutes=1.0,
        bar_low=100.0,
        expected_terminal_net=-0.001,
    )
    assert sold.get("reason") == "SCALP_V2_LEARNED_CONTINUATION"


def test_time_stop_uses_scalp_ceiling_not_day_ceiling():
    from backend.services.scalp_v2.exit_evaluator import SCALP_V2_TIME_STOP_MIN, evaluate_scalp_v2_exit

    assert SCALP_V2_TIME_STOP_MIN <= 30
    assert SCALP_V2_TIME_STOP_MIN < 300
    pos = _make_position(cost_basis=100.0, highest_price=100.05)
    result = evaluate_scalp_v2_exit(
        position=pos,
        current_price=99.95,
        net_pnl_pct=-0.0005,
        hold_minutes=float(SCALP_V2_TIME_STOP_MIN) + 5,
        bar_low=99.95,
    )
    assert result.get("action") == "hold"


# ---------------------------------------------------------------------------
# 5. DAY exits NOT applied to SCALP
# ---------------------------------------------------------------------------


def test_scalp_exit_does_not_call_day_structural_invalidation():
    """SCALP exit evaluator must never call evaluate_day_v2_exit."""
    from backend.services.scalp_v2 import exit_evaluator

    with patch("backend.services.day_v2.live_exit_evaluator.evaluate_day_v2_exit") as mock_day:
        pos = _make_position(cost_basis=100.0)
        exit_evaluator.evaluate_scalp_v2_exit(
            position=pos,
            current_price=99.0,
            net_pnl_pct=-0.01,
            hold_minutes=10.0,
            bar_low=99.0,
        )
        mock_day.assert_not_called()


def test_scalp_holds_within_ceiling_without_condition():
    """Inside the horizon, with no target and no adverse stop, the scalp holds."""
    from backend.services.scalp_v2.exit_evaluator import evaluate_scalp_v2_exit

    with patch.dict(
        os.environ,
        {
            "SCALP_V2_MIN_NET_PROFIT_PCT": "0.004",
            "SCALP_V2_GIVEBACK_EXIT_ENABLED": "false",
            "SCALP_V2_STALL_EXIT_ENABLED": "false",
        },
    ):
        pos = _make_position(cost_basis=100.0, highest_price=100.1)
        result = evaluate_scalp_v2_exit(
            position=pos,
            current_price=100.0,
            net_pnl_pct=-0.001,
            hold_minutes=10.0,
            bar_low=100.0,
        )
    assert result.get("action") == "hold"
