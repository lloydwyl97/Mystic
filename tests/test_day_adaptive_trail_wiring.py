"""day_adaptive_trail.py wiring into the live DAY exit path (item p4).

apply_break_even_and_mfe_trail() must use the arm's learned MFE-giveback
trail width once (symbol, setup, regime) has enough history, and must fall
back cleanly to the fixed tier constants when history is insufficient/module
disabled/an error occurs — never crash, never widen a stop.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.day_controlled_exits import apply_break_even_and_mfe_trail


def _position(**overrides) -> SimpleNamespace:
    base = dict(
        symbol="BTC/USDT",
        entry_price=100.0,
        highest_price=101.5,  # 1.5% MFE -> clears both tiers
        stop_price=99.0,
        trailing_stop_price=0.0,
        entry_thesis="HTF_TREND_PULLBACK",
        day_route_regime_at_entry="bull",
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_uses_adaptive_trail_when_arm_history_sufficient():
    """CONSTANT: learned widths must not shrink the coin-profile trail."""
    pos = _position()
    with mock.patch(
        "backend.services.day_adaptive_trail.adaptive_trail_pct_for_arm",
        return_value={"trail_pct": 0.0123, "source": "arm_history", "n_obs": 12, "giveback_p60": 0.31, "giveback_p80": 0.4, "capped": False},
    ):
        changed = apply_break_even_and_mfe_trail(pos, current_price=101.4)
    assert changed is True
    assert pos.trailing_stop_price == pytest.approx(100.05, rel=1e-9)
    assert pos.trailing_stop_price != pos.highest_price * (1.0 - 0.0123)


def test_falls_back_to_fixed_tier_when_history_insufficient():
    pos = _position()
    with mock.patch(
        "backend.services.day_adaptive_trail.adaptive_trail_pct_for_arm",
        return_value={"trail_pct": 0.008, "source": "insufficient_history_n=1", "n_obs": 1, "giveback_p60": 0.0, "giveback_p80": 0.0, "capped": False},
    ):
        changed = apply_break_even_and_mfe_trail(pos, current_price=101.4)
    assert changed is True
    # CONSTANT: no 0.20% tier-2 tighten. BE lift only.
    assert pos.trailing_stop_price == pytest.approx(100.05, rel=1e-9)
    assert abs(pos.trailing_stop_price - pos.highest_price * (1.0 - 0.0020)) > 1e-6


def test_never_crashes_when_adaptive_module_raises():
    pos = _position()
    with mock.patch(
        "backend.services.day_adaptive_trail.adaptive_trail_pct_for_arm",
        side_effect=RuntimeError("db unavailable"),
    ):
        changed = apply_break_even_and_mfe_trail(pos, current_price=101.4)
    assert changed is True
    assert pos.trailing_stop_price == pytest.approx(100.05, rel=1e-9)
