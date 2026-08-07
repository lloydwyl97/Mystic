"""Batch 3: break-even ratchet after cost recovery + MFE-tiered trailing.

These tests exercise apply_break_even_and_mfe_trail directly and
refresh_trailing_stop end-to-end. They confirm:
1. Below MFE trigger, no ratchet fires.
2. At MFE ≥ trigger, stop AND trailing_stop_price ratchet to entry + offset.
3. At MFE ≥ tier-1, trailing distance tightens to tier-1 trail_pct.
4. At MFE ≥ tier-2, trailing distance tightens to tier-2 trail_pct.
5. Ratchet is monotone — later calls never lower stop/trail.
6. Feature can be disabled with DAY_BREAK_EVEN_TRAIL_ENABLED=false.
7. Break-even is scoped to profitable excursions and never lifts stop above
   entry+2% safety cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.services.day_controlled_exits import (
    apply_break_even_and_mfe_trail,
    refresh_trailing_stop,
)


@dataclass
class _Pos:
    entry_price: float
    highest_price: float
    stop_price: float = 0.0
    trailing_stop_price: float = 0.0
    lowest_price: float = 0.0
    trail_pct: float = 0.005
    day_route_regime_at_entry: str = ""
    max_hold_min: int = 300


def test_no_ratchet_below_break_even_trigger():
    # MFE at 0.20% (below 0.30% default trigger) — nothing should change
    pos = _Pos(entry_price=100.0, highest_price=100.2, stop_price=99.0, trailing_stop_price=99.0)
    changed = apply_break_even_and_mfe_trail(pos, current_price=100.2)
    assert changed is False
    assert pos.stop_price == 99.0
    assert pos.trailing_stop_price == 99.0


def test_break_even_ratchets_stop_when_trigger_hit():
    pos = _Pos(entry_price=100.0, highest_price=100.32, stop_price=99.0, trailing_stop_price=99.0)
    changed = apply_break_even_and_mfe_trail(pos, current_price=100.32)
    assert changed is True
    # Stop lifted to entry + 0.05% = 100.05
    assert pos.stop_price == pytest.approx(100.05, rel=1e-6)
    assert pos.trailing_stop_price == pytest.approx(100.05, rel=1e-6)


def test_tier1_tightens_trail_and_ratchets_stop():
    pos = _Pos(entry_price=100.0, highest_price=100.55, stop_price=99.0, trailing_stop_price=99.0)
    changed = apply_break_even_and_mfe_trail(pos, current_price=100.55)
    assert changed is True
    # tier-1 default trail 0.30% behind highest → 100.55 * (1 - 0.003) ≈ 100.2484
    assert pos.trailing_stop_price == pytest.approx(100.55 * (1 - 0.003), rel=1e-6)
    # Also ratcheted stop_price to the tightened trail level
    tier1_level = 100.55 * (1 - 0.003)
    assert pos.stop_price >= tier1_level - 1e-6


def test_tier2_tightens_trail_further():
    pos = _Pos(entry_price=100.0, highest_price=101.10, stop_price=99.0, trailing_stop_price=99.0)
    changed = apply_break_even_and_mfe_trail(pos, current_price=101.10)
    assert changed is True
    # tier-2 default trail 0.20% behind highest → 101.10 * (1 - 0.002) ≈ 100.898
    assert pos.trailing_stop_price == pytest.approx(101.10 * (1 - 0.002), rel=1e-6)


def test_ratchet_is_monotone():
    pos = _Pos(entry_price=100.0, highest_price=101.10, stop_price=99.0, trailing_stop_price=99.0)
    apply_break_even_and_mfe_trail(pos, current_price=101.10)
    high_stop = pos.trailing_stop_price
    # Now price drops back — previous ratchet must not un-ratchet
    changed_2 = apply_break_even_and_mfe_trail(pos, current_price=100.30)
    assert pos.trailing_stop_price >= high_stop - 1e-9
    # Highest hasn't moved so trail should stay
    assert not (pos.trailing_stop_price < high_stop)
    del changed_2


def test_disable_flag_short_circuits(monkeypatch):
    monkeypatch.setenv("DAY_BREAK_EVEN_TRAIL_ENABLED", "false")
    pos = _Pos(entry_price=100.0, highest_price=101.5, stop_price=99.0, trailing_stop_price=99.0)
    changed = apply_break_even_and_mfe_trail(pos, current_price=101.5)
    assert changed is False
    assert pos.stop_price == 99.0
    assert pos.trailing_stop_price == 99.0


def test_ratchet_capped_below_entry_plus_2pct():
    # Even at absurd MFE (5%), the safety cap keeps stop below entry+2%.
    pos = _Pos(entry_price=100.0, highest_price=105.0, stop_price=99.0, trailing_stop_price=99.0)
    apply_break_even_and_mfe_trail(pos, current_price=105.0)
    # tier-2 trail: 105 * (1 - 0.002) = 104.79 → exceeds entry * 1.02 = 102.00
    # so stop_price should be at most 102.0, but trailing_stop_price ratchets fully
    assert pos.stop_price <= 100.0 * 1.02 + 1e-6
    # trailing_stop_price is not subject to the +2% cap — it locks the gain
    assert pos.trailing_stop_price == pytest.approx(105.0 * 0.998, rel=1e-6)


def test_refresh_trailing_stop_invokes_break_even():
    """End-to-end: refresh_trailing_stop should ratchet stop above entry once MFE clears trigger."""
    pos = _Pos(entry_price=100.0, highest_price=100.60, stop_price=99.0, trailing_stop_price=0.0)
    profile = {"trail": 0.005, "sl": 0.010, "max_hold_min": 75}
    changed = refresh_trailing_stop(pos, current_price=100.60, coin_profile=profile)
    assert changed is True
    # Base trail: 100.60 * 0.995 = 100.097
    # Tier-1 trail: 100.60 * 0.997 = 100.2982 (tighter, wins)
    # → trailing_stop_price should be the tier-1 value
    assert pos.trailing_stop_price >= 100.05
    assert pos.trailing_stop_price == pytest.approx(100.60 * (1 - 0.003), rel=1e-6)
    # stop_price ratcheted to entry+offset (0.0005) at minimum
    assert pos.stop_price >= 100.0 * 1.0005


def test_custom_env_thresholds(monkeypatch):
    monkeypatch.setenv("DAY_BREAK_EVEN_TRIGGER_PCT", "0.005")
    monkeypatch.setenv("DAY_BREAK_EVEN_OFFSET_PCT", "0.001")
    # Raise tier-1 above the MFE we'll test so only break-even fires here.
    monkeypatch.setenv("DAY_MFE_TRAIL_TIER1_MFE_PCT", "0.02")
    monkeypatch.setenv("DAY_MFE_TRAIL_TIER2_MFE_PCT", "0.03")
    pos = _Pos(entry_price=100.0, highest_price=100.4, stop_price=99.0, trailing_stop_price=99.0)
    # 0.40% MFE, but trigger raised to 0.50% → should not ratchet
    changed = apply_break_even_and_mfe_trail(pos, current_price=100.4)
    assert changed is False
    # 0.60% MFE — now above raised trigger, ratchet stop to entry + 0.10%
    pos.highest_price = 100.6
    changed = apply_break_even_and_mfe_trail(pos, current_price=100.6)
    assert changed is True
    assert pos.stop_price == pytest.approx(100.10, rel=1e-6)
