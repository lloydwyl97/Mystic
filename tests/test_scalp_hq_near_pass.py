"""Unit tests for HIGH_QUALITY_NEAR_PASS pre-arm criteria."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from backend.services.binance_scalp.economics import ScalpEconomics  # noqa: E402
from backend.services.binance_scalp.protected_preflight import SPREAD_TOO_WIDE  # noqa: E402
from scripts.watch_scalp_entry_opportunity import is_high_quality_near_pass  # noqa: E402


def _row(**kwargs) -> dict:
    base = {
        "symbol": "ETHUSDT",
        "spread_pct": 0.0003,
        "buy_impact_pct": 0.0,
        "sell_impact_pct": 0.0,
        "projected_gross": 0.004,
        "required_gross": 0.0042,
        "projected_surplus": 0.0001,
        "momentum_confirmed": True,
        "breakout_confirmed": True,
        "reject_reason": "MOMENTUM_GROSS_BELOW_REQUIRED",
        "preflight_pass": False,
        "distance_to_pass": {"distance_to_pass_pct": 0.0002},
        "opportunity": "OPPORTUNITY_NEAR_PASS",
    }
    base.update(kwargs)
    return base


def test_hq_near_pass_accepts_valid_candidate():
    econ = ScalpEconomics.from_env()
    assert is_high_quality_near_pass(_row(), econ)


def test_hq_near_pass_rejects_spread_too_wide():
    econ = ScalpEconomics.from_env()
    assert not is_high_quality_near_pass(
        _row(reject_reason=SPREAD_TOO_WIDE, spread_pct=econ.spread_cap_pct + 0.001),
        econ,
    )


def test_hq_near_pass_rejects_negative_surplus():
    econ = ScalpEconomics.from_env()
    assert not is_high_quality_near_pass(_row(projected_surplus=-0.001), econ)
