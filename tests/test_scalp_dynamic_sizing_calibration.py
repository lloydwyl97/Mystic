"""Item p12 wiring check: calibration_mult dampens SCALP sizing, defaults neutral."""

from __future__ import annotations

from backend.services.binance_scalp.scalp_dynamic_sizing import compute_scalp_position_size


def _base_kwargs():
    return dict(
        base_cap=100.0,
        free_cash=1000.0,
        min_notional=5.0,
        strategy_passed=True,
        arm_penalty_mult=1.0,
        mtf_penalty_mult=1.0,
        regime_mismatch=False,
        symbol_stall_risk=False,
        spread_pct=0.0,
        impact_pct=0.0,
        realized_volatility_pct=None,
    )


def test_calibration_mult_defaults_to_neutral():
    result_default = compute_scalp_position_size(**_base_kwargs())
    result_explicit_neutral = compute_scalp_position_size(**_base_kwargs(), calibration_mult=1.0)
    assert result_default.notional == result_explicit_neutral.notional


def test_degraded_calibration_reduces_notional():
    result_neutral = compute_scalp_position_size(**_base_kwargs(), calibration_mult=1.0)
    result_degraded = compute_scalp_position_size(**_base_kwargs(), calibration_mult=0.7)
    assert result_degraded.notional < result_neutral.notional


def test_calibration_mult_never_pushes_below_floor_or_above_cap():
    result = compute_scalp_position_size(**_base_kwargs(), calibration_mult=0.01)
    assert result.notional >= 0.0
    assert result.notional <= 100.0
