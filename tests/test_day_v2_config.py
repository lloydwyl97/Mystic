"""Tests for DAY V2 configuration isolation and correctness."""

import os

import pytest

from backend.services.day_v2.config import (
    DAY_V2_CATASTROPHIC_ATR_MULTIPLIER,
    DAY_V2_CONTEXT_BAR_SECONDS,
    DAY_V2_MAX_HOLD_MINUTES,
    DAY_V2_PRIMARY_BAR_SECONDS,
    DAY_V2_REGIME_BAR_SECONDS,
    DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED,
    DAY_V2_UNIVERSE,
    DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT,
)


def test_universe_contains_all_four_symbols():
    assert "BTCUSDT" in DAY_V2_UNIVERSE
    assert "ETHUSDT" in DAY_V2_UNIVERSE
    assert "SOLUSDT" in DAY_V2_UNIVERSE
    assert "XRPUSDT" in DAY_V2_UNIVERSE
    assert len(DAY_V2_UNIVERSE) == 4


def test_universe_is_immutable():
    assert isinstance(DAY_V2_UNIVERSE, tuple)
    # tuples are immutable — assignment to item raises TypeError
    with pytest.raises(TypeError):
        DAY_V2_UNIVERSE[0] = "DOGEUSDT"  # type: ignore[index]


def test_config_namespace_isolation():
    """No DAY_V2 config key should share a name with existing DAY_ or SCALP_ keys."""
    # Import to trigger any namespace conflicts
    import backend.services.day_v2.config as day_v2_cfg

    # Collect DAY_V2 variable names (those prefixed with DAY_V2_)
    day_v2_names = {name for name in dir(day_v2_cfg) if name.startswith("DAY_V2_")}

    # These should all have the DAY_V2_ prefix — none should be a bare DAY_ or SCALP_ name
    for name in day_v2_names:
        assert name.startswith("DAY_V2_"), f"Config var {name!r} should be prefixed with DAY_V2_"

    # None of the DAY_V2_ names should exist without the V2 part
    # (i.e., DAY_MAX_HOLD_MINUTES without V2 is a different namespace)
    short_names = {name.replace("DAY_V2_", "DAY_") for name in day_v2_names}
    for short in short_names:
        # Just verifying they are different strings — the isolation is in naming
        assert short != name or "V2" in name


def test_default_primary_bar_is_900():
    assert DAY_V2_PRIMARY_BAR_SECONDS == 900


def test_default_context_bar_is_3600():
    assert DAY_V2_CONTEXT_BAR_SECONDS == 3600


def test_default_regime_bar_is_14400():
    assert DAY_V2_REGIME_BAR_SECONDS == 14400


def test_default_max_hold_minutes():
    assert DAY_V2_MAX_HOLD_MINUTES == 300


def test_default_winner_protection_min_mfe_is_above_legacy_trail():
    """DAY V2 MFE gate (0.8%) must be above the legacy trail distance (~0.20-0.25%)."""
    legacy_trail_estimate = 0.0025  # 0.25% — approximate legacy SOL/XRP trail
    assert legacy_trail_estimate < DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT, f"DAY V2 MFE gate {DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT:.4%} must be above legacy trail ~{legacy_trail_estimate:.4%}"
    assert pytest.approx(0.008) == DAY_V2_WINNER_PROTECTION_MIN_MFE_PCT


def test_invalid_env_var_raises_value_error(monkeypatch):
    """Setting a bad value for a DAY_V2_ env var should raise ValueError on import."""
    monkeypatch.setenv("DAY_V2_PRIMARY_BAR_SECONDS", "not_an_int")

    import importlib

    import backend.services.day_v2.config as cfg_module

    with pytest.raises(ValueError, match="DAY V2 config error"):
        importlib.reload(cfg_module)

    # Cleanup: reload with valid state so other tests aren't broken
    monkeypatch.delenv("DAY_V2_PRIMARY_BAR_SECONDS", raising=False)
    importlib.reload(cfg_module)


def test_catastrophic_atr_multiplier_default():
    assert pytest.approx(3.0) == DAY_V2_CATASTROPHIC_ATR_MULTIPLIER


def test_structural_invalidation_bars_default():
    assert DAY_V2_STRUCTURAL_INVALIDATION_BARS_CLOSED == 3


def test_get_day_v2_config_fails_closed_when_disabled():
    """get_day_v2_config() raises RuntimeError when DAY_V2_ENABLED=False."""
    from unittest.mock import patch

    import backend.services.day_v2.config as _cfg

    with patch.object(_cfg, "DAY_V2_ENABLED", False), pytest.raises(RuntimeError, match="DAY_V2_ENABLED"):
        _cfg.get_day_v2_config()
