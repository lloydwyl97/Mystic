"""Tests that legacy live production behavior is unchanged.

These tests verify that:
1. All key legacy entry/exit/guard functions still exist and are importable.
2. The new day_v2 package does NOT import live execution functions.
3. Importing day_v2 modules has no side effects on portfolio_engine globals.
"""

import importlib
import sys

import pytest


def test_legacy_entry_path_unchanged():
    """PortfolioEngine class must still have process_bar_candidates and execute_buy_fifo."""
    import backend.services.portfolio_engine as pe

    # These are methods on the PortfolioEngine class (the canonical entry path)
    assert hasattr(pe.PortfolioEngine, "process_bar_candidates"), "process_bar_candidates removed from PortfolioEngine class"
    assert hasattr(pe.PortfolioEngine, "execute_buy_fifo"), "execute_buy_fifo removed from PortfolioEngine class"


def test_legacy_exit_path_unchanged():
    """day_controlled_exits must still have evaluate_engine_managed_exit."""
    import backend.services.day_controlled_exits as dce

    assert hasattr(dce, "evaluate_engine_managed_exit"), "evaluate_engine_managed_exit removed from day_controlled_exits"


def test_legacy_churn_guard_unchanged():
    """day_churn_guard must still have ChurnGuardState."""
    import backend.services.day_churn_guard as dcg

    assert hasattr(dcg, "ChurnGuardState"), "ChurnGuardState removed from day_churn_guard"


def test_legacy_trailing_buy_unchanged():
    """day_trailing_buy must be importable if it exists."""
    try:
        import backend.services.day_trailing_buy
    except ImportError:
        pytest.skip("day_trailing_buy does not exist — OK if not yet created")
    # If it exists, the import itself is the test


def test_day_v2_does_not_import_live_execution():
    """None of the day_v2 modules should import live execution symbols."""
    forbidden = {"execute_buy_fifo", "execute_sell_fifo", "_place_order"}

    day_v2_modules = [
        "backend.services.day_v2",
        "backend.services.day_v2.engine_identity",
        "backend.services.day_v2.config",
        "backend.services.day_v2.opportunity",
        "backend.services.day_v2.timeframe_authority",
        "backend.services.day_v2.exit_roles",
    ]

    for mod_name in day_v2_modules:
        mod = importlib.import_module(mod_name)
        mod_globals = set(dir(mod))
        for symbol in forbidden:
            assert symbol not in mod_globals, f"Module {mod_name!r} has live execution symbol {symbol!r} in its namespace — day_v2 must never import live execution code"


def test_new_modules_have_no_side_effects_on_import():
    """Importing all day_v2 modules must not modify portfolio_engine global state."""
    # Get portfolio_engine attribute snapshot before
    import backend.services.portfolio_engine as pe

    before_attrs = set(dir(pe))

    # Import all day_v2 modules
    import backend.services.day_v2
    import backend.services.day_v2.config
    import backend.services.day_v2.engine_identity
    import backend.services.day_v2.exit_roles
    import backend.services.day_v2.opportunity
    import backend.services.day_v2.timeframe_authority

    after_attrs = set(dir(pe))

    # The attribute set of portfolio_engine should be unchanged
    added = after_attrs - before_attrs
    assert not added, f"Importing day_v2 modules added attributes to portfolio_engine: {added}"


def test_engine_id_field_added_to_open_position():
    """engine_id field must be present in OpenPosition with default LEGACY_DAY_LIVE."""
    from backend.services.portfolio_engine import OpenPosition

    pos = OpenPosition(
        symbol="ETHUSDT",
        quantity=0.01,
        entry_price=2500.0,
        entry_time=1.0,
        trade_id="test_tid",
        stop_price=2400.0,
        take_profit_1_price=2600.0,
        take_profit_2_price=2700.0,
    )
    assert pos.engine_id == "LEGACY_DAY_LIVE", "engine_id default must be LEGACY_DAY_LIVE"
    assert pos.scalp_opportunity_id == "", "scalp_opportunity_id default must be empty string"


def test_old_scalp_runner_not_imported_in_core_stack():
    """The old binance_scalp.runner must NOT be imported by portfolio_engine or integration."""
    import inspect

    import backend.services.portfolio_engine as pe

    source = inspect.getsource(pe)
    assert "binance_scalp.runner" not in source, "portfolio_engine must not import binance_scalp.runner — it is disabled from core mode"

    try:
        import backend.services.portfolio_engine_integration as pei

        source2 = inspect.getsource(pei)
        assert "binance_scalp.runner" not in source2
    except ImportError:
        pass  # If it doesn't exist, test passes
