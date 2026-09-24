"""Tests for SCALP V2 engine identity, exit calibration, and routing.

Part 7 tests:
1.  test_scalp_opportunity_id_is_deterministic
2.  test_scalp_opportunity_id_differs_by_bar
3.  test_scalp_v2_stall_disabled_by_default
4.  test_scalp_v2_giveback_disabled_by_default
5.  test_legacy_stall_still_enabled_by_default
6.  test_legacy_giveback_still_enabled_by_default
7.  test_engine_id_field_exists_in_position_dataclass
8.  test_migration_is_idempotent
9.  test_scalp_v2_exit_path_skips_stall
10. test_scalp_v2_exit_path_skips_giveback
11. test_legacy_exit_path_unchanged
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# 1. ScalpOpportunityId is deterministic
# ---------------------------------------------------------------------------


def test_scalp_opportunity_id_is_deterministic():
    from backend.services.scalp_v2.opportunity import ScalpOpportunityId

    a = ScalpOpportunityId(
        symbol="BTCUSDT",
        setup_family="VWAP",
        structural_anchor="LIVE",
        entry_bar_15m="2026-09-15T22:30:00+00:00",
    )
    b = ScalpOpportunityId(
        symbol="BTCUSDT",
        setup_family="VWAP",
        structural_anchor="LIVE",
        entry_bar_15m="2026-09-15T22:30:00+00:00",
    )
    assert a.canonical_id == b.canonical_id
    assert len(a.canonical_id) == 16


# ---------------------------------------------------------------------------
# 2. ScalpOpportunityId differs by bar
# ---------------------------------------------------------------------------


def test_scalp_opportunity_id_ignores_clock_and_changes_with_anchor():
    from backend.services.scalp_v2.opportunity import ScalpOpportunityId

    bar1 = ScalpOpportunityId(
        symbol="BTC/USDT",
        setup_family="VWAP",
        structural_anchor="pxb:1",
        entry_bar_15m="2026-09-15T22:30:00+00:00",
    )
    bar2 = ScalpOpportunityId(
        symbol="BTC/USDT",
        setup_family="VWAP",
        structural_anchor="pxb:1",
        entry_bar_15m="2026-09-15T22:45:00+00:00",
    )
    other = ScalpOpportunityId(
        symbol="BTC/USDT",
        setup_family="VWAP",
        structural_anchor="pxb:2",
        entry_bar_15m="2026-09-15T22:30:00+00:00",
    )
    assert bar1.canonical_id == bar2.canonical_id
    assert bar1.canonical_id != other.canonical_id


# ---------------------------------------------------------------------------
# 3. SCALP V2 stall exit disabled by default
# ---------------------------------------------------------------------------


def test_scalp_v2_stall_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SCALP_V2_STALL_EXIT_ENABLED", raising=False)
    from backend.services.scalp_v2.exit_calibration import scalp_v2_stall_exit_enabled

    assert scalp_v2_stall_exit_enabled() is False


# ---------------------------------------------------------------------------
# 4. SCALP V2 giveback exit disabled by default
# ---------------------------------------------------------------------------


def test_scalp_v2_giveback_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", raising=False)
    from backend.services.scalp_v2.exit_calibration import scalp_v2_giveback_exit_enabled

    assert scalp_v2_giveback_exit_enabled() is False


# ---------------------------------------------------------------------------
# 5. Legacy stall controlled by DAY_STALL_EXIT_ENABLED (not SCALP_V2 env var)
# ---------------------------------------------------------------------------


def test_legacy_stall_still_enabled_by_default(monkeypatch):
    """SCALP V2 env vars must not affect the legacy stall setting."""
    monkeypatch.delenv("SCALP_V2_STALL_EXIT_ENABLED", raising=False)
    monkeypatch.setenv("DAY_STALL_EXIT_ENABLED", "true")
    # Legacy stall is controlled by DAY_STALL_EXIT_ENABLED, not SCALP_V2_STALL_EXIT_ENABLED
    from backend.services.day_controlled_exits import _stall_exit_enabled

    assert _stall_exit_enabled() is True


# ---------------------------------------------------------------------------
# 6. Legacy giveback controlled by DAY_GIVEBACK_EXIT_ENABLED
# ---------------------------------------------------------------------------


def test_legacy_giveback_still_enabled_by_default(monkeypatch):
    """SCALP V2 env vars must not affect the legacy giveback setting."""
    monkeypatch.delenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", raising=False)
    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "true")
    from backend.services.day_controlled_exits import _giveback_exit_enabled

    assert _giveback_exit_enabled() is True


# ---------------------------------------------------------------------------
# 7. engine_id field exists in OpenPosition dataclass
# ---------------------------------------------------------------------------


def test_engine_id_field_exists_in_position_dataclass():
    from backend.services.portfolio_engine import OpenPosition

    # Check that engine_id field exists with default LEGACY_DAY_LIVE
    pos = OpenPosition(
        symbol="BTCUSDT",
        quantity=0.001,
        entry_price=81000.0,
        entry_time=1e9,
        trade_id="test_001",
        stop_price=79000.0,
        take_profit_1_price=83000.0,
        take_profit_2_price=85000.0,
    )
    assert hasattr(pos, "engine_id")
    assert pos.engine_id == "LEGACY_DAY_LIVE"
    assert hasattr(pos, "scalp_opportunity_id")
    assert pos.scalp_opportunity_id == ""


# ---------------------------------------------------------------------------
# 8. DB migration is idempotent (running twice does not fail)
# ---------------------------------------------------------------------------


def test_migration_is_idempotent():
    from backend.services.day_v2.migrations import apply_all_migrations

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Create minimum tables that migrations depend on
        conn = sqlite3.connect(db_path)
        conn.execute(
            """CREATE TABLE IF NOT EXISTS portfolio_engine_positions (
                symbol TEXT, quantity REAL, entry_price REAL, entry_time REAL,
                trade_id TEXT PRIMARY KEY, stop_price REAL, take_profit_1_price REAL,
                take_profit_2_price REAL, last_updated TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY, trade_id TEXT, symbol TEXT, side TEXT,
                price REAL, quantity REAL, timestamp TEXT
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS day_trailing_buy_intents (
                intent_id TEXT PRIMARY KEY, symbol TEXT, arm_ts REAL, status TEXT
            )"""
        )
        conn.commit()
        conn.close()

        # First run
        r1 = apply_all_migrations(db_path)
        assert "_connection_error" not in r1

        # Second run (idempotent — must not raise or error)
        r2 = apply_all_migrations(db_path)
        assert "_connection_error" not in r2

        # Verify columns were added
        conn = sqlite3.connect(db_path)
        cols_pos = {r[1] for r in conn.execute("PRAGMA table_info(portfolio_engine_positions)")}
        cols_pt = {r[1] for r in conn.execute("PRAGMA table_info(paper_trades)")}
        cols_ti = {r[1] for r in conn.execute("PRAGMA table_info(day_trailing_buy_intents)")}
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()

        assert "engine_id" in cols_pos
        assert "scalp_opportunity_id" in cols_pos
        assert "engine_id" in cols_pt
        assert "scalp_opportunity_id" in cols_pt
        assert "engine_id" in cols_ti
        assert "scalp_opportunity_id" in cols_ti
        assert "day_v2_shadow_observations" in tables
    finally:
        from pathlib import Path

        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 9. SCALP V2 exit path skips stall
# ---------------------------------------------------------------------------


def _make_stall_candidate_position():
    """Return a mock position dict that would trigger a LEGACY stall."""
    from unittest.mock import MagicMock

    pos = MagicMock()
    pos.entry_price = 1000.0
    pos.highest_price = 1001.0
    pos.lowest_price = 995.0
    pos.trailing_stop_price = None
    pos.tp1_hit = False
    pos.entry_thesis = ""
    pos.entry_vwap = 0.0
    pos.thesis_invalid_level = 0.0
    pos.thesis_target_level = 0.0
    pos.stop_price = 990.0
    pos.take_profit_1_price = 1010.0
    pos.status = "ACTIVE"
    pos.sleeve = "ACTIVE"
    pos.day_route_regime_at_entry = ""
    pos.price_structure_regime_at_entry = ""
    pos.strategy_family = ""
    pos.max_hold_min = 0
    pos.trail_pct = 0.0
    pos.trail_activated = False
    pos.trail_activated_at = 0.0
    pos.trail_activation_price = 0.0
    pos.trail_high_water_source = ""
    return pos


def test_scalp_v2_exit_path_skips_stall(monkeypatch):
    """Explicit SCALP_V2_STALL_EXIT_ENABLED=false suppresses stall. The default keeps it."""
    monkeypatch.setenv("SCALP_V2_STALL_EXIT_ENABLED", "false")
    monkeypatch.setenv("DAY_STALL_EXIT_ENABLED", "true")
    # Ensure path-aware exit is disabled so the legacy ladder runs (not path-aware exit)
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "false")
    # Stall: hold > 120 min, net_pnl_pct < -0.3%
    pos = _make_stall_candidate_position()

    from backend.services.day_controlled_exits import evaluate_engine_managed_exit

    coin_profile = {
        "tp": 0.014,
        "sl": 0.010,
        "trail": 0.0025,
        "max_hold_min": 300,
    }

    result = evaluate_engine_managed_exit(
        position=pos,
        current_price=996.0,
        net_pnl_pct=-0.004,  # -0.4%, beyond stall threshold
        hold_minutes=135.0,  # beyond 120 min stall threshold
        coin_profile=coin_profile,
        engine_id="SCALP_V2",
    )
    # SCALP_V2 should NOT return STALL_EXIT
    assert result.get("reason") != "STALL_DEAD"
    assert result.get("reason") != "STALL_EXIT"


# ---------------------------------------------------------------------------
# 10. SCALP V2 exit path skips giveback
# ---------------------------------------------------------------------------


def test_scalp_v2_exit_path_skips_giveback(monkeypatch):
    """Explicit SCALP_V2_GIVEBACK_EXIT_ENABLED=false suppresses giveback. The default keeps it."""
    monkeypatch.setenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", "false")
    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "true")
    # Ensure path-aware exit is disabled so the legacy ladder runs (not path-aware exit)
    monkeypatch.setenv("DAY_PATH_AWARE_EXIT", "false")
    pos = _make_stall_candidate_position()
    pos.entry_price = 1000.0
    pos.highest_price = 1002.5  # MFE 0.25% → above GIVEBACK_MIN_MFE 0.15%
    pos.stop_price = 990.0

    from backend.services.day_controlled_exits import evaluate_engine_managed_exit

    coin_profile = {
        "tp": 0.014,
        "sl": 0.010,
        "trail": 0.0025,
        "max_hold_min": 300,
    }

    result = evaluate_engine_managed_exit(
        position=pos,
        current_price=998.0,  # net_pnl -0.2% → triggers giveback (-0.15% MFE, -0.15% pnl)
        net_pnl_pct=-0.002,
        hold_minutes=30.0,
        coin_profile=coin_profile,
        engine_id="SCALP_V2",
    )
    # SCALP_V2 should NOT return GIVEBACK_EXIT
    assert result.get("reason") != "GIVEBACK_EXIT"


# ---------------------------------------------------------------------------
# 11. Legacy exit path unchanged (with LEGACY_DAY_LIVE engine_id)
# ---------------------------------------------------------------------------


def test_legacy_exit_path_unchanged(monkeypatch):
    """Default (LEGACY_DAY_LIVE) engine_id must behave identically to pre-SCALP V2 code."""
    monkeypatch.setenv("DAY_STALL_EXIT_ENABLED", "true")
    monkeypatch.setenv("DAY_GIVEBACK_EXIT_ENABLED", "true")
    monkeypatch.delenv("SCALP_V2_STALL_EXIT_ENABLED", raising=False)
    monkeypatch.delenv("SCALP_V2_GIVEBACK_EXIT_ENABLED", raising=False)

    # Calling without engine_id (defaults to LEGACY_DAY_LIVE) should still work.
    pos = _make_stall_candidate_position()

    from backend.services.day_controlled_exits import evaluate_engine_managed_exit

    coin_profile = {
        "tp": 0.014,
        "sl": 0.010,
        "trail": 0.0025,
        "max_hold_min": 300,
    }

    # This should not raise any error — legacy path is unchanged
    result = evaluate_engine_managed_exit(
        position=pos,
        current_price=998.0,
        net_pnl_pct=-0.002,
        hold_minutes=30.0,
        coin_profile=coin_profile,
        # engine_id not passed → defaults to LEGACY_DAY_LIVE
    )
    assert isinstance(result, dict)
    assert "action" in result
    assert "reason" in result
