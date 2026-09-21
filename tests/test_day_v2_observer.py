"""Tests for the DAY V2 shadow observer.

Part 7 tests:
1. test_observer_does_not_place_orders
2. test_observer_starts_only_when_enabled
3. test_observer_writes_to_shadow_table_only
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile

import pytest

# ---------------------------------------------------------------------------
# 1. Observer does not place orders — verify no order-placing functions are imported
# ---------------------------------------------------------------------------


def test_observer_does_not_place_orders():
    """Verify observer module does not IMPORT or CALL order-placement functions."""
    import ast
    import inspect

    from backend.services.day_v2 import observer

    source = inspect.getsource(observer)

    # Parse AST to check for function calls and imports (not doc comments/strings)
    tree = ast.parse(source)

    call_names: set[str] = set()
    import_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.add(node.func.attr)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                import_names.add(alias.name)

    forbidden_callable = {
        "execute_buy_fifo",
        "submit_order",
        "place_order",
        "create_order",
        "_commit_atomic_day_open_sync",
        "execute_sell_fifo",
    }
    for fn in forbidden_callable:
        assert fn not in call_names, f"observer.py CALLS {fn!r} — it has ZERO order authority"
        assert fn not in import_names, f"observer.py IMPORTS {fn!r} — it has ZERO order authority"


# ---------------------------------------------------------------------------
# 2. Observer does not start when DAY_V2_ENABLED is false
# ---------------------------------------------------------------------------


def test_observer_starts_only_when_enabled(monkeypatch):
    """When DAY_V2_ENABLED is false (default), run_day_v2_observer returns immediately."""
    monkeypatch.delenv("DAY_V2_ENABLED", raising=False)

    from backend.services.day_v2.observer import run_day_v2_observer

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Should return immediately without error when disabled
        asyncio.run(asyncio.wait_for(run_day_v2_observer(db_path), timeout=2.0))
        # If we get here, the coroutine returned (disabled path)
    except asyncio.TimeoutError:
        pytest.fail("Observer ran when DAY_V2_ENABLED was false — it should have returned")
    except Exception as exc:
        pytest.fail(f"Observer raised unexpected error: {exc}")
    finally:
        from pathlib import Path

        Path(db_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 3. Observer writes to shadow table only, not positions/paper_trades
# ---------------------------------------------------------------------------


def test_observer_writes_to_shadow_table_only():
    """Observer must only write to day_v2_shadow_observations."""
    from backend.services.day_v2.observer import _ensure_shadow_table, _write_observation

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        # Create the shadow table
        _ensure_shadow_table(db_path)

        # Write one observation
        _write_observation(
            db_path,
            {
                "symbol": "BTCUSDT",
                "opportunity_id": "test_opp_001",
                "state": "SHADOW_POSITION_OBSERVED",
                "exit_role_evaluated": "trail",
                "exit_should_fire": False,
                "net_pnl_pct": 0.001,
                "hold_minutes": 45.0,
                "bar_ts": "2026-09-21T00:00:00Z",
            },
        )

        conn = sqlite3.connect(db_path)

        # shadow table has one row
        count = conn.execute("SELECT COUNT(*) FROM day_v2_shadow_observations").fetchone()[0]
        assert count == 1

        # positions table does NOT exist (observer never creates it)
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "portfolio_engine_positions" not in tables
        assert "paper_trades" not in tables
        assert "day_v2_shadow_observations" in tables

        conn.close()
    finally:
        from pathlib import Path

        Path(db_path).unlink(missing_ok=True)
