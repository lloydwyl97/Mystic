"""21-hour acceptance regressions: unbooked fills, fresh DAY asks, engine identity."""

from __future__ import annotations

import inspect
import sqlite3

import pytest

from backend.services.filled_asset_custody import PENDING_BOOK, RECOVERED_ACTIVE_LOT, open_rows, remember


async def test_unbooked_fill_restart_books_one_lot_and_a_retry_does_not(tmp_path):
    from backend.services.portfolio_engine import PortfolioEngine

    db = str(tmp_path / "own.db")
    engine = PortfolioEngine(db_path=db, principal=1000.0, test_mode=True)
    engine._ensure_db_schema()
    remember(
        db,
        engine_id="DAY_V2",
        symbol="BTC/USDT",
        trade_id="mystic_BTC/USDT_crash",
        quantity=0.01,
        fill_price=80000.0,
        fee=1.6,
        cash_debit=801.6,
        client_order_id="cid-1",
        exchange_order_id="oid-1",
    )
    assert open_rows(db)[0]["status"] == PENDING_BOOK
    assert await engine.restore_filled_asset_custody() == 1
    assert await engine.restore_filled_asset_custody() == 0
    conn = sqlite3.connect(db)
    rows = conn.execute("SELECT engine_id, trade_id, quantity FROM portfolio_engine_positions WHERE symbol='BTC/USDT'").fetchall()
    status = conn.execute("SELECT status, cash_applied FROM filled_asset_custody").fetchone()
    learned = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE name='adaptive_metric_state'").fetchone()
    conn.close()
    assert rows == [("DAY_V2", "mystic_BTC/USDT_crash", pytest.approx(0.01))]
    assert status == (RECOVERED_ACTIVE_LOT, 1)
    assert learned[0] in {0, 1}
    if learned[0]:
        conn = sqlite3.connect(db)
        assert conn.execute("SELECT COUNT(*) FROM adaptive_metric_state").fetchone()[0] == 0
        conn.close()


def test_remember_is_idempotent_and_never_invents_quantity(tmp_path):
    db = str(tmp_path / "c.db")
    remember(db, engine_id="SCALP_V2", symbol="ETH/USDT", trade_id="t1", quantity=0.2, fill_price=2500.0, exchange_order_id="o1")
    remember(db, engine_id="SCALP_V2", symbol="ETH/USDT", trade_id="t1", quantity=9.0, fill_price=1.0, exchange_order_id="o1")
    row = open_rows(db)[0]
    assert row["quantity"] == pytest.approx(0.2)
    assert len(open_rows(db)) == 1


def test_day_funding_refreshes_the_ask_before_sizing():
    from backend.services.portfolio_engine_integration import PortfolioEngineIntegration

    src = inspect.getsource(PortfolioEngineIntegration._fund_day_v2_candidate)
    fresh = src.index("_resolve_day_executable_price")
    size = src.index("calculate_position_size")
    submit = src.index("submit_day_v2_direct_entry")
    assert fresh < size < submit
    assert "STALE_EXECUTABLE_ASK" in src
    assert src.index('cand["ask_price"] = ask_price') < size


def test_live_buy_refuses_a_missing_engine_before_preflight():
    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._execute_buy_fifo_locked)
    gate = src.index("BUY_BLOCKED_ENGINE_IDENTITY")
    preflight = src.index("preflight = await run_protected_preflight")
    assert gate < preflight


def test_trailing_submit_names_day_explicitly():
    from backend.services.day_trailing_buy import _submit_claimed

    src = inspect.getsource(_submit_claimed)
    assert 'fill_engine_id="DAY_V2"' in src


def test_debug_env_is_restricted_in_production(monkeypatch):
    from backend.app_factory import _debug_env_restricted

    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("DEBUG", "false")
    monkeypatch.delenv("PRODUCTION", raising=False)
    assert _debug_env_restricted() is True
    monkeypatch.setenv("DEBUG", "true")
    assert _debug_env_restricted() is False


def test_entry_fee_backfill_is_engine_scoped():
    from backend.services.portfolio_engine import PortfolioEngine

    src = inspect.getsource(PortfolioEngine._backfill_entry_fee_from_paper_trades)
    assert "AND engine_id = ?" in src
