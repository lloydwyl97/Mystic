"""Operator display follows account DAY LIVE / SCALP PAPER authority."""

from __future__ import annotations

import asyncio

from backend.services.operator_account_status import account_operator_labels
from backend.services.portfolio_engine import PortfolioEngine


def test_labels_day_live_scalp_paper(monkeypatch):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    monkeypatch.delenv("LIVE_TRADES_ALLOWED", raising=False)
    labels = account_operator_labels(live_client_present=False)
    assert labels["mode"] == "LIVE"
    assert labels["day_mode_display"] == "DAY LIVE"
    assert labels["scalp_mode_display"] == "SCALP PAPER"
    assert labels["account_execution_live"] is True
    assert labels["live_service_connected"] is True
    assert labels["real_orders_enabled"] is True
    assert labels["trailing_buy_execution_mode"] is True


def test_engine_operator_status_not_paper_when_account_live(monkeypatch, tmp_path):
    monkeypatch.setenv("MYSTIC_TRADING_MODE", "live")
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("TRADING_MODE", "live")
    monkeypatch.setenv("DAY_ENTRY_EXECUTION_MODE", "trailing_buy")
    monkeypatch.setenv("LIVE_TRADES_ALLOWED", "false")
    from backend.database_schema import initialize_paper_trading_schema

    db = tmp_path / "op.db"
    engine = PortfolioEngine(db_path=str(db), principal=228.0, test_mode=True)
    engine._ensure_db_schema()
    initialize_paper_trading_schema(str(db))
    engine._live_service = None
    engine._live_execution_enabled = False
    op = asyncio.run(engine.get_operator_status())
    assert op["mode"] == "LIVE"
    assert op["day_mode_display"] == "DAY LIVE"
    assert op["scalp_mode_display"] == "SCALP PAPER"
    assert op["real_orders_enabled"] is True
    assert op["live_service_connected"] is True
