"""A filled SCALP V2 live BUY must stay a SCALP V2 lot through restart and reconciliation."""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import pytest

ORDER_ID = "1597208579"
CLIENT_ID = "x-TKT5PX2Fb987c51891a7e4ba7d53d9"


def _filled_order() -> dict:
    return {
        "id": ORDER_ID,
        "clientOrderId": CLIENT_ID,
        "status": "closed",
        "filled": 0.0095,
        "average": 2676.23,
        "info": {"fills": [{"qty": "0.0095", "price": "2676.23", "commission": "0.0000019", "commissionAsset": "ETH"}]},
    }


def _schema(db_path: str) -> None:
    from backend.database_schema import ensure_paper_trades_columns
    from backend.services.portfolio_engine import PortfolioEngine
    from backend.services.protected_external_inventory import ensure_schema as ensure_protected_schema

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = db_path
    conn = sqlite3.connect(db_path)
    engine._ensure_schema(conn.cursor())
    ensure_paper_trades_columns(conn)
    ensure_protected_schema(conn)
    conn.commit()
    conn.close()


def _engine(tmp_path, monkeypatch, adapter):
    from backend.services.portfolio_engine import PortfolioEngine

    engine = PortfolioEngine.__new__(PortfolioEngine)
    engine.db_path = str(tmp_path / "live.db")
    _schema(engine.db_path)
    engine.open_positions = {}
    engine.cash_balance = 100.0
    engine._available_balance = 100.0
    engine._positions_value = 0.0
    engine._total_equity = 100.0
    engine._realized_pnl = 0.0
    engine._unrealized_pnl = 0.0
    engine._global_cash_lock = asyncio.Lock()
    engine.last_buy_reject_reason = ""
    engine._check_kill_switch_buy = lambda: (True, "")
    engine._day_entry_held_count = lambda: 0
    engine._normalize_order_amount = lambda **_k: (0.0095, "ok", 25.4)
    engine._live_execution_enabled = True
    engine._live_service = object()
    engine._now_ms = lambda: 1790309062000
    engine._verify_order_fill = AsyncMock(side_effect=lambda order, *_a: order)
    preflight = MagicMock(passed=True, protected_limit_price=2676.23, reject_reason="")
    monkeypatch.setattr("backend.config.live_test_mode.can_place_live_orders_sync", lambda: (True, ""))
    monkeypatch.setattr("backend.services.execution_mode_service.is_live_execution_allowed_sync", lambda: True)
    monkeypatch.setattr("backend.services.protected_limit_execution.run_protected_preflight", AsyncMock(return_value=preflight))
    monkeypatch.setattr("backend.services.protected_limit_execution.execute_protected_limit_live", adapter)
    return engine


def _arm(engine) -> str:
    from backend.services.scalp_v2.opportunity import arm_opportunity

    opp_id, blocked = arm_opportunity(engine.db_path, "ETH/USDT", "SCALP", 2676.23, engine_id="SCALP_V2")
    assert blocked is False
    return opp_id


async def _buy(engine, opp_id):
    return await engine.execute_scalp_v2_buy_live(
        "ETH/USDT",
        0.0095,
        2676.23,
        atr=10.0,
        opportunity_id=opp_id,
        entry_authority="SCALP_V2_CONFIRMED",
    )


def _one(db_path: str, sql: str, args: tuple = ()):
    conn = sqlite3.connect(db_path)
    try:
        return conn.execute(sql, args).fetchone()
    finally:
        conn.close()


@pytest.mark.asyncio
async def test_filled_buy_creates_a_durable_scalp_lot(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_filled_order()))
    opp_id = _arm(engine)

    result = await _buy(engine, opp_id)

    assert result is not None
    assert result["order_id"] == ORDER_ID
    net = 0.0095 - 0.0000019
    assert result["quantity"] == pytest.approx(net, abs=1e-12)
    pos = _one(
        engine.db_path,
        "SELECT quantity, engine_id, entry_order_id, entry_client_order_id, entry_decision_id, entry_reservation_id,"
        " take_profit_1_price, status FROM portfolio_engine_positions WHERE symbol='ETH/USDT'",
    )
    assert pos[0] == pytest.approx(net, abs=1e-12)
    assert pos[1:4] == ("SCALP_V2", ORDER_ID, CLIENT_ID)
    assert pos[4] == opp_id
    assert pos[5].startswith("res_")
    assert pos[6] == 0
    assert pos[7] == "ACTIVE"
    buy = _one(
        engine.db_path,
        "SELECT paper_run_id, order_id, remaining_position, decision_id FROM paper_trades WHERE side='BUY' AND symbol='ETH/USDT'",
    )
    assert buy[0] == "scalp_v2_live"
    assert buy[1] == ORDER_ID
    assert buy[2] == pytest.approx(net, abs=1e-12)
    assert buy[3] == opp_id
    assert _one(engine.db_path, "SELECT state FROM scalp_v2_opportunities WHERE opportunity_id=?", (opp_id,))[0] == "OPEN"
    assert _one(engine.db_path, "SELECT status FROM day_entry_reservations WHERE reservation_id=?", (pos[5],))[0] == "CONSUMED"
    assert engine.open_positions["ETH/USDT"].engine_id == "SCALP_V2"


@pytest.mark.asyncio
async def test_restart_preserves_scalp_ownership_and_provenance(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_filled_order()))
    opp_id = _arm(engine)
    assert await _buy(engine, opp_id) is not None

    engine.open_positions = {}
    await engine._load_positions_from_sqlite(allow_mutations=False)

    pos = engine.open_positions["ETH/USDT"]
    assert pos.engine_id == "SCALP_V2"
    assert pos.entry_order_id == ORDER_ID
    assert pos.entry_client_order_id == CLIENT_ID
    assert pos.scalp_opportunity_id == opp_id
    assert pos.quantity == pytest.approx(0.0095 - 0.0000019, abs=1e-12)


@pytest.mark.asyncio
async def test_post_fill_failure_keeps_provenance_instead_of_losing_the_lot(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=_filled_order()))
    opp_id = _arm(engine)

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(type(engine), "_scalp_v2_write_position_row", staticmethod(_boom))
    result = await _buy(engine, opp_id)

    assert result is None
    assert engine.last_buy_reject_reason == "POST_FILL_BIND_FAILED"
    assert "ETH/USDT" not in engine.open_positions
    buy = _one(engine.db_path, "SELECT paper_run_id, order_id, remaining_position FROM paper_trades WHERE side='BUY'")
    assert buy[:2] == ("scalp_v2_live", ORDER_ID)
    assert buy[2] > 0
    assert _one(engine.db_path, "SELECT COUNT(*) FROM day_entry_reservations WHERE status='ACTIVE'")[0] == 0


def _reconcile_engine(tmp_path, monkeypatch):
    engine = _engine(tmp_path, monkeypatch, AsyncMock(return_value=None))
    engine._symbol_in_fixed_universe = lambda _s: True
    engine._ensure_symbol_constraints = AsyncMock(return_value=None)
    engine._symbol_constraints = {"ETH/USDT": {"min_notional": 10.0, "qty_step": 0.0001}}
    live = MagicMock()
    live.get_market_price = AsyncMock(return_value={"price": 2700.0})
    engine._live_service = live
    engine._dust_check = lambda _sym, _qty, _px: (False, None, "", None)
    engine._retain_exchange_dust = AsyncMock(return_value=None)
    engine._recompute_positions_values = AsyncMock(return_value=None)
    engine._persist_ledger_to_sqlite = AsyncMock(return_value=None)
    return engine


def _seed_buy(db_path: str, *, remaining: float, run: str = "scalp_v2_live", order_id: str = ORDER_ID) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO paper_trades
            (trade_id, paper_run_id, mode, symbol, side, quantity, price, remaining_position,
             stop_price, atr_at_entry, fees_paid, timestamp, entry_timestamp, status, strategy_id, order_id, decision_id)
        VALUES ('scalp_v2_ETHUSDT_1', ?, 'live', 'ETH/USDT', 'BUY', 0.0094981, 2676.23, ?,
                0, 10.0, 0.005, '2026-09-25T04:04:22+00:00', '2026-09-25T04:04:22+00:00', 'executed', 'SCALP_V2', ?, 'opp-eth')
        """,
        (run, remaining, order_id),
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_reconciliation_restores_known_mystic_fill_instead_of_protecting_it(tmp_path, monkeypatch):
    engine = _reconcile_engine(tmp_path, monkeypatch)
    _seed_buy(engine.db_path, remaining=0.0094981)

    await engine._import_missing_exchange_positions({"ETH": 0.0094981}, {"ETH": 0.0094981})

    pos = engine.open_positions["ETH/USDT"]
    assert pos.engine_id == "SCALP_V2"
    assert pos.entry_order_id == ORDER_ID
    assert pos.quantity == pytest.approx(0.0094981, abs=1e-12)
    row = _one(engine.db_path, "SELECT engine_id, entry_order_id, trade_id FROM portfolio_engine_positions WHERE symbol='ETH/USDT'")
    assert row == ("SCALP_V2", ORDER_ID, "scalp_v2_ETHUSDT_1")
    assert _one(engine.db_path, "SELECT COUNT(*) FROM protected_external_inventory")[0] == 0


@pytest.mark.asyncio
async def test_true_external_holding_stays_protected(tmp_path, monkeypatch):
    engine = _reconcile_engine(tmp_path, monkeypatch)

    await engine._import_missing_exchange_positions({"ETH": 0.02}, {"ETH": 0.02})

    assert "ETH/USDT" not in engine.open_positions
    assert _one(engine.db_path, "SELECT quantity FROM protected_external_inventory WHERE symbol='ETH/USDT'")[0] == pytest.approx(0.02)
    assert _one(engine.db_path, "SELECT COUNT(*) FROM portfolio_engine_positions")[0] == 0


@pytest.mark.asyncio
async def test_restore_uses_only_the_unsold_remainder_and_protects_the_rest(tmp_path, monkeypatch):
    engine = _reconcile_engine(tmp_path, monkeypatch)
    _seed_buy(engine.db_path, remaining=0.005)

    await engine._import_missing_exchange_positions({"ETH": 0.0150}, {"ETH": 0.0150})

    assert engine.open_positions["ETH/USDT"].quantity == pytest.approx(0.005)
    assert _one(engine.db_path, "SELECT quantity FROM protected_external_inventory WHERE symbol='ETH/USDT'")[0] == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_fully_sold_or_dust_lot_is_not_restored(tmp_path, monkeypatch):
    from backend.services.protected_external_inventory import unsold_scalp_v2_lot

    engine = _reconcile_engine(tmp_path, monkeypatch)
    _seed_buy(engine.db_path, remaining=0.0)
    conn = sqlite3.connect(engine.db_path)
    assert unsold_scalp_v2_lot(conn, "ETH/USDT") is None
    conn.execute("UPDATE paper_trades SET remaining_position=0.0000981")
    conn.commit()
    conn.close()

    assert await engine._restore_unsold_scalp_lot("ETH/USDT", 0.0000981, 2700.0, 10.0) == 0.0
    assert "ETH/USDT" not in engine.open_positions


def test_only_live_scalp_rows_with_a_venue_order_count_as_provenance(tmp_path):
    from backend.services.protected_external_inventory import unsold_scalp_v2_lot

    db = str(tmp_path / "prov.db")
    _schema(db)
    _seed_buy(db, remaining=0.009, run="day_live")
    conn = sqlite3.connect(db)
    assert unsold_scalp_v2_lot(conn, "ETH/USDT") is None
    conn.execute("UPDATE paper_trades SET paper_run_id='scalp_v2_live', order_id=''")
    assert unsold_scalp_v2_lot(conn, "ETH/USDT") is None
    conn.execute("UPDATE paper_trades SET order_id=?", (ORDER_ID,))
    lot = unsold_scalp_v2_lot(conn, "ETH/USDT")
    conn.close()
    assert lot is not None
    assert lot["order_id"] == ORDER_ID
    assert lot["remaining"] == pytest.approx(0.009)
